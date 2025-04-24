import torch, json, functools, random, argparse, os
import decord
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
apply_liger_kernel_to_qwen2_vl()
from transformers import Trainer, Qwen2VLForConditionalGeneration, AutoProcessor, TrainingArguments, logging

from data.tos import get_tos_client, get_tos_bytes

logger = logging.get_logger(__name__)
tos_loader_with_length_check = functools.partial(get_tos_bytes, client=get_tos_client(), md5_check=True)

class VideoFramesDataset(Dataset):
    def __init__(self, videos: list[str], processor: AutoProcessor):
        self.videos = videos
        self.processor = processor

    def __len__(self):
        return len(self.videos)

    def getitem(self, i):
        video = self.videos[i]
        num_frames = 8
        reader = decord.VideoReader(tos_loader_with_length_check(video), num_threads=2)
        idxs = torch.linspace(0, len(reader) - 1, num_frames).round().long()
        images = torch.from_numpy(reader.get_batch(idxs).asnumpy()).permute(0, 3, 1, 2)
        images = resize(images, (320, 180), interpolation=InterpolationMode.BICUBIC, antialias=True)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image} for image in images
                ] + [{"type": "text", "text": f"Here are {num_frames} evenly sampled frames from a YouTube video. Are there someone always showing their faces and talking? Answer Yes or No."}]
            }
        ]
        return conversation, images

    def __getitem__(self, index):
        max_tries = 100
        for _ in range(max_tries):
            try:
                return self.getitem(index)
            except Exception as e:
                logger.warning(f"Failed {_}-th try to get item {index}: {e}")
                index = random.randint(0, self.__len__() - 1)
                logger.warning(f"Retrying to get item {index}")
        raise Exception(f"Failed to get item after {max_tries} retries")


def data_collator(batch, processor):
    conversations, image_inputs = zip(*batch)
    texts = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    return inputs

def preprocess_logits_for_metrics(logits, labels): 
    return logits[:, -1].softmax(dim=-1)[:, 9454]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_index', type=int, default=None)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--outputs', type=str, default='datasets/lmm4asd/')
    args = parser.parse_args()
    local = int(os.getenv('ARNOLD_ID')) if args.node_index is None else args.node_index
    print(f'{local=}, {args.num_nodes=}')
    data_path = 'livecc_30-240s_wps1-4_ppl1.5-6.5_videos.json'
    videos = json.load(open(data_path))
    # ---
    idxs = list(range(3, len(videos), 4))
    # ---
    idxs = [idxs[i] for i in range(local, len(idxs), args.num_nodes)]
    videos = [videos[i] for i in idxs]
    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto",  attn_implementation='flash_attention_2')
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", padding_side='left', use_fast=True)
    trainer = Trainer(
        model=model, 
        args=TrainingArguments(output_dir='outputs/', do_eval=True, per_device_eval_batch_size=64, dataloader_num_workers=8, report_to='none', do_predict=True), 
        data_collator=functools.partial(data_collator, processor=processor),
        processing_class=processor,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    chunk_size = 12500
    for i in range(0, len(videos), chunk_size):
        idxs_chunk = idxs[i:i+chunk_size]
        videos_chunk = videos[i:i+chunk_size]
        dataset = VideoFramesDataset(videos_chunk, processor)
        yesno_logits = trainer.predict(dataset, ignore_keys=['past_key_values', 'hidden_states', 'attentions', 'rope_deltas']).predictions
        yesno_logits = yesno_logits.tolist()
        yesno_logits_with_idxs = list(zip(idxs_chunk, yesno_logits))
        if trainer.args.process_index == 0:
            path = f'yesno_logits_with_idxs_part3_local{local}-{args.num_nodes}_chunk{i}+{chunk_size}.json'
            with open(path, 'w') as f:
                json.dump(yesno_logits_with_idxs, f)
            hput('./' + path, args.outputs)
    # deepspeed --num_gpus=8 --master_port=11822 data/production/distributed_lmm4asd.py