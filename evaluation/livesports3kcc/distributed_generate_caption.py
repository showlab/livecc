import os
import json
import tqdm
import shutil
import argparse
import transformers
import multiprocessing
from functools import partial
from datasets import load_dataset
from utils.multiprocessor import local_mp
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor

def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed offline caption generation over the LiveSports-3K CC split"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, required=True,
        help="HuggingFace model path, e.g., Qwen/Qwen2.5-VL-7B-Instruct"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8,
        help="Number of parallel processes/gpus to use"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="evaluation/livesports3kcc/captions",
        help="Directory to write generated JSON outputs"
    )
    return parser.parse_args()

def caption_worker(
    device_id: int,
    model_name_or_path: str,
    save_dir: str,
    num_workers: int
):
    ds = load_dataset('stdKonjac/LiveSports-3K', name='LiveSports_3K_CC', split="test")
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = getattr(transformers, config.architectures[0]).from_pretrained(
        model_name_or_path, 
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        device_map=f'cuda:{device_id}'
    )
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    idxs = list(range(len(ds)))
    idxs_on_device = idxs[device_id::num_workers]

    # Prepare temporary save folder for this model
    os.makedirs(save_dir, exist_ok=True)

    for idx in tqdm.tqdm(idxs_on_device, desc=f"Device {device_id}", total=len(idxs_on_device)):
        save_path = os.path.join(save_dir, f"{idx}.json")
        if os.path.exists(save_path):
            continue

        record = ds[idx]
        video = record.get("video")
        video_id = record.get("video_id")
        event_id = record.get("event_id")
        video_start = record.get("begin")
        video_end = record.get("end")
        title = record.get("event_title")
        preasr = record.get("preasr_text")

        commentary_prompt = (
            "You are an expert video commentator providing real-time, insightful, "
            "and engaging commentary on visual content.\n"
        )
        overall_prompt = commentary_prompt
        if title:
            overall_prompt += f"This is a video titled \"{title}\".\n"
        if preasr:
            overall_prompt += f"Here is previous commentary of the video:\n\n{preasr}\n\n"
            overall_prompt += "Please continue to comment the video."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video},
                    {"type": "text", "text": overall_prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        output_ids = model.generate(**inputs, max_new_tokens=512)[0, inputs.input_ids.size(1):]
        caption = processor.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        with open(save_path, 'w') as wf:
            json.dump({
                "video_id": video_id,
                'event_id': event_id,
                "begin": video_start,
                "end": video_end,
                "pred": caption
            }, wf)

if __name__ == "__main__":
    args = parse_args()
    multiprocessing.set_start_method('spawn', force=True)
    save_dir = os.path.join(args.output_dir, os.path.basename(args.model_name_or_path))
    worker_fn = partial(
        caption_worker,
        model_name_or_path=args.model_name_or_path,
        save_dir=save_dir,
        num_workers=args.num_workers,
    )
    local_mp(
        list(range(args.num_workers)),
        worker_fn,
        desc="caption_generation",
        num_workers=args.num_workers
    )
    # jsons -> jsonl
    save_path = save_dir + '.jsonl'
    with open(save_path, 'w') as wf:
        for file in os.listdir(save_dir):
            datum = json.load(open(os.path.join(save_dir, file))) 
            wf.write(json.dumps(datum) + '\n')
    # remove save_dir
    shutil.rmtree(save_dir)

