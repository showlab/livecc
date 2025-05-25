import torch, json, functools, tqdm, random, os
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, logging

from livecc_utils import _read_video_decord_plus
from qwen_vl_utils import process_vision_info
# import qwen_omni_utils
# qwen_omni_utils.vision_process.FPS_MAX_FRAMES = int(os.environ.get('FPS_MAX_FRAMES', 480)) # decrease this for gpu memory... 
# from qwen_omni_utils.v2_5.vision_process import VIDEO_READER_BACKENDS
# VIDEO_READER_BACKENDS['decord+'] = _read_video_decord_plus
# from qwen_omni_utils import process_mm_info

logger = logging.get_logger(__name__)

class MCQDataset(Dataset):
    def __init__(self, remote_loader, path, question_prefix, question_postfix, answer_prefix, with_subtitles: bool = False, sample: int = None):
        lines = open(path).readlines()
        if sample is not None:
            random.seed(42)
            lines = random.sample(lines, sample)
        self.datums = [json.loads(line) for line in tqdm.tqdm(lines, desc='load datums')]
        if isinstance(self.datums[0], str):
            self.datums = [json.loads(datum) for datum in tqdm.tqdm(self.datums, desc='load datumsx2')]
        self.question_prefix = question_prefix
        self.question_postfix = question_postfix
        self.answer_prefix = answer_prefix
        self.remote_loader = remote_loader
        self.with_subtitles = with_subtitles
        
    def __len__(self):
        return len(self.datums)
        
    def __getitem__(self, i):
        datum = self.datums[i]
        query = self.question_prefix + datum['question'] + '\n' + '\n'.join(datum['options']) + self.question_postfix
        conversation = [{"role": "user", "content": []}]
        if 'video' in datum:
            conversation[0]['content'].append(
                {"type": "video", "video": datum['video'], 'remote_loader': self.remote_loader},
            )
            if 'video_start' in datum:
                conversation[0]['content'][-1]['video_start'] = datum['video_start']
            if 'video_end' in datum:
                conversation[0]['content'][-1]['video_end'] = datum['video_end']
        if not self.with_subtitles:
            conversation[0]['content'].append({"type": "text", "text": query})
        else:
            query = f"This video's subtitles are listed below:\n{datum['subtitles']}\nAccording to the video and subtitles, "  + query
            conversation[0]['content'].append({"type": "text", "text": query})
        return conversation

    def data_collator(self, conversations, processor):
        texts = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        texts = [text + self.answer_prefix for text in texts]
        for _ in range(10):
            try:
                image_inputs, video_inputs = process_vision_info(conversations)
                # audio_inputs, image_inputs, video_inputs = process_mm_info(conversations, use_audio_in_video=False)
                break
            except:
                print(f"{_}-th process_vision_info failed. retry...")
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            # use_audio_in_video=False
        )
        return inputs

def preprocess_logits_for_metrics(logits, labels, strict_letter_ids): 
    return torch.stack([logit[(logit[:, 0] != -100).nonzero().squeeze()[-1], strict_letter_ids] for logit in logits]).argmax(dim=-1)

def mcq_predict(
    model, 
    processor, 
    benchmark_path: str, 
    letters: list[str], 
    remote_loader: callable = None,
    question_prefix: str = '', 
    question_postfix: str = '\nPlease select the correct answer.', 
    answer_prefix: str = 'Answer:', 
    abcd_previous_str: str = ': ',
    use_liger_kernel: bool = True,
    per_device_eval_batch_size: int = 2,
    dataloader_num_workers: int = 4,
    with_subtitles: bool = False
):
    strict_letter_ids = [processor.tokenizer(f'{abcd_previous_str}{_}').input_ids[-1] for _ in letters] 
    dataset = MCQDataset(remote_loader, benchmark_path, question_prefix=question_prefix, question_postfix=question_postfix, answer_prefix=answer_prefix, with_subtitles=with_subtitles)
    trainer = Trainer(
        model=model, 
        args=TrainingArguments(
            output_dir='outputs/', do_predict=True, 
            per_device_eval_batch_size=per_device_eval_batch_size, 
            dataloader_num_workers=dataloader_num_workers, 
            report_to='none', use_liger_kernel=use_liger_kernel
        ), 
        data_collator=functools.partial(dataset.data_collator, processor=processor),
        processing_class=processor,
        preprocess_logits_for_metrics=functools.partial(preprocess_logits_for_metrics, strict_letter_ids=strict_letter_ids),
    )
    letter_idxs_predictions = trainer.predict(dataset, ignore_keys=['past_key_values', 'hidden_states', 'attentions', 'rope_deltas']).predictions
    return letter_idxs_predictions, dataset.datums, trainer.args.process_index
