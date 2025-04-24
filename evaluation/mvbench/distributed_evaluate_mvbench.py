import torch, json, functools, tqdm, random, sys, os
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, logging, Qwen2VLForConditionalGeneration, AutoProcessor

from livecc_utils import _read_video_decord_plus, _spatial_resize_video
from qwen_vl_utils.vision_process import process_vision_info, smart_nframes, FPS
from data.lmm_dataset import bytes_to_pil, pil_to_tensor
from data.tos import tos_loader

logger = logging.get_logger(__name__)

def save_function_print(function: callable, save_path: str, *args, **kwargs):
    original_stdout = sys.stdout
    try:
        with open(save_path, 'w') as f:
            sys.stdout = f  
            function(*args, **kwargs)          
    finally:
        sys.stdout = original_stdout 

class MVBenchMCQDataset(Dataset):
    def __init__(self, remote_loader, path, question_prefix, question_postfix, answer_prefix, sample: int = None):
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
        
    def __len__(self):
        return len(self.datums)

    def __getitem__(self, i):
        datum = self.datums[i]
        query = self.question_prefix + datum['question'] + '\n' + '\n'.join(datum['options']) + self.question_postfix
        conversation = [{"role": "user", "content": []}]
        video_inputs = None
        if 'tos_key' in datum:
            if 'tvqa' in datum['tos_key']:
                nframes = smart_nframes({'fps': FPS}, total_frames=len(datum['frames']), video_fps=FPS) # suggest this has been fpsed
                sampler = torch.linspace(0, len(datum['frames']) - 1, nframes).round().long()
                images = [pil_to_tensor(bytes_to_pil(tos_loader(os.path.join(datum['tos_key'], datum['frames'][i]), return_io=False, length_check=True))) for i in sampler]
                video = torch.stack(images)
                video = _spatial_resize_video(video)
                conversation[0]['content'].append({"type": "video", "video": video})
                video_inputs = [video]
            else:
                conversation[0]['content'].append(
                    {"type": "video", "video": datum['tos_key'], 'remote_loader': self.remote_loader},
                )
        conversation[0]['content'].append({"type": "text", "text": query})
        if video_inputs is None:
            for _ in range(10):
                try:
                    _, video_inputs = process_vision_info(conversation)
                    break
                except:
                    print(f"{_}-th process_vision_info failed. retry...")
        return conversation, video_inputs[0]

    def data_collator(self, batch, processor):
        conversations, video_inputs = zip(*batch)
        texts = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        texts = [text + self.answer_prefix for text in texts]
        inputs = processor(
            text=texts,
            images=None,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs

def preprocess_logits_for_metrics(logits, labels, strict_letter_ids): 
    return torch.stack([logit[(logit[:, 0] != -100).nonzero().squeeze()[-1], strict_letter_ids] for logit in logits]).argmax(dim=-1)

def mcq_predict(
    model, 
    processor, 
    benchmark_path: str, 
    letters: list[str], 
    remote_loader: callable,
    question_prefix: str = '', 
    question_postfix: str = '\nPlease select the correct answer.', 
    answer_prefix: str = 'Answer:', 
    abcd_previous_str: str = ': ',
    use_liger_kernel: bool = True,
    per_device_eval_batch_size: int = 2,
    dataloader_num_workers: int = 4,
):
    strict_letter_ids = [processor.tokenizer(f'{abcd_previous_str}{_}').input_ids[-1] for _ in letters] 
    dataset = MVBenchMCQDataset(remote_loader, benchmark_path, question_prefix=question_prefix, question_postfix=question_postfix, answer_prefix=answer_prefix)
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

def evaluate_mvbench_results(results: list):
    task_type_to_counts = {}
    for video_item in results:
        for question_item in video_item['questions']:
            task_type = question_item['task_type']
            if task_type not in task_type_to_counts:
                task_type_to_counts[task_type] = {'correct': 0, 'total': 0}
            task_type_to_counts[task_type]['total'] += 1
            if question_item['response'][0] == question_item['answer']:
                task_type_to_counts[task_type]['correct'] += 1
    accs = []
    for task_type, counts in task_type_to_counts.items():
        print(f'{task_type}: {counts["correct"]}/{counts["total"]}={counts["correct"]/counts["total"]}')
        accs.append(counts["correct"]/counts["total"])
    print(f'Average: {sum(accs)/len(accs)}')

if __name__ == '__main__':
    model_path = "chenjoya/LiveCC-7B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", attn_implementation='flash_attention_2')
    processor = AutoProcessor.from_pretrained(model_path, padding_side='left')
    letter_idxs_predictions, benchmark_datums, process_index = mcq_predict(
        model=model, processor=processor, benchmark_path='mvbench_video_existed.jsonl',
        remote_loader=tos_loader, letters=['A', 'B', 'C', 'D', 'E'], use_liger_kernel='LiveCC' in model_path,
    )
    if process_index == 0:
        video_to_results = {}
        for datum, letter_idx_prediction in zip(benchmark_datums, letter_idxs_predictions):
            video = datum['video']
            if video not in video_to_results:
                video_to_results[video] = {
                    'video': video,
                    'questions': [],
                }
            video_to_results[video]['questions'].append(
                {
                    "task_type": datum['task_type'],
                    "question": datum['question'],
                    "options": datum['options'],
                    "answer": datum['answer'],
                    "response": datum['options'][letter_idx_prediction],
                },
            )
        results = list(video_to_results.values())
        save_json_path = f'evaluation/mvbench/results/{os.path.basename(model_path)}.json'
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        json.dump(results, open(save_json_path, 'w'))
        save_txt_path = save_json_path.replace('.json', '.txt')
        save_function_print(
            evaluate_mvbench_results,
            save_txt_path,
            results
        )

# torchrun --standalone --nproc_per_node=8 evaluation/mvbench/distributed_evaluate_mvbench.py