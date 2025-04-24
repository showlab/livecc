import json, os
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from data.tos import videomme_tos_loader
from evaluation.videomme.eval_your_results import eval_your_results
from evaluation.distributed_mcq_predictor import mcq_predict
from evaluation.utils import save_function_print

if __name__ == '__main__':
    model_path = "chenjoya/LiveCC-7B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", attn_implementation='flash_attention_2')
    processor = AutoProcessor.from_pretrained(model_path if model_path != 'Qwen/Qwen2-VL-7B' else 'Qwen/Qwen2-VL-7B-Instruct', padding_side='left')
    benchmark_path = 'videomme.jsonl' # _with_subtitles.jsonl
    letter_idxs_predictions, benchmark_datums, process_index = mcq_predict(
        model=model, processor=processor, benchmark_path=benchmark_path, 
        remote_loader=videomme_tos_loader, letters=['A', 'B', 'C', 'D'], use_liger_kernel='LiveCC' in model_path,
        per_device_eval_batch_size=2,
    )
    if process_index == 0:
        video_id_to_results = {}
        for datum, letter_idx_prediction in zip(benchmark_datums, letter_idxs_predictions):
            video_id = datum['video_id']
            if video_id not in video_id_to_results:
                video_id_to_results[video_id] = {
                    'video_id': video_id,
                    'duration': datum['duration'],
                    'domain': datum['domain'],
                    'sub_category': datum['sub_category'],
                    'questions': [],
                }
            video_id_to_results[video_id]['questions'].append(
                {
                    "question_id": datum['question_id'],
                    "task_type": datum['task_type'],
                    "question": datum['question'],
                    "options": datum['options'],
                    "answer": datum['answer'],
                    "response": datum['options'][letter_idx_prediction],
                },
            )
        results = list(video_id_to_results.values())
        if 'subtitles' in benchmark_path:
            save_json_path = f'evaluation/videomme/results_with_subtitles/{os.path.basename(model_path)}.json'
        else:
            save_json_path = f'evaluation/videomme/results/{os.path.basename(model_path)}.json'
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        json.dump(results, open(save_json_path, 'w'))
        save_txt_path = save_json_path.replace('.json', '.txt')
        save_function_print(
            eval_your_results, 
            save_txt_path,
            save_json_path, 
            video_types=['short', 'medium', 'long'], 
            return_categories_accuracy=True, 
            return_sub_categories_accuracy=True, 
            return_task_types_accuracy=True,
        )

# torchrun --standalone --nproc_per_node=8 evaluation/videomme/distributed_evaluate_videomme.py