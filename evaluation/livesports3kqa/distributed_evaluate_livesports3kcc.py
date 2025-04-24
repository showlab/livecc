import json, os
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor

from data.tos import tos_loader
from evaluation.distributed_mcq_predictor import mcq_predict, save_function_print

def evaluate_livesports3k_results(results: list):
    q_type_to_counts = {}
    ocr_to_counts = {'correct': 0, 'total': 0}
    for video_item in results:
        for question_item in video_item['questions']:
            q_type = question_item['q_type']
            if q_type not in q_type_to_counts:
                q_type_to_counts[q_type] = {'correct': 0, 'total': 0}
            if question_item['OCR'] == 1:
                ocr_to_counts['total'] += 1
            q_type_to_counts[q_type]['total'] += 1
            if question_item['response'][0] == question_item['answer']:
                q_type_to_counts[q_type]['correct'] += 1
                if question_item['OCR'] == 1:
                    ocr_to_counts['correct'] += 1
    correct, total = 0, 0
    for q_type, counts in q_type_to_counts.items():
        correct += counts['correct']
        total += counts['total']
        print(f'{q_type}: {counts["correct"]}/{counts["total"]}={counts["correct"]/counts["total"]}')
    print(f'OCR: {ocr_to_counts["correct"]}/{ocr_to_counts["total"]}={ocr_to_counts["correct"]/ocr_to_counts["total"]}')
    print(f'Overall: {correct}/{total}={correct/total}')

if __name__ == '__main__':
    model_path = "chenjoya/LiveCC-7B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", attn_implementation='flash_attention_2')
    processor = AutoProcessor.from_pretrained(model_path if model_path != 'Qwen/Qwen2-VL-7B' else 'Qwen/Qwen2-VL-7B-Instruct', padding_side='left')
    letter_idxs_predictions, benchmark_datums, process_index = seq_predict(
        model=model, processor=processor, benchmark_path='sports3k-qa.jsonl', 
        remote_loader=tos_loader,letters=['A', 'B', 'C', 'D'], use_liger_kernel='LiveCC' in model_path
    )
    if process_index == 0:
        video_id_to_results = {}
        for datum, letter_idx_prediction in zip(benchmark_datums, letter_idxs_predictions):
            video_id = datum['video_id']
            if video_id not in video_id_to_results:
                video_id_to_results[video_id] = {
                    'video_id': video_id,
                    'questions': [],
                }
            video_id_to_results[video_id]['questions'].append(
                {
                    "question_id": datum['question_id'],
                    "q_type": datum['q_type'],
                    'OCR': datum['OCR'],
                    "question": datum['question'],
                    "options": datum['options'],
                    "answer": datum['answer'],
                    "response": datum['options'][letter_idx_prediction],
                },
            )
        results = list(video_id_to_results.values())
        save_json_path = f'evaluation/livesports3k/qa_results/{os.path.basename(model_path)}.json'
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        json.dump(results, open(save_json_path, 'w'))
        save_txt_path = save_json_path.replace('.json', '.txt')
        save_function_print(
            evaluate_livesports3k_results,
            save_txt_path,
            results
        )

# torchrun --standalone --nproc_per_node=8 evaluation/livesports3k/distributed_evaluate_livesports3kqa.py