import json
import os
import argparse
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from data.tos import videomme_tos_loader
from evaluation.videomme.eval_your_results import eval_your_results
from evaluation.distributed_mcq_predictor import mcq_predict
from evaluation.utils import save_function_print

def main():
    parser = argparse.ArgumentParser(
        description="Distributed evaluation for VideoMME models"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path or identifier of the pretrained model"
    )
    parser.add_argument(
        "--benchmark_path", type=str, required=True,
        help="Path to the benchmark JSONL file"
    )
    parser.add_argument(
        "--with_subtitles", action="store_true",
        help="Flag to indicate evaluation on subtitles-enabled benchmark"
    )
    args = parser.parse_args()

    # Load model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        attn_implementation="flash_attention_2"
    )
    processor_name = (
        args.model_path
        if args.model_path != 'Qwen/Qwen2-VL-7B'
        else 'Qwen/Qwen2-VL-7B-Instruct'
    )
    processor = AutoProcessor.from_pretrained(
        processor_name,
        padding_side='left'
    )

    # Run distributed prediction
    letter_idxs_predictions, benchmark_datums, process_index = mcq_predict(
        model=model,
        processor=processor,
        benchmark_path=args.benchmark_path,
        remote_loader=videomme_tos_loader,
        letters=['A', 'B', 'C', 'D'],
        use_liger_kernel='LiveCC' in args.model_path,
        per_device_eval_batch_size=1,
        with_subtitles=args.with_subtitles
    )

    # Only process rank 0 for result aggregation and saving
    if process_index == 0:
        video_id_to_results = {}
        for datum, letter_idx_prediction in zip(
            benchmark_datums, letter_idxs_predictions
        ):
            vid = datum['video_id']
            if vid not in video_id_to_results:
                video_id_to_results[vid] = {
                    'video_id': vid,
                    'duration': datum['duration'],
                    'domain': datum['domain'],
                    'sub_category': datum['sub_category'],
                    'questions': [],
                }
            video_id_to_results[vid]['questions'].append({
                "question_id": datum['question_id'],
                "task_type": datum['task_type'],
                "question": datum['question'],
                "options": datum['options'],
                "answer": datum['answer'],
                "response": datum['options'][letter_idx_prediction],
            })

        results = list(video_id_to_results.values())

        # Determine output paths
        suffix = 'with_subtitles' if args.with_subtitles else 'no_subtitles'
        out_dir = f'evaluation/videomme/results'
        os.makedirs(out_dir, exist_ok=True)
        save_json_path = os.path.join(
            out_dir,
            f"{os.path.basename(args.model_path)}_{suffix}.json"
        )

        # Save JSON
        with open(save_json_path, 'w') as f:
            json.dump(results, f)

        # Save evaluation text report
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

if __name__ == '__main__':
    main()