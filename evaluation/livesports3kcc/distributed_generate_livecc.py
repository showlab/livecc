import argparse
import os
import json
import tqdm
import shutil
import multiprocessing
from functools import partial
from datasets import load_dataset, Dataset
from demo.infer import LiveCCDemoInfer
from data.tos import tos_loader
from utils.multiprocessor import local_mp

def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed LiveCC generation over the LiveSports-3K CC split"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, required=True,
        help="HuggingFace model path, e.g., chenjoya/LiveCC-7B-Instruct"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8,
        help="Number of parallel processes/gpus to use"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="evaluation/livesports3kcc/livecc",
        help="Directory to write generated JSON outputs"
    )
    return parser.parse_args()

def livecc_worker(
    device_id: int,
    model_name_or_path: str,
    save_dir: str,
    num_workers: int
):
    ds_val = load_dataset('stdKonjac/LiveSports-3K', name='LiveSports_3K_CC', split="val")
    ds_test = load_dataset('stdKonjac/LiveSports-3K', name='LiveSports_3K_CC', split="test")
    records = [dict(r) for r in ds_val] + [dict(r) for r in ds_test]
    ds = Dataset.from_list(records)

    infer = LiveCCDemoInfer(model_name_or_path=model_name_or_path, device=f'cuda:{device_id}')
    total = len(ds)
    idxs = list(range(total))
    idxs_on_device = idxs[device_id::num_workers]

    # Prepare save folder for this model
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
        preasr = record.get("event_asr")

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

        responses = infer.live_cc_once_for_evaluation(
            query=overall_prompt,
            video=video, video_start=video_start, video_end=video_end,
            remote_loader=tos_loader,
            max_new_tokens=32,
            repetition_penalty=1.15
        )

        overall_cc = (
            ' '.join(cc.replace(' ...', '') for _, _, cc in responses if cc)
            .strip() + '...'
        )

        with open(save_path, 'w') as wf:
            json.dump({
                "video_id": video_id,
                'event_id': event_id,
                "begin": video_start,
                "end": video_end,
                "pred": overall_cc
            }, wf)

if __name__ == "__main__":
    args = parse_args()
    multiprocessing.set_start_method('spawn', force=True)
    save_dir = os.path.join(args.output_dir, os.path.basename(args.model_name_or_path))
    worker_fn = partial(
        livecc_worker,
        model_name_or_path=args.model_name_or_path,
        save_dir=save_dir,
        num_workers=args.num_workers
    )
    local_mp(
        list(range(args.num_workers)),
        worker_fn,
        desc="livecc_generation",
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

