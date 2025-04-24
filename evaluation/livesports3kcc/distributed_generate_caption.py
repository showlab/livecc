import json, tqdm, os, multiprocessing
from demo.infer import LiveCCInfer
from data.tos import tos_loader
from utils.multiprocessor import local_mp

model_path = 'Qwen/Qwen2.5-VL-72B-Instruct'
lines = open('sports3k-cc.jsonl').readlines()
idxs = list(range(len(lines)))
dirpath = 'evaluation/livesports3kcc/captions'
num_workers = 1

def livecc_on_device(device_id: int):
    idxs_on_device = idxs[device_id::num_workers]
    infer = LiveCCInfer(model_path=model_path, device_id=device_id, use_liger_kernel='LiveCC' in model_path)
    for idx in tqdm.tqdm(idxs_on_device, desc=f'livecc_on_device{device_id}'):
        save_json_path = os.path.join(dirpath, os.path.basename(model_path), f'{idx}.json')
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        if os.path.exists(save_json_path):
            continue
        conversation = json.loads(lines[idx])
        video, video_start, video_end = conversation[0]['content'][0]['video'], conversation[0]['content'][0]['video_start'], conversation[0]['content'][0]['video_end']
        title, preasr = conversation[0]['content'][1]['title'], conversation[0]['content'][1]['previous']
        overall_prompt = "You are an expert video commentator providing real-time, insightful, and engaging commentary on visual content.\n"
        if title is not None:
            overall_prompt += f"This is a video titled with \"{title}\".\n"
        if preasr is not None:
            overall_prompt += f"Here is previous commentary of the video:\n\n{preasr}\n\n"
            overall_prompt += f"Please continue to comment the video."
        responses = infer.video_qa_once(
            query=overall_prompt, 
            video=video, video_start=video_start, video_end=video_end, remote_loader=tos_loader,
            do_sample=False, top_p=None, top_k=None, temperature=None, max_new_tokens=1024,
        )
        json.dump(dict(video=video, video_start=video_start, video_end=video_end, commentary=responses), open(save_json_path, 'w'))

if __name__ == '__main__':
    if num_workers > 1:
        multiprocessing.set_start_method('spawn', force=True)
        local_mp(range(num_workers), livecc_on_device, desc='livecc_on_device', num_workers=num_workers)
    else:
        livecc_on_device(0)