import json, tqdm, os, multiprocessing
from demo.infer import LiveCCInfer
from data.tos import tos_loader
from utils.multiprocessor import local_mp

model_path = 'chenjoya/Qwen2-VL-7B-LiveCCInstruct'
lines = open('sports3k-cc.jsonl').readlines()
idxs = list(range(len(lines)))
dirpath = 'evaluation/livesports3kcc/commentaries'

def livecc_on_device(device_id: int):
    idxs_on_device = idxs[device_id::8]
    infer = LiveCCInfer(model_path=model_path, device_id=device_id)
    for idx in tqdm.tqdm(idxs_on_device, desc=f'livecc_on_device{device_id}'):
        save_json_path = os.path.join(dirpath, os.path.basename(model_path), f'{idx}.json')
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        if os.path.exists(save_json_path):
            continue
        conversation = json.loads(lines[idx])
        video, video_start, video_end = conversation[0]['content'][0]['video'], conversation[0]['content'][0]['video_start'], conversation[0]['content'][0]['video_end']
        title, preasr = conversation[0]['content'][1]['title'], conversation[0]['content'][1]['previous']
        commentary_prompt = "You are an expert video commentator providing real-time, insightful, and engaging commentary on visual content.\n"
        overall_prompt = commentary_prompt
        if title is not None:
            title_prompt = f"This is a video titled with \"{title}\".\n"
            overall_prompt += title_prompt
        if preasr is not None:
            asr_pre_prompt = f"Here is previous commentary of the video:\n\n{preasr}\n\n"
            overall_prompt += asr_pre_prompt
            asr_post_prompt = f"Please continue to comment the video."
            overall_prompt += asr_post_prompt
        responses = infer.live_cc_once(
            query=overall_prompt, 
            video=video, video_start=video_start, video_end=video_end, remote_loader=tos_loader,
            do_sample=False, top_p=None, top_k=None, temperature=None, max_new_tokens=32, repetition_penalty=1.15
        )
        overall_cc = ' '.join(cc.replace(' ...', '') for start, stop, cc in responses if cc).strip() + '...'
        json.dump(dict(video=video, video_start=video_start, video_end=video_end, commentary=overall_cc), open(save_json_path, 'w'))

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    local_mp(range(8), livecc_on_device, desc='livecc_on_device', num_workers=8)