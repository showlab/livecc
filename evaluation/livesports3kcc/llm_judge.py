import os, openai, json, functools
from utils.multiprocessor import local_mt

model_to_predir = {
    'GPT-4o': 'evaluation/livesports3kcc/captions/GPT-4o.jsonl',
    'Gemini-1.5-pro': 'evaluation/livesports3kcc/captions/Gemini-1.5-pro.jsonl',
    'VideoLLaMA2-72B': 'evaluation/livesports3kcc/captions/VideoLLaMA2-72B.jsonl',
    'llava-onevision-qwen2-72b-ov-sft': 'evaluation/livesports3kcc/captions/llava-onevision-qwen2-72b-ov-sft.jsonl',
    'LLaVA-Video-72B-Qwen2': 'evaluation/livesports3kcc/captions/LLaVA-Video-72B-Qwen2.jsonl',
    'internlm-xcomposer2d5-7b': 'evaluation/livesports3kcc/captions/internlm-xcomposer2d5-7b.jsonl',
    'LLaVA-Video-7B-Qwen2': 'evaluation/livesports3kcc/captions/LLaVA-Video-7B-Qwen2.jsonl',
    'llava-onevision-qwen2-7b-ov': 'evaluation/livesports3kcc/captions/llava-onevision-qwen2-7b-ov.jsonl',
    'Qwen2.5-VL-7B-Instruct': 'evaluation/livesports3kcc/captions/Qwen2.5-VL-7B-Instruct.jsonl',
    'Qwen2-VL-7B-Instruct': 'evaluation/livesports3kcc/captions/Qwen2-VL-7B-Instruct.jsonl',
    'Qwen2.5-Omni-7B': 'evaluation/livesports3kcc/captions/Qwen2.5-Omni-7B.jsonl',
    'Qwen2-VL-7B': 'evaluation/livesports3kcc/captions/Qwen2-VL-7B.jsonl',
    'Qwen2-VL-72B-Instruct': 'evaluation/livesports3kcc/captions/Qwen2-VL-72B-Instruct.jsonl',
    'Qwen2.5-VL-72B-Instruct': 'evaluation/livesports3kcc/captions/Qwen2.5-VL-72B-Instruct.jsonl',
    'Qwen2-VL-7B-LLaVAInstruct': 'evaluation/livesports3kcc/captions/Qwen2-VL-72B-Instruct.jsonl',
    'Qwen2-VL-7B-LiveCCInstruct': 'evaluation/livesports3kcc/livecc/Qwen2-VL-7B-LiveCCInstruct.jsonl',
    'LiveCC-7B-Base': 'evaluation/livesports3kcc/livecc/LiveCC-7B-Base.jsonl',
    'LiveCC-7B-Instruct': 'evaluation/livesports3kcc/livecc/LiveCC-7B-Instruct.jsonl',
}

baseline_id = 'GPT-4o'
baseline_jsonl = model_to_predir[baseline_id]
video_event_id_to_baseline_pred, video_event_id_to_gt_asr, video_start_end_to_video_event_id = {}, {}, {}

for line in open(baseline_jsonl):
    datum = json.loads(line)
    video_event_id = datum['video_id'] + '_' + str(datum['event_id'])
    video_event_id_to_gt_asr[video_event_id] = datum['gt_asr']
    video_start_end = (datum['video_id'], datum['begin'], datum['end'])
    video_start_end_to_video_event_id[video_start_end] = video_event_id
    video_event_id_to_baseline_pred[video_event_id] = datum['pred']

gpt = openai.AzureOpenAI(
    azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
    api_version="2024-08-06",
    api_key=os.environ.get('AZURE_OPENAI_API_KEY')
)

def judge_ab(a_id_with_pred, b_id_with_pred, gt_asr):
    a_id, a_pred = a_id_with_pred
    b_id, b_pred = b_id_with_pred
    ab_prompt = (
        'You are an expert in video commentary. '
        'Your task is to review two commentaries (Commentary A and Commentary B), and select the one that better aligns with the human commentary. '
        'You should consider the criteria:\n'
        '1. Semantic Alignment: The commentary should convey the same meaning, details, and key points as the human commentary.\n'
        'If the above criteria is not enough to judge, then consider:\n'
        '2. Stylistic Consistency: The commentary should maintain a tone, word choice, and structure similar to the human commentary.\n'
        f'\n---Commentary A---\n{a_pred}\n----------\n'
        f'\n---Commentary B---\n{b_pred}\n----------\n'
        f'\n---Human Commentary---\n{gt_asr}\n----------\n'
        '\nYour response should be "Commentary A is better aligned with the human commentary" or "Commentary B is better aligned with the human commentary".\n'
    )
    while True:
        try:
            ab_resp = gpt.chat.completions.create(
                model='gpt-4o-2024-08-06',
                messages=[{"role": "user", "content": [{'type': 'text', 'text': ab_prompt}]}],
                seed=42,
                temperature=0,
            ).choices[0].message.content
            break
        except Exception as e:
            print('Failed to get response...', e)
    if 'Commentary A' in ab_resp:
        ab_winner = a_id
    elif 'Commentary B' in ab_resp:
        ab_winner = b_id
    else:
        ab_winner = 'tie'
    return ab_winner

def judge(item, model_id):
    video_event_id, model_pred = item
    gt_asr = video_event_id_to_gt_asr[video_event_id]
    baseline_pred = video_event_id_to_baseline_pred[video_event_id]
    return {
        'video_event_id': video_event_id, 
        'ab_winner': judge_ab([model_id, model_pred], [baseline_id, baseline_pred], gt_asr), 
        'ba_winner': judge_ab([baseline_id, baseline_pred], [model_id, model_pred], gt_asr)
    }

if __name__ == '__main__':
    for model_id in model_to_predir:
        print(f'{model_id} vs. {baseline_id}')
        save_path = f'./evaluation/livesports3kcc/judges/{baseline_id}_{model_id}.jsonl'
        if os.path.exists(save_path):
            print(f'{save_path} exists, skip')
            continue
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model_dir_or_jsonl = model_to_predir[model_id]
        video_event_id_to_model_pred = {}
        if os.path.exists(model_dir_or_jsonl) and os.path.isfile(model_dir_or_jsonl):
            for line in open(model_dir_or_jsonl):
                datum = json.loads(line)
                video_event_id = datum['video_id'] + '_' + str(datum['event_id'])
                video_event_id_to_model_pred[video_event_id] = datum['pred']
        else:
            print(f'{model_id} not found, skip')
            continue

        for video_event_id in video_event_id_to_baseline_pred:
            assert video_event_id_to_model_pred[video_event_id] is not None
        
        winner_results = local_mt(
            video_event_id_to_model_pred.items(), 
            functools.partial(judge, model_id=model_id), 
            desc=f'Call gpt4o for {model_id} vs. {baseline_id}', 
            num_workers=16
        )
        
        with open(save_path, 'w') as f:
            for winner_result in winner_results:
                f.write(json.dumps(winner_result) + '\n')
        
        win_count, count = 0, 0
        for winner_result in winner_results:
            if winner_result['ab_winner'] == model_id:
                win_count += 1
            if winner_result['ba_winner'] == model_id:
                win_count += 1
            count += 2
        
        win_rate = win_count / count * 100
        output = f'Winning Rate for {model_id} vs. {baseline_id}: {win_rate:.2f}%'
        print(output)
        with open('evaluation/livesports3kcc/judges/judges.txt', 'a') as f:
            f.write(output + '\n')