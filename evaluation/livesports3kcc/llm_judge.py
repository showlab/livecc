import os, openai, json, functools, argparse
from datasets import load_dataset
from utils.multiprocessor import local_mt

baseline_id = 'GPT-4o'
baseline_jsonl = 'evaluation/livesports3kcc/captions/GPT-4o.jsonl'
video_event_id_to_baseline_pred, video_event_id_to_gt_asr, video_start_end_to_video_event_id = {}, {}, {}

for line in open(baseline_jsonl):
    datum = json.loads(line)
    video_event_id = datum['video_id'] + '_' + str(datum['event_id'])
    video_start_end = (datum['video_id'], datum['begin'], datum['end'])
    video_start_end_to_video_event_id[video_start_end] = video_event_id
    video_event_id_to_baseline_pred[video_event_id] = datum['pred']

for datum in load_dataset('stdKonjac/LiveSports-3K', name='LiveSports_3K_CC', split="test"):
    video_event_id = datum['video_id'] + '_' + str(datum['event_id'])
    video_event_id_to_gt_asr[video_event_id] = datum['event_asr_text']

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='Model name to compare against baseline')
    parser.add_argument('--prediction_jsonl', type=str, required=True, help='Path to model predictions in JSONL format')
    parser.add_argument('--output_dir', type=str, default='evaluation/livesports3kcc/judges/', help='Directory to save judgment results')
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()

    model_id = args.model_id
    prediction_jsonl = args.prediction_jsonl
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{baseline_id}_{model_id}.jsonl')

    print(f'{model_id} vs. {baseline_id}')

    video_event_id_to_model_pred = {}
    assert os.path.exists(prediction_jsonl), f'{prediction_jsonl} not found'
    for line in open(prediction_jsonl):
        datum = json.loads(line)
        video_event_id = datum['video_id'] + '_' + str(datum['event_id'])
        video_event_id_to_model_pred[video_event_id] = datum['pred']

    for video_event_id in video_event_id_to_baseline_pred:
        assert video_event_id in video_event_id_to_model_pred, f'Missing prediction for {video_event_id}'

    winner_results = local_mt(
        video_event_id_to_model_pred.items(), 
        functools.partial(judge, model_id=model_id), 
        desc=f'Call gpt4o for {model_id} vs. {baseline_id}', 
        num_workers=args.num_workers
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
    with open(os.path.join(output_dir, 'log.txt'), 'a') as f:
        f.write(output + '\n')