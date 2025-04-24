import json, tqdm
from transformers import pipeline
from utils.multiprocessor import local_mt

lines = open('live_whisperx_528k_with_seeks.jsonl').readlines()[:-1]

def language_detect_on_device(device_id: int):
    model_ckpt = "papluca/xlm-roberta-base-language-detection"
    pipe = pipeline("text-classification", model=model_ckpt, device=f'cuda:{device_id}')
    results = []
    local_lines = lines[device_id::8]
    path = f'language_detect_result_{device_id}.json'
    for line in tqdm.tqdm(local_lines, desc=path):
        conversation = json.loads(line)
        paragraph = ' '.join(w for s, e, w, in conversation[1]['content'][0]['text_stream'])
        result = pipe(paragraph, top_k=1, truncation=True)[0]
        results.append(result)
    json.dump(results, open(path, 'w'))

def gather_and_filter():
    results = [None] * 527583
    for i in range(7):
        path = f'language_detect_results_{i}.json'
        results[i::8] = json.load(open(path))
    results_part7 = [None] * len(range(7, 527583, 8))
    for i in range(8):
        path = f'language_detect_results_part7_{i}.json'
        results_part7[i::8] = json.load(open(path))
    results[7::8] = results_part7
    english_lines = [line for line, result in zip(lines, results) if result['label'] == 'en' and result['score'] >= 0.9]

if __name__ == '__main__':
    # local_mt(range(8), language_detect_on_device, desc='language_detect', num_workers=8)
    gather_and_filter()