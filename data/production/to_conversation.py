import os, json, torch, tqdm
from utils.multiprocessor import local_mt

lines = open('live_whisperx_7c_30-240s_lmloss1.5-5_asd0-0.05_prompted.jsonl').readlines()
datums = local_mt(lines, json.loads, desc='json.loads', num_workers=8)

conversations = [
    [
        {'role': 'user', 'content': [
            {'type': 'video', 'video': datum['video'], 'video_start': datum['content'][0][0], 'video_end': datum['content'][-1][1]}, 
            {'type': 'text', 'text': datum['query'], 'previous': datum['preasr'], 'title': datum['title'], 'category': datum['category']}
        ]},
        {'role': 'assistant', 'content': [{'type': 'text_stream', 'text_stream': datum['content']}]},
    ] for datum in tqdm.tqdm(datums)
]

with open('live_whisperx_with_seeks.jsonl', 'w') as f:
    lengths = []
    for conversation in conversations:
        line = json.dumps(conversation) + '\n'
        lengths.append(len(line))
        f.write(line)
    seeks = [0] + torch.tensor(lengths).cumsum(dim=-1).tolist()[:-1]
    f.write(json.dumps(seeks))