import json, argparse, functools, tqdm
from utils.multiprocessor import local_mt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=str, default='ytcc8m.jsonl')
    parser.add_argument('--part', type=str, default='1/1')
    parser.add_argument('--min_clip_sec', type=int, default=30)
    parser.add_argument('--max_clip_sec', type=int, default=240)
    parser.add_argument('--max_empty_sec', type=int, default=3)
    parser.add_argument('--min_wps', type=int, default=1)
    parser.add_argument('--max_wps', type=int, default=4)
    return parser.parse_args()

def split2words(datum: dict):
    subtitles = datum.pop('subtitles')
    content = []
    for start, end, subtitle in subtitles:
        if '[' in subtitle or ']' in subtitle:
            continue
        words = []
        for word in subtitle.split(' '):
            if not words or words[-1] != word:
                words.append(word)
        duration = end - start
        duration_per_word = duration / len(words)
        for i, word in enumerate(words):
            content.append([round(start + i * duration_per_word, 1), round(start + (i+1) * duration_per_word, 1), word])
    datum['content'] = content
    return datum

def clip4pretrain(datum: dict, args):
    words, title = datum['content'], datum['title']
    clips, contexts, i = [], [], 0
    while i < len(words):
        j = None
        for j in range(i+1, len(words)):
            if words[j][1] - words[i][1] > args.max_clip_sec:
                break
            if words[j][1] - words[j-1][1] > args.max_empty_sec:
                break
        if j is not None and j > i and words[j-1][1] - words[i][1] >= args.min_clip_sec:
            clips.append(words[i:j])
            contexts.append(' '.join(word[2] for word in words[:i]))
        if j is not None:
            i = j
        else:
            break
    return [{'video': datum['tos_key'], 'content': clip, 'previous': context, 'title': title, 'category': datum['category']} for clip, context in zip(clips, contexts)]

def check(datum: dict, args):
    subtitles = datum['content']
    duration = subtitles[-1][1] - subtitles[0][1]
    wps = len(subtitles) / duration
    if wps < args.min_wps or wps > args.max_wps:
        return False
    return True

def process(datum: dict, args):
    datum = split2words(datum)
    clips_datum = clip4pretrain(datum, args)
    clips_datum = [clip_datum for clip_datum in clips_datum if check(clip_datum, args)]
    return clips_datum

if __name__ == '__main__':
    args = get_args()
    index, total = args.part.split('/')
    index, total = int(index), int(total)

    print(f'open({args.inputs}).readlines() could cost some times...')
    lines = open(args.inputs).readlines()
    print(f'done open({args.inputs}).readlines(). json.loads them...')
    datums = local_mt(lines, json.loads, desc='json.loads')
    print(f'{len(datums)} datums. continuing...')

    datums = datums[index-1::total]
    print(f'{len(datums)} datums in this part. continuing...')
    
    clips_datums = local_mt(datums, functools.partial(process, args=args), 'process')
    with open('livecc_30-240s_wps1-4_clips.jsonl', 'w') as f:
        for clips_datum in tqdm.tqdm(clips_datums):
            for clip_datum in clips_datum:
                f.write(json.dumps(clip_datum) + '\n')