import json, tqdm
from utils.multiprocessor import local_mt

def clipping(line: str, min_clip_sec: int = 30, max_clip_sec: int = 240, max_silence_sec: int = 3):
    datum = json.loads(line)
    words, title = datum['content'], datum['title']
    clips, preasrs, i = [], [], 0
    while i < len(words):
        can_be_start = (i == 0) or ((any(words[i-1][-1].endswith(e) for e in ['.', '?', '!'])) and words[i][-1].isupper())
        if not can_be_start:
            i += 1
            continue
        j = None
        for j in range(i+1, len(words)): # word = [start, end, word_str]
            if words[j][0] - words[i][0] > max_clip_sec:
                break
            if words[j][0] - words[j-1][0] > max_silence_sec:
                break
        if j is not None and j > i and words[j-1][0] - words[i][0] >= min_clip_sec:
            clips.append(words[i:j])
            preasrs.append(' '.join(word[2].strip() for word in words[:i]))
        if j is not None:
            i = j
        else:
            break
    return [{'video': datum['video'], 'content': clip, 'preasr': preasr, 'title': title, 'category': datum['category']} for clip, preasr in zip(clips, preasrs)]

if __name__ == '__main__':
    path = 'live_whisperx_3.1m.jsonl'
    lines = open(path).readlines()[:-1]
    all_clips = local_mt(lines, clipping, desc='clipping', num_workers=8)
    with open(f'live_whisperx_30-240s.jsonl', 'w') as f:
        for clips in tqdm.tqdm(all_clips, desc='write to jsonl'):
            for clip in clips:
                f.write(json.dumps(clip) + '\n')

