import json, tqdm, pysubs2, os

lines = open('videomme.jsonl').readlines()
with open('videomme_with_subtitles.jsonl', 'w') as f:
    for line in tqdm.tqdm(lines):
        datum = json.loads(json.loads(line))
        srt_path = 'videomme_subtitle/' + datum['videoID'] + '.srt'
        if os.path.exists(srt_path):
            subs = pysubs2.load(srt_path, encoding="utf-8")
            subtitles = []
            for sub in subs:
                sub_text = sub.text.replace("\\N", " ")
                if sub_text.strip():
                    subtitles.append(sub_text)
            subtitles = " ".join(subtitles)
        else:
            subtitles = ""
        datum['subtitles'] = subtitles
        f.write(json.dumps(datum) + '\n')
