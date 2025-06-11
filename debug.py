import json, tqdm, os, collections
from utils.multiprocessor import local_mt

# Howto & Style, Sports, Education, Autos & Vehicles, Science & Technology, Gaming, News & Politics
# --- filter category ---
def filter_category():
    path = 'live_whisperx_30-240s_3.5m.jsonl'

    with open(path) as f:
        lines = f.readlines()

    def process(line: str):
        datum = json.loads(line)
        if datum['category'] in ['Howto & Style', 'Sports', 'Education', 'Autos & Vehicles', 'Science & Technology', 'Gaming', 'News & Politics']:
            return line
        return None

    lines = local_mt(lines, process, desc='process', num_workers=4)
    lines = [line for line in lines if line is not None]
    path = 'live_whisperx_7c_30-240s.jsonl'
    with open(path, 'w') as f:
        f.writelines(lines)
    print('done')

# --- correct category ---
def correct_category():
    path = 'live_whisperx_30-240s_lmloss1.5-5_2.8m.jsonl'

    with open(path) as f:
        lines = f.readlines()
    video2cat = json.load(open('video2cat.json'))

    def process(line: str):
        datum = json.loads(line)
        datum['category'] = video2cat[datum['video']]
        line = json.dumps(datum) + '\n'
        return line

    lines = local_mt(lines, process, desc='process', num_workers=4)
    with open(path, 'w') as f:
        f.writelines(lines)
    print('done')

def select_asd(threshold=0.05):
    dirname = 'live_whisperx_7c_30-240s_lmloss1.5-5_1.54m_idx2asd'
    lines = open('live_whisperx_7c_30-240s_lmloss1.5-5_1.54m.jsonl').readlines()
    selected_idx_and_ratios = []
    for file in tqdm.tqdm(os.listdir(dirname)):
        path = f'{dirname}/{file}'
        idx_and_ratios = json.load(open(path))
        selected_idx_and_ratios.extend([(idx, ratio) for idx, ratio in idx_and_ratios if 0 <= ratio <= threshold])
    selected_lines = [lines[i] for i, r in selected_idx_and_ratios]
    with open(f'live_whisperx_7c_30-240s_lmloss1.5-5_asd0-{threshold}.jsonl', 'w') as f:
        f.writelines(selected_lines)
    return selected_lines

def category_statistics():
    lines = open('live_whisperx_526k_with_seeks.jsonl').readlines()
    datums = local_mt(lines[:-1], json.loads, desc='json.loads', num_workers=8)
    categories = [conversations[0]['content'][1]['category'] for conversations in datums]
    print(collections.Counter(categories))

def unknown():
    lines = open('live_whisperx_528k_with_seeks.jsonl').readlines()[:-1]
    get_video = lambda line: line[57:line.index('video_start')-4]
    videos = [get_video(line) for line in lines]
    json.dump(videos, open('live_whisperx_528k_videos.json', 'w'))

def make_preview():
    lines = open('live_whisperx_526k_with_seeks.jsonl').readlines()[:100]
    datums = [json.loads(line) for line in lines]
    previews = [{'video': datum[0]['content'][0]['video'], 'video_start': datum[0]['content'][0]['video_start'], 'video_end': datum[0]['content'][0]['video_end'], 'query': datum[0]['content'][1]['text'], 'text_stream': json.dumps(datum[1]['content'][0]['text_stream'])} for datum in datums]
    json.dump(previews, open(f'live_whisperx_100_for_preview.json', 'w'))
    
def push_to_hf(folder_path, repo_id, repo_type):
    import os
    from huggingface_hub import HfApi
    HfApi(token=os.environ.get('HF_TOKEN')).upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )

def remove_7c():
    lines = open('live_whisperx_2.6m.jsonl').readlines()
    datums = local_mt(lines, json.loads, desc='json.loads', num_workers=8)
    datums_7c = [datum for datum in datums if datum['category'] in ['Howto & Style', 'Sports', 'Education', 'Autos & Vehicles', 'Science & Technology', 'Gaming', 'News & Politics']]
    videos = set(datum['video'] for datum in datums_7c)
    print(len(videos))
    with open('live_whisperx_2.6m_7c.jsonl', 'w') as f:
        f.writelines(list(local_mt(datums_7c, json.dumps, desc='json.dumps', num_workers=8)))

def just_ffmpeg(input_path: str, output_path: str):
    # os.system(f"""ffmpeg -ss 3580 -t 30 -i spacex_falcon9.mp4 -vf "scale='if(gt(a,1),-2,1080)':'if(gt(a,1),1080,-2)'" -an demo/sources/spacex_falcon9_mute_1080p.mp4 -y""")
    os.system(f"""ffmpeg -ss 16 -t 31  -i {input_path} -vf "setpts=PTS*1.1,scale='if(gt(a,1),-2,1080)':'if(gt(a,1),1080,-2)'" -an {output_path} -y""") # setpts=PTS/1.7,
    # os.system(f"""ffmpeg  -i {input_path} -vf "setpts=PTS/2,scale='if(gt(a,1),-2,1080)':'if(gt(a,1),1080,-2)'" -an {output_path} -y""") # setpts=PTS/1.7,

# if __name__ == '__main__':
#     # for file in os.listdir('spc_demo_videos'):
#         input_path = f'dota2_facelessvoid.mp4'
#         output_path = f"demo/sources/dota2_facelessvoid_mute_1080p.mp4"
#         just_ffmpeg(input_path, output_path)
