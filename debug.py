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

import functools, torch, os, tqdm
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
apply_liger_kernel_to_qwen2_vl() # important. our model is trained with this. keep consistency
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LogitsProcessor, logging
from livecc_utils import prepare_multiturn_multimodal_inputs_for_generation, get_smart_resized_clip, get_smart_resized_video_reader
from qwen_vl_utils import process_vision_info

class LiveCCDemoInfer:
  fps = 2
  initial_fps_frames = 6
  streaming_fps_frames = 2
  initial_time_interval = initial_fps_frames / fps
  streaming_time_interval = streaming_fps_frames / fps
  frame_time_interval = 1 / fps
  def __init__(self, model_path: str = None, device_id: int = 0):
      self.model = Qwen2VLForConditionalGeneration.from_pretrained(
          model_path, torch_dtype="auto", 
          device_map=f'cuda:{device_id}', 
          attn_implementation='flash_attention_2'
      )
      self.processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
      self.model.prepare_inputs_for_generation = functools.partial(prepare_multiturn_multimodal_inputs_for_generation, self.model)
      message = {
          "role": "user",
          "content": [
              {"type": "text", "text": 'livecc'},
          ]
      }
      texts = self.processor.apply_chat_template([message], tokenize=False)
      self.system_prompt_offset = texts.index('<|im_start|>user')
      self._cached_video_readers_with_hw = {}


  def live_cc(
      self,
      query: str,
      state: dict,
      max_pixels: int = 384 * 28 * 28,
      default_query: str = 'Please describe the video.',
      do_sample: bool = False,
      repetition_penalty: float = 1.05,
      **kwargs,
  ): 
      """
      state: dict, (maybe) with keys:
          video_path: str, video path
          video_timestamp: float, current video timestamp
          last_timestamp: float, last processed video timestamp
          last_video_pts_index: int, last processed video frame index
          video_pts: np.ndarray, video pts
          last_history: list, last processed history
          past_key_values: llm past_key_values
          past_ids: past generated ids
      """
      # 1. preparation: video_reader, and last processing info
      video_timestamp, last_timestamp = state.get('video_timestamp', 0), state.get('last_timestamp', -1 / self.fps)
      video_path = state['video_path']
      if video_path not in self._cached_video_readers_with_hw:
          self._cached_video_readers_with_hw[video_path] = get_smart_resized_video_reader(video_path, max_pixels)
          video_reader = self._cached_video_readers_with_hw[video_path][0]
          video_reader.get_frame_timestamp(0)
          state['video_pts'] = torch.from_numpy(video_reader._frame_pts[:, 1])
          state['last_video_pts_index'] = -1
      video_pts = state['video_pts']
      if last_timestamp + self.frame_time_interval > video_pts[-1]:
          state['video_end'] = True
          return 
      video_reader, resized_height, resized_width = self._cached_video_readers_with_hw[video_path]
      last_video_pts_index = state['last_video_pts_index']

      # 2. which frames will be processed
      initialized = last_timestamp >= 0
      if not initialized:
          video_timestamp = max(video_timestamp, self.initial_time_interval)
      if video_timestamp <= last_timestamp + self.frame_time_interval:
          return
      timestamps = torch.arange(last_timestamp + self.frame_time_interval, video_timestamp, self.frame_time_interval) # add compensation
      
      # 3. fetch frames in required timestamps
      clip, clip_timestamps, clip_idxs = get_smart_resized_clip(video_reader, resized_height, resized_width, timestamps, video_pts, video_pts_index_from=last_video_pts_index+1)
      state['last_video_pts_index'] = clip_idxs[-1]
      state['last_timestamp'] = clip_timestamps[-1]

      # 4. organize to interleave frames
      interleave_clips, interleave_timestamps = [], []
      if not initialized:
          interleave_clips.append(clip[:self.initial_fps_frames])
          interleave_timestamps.append(clip_timestamps[:self.initial_fps_frames])
          clip = clip[self.initial_fps_frames:]
          clip_timestamps = clip_timestamps[self.initial_fps_frames:]
      if len(clip) > 0:
          interleave_clips.extend(list(clip.split(self.streaming_fps_frames)))
          interleave_timestamps.extend(list(clip_timestamps.split(self.streaming_fps_frames)))

      # 5. make conversation and send to model
      for clip, timestamps in zip(interleave_clips, interleave_timestamps):
          start_timestamp, stop_timestamp = timestamps[0].item(), timestamps[-1].item() + self.frame_time_interval
          message = {
              "role": "user",
              "content": [
                  {"type": "text", "text": f'Time={start_timestamp:.1f}-{stop_timestamp:.1f}s'},
                  {"type": "video", "video": clip}
              ]
          }
          if not query and not state.get('query', None):
              query = default_query
              print(f'No query provided, use default_query={default_query}')
          if query and state.get('query', None) != query:
              message['content'].append({"type": "text", "text": query})
              state['query'] = query
          texts = self.processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True, return_tensors='pt')
          past_ids = state.get('past_ids', None)
          if past_ids is not None:
              texts = '<|im_end|>\n' + texts[self.system_prompt_offset:]
          inputs = self.processor(
              text=texts,
              images=None,
              videos=[clip],
              return_tensors="pt",
              return_attention_mask=False
          )
          inputs.to('cuda')
          if past_ids is not None:
              inputs['input_ids'] = torch.cat([past_ids, inputs.input_ids], dim=1) 
          outputs = self.model.generate(
              **inputs, past_key_values=state.get('past_key_values', None), 
              return_dict_in_generate=True, do_sample=do_sample, 
              repetition_penalty=repetition_penalty,
              eos_token_id=self.model.config.eos_token_id,
              pad_token_id=0,
          )
          state['past_key_values'] = outputs.past_key_values
          state['past_ids'] = outputs.sequences[:, :-1]
          yield (start_timestamp, stop_timestamp), self.processor.decode(outputs.sequences[0, inputs.input_ids.size(1):], skip_special_tokens=True), state

model_path = 'chenjoya/LiveCC-7B-Instruct'
# download a test video at: https://github.com/showlab/livecc/blob/main/demo/sources/howto_fix_laptop_mute_1080p.mp4
video_path = "demo/sources/howto_fix_laptop_mute_1080p.mp4"
query = "Please describe the video."

infer = LiveCCDemoInfer(model_path=model_path)
state = {'video_path': video_path}
commentaries = []
t = 0
for t in range(31):
    state['video_timestamp'] = t
    for (start_t, stop_t), response, state in infer.live_cc(
        query=query, state=state, 
        max_pixels = 384 * 28 * 28, repetition_penalty=1.05, 
        streaming_eos_base_threshold=0.0, streaming_eos_threshold_step=0
    ):
        print(f'{start_t}s-{stop_t}s: {response}')
        commentaries.append([start_t, stop_t, response])
    if state.get('video_end', False):
        break
    t += 1