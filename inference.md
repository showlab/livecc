### LiveCC Hands-on Inference

Like qwen-vl-utils, we offer a toolkit to help you handle various types of visual input more conveniently, **especially on video streaming inputs**. You can install it using the following command:

```bash
pip install qwen-vl-utils livecc-utils liger_kernel
```

Here we show a code snippet to show you how to do **real-time video commentary** with `transformers` and the above utils:

```python
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
              pad_token_id=self.model.config.eos_token_id,
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
```

Here we show a code snippet to show you how to do **common video (multi-turn) qa** with `transformers` and the above utils:
```python
import functools, torch
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

  def __init__(self, model_path: str = None, device: str = 'cuda'):
      self.model = Qwen2VLForConditionalGeneration.from_pretrained(
          model_path, torch_dtype="auto", 
          device_map=device, 
          attn_implementation='flash_attention_2'
      )
      self.processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
      self.streaming_eos_token_id = self.processor.tokenizer(' ...').input_ids[-1]
      self.model.prepare_inputs_for_generation = functools.partial(prepare_multiturn_multimodal_inputs_for_generation, self.model)
      message = {
          "role": "user",
          "content": [
              {"type": "text", "text": 'livecc'},
          ]
      }
      texts = self.processor.apply_chat_template([message], tokenize=False)
      self.system_prompt_offset = texts.index('<|im_start|>user')

  def video_qa(
      self,
      message: str,
      state: dict,
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
      video_path = state.get('video_path', None)
      conversation = []
      past_ids = state.get('past_ids', None)
      content = [{"type": "text", "text": message}]
      if past_ids is None and video_path: # only use once
          content.insert(0, {"type": "video", "video": video_path})
      conversation.append({"role": "user", "content": content})
      image_inputs, video_inputs = process_vision_info(conversation)
      texts = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, return_tensors='pt')
      if past_ids is not None:
          texts = '<|im_end|>\n' + texts[self.system_prompt_offset:]
      inputs = self.processor(
          text=texts,
          images=image_inputs,
          videos=video_inputs,
          return_tensors="pt",
          return_attention_mask=False
      )
      inputs.to(self.model.device)
      if past_ids is not None:
          inputs['input_ids'] = torch.cat([past_ids, inputs.input_ids], dim=1) 
      outputs = self.model.generate(
          **inputs, past_key_values=state.get('past_key_values', None), 
          return_dict_in_generate=True, do_sample=do_sample, 
          repetition_penalty=repetition_penalty,
          max_new_tokens=512,
          pad_token_id=self.model.config.eos_token_id,
      )
      state['past_key_values'] = outputs.past_key_values
      state['past_ids'] = outputs.sequences[:, :-1]
      response = self.processor.decode(outputs.sequences[0, inputs.input_ids.size(1):], skip_special_tokens=True)
      return response, state

model_path = 'chenjoya/LiveCC-7B-Instruct'
# download a test video at: https://github.com/showlab/livecc/blob/main/demo/sources/howto_fix_laptop_mute_1080p.mp4
video_path = "demo/sources/howto_fix_laptop_mute_1080p.mp4"

infer = LiveCCDemoInfer(model_path=model_path)
state = {'video_path': video_path}
# first round
query1 = 'What is the video?'
response1, state = infer.video_qa(message=query1, state=state)
print(f'Q1: {query1}\nA1: {response1}')
# second round
query2 = 'How do you know that?'
response2, state = infer.video_qa(message=query2, state=state)
print(f'Q2: {query2}\nA2: {response2}')
```
