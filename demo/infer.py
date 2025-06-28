import functools, torch
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
apply_liger_kernel_to_qwen2_vl()
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LogitsProcessor, logging
from livecc_utils import prepare_multiturn_multimodal_inputs_for_generation, get_smart_resized_clip, get_smart_resized_video_reader, _read_video_decord_plus, _spatial_resize_video
from qwen_vl_utils import process_vision_info

logger = logging.get_logger(__name__)

class ThresholdLogitsProcessor(LogitsProcessor):
    def __init__(self, token_id: int, base_threshold: float, step: float):
        self.token_id = token_id
        self.base_threshold = base_threshold
        self.step = step
        self.count = 0
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        threshold = self.base_threshold + self.step * self.count 
        low_confidence = torch.softmax(scores, dim=-1)[:, self.token_id] <= threshold
        if low_confidence.any():
            scores[low_confidence, self.token_id] = -float("inf")
        self.count += 1
        return scores
    
class LiveCCDemoInfer:
    VIDEO_PLAY_END = object()
    VIDEO_PLAY_CONTINUE = object()
    fps = 2
    initial_fps_frames = 6
    streaming_fps_frames = 2
    initial_time_interval = initial_fps_frames / fps
    streaming_time_interval = streaming_fps_frames / fps
    frame_time_interval = 1 / fps

    def __init__(self, model_path: str = None, device: str = None):
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", 
            device_map=device, 
            attn_implementation='flash_attention_2' if device == 'cuda' else None
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
        self._cached_video_readers_with_hw = {}

    @torch.inference_mode()
    def live_cc(
        self,
        message: str,
        state: dict,
        max_pixels: int = 384 * 28 * 28,
        default_query: str = 'Please describe the video.',
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        streaming_eos_base_threshold: float = None, 
        streaming_eos_threshold_step: float = None, 
        hf_spaces: bool = False,
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
        """
        # 1. preparation: video_reader, and last processing info
        video_timestamp, last_timestamp = state.get('video_timestamp', 0), state.get('last_timestamp', -1 / self.fps)
        video_path = state.get('video_path', None)
        if not video_path:
            return
        if video_path not in self._cached_video_readers_with_hw:
            self._cached_video_readers_with_hw[video_path] = get_smart_resized_video_reader(video_path, max_pixels)
            video_reader = self._cached_video_readers_with_hw[video_path][0]
            video_reader.get_frame_timestamp(0)
            state['video_pts'] = torch.from_numpy(video_reader._frame_pts[:, 1])
            state['last_video_pts_index'] = -1
        video_pts = state.get('video_pts', None)
        if video_pts is None:
            return
        video_timestamp = min(video_timestamp, video_pts[-1])
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
        if len(clip_idxs) == 0:
            return
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
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": f'Time={start_timestamp:.1f}-{stop_timestamp:.1f}s'},
                    {"type": "video", "video": clip}
                ]
            }]
            if not message and not state.get('message', None):
                message = default_query
                logger.warning(f'No query provided, use default_query={default_query}')
            if message and state.get('message', None) != message:
                conversation[0]['content'].append({"type": "text", "text": message})
                state['message'] = message
            texts = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, return_tensors='pt')
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
            inputs.to(self.model.device)
            if past_ids is not None:
                inputs['input_ids'] = torch.cat([past_ids, inputs.input_ids], dim=1) 
            if streaming_eos_base_threshold is not None:
                logits_processor = [ThresholdLogitsProcessor(self.streaming_eos_token_id, streaming_eos_base_threshold, streaming_eos_threshold_step)]
            else:
                logits_processor = None
            outputs = self.model.generate(
                **inputs, past_key_values=state.get('past_key_values', None), 
                return_dict_in_generate=True, do_sample=do_sample, 
                repetition_penalty=repetition_penalty,
                logits_processor=logits_processor,
                max_new_tokens=16,
                pad_token_id=self.model.config.eos_token_id,
            )
            state['past_key_values'] = outputs.past_key_values
            state['past_ids'] = outputs.sequences[:, :-1]
            response = self.processor.decode(outputs.sequences[0, inputs.input_ids.size(1):], skip_special_tokens=True)
            if hf_spaces:
                light_state = {k: v for k, v in state.items() if k not in ['past_ids', 'past_key_values']}
                yield (start_timestamp, stop_timestamp), response, light_state
            else:
                yield (start_timestamp, stop_timestamp), response, state

    @torch.inference_mode()
    def video_qa(
        self,
        message: str,
        history: list,
        state: dict,
        do_sample: bool = False,
        repetition_penalty: float = 1.05,
        hf_spaces: bool = False,
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
        """
        video_path = state.get('video_path', None)
        conversation = []
        if hf_spaces:
            for past_message in history:
                content = [{"type": "text", "text": past_message['content']}]
                if video_path: # only use once
                    content.insert(0, {"type": "video", "video": video_path})
                    video_path = None
                conversation.append({"role": past_message["role"], "content": content})
        else:
            pass # use past_key_values
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
        state['past_key_values'] = outputs.past_key_values if not hf_spaces else None
        state['past_ids'] = outputs.sequences[:, :-1] if not hf_spaces else None
        response = self.processor.decode(outputs.sequences[0, inputs.input_ids.size(1):], skip_special_tokens=True)
        return response, state

    @torch.inference_mode()
    def live_cc_once_for_evaluation(
        self,
        query: str,
        video: str,
        video_start: float = None,
        video_end: float = None,
        remote_loader: callable = None,
        max_new_tokens: int = 32,
        repetition_penalty: float = 1.05,
    ): 
        # 1. read video clip
        clip, _ = _read_video_decord_plus({'video': video, 'video_start': video_start, 'video_end': video_end, 'remote_loader': remote_loader})
        clip = _spatial_resize_video(clip)

        # 2. organize to interleave frames
        interleave_clips = []
        ## 2.1 initial_fps_frames
        interleave_clips.append(clip[:self.initial_fps_frames])
        clip = clip[self.initial_fps_frames:]
        ## 2.2 streaming_fps_frames
        if len(clip) > 0:
            interleave_clips.extend(list(clip.split(self.streaming_fps_frames)))
        
        # 3. make conversation and send to model
        past_key_values = None
        responses = []
        for i, clip in enumerate(interleave_clips):
            if i == 0:
                start_timestamp, stop_timestamp = 0, self.initial_time_interval
            else:
                start_timestamp, stop_timestamp = stop_timestamp, stop_timestamp + self.streaming_time_interval
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": f'Time={start_timestamp:.1f}-{stop_timestamp:.1f}s'},
                    {"type": "video", "video": clip}
                ]
            }
            if not past_key_values:
                message['content'].append({"type": "text", "text": query})
            texts = self.processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True, return_tensors='pt')
            if past_key_values:
                texts = '<|im_end|>\n' + texts[self.system_prompt_offset:]
            inputs = self.processor(
                text=texts,
                images=None,
                videos=[clip],
                return_tensors="pt",
            )
            inputs.to(self.model.device)
            if past_key_values:
                inputs['input_ids'] = torch.cat([past_ids, inputs.input_ids], dim=1) 
            outputs = self.model.generate(
                **inputs, past_key_values=past_key_values, 
                return_dict_in_generate=True, 
                max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty, 
                pad_token_id=self.model.config.eos_token_id,
            )
            past_key_values = outputs.past_key_values
            past_ids = outputs.sequences[:, :-1]
            responses.append([
                video_start + start_timestamp, 
                video_start + stop_timestamp, 
                self.processor.decode(outputs.sequences[0, inputs.input_ids.size(1):], skip_special_tokens=True)
            ])
        return responses
