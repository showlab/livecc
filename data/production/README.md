## Data Production for LiveCC

![image](https://github.com/user-attachments/assets/2d0024d4-e1a2-4045-95ca-d70de962d211)

Here we have the implementation of pre-training & SFT data production pipeline for real-time video commentary. 

### Pre-training

#### A1. Video ASR Clipping

Please refer to [pretrain_to_clips.py](https://github.com/showlab/livecc/blob/main/data/production/pretrain_to_clips.py)

- Split consistent YouTube CC into words: https://github.com/showlab/livecc/blob/1dee25315ead4f7d641fa64cd6abd3f86024da8c/data/production/pretrain_to_clips.py#L15

- `min_clip_sec` ~ `max_clip_sec` clipping, and preserve previous ASR: https://github.com/showlab/livecc/blob/1dee25315ead4f7d641fa64cd6abd3f86024da8c/data/production/pretrain_to_clips.py#L32

- Check word speed: https://github.com/showlab/livecc/blob/1dee25315ead4f7d641fa64cd6abd3f86024da8c/data/production/pretrain_to_clips.py#L51

#### A2. Pure Text Filtering (by causal LM loss, Distributed)

Please refer to [lm_loss.py](https://github.com/showlab/livecc/blob/main/data/production/lm_loss.py)

- Use `Qwen/Qwen2-1.5B-Instruct`: https://github.com/showlab/livecc/blob/1dee25315ead4f7d641fa64cd6abd3f86024da8c/data/production/lm_loss.py#L62

- Easy causal LM loss: https://github.com/showlab/livecc/blob/1dee25315ead4f7d641fa64cd6abd3f86024da8c/data/production/lm_loss.py#L50

#### A3. Remove Talking-head Videos (by LMM, Distributed)

Please refer to [distributed_lmm4asd.py](https://github.com/showlab/livecc/blob/main/data/production/distributed_lmm4asd.py)

- Use `Qwen/Qwen2-VL-2B-Instruct`: https://github.com/showlab/livecc/blob/1dee25315ead4f7d641fa64cd6abd3f86024da8c/data/production/distributed_lmm4asd.py#L79

- Simply select 8 frames and low resolution per video to speed up: https://github.com/showlab/livecc/blob/1dee25315ead4f7d641fa64cd6abd3f86024da8c/data/production/distributed_lmm4asd.py#L22-L26


### SFT

#### B1. Select 7 Categories

Omitted.

#### B2. Better ASR (Distributed)

Please refer to [distributed_whisperx.py](https://github.com/showlab/livecc/blob/main/data/production/distributed_whisperx.py)

- Use `large-v3-turbo`: https://github.com/showlab/livecc/blob/1dee25315ead4f7d641fa64cd6abd3f86024da8c/data/production/distributed_whisperx.py#L10

#### B3. Video ASR Clipping

Please refer to [sft_to_clips.py](https://github.com/showlab/livecc/blob/main/data/production/sft_to_clips.py)

- During SFT, the text should always start at the beginning of a sentence, since we did not use previous ASR: https://github.com/showlab/livecc/blob/1dee25315ead4f7d641fa64cd6abd3f86024da8c/data/production/sft_to_clips.py#L9

#### B4. Pure Text Filtering (by causal LM loss, Distributed)

Follow A2.

#### B5. Remove Talking-head Videos (by Lighter-ASD, Distributed)

Please refer to [distributed_lighter_asd](https://github.com/showlab/livecc/tree/main/data/production/distributed_lighter_asd)

#### B6. Make Prompt

Please refer to [make_prompt.py](https://github.com/showlab/livecc/blob/main/data/production/make_prompt.py)

- The prompt: https://github.com/showlab/livecc/blob/1dee25315ead4f7d641fa64cd6abd3f86024da8c/data/production/make_prompt.py#L11-L27


### Some Utils Functions

- Make data jsonl to be Qwen-style conversation: [to_conversation.py](https://github.com/showlab/livecc/blob/main/data/production/to_conversation.py)

- Detect Language: [language_detect.py](https://github.com/showlab/livecc/blob/main/data/production/language_detect.py)



