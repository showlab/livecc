## LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale

<a href="https://showlab.github.io/livecc/" target="_blank"><img alt="Homepage" src="https://img.shields.io/badge/🌍 Homepage-d35400?color=d35400" /></a>
<a href="https://huggingface.co/spaces/chenjoya/livecc" target="_blank"><img alt="Demo" src="https://img.shields.io/badge/🤗 Demo-ffc107?color=ffc107" /></a>
<a href="https://huggingface.co/papers/2504.16030" target="_blank"><img alt="Paper" src="https://img.shields.io/badge/📄 Paper-28a745?color=28a745" /></a>
<a href="https://huggingface.co/chenjoya/LiveCC-7B-Instruct" target="_blank"><img alt="Checkpoint" src="https://img.shields.io/badge/🤗 Model-2980b9?color=2980b9" /></a>
<a href="https://huggingface.co/datasets/chenjoya/Live-WhisperX-526K" target="_blank"><img alt="Data" src="https://img.shields.io/badge/🤗 Dataset-8e44ad?color=8e44ad" /></a>
<a href="https://huggingface.co/datasets/stdKonjac/LiveSports-3K" target="_blank"><img alt="Data" src="https://img.shields.io/badge/🤗 Benchmark-8e44ad?color=007bff" /></a>
<a href="https://huggingface.co/collections/chenjoya/livecc-67e29b3df1b6b5c6d5d682f4" target="_blank"><img alt="Data" src="https://img.shields.io/badge/🤗 All Collections-8e44ad?color=e74c3c" /></a>

[![Watch the video](webpage/static/videos/thumbnail_yt.png)](https://www.youtube.com/watch?v=56sfodoHXo4)

### TLDR

The first video LLM capable of real-time commentary, trained with a novel video-ASR streaming method, SOTA on both streaming and offline benchmarks.

### Installation

Ensure you have Python version >= 3.11 installed.

```sh
pip install torch torchvision torchaudio
pip install "transformers>=4.52.4" accelerate deepspeed peft opencv-python decord datasets tensorboard gradio pillow-heif gpustat timm sentencepiece openai av==12.0.0 qwen_vl_utils liger_kernel numpy==1.24.4
pip install flash-attn --no-build-isolation
pip install livecc-utils==0.0.2
```

We trained our models under ```torch==2.6.0```, ```transformers==4.50.0```, ```liger-kernel==0.5.5```. But other versions should also work. 

#### Advanced

If you want to delve into our data production pipeline:

```sh
pip install insightface onnxruntime-gpu python_speech_features wavfile
```

### Quick Start

#### Gradio Demo
```
python demo/app.py --js_monitor
```
`--js_monitor` is to use javascript video timestamp monitoring (recommend to disable for environments with high latency)

<img width="1503" alt="image" src="https://github.com/user-attachments/assets/9673fe1f-a68e-4995-bb35-d07f5a8c8ffd" />

#### CLI
```
python demo/cli.py
```
<img width="770" alt="image" src="https://github.com/user-attachments/assets/5e099923-34f5-46d7-9cb6-629d8ab23803" />

#### Hands-on Inference

Please refer to [inference.md](https://github.com/showlab/livecc/blob/main/inference.md)

### Training

The following scripts are for a single node training, with the batch size of 512. If you have multiple nodes, please try to set [torchrun arguments](https://pytorch.org/docs/stable/elastic/run.html) and ```--gradient_accumulation_steps``` accordingly.

#### Pre-training

##### Data

https://huggingface.co/datasets/chenjoya/Live-CC-5M

##### Scripts

[scripts/pt_local.sh](scripts/pt_local.sh)

The explanation for the training arugments:

```bash
export VIDEO_MIN_PIXELS=78400 # 100*28*28. the minimum visual frame tokens sent to llm is 100
export FPS_MAX_FRAMES=480 # maximum number of frames for each video (480/60/2 = 4min)
export VIDEO_MAX_PIXELS=19267584 # 24576*28*28. the maximum overall video tokens sent to llm is 24k (leave 8k for language)

learning_rate=2e-5 # pretraining uses 2e-5 lr
run_name="livecc_pretrain_24kx480x100_bs512lr$learning_rate"

WANDB_PROJECT='joya.chen' TOKENIZERS_PARALLELISM=false torchrun --standalone --nproc_per_node=8 train.py \
  --deepspeed ./scripts/deepspeed_zero2.json \                       # Use DeepSpeed ZeRO-2 config
  --output_dir checkpoints/$run_name \                               # Where to save model checkpoints
  --overwrite_output_dir True \                                      # Set False to resume from existing checkpoint
  --run_name $run_name \                                             # Unique identifier for the training run (used by WandB)
  --save_on_each_node True \                                         # Set False if nodes share a filesystem
  --do_train True \                                                  # Enable training mode
  --eval_strategy no \                                               # No evaluation between training steps
  --per_device_train_batch_size 1 \                                  # Batch size per GPU
  --gradient_accumulation_steps 64 \                                 # Effective batch size = 64 × num_gpus
  --learning_rate $learning_rate \                                   # Learning rate to use
  --warmup_ratio 0.03 \                                              # Warm-up proportion of training steps
  --optim adamw_torch \                                              # Optimizer: AdamW (PyTorch implementation)
  --lr_scheduler_type cosine \                                       # Cosine decay learning rate schedule
  --num_train_epochs 1 \                                             # Number of training epochs
  --logging_steps 10 \                                               # Log training metrics every 10 steps
  --save_steps 1000 \                                                # Save checkpoint every 1000 steps
  --bf16 True \                                                      # Use BF16 mixed precision (if supported)
  --tf32 True \                                                      # Use TF32 precision on NVIDIA Ampere+ GPUs
  --gradient_checkpointing True \                                    # Enable gradient checkpointing to save memory
  --pretrained_model_name_or_path Qwen/Qwen2-VL-7B \                 # Start from pretrained Qwen2-VL-7B model
  --annotation_paths datasets/live_cc_5m_with_seeks.jsonl \          # Dataset used for training
  --dataloader_num_workers 16 \                                      # Number of parallel workers for data loading
  --freeze_modules visual \                                          # Freeze visual encoder parameters
  --use_liger_kernel True \                                          # Use Liger kernel for faster attention (must match in inference)
  --report_to wandb                                                  # Enable logging to Weights & Biases
```

#### SFT

##### Data

https://huggingface.co/datasets/chenjoya/Live-WhisperX-526K

https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K

##### Scripts

[scripts/sft_local.sh](scripts/sft_local.sh)

```bash
export VIDEO_MIN_PIXELS=78400 # 100*28*28. the minimum visual frame tokens sent to llm is 100
export FPS_MAX_FRAMES=480 # maximum number of frames for each video (480/60/2 = 4min)
export VIDEO_MAX_PIXELS=19267584 # 24576*28*28. the maximum overall video tokens sent to llm is 24k (leave 8k for language)

learning_rate=1e-5 # sft uses 1e-5 lr
run_name="livecc_sft_24k480x100_live526k+llava178k+hound+onevision_lr$learning_rate"

WANDB_PROJECT='joya.chen' TOKENIZERS_PARALLELISM=false torchrun --standalone --nproc_per_node=8 train.py \
  --deepspeed ./scripts/deepspeed_zero2.json \                       # Use DeepSpeed ZeRO-2 config
  --output_dir checkpoints/$run_name \                               # Output checkpoint directory
  --overwrite_output_dir True \                                      # Set to False to resume training
  --run_name $run_name \                                             # Wandb and checkpoint run name
  --save_on_each_node True \                                         # Set False if using shared storage
  --do_train True \                                                  # Enable training mode
  --eval_strategy no \                                               # No evaluation during training
  --per_device_train_batch_size 1 \                                  # Batch size per GPU
  --gradient_accumulation_steps 64 \                                 # Accumulate gradients for effective batch size = 64 × num_gpus
  --learning_rate $learning_rate \                                   # Learning rate to use
  --warmup_ratio 0.03 \                                              # Learning rate warm-up ratio
  --optim adamw_torch \                                              # Optimizer type
  --lr_scheduler_type cosine \                                       # Cosine learning rate scheduler
  --num_train_epochs 1 \                                             # Total number of training epochs
  --logging_steps 10 \                                               # Log every 10 steps
  --save_steps 1000 \                                                # Save checkpoint every 1000 steps
  --bf16 True \                                                      # Use BF16 mixed precision
  --tf32 True \                                                      # Enable TF32 acceleration (NVIDIA Ampere+)
  --gradient_checkpointing True \                                    # Enable gradient checkpointing for memory efficiency
  --pretrained_model_name_or_path chenjoya/LiveCC-7B-Base \          # Initialization checkpoint
  --annotation_paths \                                               # Training datasets:
      datasets/live_whisperx_526k_with_seeks.jsonl \                 # - LiveCC 526k
      datasets/llava_ov_single_image_text_mix_with_seeks.jsonl \     # - OneVision (single image)
      datasets/llava_ov_multi_image_with_seeks.jsonl \               # - OneVision (multi-image)
      datasets/llava_hound_video_with_seeks.jsonl \                  # - LLaVA-Hound video
      datasets/llava_video_178k_with_seeks.jsonl \                   # - LLaVA-Video 178k
  --dataloader_num_workers 16 \                                      # Number of workers for data loading
  --freeze_modules visual \                                          # Do not update visual encoder
  --use_liger_kernel True \                                          # Use Liger kernel for efficient attention (enable at inference too)
  --report_to wandb                                                  # Report metrics to Weights & Biases
```

### Evaluation

#### LiveSports3KCC

The following scripts will automatically download data from [LiveSports3K](https://huggingface.co/datasets/stdKonjac/LiveSports-3K).

##### Real-time Video Commentary (LiveCC)

```bash
# generate livecc
python evaluation/livesports3kcc/distributed_generate_livecc.py --model_name_or_path chenjoya/LiveCC-7B-Instruct --output_dir evaluation/livesports3kcc/livecc --num_workers 8 --repetition_penalty 1.15

# if evaluate base model, please add --not_instruct_model
python evaluation/livesports3kcc/distributed_generate_livecc.py --model_name_or_path chenjoya/LiveCC-7B-Base --output_dir evaluation/livesports3kcc/livecc --num_workers 8 --repetition_penalty 1.15 --not_instruct_model

# llm judge winning rate
AZURE_OPENAI_ENDPOINT=xxx AZURE_OPENAI_API_KEY=xxx python evaluation/livesports3kcc/llm_judge.py --model_id LiveCC-7B-Instruct --prediction_jsonl evaluation/livesports3kcc/livecc/LiveCC-7B-Instruct.jsonl --output_dir evaluation/livesports3kcc/judges --num_workers 16
```
<img width="471" alt="image" src="https://github.com/user-attachments/assets/87d752d7-663f-4e24-8f54-b78680e45a66" />

(Slightly better than our paper results, since Azure GPT-4o output is not strictly stable, even if we set ```seed=42, temperature=0```😂)

If you do not have GPT-4o quota, please submit results at [CVPR'25 LoVE Workshop Track2A](https://sites.google.com/view/loveucvpr25/track2a). We cover the GPT-4o evaluation cost 1 time per day for every participant.

##### Offline Caption (e.g. GPT-4o, Qwen2.5VL, etc)

```
python evaluation/livesports3kcc/distributed_generate_caption.py --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct --output_dir evaluation/livesports3kcc/captions --num_workers 8
```

#### LiveSports3KQA

#### VideoMME

Our fast distributed VideoMME evaluator needs ```videomme.jsonl``` with the data format of each line as:
```json
{"video_id": "001", "duration": "short", "domain": "Knowledge", "sub_category": "Humanity & History", "url": "https://www.youtube.com/watch?v=fFjv93ACGo8", "videoID": "fFjv93ACGo8", "question_id": "001-1", "task_type": "Counting Problem", "question": "When demonstrating the Germany modern Christmas tree is initially decorated with apples, candles and berries, which kind of the decoration has the largest number?", "options": ["A. Apples.", "B. Candles.", "C. Berries.", "D. The three kinds are of the same number."], "answer": "C", "subtitles": "[Music] and new at 6:00 ..."}
```

After preparation, please run:
```shell
# without subtitles
torchrun --standalone --nproc_per_node=8 evaluation/videomme/distributed_evaluate_videomme.py --model_name_or_path chenjoya/LiveCC-7B-Instruct --benchmark_path videomme.jsonl
# with subtitles
torchrun --standalone --nproc_per_node=8 evaluation/videomme/distributed_evaluate_videomme.py --model_name_or_path chenjoya/LiveCC-7B-Instruct --benchmark_path videomme.jsonl --with_subtitles
```
Typically, it costs ~40min (no subtitles) or ~50min (with subtitles) to finish the evaluation (8x80G GPUs). The results will be written to [evaluation/videomme/results](evaluation/videomme/results). We also provided the evaluation results of [LiveCC-7B-Instruct](https://huggingface.co/chenjoya/LiveCC-7B-Instruct) at [evaluation/videomme/results](evaluation/videomme/results).

#### OVOBench

First, make the data structure of OVOBench like:

```
ovobench
├── AutoEvalMetaData
├── COIN
├── cross_task
├── Ego4D
├── hirest
├── MovieNet
├── OpenEQA
├── ovo_bench_new.json
├── perception_test
├── star
├── thumos
├── youcook2
└── YouTube_Games
```

Then, do some preprocessing for annotation json to simplify the distributed evaluation:

```
python evaluation/ovobench/transfer_annotation_format.py --input .../ovobench/ovo_bench_new.json --output .../ovobench/ovo-bench-formatted.jsonl
```

The above will create the file ```ovo-bench-formatted.jsonl``` under your ovobench dir. Finally, do evaluation:

```
torchrun --standalone --nproc_per_node=8 evaluation/ovobench/distributed_evaluate_ovobench.py --benchmark_dir .../ovobench
```

The results will be stored in ```evaluation/ovobench/results```.

#### MVBench

Too busy recently 😭, will update readme as soon as possible

### Data Production Pipeline

Please refer to [data/production/README.md](https://github.com/showlab/livecc/tree/main/data/production/README.md)

### Citation

```
@inproceedings{livecc,
    author       = {Joya Chen and Ziyun Zeng and Yiqi Lin and Wei Li and Zejun Ma and Mike Zheng Shou},
    title        = {LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale},
    booktitle    = {CVPR},
    year         = {2025},
}
```
