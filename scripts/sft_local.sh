export VIDEO_MIN_PIXELS=78400
export VIDEO_MAX_PIXELS=19267584
export FPS_MAX_FRAMES=480

learning_rate=1e-5
run_name="livecc_sft_24k480x100_llava178k+hound+onevision_lr$learning_rate"

WANDB_PROJECT='joya.chen' TOKENIZERS_PARALLELISM=false torchrun --standalone --nproc_per_node=8 train.py  \
  --deepspeed ./scripts/deepspeed/deepspeed_zero2.json \
  --output_dir checkpoints/$run_name --overwrite_output_dir True \
  --run_name $run_name \
  --save_on_each_node True \
  --do_train True \
  --eval_strategy no \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate $learning_rate \
  --warmup_ratio 0.03 \
  --optim adamw_torch \
  --lr_scheduler_type cosine \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --save_steps 1000 \
  --bf16 True --tf32 True \
  --gradient_checkpointing True \
  --pretrained_model_name_or_path chenjoya/LiveCC-7B-Base \
  --annotation_paths datasets/live_whisperx_526k_with_seeks.jsonl datasets/llava_ov_single_image_text_mix_with_seeks.jsonl datasets/llava_ov_multi_image_with_seeks.jsonl datasets/llava_hound_video_with_seeks.jsonl datasets/llava_video_178k_with_seeks.jsonl \
  --dataloader_num_workers 16 \
  --freeze_modules visual \
  --use_liger_kernel True \
  --report_to wandb 