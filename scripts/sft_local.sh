export VIDEO_MIN_PIXELS=78400 # 100*28*28. the minimum visual frame tokens sent to llm is 100
export FPS_MAX_FRAMES=480 # maximum number of frames for each video (480/60/2 = 4min)
export VIDEO_MAX_PIXELS=19267584 # 24576*28*28. the maximum overall video tokens sent to llm is 24k (leave 8k for language)

learning_rate=1e-5 # sft uses 2e-5 lr
run_name="livecc_sft_24k480x100_llava178k+hound+onevision_lr$learning_rate"

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