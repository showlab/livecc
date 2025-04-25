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