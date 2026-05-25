#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/provenance_weight_training

WANDB_MODE=disabled python training/pretrain_weighted.py \
  --data_path data/scored_data.jsonl \
  --output_dir output/models/tau_0.7_seed42 \
  --weighting_method adaptive \
  --model_name models/Qwen/Qwen3-0.6B \
  --num_train_tokens 200000000 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_length 2048 \
  --learning_rate 5e-5 \
  --seed 42 \
  --logging_steps 10 \
  --save_steps 500 \
  --save_total_limit 2 \
  --save_only_model \
  --contamination_ratio 0.4 \
  --tau 0.7 \
  --bf16
