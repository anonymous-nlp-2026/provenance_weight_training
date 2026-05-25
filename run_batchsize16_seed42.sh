#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /root/provenance_weight_training
WANDB_MODE=disabled python training/pretrain_weighted.py \
  --data_path data/scored_data.jsonl \
  --output_dir output/models/batchsize16_seed42 \
  --model_name models/Qwen/Qwen3-0.6B \
  --weighting_method adaptive \
  --contamination_ratio 0.4 \
  --num_train_tokens 200000000 \
  --batch_size 16 \
  --gradient_accumulation_steps 1 \
  --n_min 2 \
  --learning_rate 5e-5 \
  --max_length 2048 \
  --seed 42 \
  --save_steps 500 \
  --logging_steps 10 \
  --bf16 \
  --save_total_limit 2 \
  --save_only_model \
  --eval_data_path data/human/eval_holdout.jsonl \
  --eval_steps 500
echo "DONE: batchsize16 seed42"
