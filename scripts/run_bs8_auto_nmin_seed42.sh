#!/bin/bash
set -e
# CUDA_VISIBLE_DEVICES set by submit_training_job
export WANDB_MODE=disabled
export HF_HOME=~/.cache/huggingface
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /root/provenance_weight_training

python training/pretrain_weighted.py \
  --data_path data/scored_data.jsonl \
  --output_dir output/models/bs8_auto_nmin_seed42/ \
  --model_name models/Qwen/Qwen3-0.6B \
  --weighting_method adaptive \
  --contamination_ratio 0.4 \
  --tau 0.8 \
  --num_train_tokens 200000000 \
  --batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --max_length 2048 \
  --seed 42 \
  --logging_steps 10 \
  --save_steps 500 \
  --eval_steps 500 \
  --save_total_limit 2 \
  --save_only_model \
  --bf16 \
  --eval_data_path data/human/eval_holdout.jsonl

echo "DONE: bs8_auto_nmin seed42 (n_min auto = max(2, 8//2) = 4)"
