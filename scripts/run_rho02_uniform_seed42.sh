#!/bin/bash
# rho=0.2 uniform baseline
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export HF_HOME=~/.cache/huggingface
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=offline

cd /root/provenance_weight_training

python training/pretrain_weighted.py \
    --data_path data/scored_data_rho02.jsonl \
    --eval_data_path data/human/eval_holdout.jsonl \
    --output_dir output/models/rho02_uniform_seed42 \
    --model_name models/Qwen/Qwen3-0.6B \
    --weighting_method uniform \
    --contamination_ratio 0.2 \
    --num_train_tokens 200000000 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --max_length 2048 \
    --seed 42 \
    --use_wandb \
    --wandb_project provenance-weight-training \
    --wandb_run_name rho02_uniform_seed42 \
    --eval_steps 500 \
    --save_steps 500 \
    --logging_steps 10 \
    --bf16 \
    --save_total_limit 3 \
    --save_only_model

echo "DONE: rho02_uniform seed42"
