#!/bin/bash
# Adaptive weighting with fixed tau=0.8, n_min=2, rho=0.4, seed=42
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export HF_HOME=~/.cache/huggingface
export WANDB_MODE=disabled

cd /root/provenance_weight_training

SEED=42

python training/pretrain_weighted.py \
    --data_path data/scored_data.jsonl \
    --eval_data_path data/human/eval_holdout.jsonl \
    --output_dir output/models/adaptive_fix_seed42 \
    --model_name models/Qwen/Qwen3-0.6B \
    --weighting_method adaptive \
    --tau 0.8 \
    --n_min 2 \
    --contamination_ratio 0.4 \
    --num_train_tokens 200000000 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --max_length 2048 \
    --seed ${SEED} \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 10 \
    --bf16 \
    --save_total_limit 3 \
    --save_only_model

echo "DONE: adaptive_fix seed${SEED} (tau=0.8, n_min=2, rho=0.4)"
