#!/bin/bash
# Batch-adaptive tau: rho=0.4, seed=42
# tau_batch = clamp(alpha_eff_natural + 0.3, 0.5, 0.8)
set -e
export CUDA_VISIBLE_DEVICES=2
export WANDB_MODE=offline
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export HF_HOME=~/.cache/huggingface

cd /root/provenance_weight_training

SEED=42

python training/pretrain_weighted.py \
    --data_path data/scored_data.jsonl \
    --eval_data_path data/human/eval_holdout.jsonl \
    --output_dir output/models/batch_adaptive_tau_rho04_seed${SEED} \
    --model_name models/Qwen/Qwen3-0.6B \
    --weighting_method adaptive \
    --tau 0.8 \
    --adaptive_tau \
    --tau_delta 0.3 \
    --tau_min 0.5 \
    --n_min 2 \
    --contamination_ratio 0.4 \
    --num_train_tokens 200000000 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --max_length 2048 \
    --seed ${SEED} \
    --eval_steps 500 \
    --save_steps 500 \
    --logging_steps 10 \
    --bf16 \
    --save_total_limit 3 \
    --save_only_model

echo "DONE: batch_adaptive_tau rho04 seed${SEED}"
