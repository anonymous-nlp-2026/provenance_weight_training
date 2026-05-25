#!/bin/bash
# Hybrid filter+reweight: remove q>0.95, then adaptive reweight on remainder (rho=0.6)
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export HF_HOME=~/.cache/huggingface
export CUDA_VISIBLE_DEVICES=3
export WANDB_MODE=offline

cd /root/provenance_weight_training

SEED=42

python training/pretrain_weighted.py \
    --data_path data/scored_data_hybrid_rho06.jsonl \
    --eval_data_path data/human/eval_holdout.jsonl \
    --output_dir output/models/hybrid_rho06_seed${SEED} \
    --model_name models/Qwen/Qwen3-0.6B \
    --weighting_method adaptive \
    --tau 0.7 \
    --n_min 2 \
    --contamination_ratio 0.6 \
    --num_train_tokens 200000000 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --max_length 2048 \
    --seed ${SEED} \
    --use_wandb \
    --wandb_project provenance-weight-training \
    --wandb_run_name hybrid_rho06_seed${SEED} \
    --eval_steps 500 \
    --save_steps 500 \
    --logging_steps 10 \
    --bf16 \
    --save_total_limit 3 \
    --save_only_model

echo "DONE: hybrid_rho06 seed${SEED}"
