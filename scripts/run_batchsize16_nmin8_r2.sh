#!/bin/bash
# batchsize16_nmin8_r2: batch_size=16, n_min=8, rho=0.4 (r2 = contamination_ratio 0.4 variant)
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export HF_HOME=~/.cache/huggingface
export WANDB_MODE=disabled

cd /root/provenance_weight_training

SEED=42

python training/pretrain_weighted.py \
    --data_path data/scored_data.jsonl \
    --output_dir output/models/batchsize16_nmin8_r2 \
    --model_name models/Qwen/Qwen3-0.6B \
    --weighting_method adaptive \
    --tau 0.8 \
    --n_min 8 \
    --contamination_ratio 0.4 \
    --num_train_tokens 200000000 \
    --batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --max_length 2048 \
    --seed ${SEED} \
    --use_wandb \
    --wandb_project provenance-weight-training \
    --wandb_run_name batchsize16_nmin8_r2 \
    --save_steps 500 \
    --logging_steps 10 \
    --bf16 \
    --save_total_limit 3 \
    --save_only_model \
    --gradient_checkpointing

echo "DONE: batchsize16_nmin8_r2"
