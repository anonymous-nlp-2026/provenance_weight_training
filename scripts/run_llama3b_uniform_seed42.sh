#!/bin/bash
# Llama-3.2-3B uniform baseline (cross-model validation)
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export HF_HOME=~/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

cd /root/provenance_weight_training

SEED=42

python training/pretrain_weighted.py \
    --data_path data/scored_data.jsonl \
    --eval_data_path data/human/eval_holdout.jsonl \
    --output_dir output/models/llama3b_uniform_seed${SEED} \
    --model_name models/meta-llama/Llama-3.2-3B \
    --weighting_method uniform \
    --contamination_ratio 0.4 \
    --num_train_tokens 200000000 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --max_length 2048 \
    --seed ${SEED} \
    --use_wandb \
    --wandb_project provenance-weight-training \
    --wandb_run_name llama3b_uniform_seed${SEED} \
    --eval_steps 500 \
    --save_steps 500 \
    --logging_steps 10 \
    --bf16 \
    --gradient_checkpointing \
    --save_total_limit 3 \
    --save_only_model

echo "DONE: llama3b_uniform seed${SEED}"
