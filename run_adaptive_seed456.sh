#!/bin/bash
# Adaptive b* training: seed 456
# Learns optimal b value during training via convergence-informed weighting
# 200M tokens, Qwen3-0.6B, scored_data.jsonl
# Output: output/models/adaptive_seed456/
# Expected runtime: ~2h

set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=~/.cache/huggingface
cd /root/provenance_weight_training

echo "=========================================="
echo "Starting adaptive b* seed=456 at $(date)"
echo "=========================================="

python training/pretrain_weighted.py \
    --data_path data/scored_data.jsonl \
    --output_dir output/models/adaptive_seed456 \
    --weighting_method adaptive \
    --model_name models/Qwen/Qwen3-0.6B \
    --num_train_tokens 200000000 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_length 2048 \
    --learning_rate 5e-5 \
    --seed 456 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 2 \
    --save_only_model \
    --contamination_ratio 0.4

echo "Completed adaptive b* seed=456 at $(date)"
