#!/bin/bash
# Grid search: fixed b=0.2, seed=42
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export HF_HOME=~/.cache/huggingface
export WANDB_MODE=disabled
# CUDA_VISIBLE_DEVICES set by submit_training_job
cd /root/provenance_weight_training

python training/pretrain_weighted.py \
    --data_path data/scored_data.jsonl \
    --eval_data_path data/human/eval_holdout.jsonl \
    --output_dir output/models/grid_b0_2_seed42 \
    --model_name models/Qwen/Qwen3-0.6B \
    --weighting_method grid \
    --grid_b_value 0.2 \
    --contamination_ratio 0.4 \
    --num_train_tokens 200000000 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --max_length 2048 \
    --seed 42 \
    --eval_steps 500 \
    --save_steps 500 \
    --logging_steps 10 \
    --bf16 \
    --save_total_limit 3 \
    --save_only_model

echo "DONE: grid_b0_2 seed42"
