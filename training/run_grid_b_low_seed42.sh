#!/bin/bash
set -e
# Grid-b low values (0.1/0.2/0.3) serial training, seed=42, bf16, no inline eval

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/provenance_weight_training

# b=0.1
WANDB_MODE=disabled python training/pretrain_weighted.py \
    --data_path data/scored_data.jsonl \
    --output_dir output/models/grid_b0_1_seed42 \
    --weighting_method grid \
    --grid_b_value 0.1 \
    --model_name models/Qwen/Qwen3-0.6B \
    --num_train_tokens 200000000 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_length 2048 \
    --learning_rate 5e-5 \
    --seed 42 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 2 \
    --save_only_model \
    --contamination_ratio 0.4 \
    --bf16

echo "b=0.1 seed42 complete"

# b=0.2
WANDB_MODE=disabled python training/pretrain_weighted.py \
    --data_path data/scored_data.jsonl \
    --output_dir output/models/grid_b0_2_seed42 \
    --weighting_method grid \
    --grid_b_value 0.2 \
    --model_name models/Qwen/Qwen3-0.6B \
    --num_train_tokens 200000000 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_length 2048 \
    --learning_rate 5e-5 \
    --seed 42 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 2 \
    --save_only_model \
    --contamination_ratio 0.4 \
    --bf16

echo "b=0.2 seed42 complete"

# b=0.3
WANDB_MODE=disabled python training/pretrain_weighted.py \
    --data_path data/scored_data.jsonl \
    --output_dir output/models/grid_b0_3_seed42 \
    --weighting_method grid \
    --grid_b_value 0.3 \
    --model_name models/Qwen/Qwen3-0.6B \
    --num_train_tokens 200000000 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_length 2048 \
    --learning_rate 5e-5 \
    --seed 42 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 2 \
    --save_only_model \
    --contamination_ratio 0.4 \
    --bf16

echo "All grid-b low seed42 training complete"
