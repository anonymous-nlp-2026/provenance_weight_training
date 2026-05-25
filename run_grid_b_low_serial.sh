#!/bin/bash
set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/provenance_weight_training

echo "=== Starting b=0.1 at $(date) ==="
python training/pretrain_weighted.py \
  --data_path data/scored_data.jsonl \
  --output_dir output/models/grid_b0_1_seed42 \
  --model_name models/Qwen/Qwen3-0.6B \
  --weighting_method grid \
  --grid_b_value 0.1 \
  --contamination_ratio 0.4 \
  --num_train_tokens 200000000 \
  --batch_size 4 --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 --max_length 2048 \
  --seed 42 --logging_steps 10 \
  --save_steps 500 --save_total_limit 3 --save_only_model --bf16

echo "=== Starting b=0.2 at $(date) ==="
python training/pretrain_weighted.py \
  --data_path data/scored_data.jsonl \
  --output_dir output/models/grid_b0_2_seed42 \
  --model_name models/Qwen/Qwen3-0.6B \
  --weighting_method grid \
  --grid_b_value 0.2 \
  --contamination_ratio 0.4 \
  --num_train_tokens 200000000 \
  --batch_size 4 --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 --max_length 2048 \
  --seed 42 --logging_steps 10 \
  --save_steps 500 --save_total_limit 3 --save_only_model --bf16

echo "=== Starting b=0.3 at $(date) ==="
python training/pretrain_weighted.py \
  --data_path data/scored_data.jsonl \
  --output_dir output/models/grid_b0_3_seed42 \
  --model_name models/Qwen/Qwen3-0.6B \
  --weighting_method grid \
  --grid_b_value 0.3 \
  --contamination_ratio 0.4 \
  --num_train_tokens 200000000 \
  --batch_size 4 --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 --max_length 2048 \
  --seed 42 --logging_steps 10 \
  --save_steps 500 --save_total_limit 3 --save_only_model --bf16

echo "=== All done at $(date) ==="
