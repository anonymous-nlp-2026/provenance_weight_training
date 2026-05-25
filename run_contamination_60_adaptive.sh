#!/bin/bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /root/provenance_weight_training

python training/pretrain_weighted.py \
  --data_path data/scored_data_60.jsonl \
  --output_dir output/models/contamination_60_seed42 \
  --model_name models/Qwen/Qwen3-0.6B \
  --weighting_method adaptive \
  --contamination_ratio 0.6 \
  --num_train_tokens 200000000 \
  --n_min 2 --tau 0.8 \
  --batch_size 4 --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 --max_length 2048 \
  --seed 42 --logging_steps 10 \
  --save_steps 500 --save_total_limit 3 --save_only_model --bf16
