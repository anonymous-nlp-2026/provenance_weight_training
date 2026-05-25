#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/provenance_weight_training

for b in 0.1 0.2 0.3; do
    b_name=$(echo $b | tr '.' '_')
    output_dir="output/models/grid_b${b_name}_seed42"
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') === Starting training b=$b → $output_dir ==="
    
    WANDB_MODE=disabled python training/pretrain_weighted.py \
        --data_path data/scored_data.jsonl \
        --output_dir "$output_dir" \
        --model_name models/Qwen/Qwen3-0.6B \
        --weighting_method grid \
        --grid_b_value $b \
        --contamination_ratio 0.4 \
        --num_train_tokens 200000000 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 5e-5 \
        --max_length 2048 \
        --seed 42 \
        --save_steps 500 \
        --logging_steps 10 \
        --bf16 \
        --save_total_limit 3 \
        --save_only_model
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') === Completed b=$b ==="
done
echo "$(date '+%Y-%m-%d %H:%M:%S') === All 3 b-values done ==="
