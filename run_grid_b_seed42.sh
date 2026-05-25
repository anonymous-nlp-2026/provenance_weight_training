#!/bin/bash
# Grid-b serial runner: trains 6 b values for seed 42
# Each run: 200M tokens, Qwen3-0.6B, weighted pretraining
# Input: data/scored_data.jsonl (371K docs with q_score)
# Output: output/models/grid_b{B}_seed42/ for each b

set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=~/.cache/huggingface
cd /root/provenance_weight_training

for b in 0.5 1.0 1.5 2.0 3.0 5.0; do
    echo "=========================================="
    echo "Starting grid-b=${b} seed=42 at $(date)"
    echo "=========================================="
    
    # Replace dots with underscores for directory name
    b_dir=$(echo $b | tr '.' '_')
    
    python training/pretrain_weighted.py \
        --data_path data/scored_data.jsonl \
        --output_dir output/models/grid_b${b_dir}_seed42 \
        --weighting_method grid \
        --grid_b_value $b \
        --model_name models/Qwen/Qwen3-0.6B \
        --num_train_tokens 200000000 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --max_length 2048 \
        --learning_rate 5e-5 \
        --seed 42 \
        --logging_steps 10 \
        --save_steps 500 \
        --eval_steps 500 \
        --save_total_limit 2 \
        --save_only_model \
        --contamination_ratio 0.4
    
    echo "Completed grid-b=${b} seed=42 at $(date)"
    echo ""
done

echo "All grid-b runs for seed 42 complete at $(date)"
