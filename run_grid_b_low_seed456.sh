#!/bin/bash
# Grid-b low range runner: trains b=0.1/0.2/0.3 for seed 456
# Purpose: extend grid search below b=0.5 boundary
# Input: data/scored_data.jsonl (371K docs with q_score)
# Output: output/models/grid_b{B}_seed456/ for b in {0_1, 0_2, 0_3}
# Depends: training/pretrain_weighted.py, models/Qwen/Qwen3-0.6B

set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=~/.cache/huggingface
cd /root/provenance_weight_training

for b in 0.1 0.2 0.3; do
    echo "=========================================="
    echo "Starting grid-b=${b} seed=456 at $(date)"
    echo "=========================================="
    
    b_dir=$(echo $b | tr '.' '_')
    
    WANDB_MODE=disabled python training/pretrain_weighted.py \
        --data_path data/scored_data.jsonl \
        --output_dir output/models/grid_b${b_dir}_seed456 \
        --weighting_method grid \
        --grid_b_value $b \
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
    
    echo "=== b=${b} DONE ==="
    echo "Completed grid-b=${b} seed=456 at $(date)"
    echo ""
done

echo "All grid-b low runs for seed 456 complete at $(date)"
