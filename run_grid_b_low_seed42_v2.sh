#!/bin/bash
set -e
# Grid-b low range (0.1/0.2/0.3) serial training + eval, seed=42
# Each run: ~200M tokens, ~3.6h on single GPU
# After each training: holdout perplexity eval

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=~/.cache/huggingface
cd /root/provenance_weight_training

mkdir -p output/eval

for b in 0.1 0.2 0.3; do
    echo "=========================================="
    echo "Starting grid-b=${b} seed=42 at $(date)"
    echo "=========================================="

    b_dir=$(echo $b | tr '.' '_')

    WANDB_MODE=disabled python training/pretrain_weighted.py \
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
        --tau 0.8 \
        --seed 42 \
        --logging_steps 10 \
        --save_steps 500 \
        --eval_steps 500 \
        --save_total_limit 2 \
        --save_only_model \
        --contamination_ratio 0.4 \
        --bf16

    echo "=== Training b=${b} DONE, running eval ==="

    python eval_perplexity.py \
        --checkpoints output/models/grid_b${b_dir}_seed42/final \
        --data_path data/human/eval_holdout.jsonl \
        --max_docs 5000 \
        --output_path output/eval/grid_b${b_dir}_seed42_perplexity.json

    echo "=== b=${b} fully complete at $(date) ==="
    echo ""
done

echo "All grid-b low seed42 training + eval complete at $(date)"
