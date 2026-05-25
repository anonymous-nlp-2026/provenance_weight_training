#!/bin/bash
# Manual batch eval for seed42 grid-b checkpoints
# Usage: bash scripts/run_eval_seed42.sh [gpu_id]
# The auto_eval_watcher.py should handle this automatically,
# but this script can be used as a fallback or for re-runs.

set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HOME=~/.cache/huggingface

cd /root/provenance_weight_training

EVAL_SCRIPT="eval_perplexity.py"
DATA_PATH="data/human/eval_holdout.jsonl"
EVAL_DIR="output/eval_results"
mkdir -p "$EVAL_DIR"

B_VALUES="0.5 1.0 1.5 2.0 3.0 5.0"

for b in $B_VALUES; do
    b_dir=$(echo "$b" | tr '.' '_')
    exp_id="grid_b${b_dir}_seed42"
    ckpt_path="output/models/${exp_id}/final"
    result_path="${EVAL_DIR}/${exp_id}_eval.json"

    if [ -f "$result_path" ]; then
        echo "[SKIP] ${exp_id}: eval result already exists at ${result_path}"
        continue
    fi

    if [ ! -f "${ckpt_path}/model.safetensors" ]; then
        echo "[SKIP] ${exp_id}: checkpoint not ready (no ${ckpt_path}/model.safetensors)"
        continue
    fi

    echo "=========================================="
    echo "Evaluating ${exp_id} on GPU ${GPU_ID} at $(date)"
    echo "=========================================="

    python "$EVAL_SCRIPT" \
        --checkpoints "$ckpt_path" \
        --data_path "$DATA_PATH" \
        --output_path "$result_path"

    echo "Completed ${exp_id} at $(date)"
    echo ""
done

echo "All seed42 evals complete at $(date)"

echo ""
echo "=== Summary ==="
for b in $B_VALUES; do
    b_dir=$(echo "$b" | tr '.' '_')
    result_path="${EVAL_DIR}/grid_b${b_dir}_seed42_eval.json"
    if [ -f "$result_path" ]; then
        echo "  b=${b}: $(cat "$result_path")"
    else
        echo "  b=${b}: NO RESULT"
    fi
done
