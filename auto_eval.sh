#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /root/provenance_weight_training

EVAL_DIR=eval_results
HOLDOUT=data/human/eval_holdout.jsonl
OWT=data/eval_openwebtext.jsonl
WIKI=data/eval_wikipedia.jsonl

run_eval() {
    local GPU=$1 PID=$2 MODEL_DIR=$3 EXP_NAME=$4
    
    if [ "$PID" != "0" ]; then
        echo "[$(date)] Waiting for PID $PID ($EXP_NAME) on GPU $GPU..."
        while kill -0 $PID 2>/dev/null; do sleep 30; done
        echo "[$(date)] PID $PID exited. Checking final model..."
    fi
    
    FINAL="${MODEL_DIR}/final"
    if [ ! -f "${FINAL}/config.json" ]; then
        echo "[$(date)] ERROR: ${FINAL}/config.json not found for $EXP_NAME"
        return 1
    fi
    
    echo "[$(date)] Starting eval for $EXP_NAME on GPU $GPU"
    for dataset_name in holdout owt wiki; do
        case $dataset_name in
            holdout) DATA=$HOLDOUT ;;
            owt) DATA=$OWT ;;
            wiki) DATA=$WIKI ;;
        esac
        OUT="${EVAL_DIR}/${EXP_NAME}_${dataset_name}.json"
        echo "[$(date)] Eval $EXP_NAME on $dataset_name..."
        CUDA_VISIBLE_DEVICES=$GPU python eval_perplexity.py \
            --checkpoints "$FINAL" \
            --data_path "$DATA" \
            --max_length 2048 --max_docs 5000 \
            --output_path "$OUT"
        echo "[$(date)] Done: $OUT (exit=$?)"
    done
    echo "[$(date)] All evals done for $EXP_NAME"
}

case "$1" in
    rho03_uniform_seed456)
        run_eval 2 0 output/models/rho03_uniform_seed456 rho03_uniform_seed456
        ;;
    bs8_nmin4_uniform_seed123)
        run_eval 3 835338 output/models/bs8_nmin4_uniform_seed123 bs8_nmin4_uniform_seed123
        ;;
    bs8_nmin4_adaptive_seed123)
        run_eval 1 835245 output/models/bs8_nmin4_adaptive_seed123 bs8_nmin4_adaptive_seed123
        ;;
    rho03_adaptive_seed456_r2)
        run_eval 0 834750 output/models/rho03_adaptive_seed456_r2 rho03_adaptive_seed456_r2
        ;;
esac
