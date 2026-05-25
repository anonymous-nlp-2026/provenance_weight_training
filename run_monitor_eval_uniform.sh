#!/bin/bash
set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

MODEL_DIR="output/models/uniform_epoch_matched_indep_lr_seed42"
FINAL_DIR="${MODEL_DIR}/final"
DATA_DIR="data"
CODE_DIR="/root/provenance_weight_training"
LOG="/root/runs/provenance_weight_training/uniform_epoch_matched_eval.log"

export CUDA_VISIBLE_DEVICES=2

echo "[$(date)] Waiting for training to complete..." | tee -a "$LOG"

while true; do
  if [ -d "$FINAL_DIR" ] && [ -f "$FINAL_DIR/config.json" ]; then
    echo "[$(date)] TRAINING COMPLETE - final checkpoint found" | tee -a "$LOG"
    break
  fi
  if ! ps aux | grep -v grep | grep "uniform_epoch_matched_indep_lr" > /dev/null 2>&1; then
    if [ -d "$FINAL_DIR" ] && [ -f "$FINAL_DIR/config.json" ]; then
      echo "[$(date)] Process ended, final checkpoint exists" | tee -a "$LOG"
      break
    else
      echo "[$(date)] ERROR: Process ended but no final checkpoint!" | tee -a "$LOG"
      ls -la "$MODEL_DIR/" | tee -a "$LOG"
      exit 1
    fi
  fi
  sleep 30
done

echo "[$(date)] Starting eval on holdout..." | tee -a "$LOG"

cd "$CODE_DIR"

# Eval 1: Holdout
python eval_perplexity.py \
  --checkpoints "$FINAL_DIR" \
  --data_path "$DATA_DIR/human/eval_holdout.jsonl" \
  --max_docs 5000 --max_length 2048 \
  --output_path "$MODEL_DIR/eval_holdout.json" \
  2>&1 | tee -a "$LOG"

echo "[$(date)] Starting eval on OOD OpenWebText..." | tee -a "$LOG"

# Eval 2: OOD OpenWebText
python eval_perplexity.py \
  --checkpoints "$FINAL_DIR" \
  --data_path "$DATA_DIR/eval_openwebtext.jsonl" \
  --max_docs 5000 --max_length 2048 \
  --output_path "$MODEL_DIR/eval_owt.json" \
  2>&1 | tee -a "$LOG"

echo "[$(date)] Starting eval on OOD Wikipedia..." | tee -a "$LOG"

# Eval 3: OOD Wikipedia
python eval_perplexity.py \
  --checkpoints "$FINAL_DIR" \
  --data_path "$DATA_DIR/eval_wikipedia.jsonl" \
  --max_docs 5000 --max_length 2048 \
  --output_path "$MODEL_DIR/eval_wiki.json" \
  2>&1 | tee -a "$LOG"

echo "[$(date)] ALL EVALS DONE" | tee -a "$LOG"

# Summary
echo "" | tee -a "$LOG"
echo "=== RESULTS SUMMARY ===" | tee -a "$LOG"
for f in "$MODEL_DIR"/eval_*.json; do
  fname=$(basename "$f")
  python3 -c "
import json
d = json.load(open('$f'))
for k,v in d.items():
    print(f'  {fname}: ppl={v[\"perplexity\"]:.6f}, loss={v[\"loss\"]:.6f}, tokens={v[\"num_tokens\"]}')
" | tee -a "$LOG"
done
