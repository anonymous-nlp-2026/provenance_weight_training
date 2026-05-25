#!/bin/bash
set -e
cd /root/provenance_weight_training
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base

MODELS_DIR=output/models
DATA_DIR=data
OUT_DIR=output/eval_results

mkdir -p $OUT_DIR/bs8_nmin4_adaptive_seed42
mkdir -p $OUT_DIR/bs8_nmin4_uniform_seed42

echo "=== Holdout eval ==="
python eval_perplexity.py \
  --checkpoints $MODELS_DIR/bs8_nmin4_adaptive_seed42/final/ $MODELS_DIR/bs8_nmin4_uniform_seed42/final/ \
  --data_path $DATA_DIR/human/eval_holdout.jsonl \
  --output_path /tmp/bs8_holdout.json

echo "=== OWT eval ==="
python eval_perplexity.py \
  --checkpoints $MODELS_DIR/bs8_nmin4_adaptive_seed42/final/ $MODELS_DIR/bs8_nmin4_uniform_seed42/final/ \
  --data_path $DATA_DIR/eval_openwebtext.jsonl \
  --output_path /tmp/bs8_owt.json

echo "=== Wiki eval ==="
python eval_perplexity.py \
  --checkpoints $MODELS_DIR/bs8_nmin4_adaptive_seed42/final/ $MODELS_DIR/bs8_nmin4_uniform_seed42/final/ \
  --data_path $DATA_DIR/eval_wikipedia.jsonl \
  --output_path /tmp/bs8_wiki.json

echo "=== ALL DONE ==="
echo "--- HOLDOUT ---"
cat /tmp/bs8_holdout.json
echo ""
echo "--- OWT ---"
cat /tmp/bs8_owt.json
echo ""
echo "--- WIKI ---"
cat /tmp/bs8_wiki.json
