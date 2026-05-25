#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /root/provenance_weight_training
export CUDA_VISIBLE_DEVICES=3

CKPT=output/models/adaptive_1_7b_seed456/final
OUTDIR=output/models/adaptive_1_7b_seed456/eval_results

echo "[$(date)] === Starting eval suite for adaptive_1_7b_seed456 ==="

echo "[$(date)] [1/4] Holdout PPL"
python -u eval_perplexity.py \
    --checkpoints $CKPT \
    --data_path data/human/eval_holdout.jsonl \
    --max_docs 5000 --max_length 2048 \
    --output_path $OUTDIR/ppl_holdout.json
echo "[$(date)] [1/4] DONE"

echo "[$(date)] [2/4] OWT PPL"
python -u eval_perplexity.py \
    --checkpoints $CKPT \
    --data_path data/eval_openwebtext.jsonl \
    --max_docs 5000 --max_length 2048 \
    --output_path $OUTDIR/ppl_openwebtext.json
echo "[$(date)] [2/4] DONE"

echo "[$(date)] [3/4] Wikipedia PPL"
python -u eval_perplexity.py \
    --checkpoints $CKPT \
    --data_path data/eval_wikipedia.jsonl \
    --max_docs 5000 --max_length 2048 \
    --output_path $OUTDIR/ppl_wikipedia.json
echo "[$(date)] [3/4] DONE"

echo "[$(date)] [4/4] Downstream 5-task"
python -m lm_eval \
    --model hf \
    --model_args pretrained=$CKPT,dtype=float16 \
    --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande \
    --batch_size auto \
    --num_fewshot 0 \
    --output_path $OUTDIR/downstream/
echo "[$(date)] [4/4] DONE"

echo "[$(date)] === ALL EVAL COMPLETE ==="
