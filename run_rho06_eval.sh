#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export CUDA_VISIBLE_DEVICES=2
CKPT=output/models/contamination_60_adaptive_seed456/final
OUTDIR=output/eval_results

echo "=== [$(date)] OOD: openwebtext ==="
cd /root/provenance_weight_training
python eval_perplexity.py \
  --checkpoints $CKPT \
  --data_path data/eval_openwebtext.jsonl \
  --max_docs 5000 --max_length 2048 \
  --output_path $OUTDIR/contamination_60_adaptive_seed456_ood_owt.json

echo "=== [$(date)] OOD: wikipedia ==="
python eval_perplexity.py \
  --checkpoints $CKPT \
  --data_path data/eval_wikipedia.jsonl \
  --max_docs 5000 --max_length 2048 \
  --output_path $OUTDIR/contamination_60_adaptive_seed456_ood_wiki.json

echo "=== [$(date)] Downstream: lm_eval ==="
python -m lm_eval \
  --model hf \
  --model_args pretrained=$CKPT,dtype=float16 \
  --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande \
  --batch_size auto \
  --num_fewshot 0 \
  --output_path $OUTDIR/contamination_60_adaptive_seed456_downstream/

echo "=== [$(date)] ALL DONE ==="
