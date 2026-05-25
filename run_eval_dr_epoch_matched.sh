#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
cd /root/provenance_weight_training

MODEL=output/models/dr_epoch_matched_indep_lr_seed42
EVAL_DIR=${MODEL}/eval_results
mkdir -p ${EVAL_DIR}

# Wait for holdout eval to finish (PID 609688)
echo "Waiting for holdout eval (PID 609688) to finish..."
while kill -0 609688 2>/dev/null; do sleep 10; done
echo "Holdout eval done."

# OOD OpenWebText
echo "Starting OOD OpenWebText eval..."
CUDA_VISIBLE_DEVICES=1 python eval_perplexity.py \
  --checkpoints ${MODEL}/final \
  --data_path data/eval_openwebtext.jsonl \
  --max_docs 5000 --max_length 2048 \
  --output_path ${EVAL_DIR}/ppl_openwebtext.json
echo "OOD OpenWebText eval done."

# OOD Wikipedia
echo "Starting OOD Wikipedia eval..."
CUDA_VISIBLE_DEVICES=1 python eval_perplexity.py \
  --checkpoints ${MODEL}/final \
  --data_path data/eval_wikipedia.jsonl \
  --max_docs 5000 --max_length 2048 \
  --output_path ${EVAL_DIR}/ppl_wikipedia.json
echo "OOD Wikipedia eval done."

echo "ALL EVALS COMPLETE"
