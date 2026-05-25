#!/bin/bash
# GPU 0: adaptive model - all 3 eval sets
# GPU 1: uniform model - all 3 eval sets
# Run from /root/provenance_weight_training

ADAPTIVE=output/models/adaptive_fix_seed42/final
UNIFORM=output/models/qwen06b_uniform_seed42_v2/final
PY=/root/miniconda3/bin/python

run_eval() {
    local gpu=$1 ckpt=$2 data=$3 out=$4 log=$5
    echo "[$(date)] Starting: GPU=$gpu data=$data" | tee -a /tmp/eval_all.log
    CUDA_VISIBLE_DEVICES=$gpu $PY -u eval_perplexity.py \
        --checkpoints "$ckpt" \
        --data_path "$data" \
        --max_docs 5000 --max_length 2048 \
        --output_path "$out" 2>&1 | tee -a "$log"
    echo "[$(date)] Done: $out" | tee -a /tmp/eval_all.log
}

# GPU 0: adaptive
(
    run_eval 0 $ADAPTIVE data/human/eval_holdout.jsonl /tmp/eval_fineweb_adaptive.json /tmp/eval_fineweb_adaptive.log
    run_eval 0 $ADAPTIVE data/eval_openwebtext.jsonl /tmp/eval_owt_adaptive.json /tmp/eval_owt_adaptive.log
    run_eval 0 $ADAPTIVE data/eval_wikipedia.jsonl /tmp/eval_wiki_adaptive.json /tmp/eval_wiki_adaptive.log
) &
PID0=$!

# GPU 2: uniform
(
    run_eval 2 $UNIFORM data/human/eval_holdout.jsonl /tmp/eval_fineweb_uniform.json /tmp/eval_fineweb_uniform.log
    run_eval 2 $UNIFORM data/eval_openwebtext.jsonl /tmp/eval_owt_uniform.json /tmp/eval_owt_uniform.log
    run_eval 2 $UNIFORM data/eval_wikipedia.jsonl /tmp/eval_wiki_uniform.json /tmp/eval_wiki_uniform.log
) &
PID2=$!

echo "Adaptive evals on GPU 0: PID=$PID0"
echo "Uniform evals on GPU 2: PID=$PID2"
wait $PID0 $PID2
echo "[$(date)] All evals complete!" | tee -a /tmp/eval_all.log
