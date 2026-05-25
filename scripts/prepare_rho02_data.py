"""Generate scored_data_rho02.jsonl: rho=0.2 (20% synthetic, 80% human).

Adjusted total rows to 206111 (= 164889 / 0.8) to fit available human data.
Still > 200M tokens needed for training (206111 * 2048 ≈ 422M tokens).

Input:
  - data/human/train.jsonl          (164889 human docs)
  - data/synthetic/depth_1_merged.jsonl (340992 synthetic docs)
  - output/detector_tfidf/{tfidf_vectorizer,lr_model,calibration_results}.joblib

Output:
  - data/scored_data_rho02.jsonl    (206111 rows)
"""

import json
import math
import os
import random
import sys

import joblib
import numpy as np
from scipy.special import expit

SEED = 42
RHO = 0.2
N_HUMAN_AVAILABLE = 164889
TOTAL_ROWS = int(N_HUMAN_AVAILABLE / (1 - RHO))  # 206111

HUMAN_PATH = "data/human/train.jsonl"
SYNTH_PATH = "data/synthetic/depth_1_merged.jsonl"
MIX_OUT = "data/mixed_data_rho02.jsonl"
SCORED_OUT = "data/scored_data_rho02.jsonl"
VECTORIZER_PATH = "output/detector_tfidf/tfidf_vectorizer.joblib"
MODEL_PATH = "output/detector_tfidf/lr_model.joblib"
CAL_PATH = "output/detector_tfidf/calibration_results.json"

random.seed(SEED)
np.random.seed(SEED)

def load_jsonl(path, max_lines=None):
    docs = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            docs.append(json.loads(line))
    return docs

n_synth_target = math.ceil(TOTAL_ROWS * RHO)
n_human_target = TOTAL_ROWS - n_synth_target
print(f"Target: {TOTAL_ROWS} rows, rho={RHO}")
print(f"  Synthetic needed: {n_synth_target}")
print(f"  Human needed: {n_human_target}")

print("Loading human data...")
human_all = load_jsonl(HUMAN_PATH)
print(f"  {len(human_all)} human docs available")

print("Loading synthetic data...")
synth_all = load_jsonl(SYNTH_PATH)
print(f"  {len(synth_all)} synthetic docs available")

assert len(human_all) >= n_human_target, \
    f"Not enough human docs: need {n_human_target}, have {len(human_all)}"
assert len(synth_all) >= n_synth_target, \
    f"Not enough synthetic docs: need {n_synth_target}, have {len(synth_all)}"

random.shuffle(human_all)
random.shuffle(synth_all)
human = human_all[:n_human_target]
synth = synth_all[:n_synth_target]

mixed = human + synth
random.shuffle(mixed)
actual_synth = sum(1 for d in mixed if d.get("depth", 0) == 1)
actual_rho = actual_synth / len(mixed)
print(f"Mixed: {len(mixed)} docs (actual rho={actual_rho:.4f})")

print(f"Writing {MIX_OUT}...")
with open(MIX_OUT, "w") as f:
    for doc in mixed:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

print("Loading detector for scoring...")
vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)
with open(CAL_PATH) as f:
    cal = json.load(f)
T = cal["temperature"]

def score_text(text):
    words = text.lower().split()[:100]
    truncated = " ".join(words)
    X = vectorizer.transform([truncated])
    logit = model.decision_function(X)[0]
    return float(expit(logit / T))

print(f"Scoring {len(mixed)} docs -> {SCORED_OUT}...")
with open(SCORED_OUT, "w") as f:
    for i, doc in enumerate(mixed):
        doc["q_score"] = score_text(doc["text"])
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        if (i + 1) % 50000 == 0:
            print(f"  Scored {i + 1}/{len(mixed)}")

print(f"\nDone. Output: {SCORED_OUT}")
print(f"Total: {len(mixed)} rows, actual rho={actual_rho:.4f}")
