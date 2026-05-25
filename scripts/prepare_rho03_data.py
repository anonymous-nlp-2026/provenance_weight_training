"""Generate scored_data_rho03.jsonl: rho=0.3 (30% synthetic, 70% human).

Resamples from scored_data.jsonl which already contains q_scores.
Uses same SEED=42 as other prepare scripts for reproducibility.

Input:
  - data/scored_data.jsonl (371499 rows: 164889 human + 206610 synthetic)

Output:
  - data/scored_data_rho03.jsonl (235555 rows: 164888 human + 70667 synthetic)
"""

import json
import math
import random

SEED = 42
RHO = 0.3
N_HUMAN_AVAILABLE = 164889
TOTAL_ROWS = int(N_HUMAN_AVAILABLE / (1 - RHO))  # 235555

INPUT_PATH = "data/scored_data.jsonl"
OUTPUT_PATH = "data/scored_data_rho03.jsonl"

random.seed(SEED)

n_synth_target = math.ceil(TOTAL_ROWS * RHO)
n_human_target = TOTAL_ROWS - n_synth_target
print(f"Target: {TOTAL_ROWS} rows, rho={RHO}")
print(f"  Synthetic needed: {n_synth_target}")
print(f"  Human needed: {n_human_target}")

print(f"Loading {INPUT_PATH}...")
human_docs = []
synth_docs = []
with open(INPUT_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        doc = json.loads(line)
        if doc.get("depth", 0) == 0:
            human_docs.append(doc)
        else:
            synth_docs.append(doc)

print(f"  {len(human_docs)} human docs, {len(synth_docs)} synthetic docs")

assert len(human_docs) >= n_human_target, \
    f"Not enough human docs: need {n_human_target}, have {len(human_docs)}"
assert len(synth_docs) >= n_synth_target, \
    f"Not enough synthetic docs: need {n_synth_target}, have {len(synth_docs)}"

random.shuffle(human_docs)
random.shuffle(synth_docs)
human = human_docs[:n_human_target]
synth = synth_docs[:n_synth_target]

mixed = human + synth
random.shuffle(mixed)
actual_synth = sum(1 for d in mixed if d.get("depth", 0) == 1)
actual_rho = actual_synth / len(mixed)
print(f"Mixed: {len(mixed)} docs (actual rho={actual_rho:.4f})")

print(f"Writing {OUTPUT_PATH}...")
with open(OUTPUT_PATH, "w") as f:
    for doc in mixed:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

print(f"\nDone. Output: {OUTPUT_PATH}")
print(f"Total: {len(mixed)} rows, actual rho={actual_rho:.4f}")
