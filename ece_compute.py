import json
import numpy as np

def compute_ece(qscores, depths, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (qscores >= bins[i]) & (qscores < bins[i+1])
        if mask.sum() == 0:
            continue
        avg_confidence = qscores[mask].mean()
        avg_accuracy = depths[mask].mean()
        ece += mask.sum() * abs(avg_confidence - avg_accuracy)
    return ece / len(qscores)

datasets = [
    ("TF-IDF (对照)", "data/scored_data.jsonl"),
    ("GPT-2 PPL", "data/scored_data_gpt2ppl_qscore.jsonl"),
    ("N-gram PPL", "data/scored_data_ngram_qscore.jsonl"),
]

for name, path in datasets:
    data = [json.loads(l) for l in open(path)]
    qs = np.array([d["q_score"] for d in data])
    depths = np.array([float(d["depth"] > 0) for d in data])
    ece = compute_ece(qs, depths)
    print(f"{name}: ECE = {ece:.4f} (N={len(data)})")
