"""Score mixed_data.jsonl with TF-IDF + LR detector."""

import json
import joblib
import numpy as np
from scipy.special import expit

vectorizer = joblib.load("output/detector_tfidf/tfidf_vectorizer.joblib")
model = joblib.load("output/detector_tfidf/lr_model.joblib")

with open("output/detector_tfidf/calibration_results.json") as f:
    cal = json.load(f)
T = cal["temperature"]

def score_text(text):
    words = text.lower().split()[:100]
    truncated = " ".join(words)
    X = vectorizer.transform([truncated])
    logit = model.decision_function(X)[0]
    q_score = float(expit(logit / T))
    return q_score

with open("data/mixed_data.jsonl") as fin, open("data/scored_data.jsonl", "w") as fout:
    for i, line in enumerate(fin):
        rec = json.loads(line)
        rec["q_score"] = score_text(rec["text"])
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (i + 1) % 50000 == 0:
            print(f"Scored {i+1} docs...")

print(f"Done. Total docs scored: {i+1}")
