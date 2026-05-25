"""
Detector miscalibration via temperature scaling.
Applies T-scaling to q_scores: q_scaled = sigmoid(logit(q) / T)
T > 1.0 makes scores more uniform (less confident)
T < 1.0 makes scores more extreme (more confident)

Input: scored_data.jsonl (371K samples with 'q_score' field)
Output: scored_data_T2.0.jsonl and scored_data_T3.0.jsonl
"""
import json
import math
import sys

def logit(p, eps=1e-7):
    p = max(eps, min(1 - eps, p))
    return math.log(p / (1 - p))

def sigmoid(x):
    if x > 500: return 1.0
    if x < -500: return 0.0
    return 1.0 / (1.0 + math.exp(-x))

def rescale_score(q, temperature):
    return sigmoid(logit(q) / temperature)

def process(input_path, output_path, temperature):
    print(f"Processing T={temperature}: {input_path} -> {output_path}")
    count = 0
    q_orig_sum = 0
    q_new_sum = 0
    
    with open(input_path) as fin, open(output_path, 'w') as fout:
        for line in fin:
            obj = json.loads(line)
            q_orig = obj['q_score']
            q_new = rescale_score(q_orig, temperature)
            q_orig_sum += q_orig
            q_new_sum += q_new
            obj['q_score_original'] = q_orig
            obj['q_score'] = round(q_new, 6)
            fout.write(json.dumps(obj) + '\n')
            count += 1
            if count % 100000 == 0:
                print(f"  {count} samples processed")
    
    print(f"Done: {count} samples")
    print(f"  q_score mean: {q_orig_sum/count:.4f} -> {q_new_sum/count:.4f}")
    return count

def main():
    input_path = '/root/provenance_weight_training/data/scored_data.jsonl'
    
    for T in [2.0, 3.0]:
        output_path = f'/root/provenance_weight_training/data/scored_data_T{T}.jsonl'
        process(input_path, output_path, T)
    
    print("\nAll done. Files created:")
    print("  scored_data_T2.0.jsonl")
    print("  scored_data_T3.0.jsonl")

if __name__ == '__main__':
    main()
