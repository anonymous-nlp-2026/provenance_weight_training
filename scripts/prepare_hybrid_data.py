"""Hybrid filter: remove q>threshold samples, keep rest for adaptive reweighting."""
import json
import argparse

def filter_data(input_path, output_path, q_threshold=0.95):
    kept = 0
    removed = 0
    with open(input_path) as fin, open(output_path, 'w') as fout:
        for line in fin:
            d = json.loads(line)
            if d['q_score'] <= q_threshold:
                fout.write(line)
                kept += 1
            else:
                removed += 1
    print(f"Input: {kept+removed}, Kept: {kept}, Removed: {removed} ({removed/(kept+removed)*100:.1f}%)")
    return kept

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()
    filter_data(args.input, args.output, args.threshold)
