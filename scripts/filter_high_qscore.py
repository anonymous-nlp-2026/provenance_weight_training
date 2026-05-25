"""Filter scored_data.jsonl by q_score threshold (detect-and-remove baseline)."""
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Remove samples with q_score > threshold")
    args = parser.parse_args()

    kept, removed = 0, 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("q_score", 0.0) <= args.threshold:
                fout.write(json.dumps(item) + "\n")
                kept += 1
            else:
                removed += 1

    print(f"Threshold: {args.threshold}")
    print(f"Kept: {kept}, Removed: {removed}, Total: {kept + removed}")
    print(f"Output: {args.output}")

if __name__ == "__main__":
    main()
