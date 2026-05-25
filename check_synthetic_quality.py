"""
Synthetic data quality check.

Computes:
1. Distinct-3 ratio (distinct trigrams / total trigrams) — corpus level
2. Degeneration rate: fraction of documents with distinct-3 < 0.1
3. Average document length vs effective (non-repeated) length
4. Sample of worst degenerate documents for manual inspection

Usage:
    python check_synthetic_quality.py --input data/synthetic/depth_1_merged.jsonl
"""

import argparse
import json
import sys
from collections import Counter


def get_trigrams(tokens):
    return [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]


def distinct_n_ratio(tokens, n=3):
    if len(tokens) < n:
        return 1.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 1.0
    return len(set(ngrams)) / len(ngrams)


def effective_length(tokens, n=5):
    """Non-repeated length: count tokens not in any repeated n-gram (n>=5)."""
    if len(tokens) < n:
        return len(tokens)
    ngram_counts = Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    repeated_positions = set()
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i:i+n])
        if ngram_counts[ng] > 1:
            for j in range(i, i + n):
                repeated_positions.add(j)
    return len(tokens) - len(repeated_positions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to merged synthetic JSONL")
    parser.add_argument("--max_docs", type=int, default=0, help="Max docs to process (0=all)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    docs = []
    with open(args.input) as f:
        for i, line in enumerate(f):
            if args.max_docs > 0 and i >= args.max_docs:
                break
            try:
                rec = json.loads(line)
                docs.append(rec)
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(docs)} documents")

    # Per-document metrics
    all_tokens = []
    doc_d3 = []
    doc_lengths = []
    doc_eff_lengths = []
    degenerate_samples = []

    for doc in docs:
        text = doc.get("text", "")
        tokens = text.lower().split()
        doc_lengths.append(len(tokens))
        all_tokens.extend(tokens)

        d3 = distinct_n_ratio(tokens, n=3)
        doc_d3.append(d3)

        eff_len = effective_length(tokens, n=5)
        doc_eff_lengths.append(eff_len)

        if d3 < 0.1 and len(degenerate_samples) < 10:
            degenerate_samples.append({
                "index": len(doc_d3) - 1,
                "distinct_3": round(d3, 4),
                "length": len(tokens),
                "effective_length": eff_len,
                "text_preview": text[:500],
            })

    # Corpus-level distinct-3
    corpus_trigrams = get_trigrams(all_tokens)
    corpus_d3 = len(set(corpus_trigrams)) / len(corpus_trigrams) if corpus_trigrams else 0

    # Degeneration rate
    degen_count = sum(1 for d in doc_d3 if d < 0.1)
    degen_rate = degen_count / len(doc_d3) if doc_d3 else 0

    # Length stats
    avg_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
    avg_eff = sum(doc_eff_lengths) / len(doc_eff_lengths) if doc_eff_lengths else 0
    eff_ratio = avg_eff / avg_len if avg_len > 0 else 0

    # Mean doc distinct-3
    mean_doc_d3 = sum(doc_d3) / len(doc_d3) if doc_d3 else 0

    results = {
        "num_documents": len(docs),
        "corpus_distinct_3": round(corpus_d3, 4),
        "mean_doc_distinct_3": round(mean_doc_d3, 4),
        "degenerate_count": degen_count,
        "degenerate_rate": round(degen_rate, 4),
        "avg_doc_length_tokens": round(avg_len, 1),
        "avg_effective_length_tokens": round(avg_eff, 1),
        "effective_length_ratio": round(eff_ratio, 4),
        "degenerate_samples": degenerate_samples,
    }

    print(f"\n=== Synthetic Data Quality Report ===")
    print(f"Documents:           {len(docs)}")
    print(f"Corpus distinct-3:   {corpus_d3:.4f}")
    print(f"Mean doc distinct-3: {mean_doc_d3:.4f}")
    print(f"Degenerate docs:     {degen_count} / {len(docs)} ({degen_rate:.1%})")
    print(f"Avg doc length:      {avg_len:.0f} tokens")
    print(f"Avg effective length:{avg_eff:.0f} tokens ({eff_ratio:.1%} non-repeated)")
    
    if degen_rate > 0.3:
        print(f"\n⚠️  WARNING: Degeneration rate {degen_rate:.1%} > 30% threshold")
    else:
        print(f"\n✓ Degeneration rate {degen_rate:.1%} within acceptable range (<30%)")

    if degenerate_samples:
        print(f"\n--- Worst degenerate samples (top {len(degenerate_samples)}) ---")
        for s in degenerate_samples[:3]:
            print(f"  [idx={s['index']}] d3={s['distinct_3']}, len={s['length']}, eff={s['effective_length']}")
            print(f"    {s['text_preview'][:200]}...")

    out_path = args.output or args.input.replace(".jsonl", "_quality.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
