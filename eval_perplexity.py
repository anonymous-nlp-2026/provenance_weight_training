"""Batch eval: compute held-out perplexity for multiple checkpoints."""
import argparse
import json
import math
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_texts(path, max_docs=None):
    texts = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_docs and i >= max_docs:
                break
            rec = json.loads(line)
            texts.append(rec["text"])
    return texts


def eval_perplexity(model, tokenizer, texts, max_length=2048, batch_size=8):
    stride = max_length // 2
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0

    for doc_idx, text in enumerate(texts):
        encodings = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = encodings.input_ids[0]
        seq_len = input_ids.size(0)
        if seq_len < 2:
            continue

        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            target_begin = max(begin, stride) if begin > 0 else 1

            ids = input_ids[begin:end].unsqueeze(0).to(device)
            target_ids = ids.clone()
            target_ids[0, :target_begin - begin] = -100

            with torch.no_grad(), torch.amp.autocast("cuda"):
                outputs = model(ids, labels=target_ids)

            n_tokens = (target_ids[0] != -100).sum().item()
            if n_tokens > 0:
                total_nll += outputs.loss.float().item() * n_tokens
                total_tokens += n_tokens

            if end == seq_len:
                break

        if (doc_idx + 1) % 500 == 0:
            interim_loss = total_nll / total_tokens if total_tokens else 0
            print(f"  [{doc_idx+1}/{len(texts)}] tokens={total_tokens}, loss={interim_loss:.4f}")

    avg_loss = total_nll / total_tokens if total_tokens > 0 else float("inf")
    avg_ppl = math.exp(avg_loss) if avg_loss < 30 else float("inf")
    return {"loss": avg_loss, "perplexity": avg_ppl, "num_tokens": total_tokens}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--max_docs", type=int, default=5000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--output_path", type=str, default="eval_perplexity_results.json")
    args = parser.parse_args()

    print(f"Loading eval data: {args.data_path} (max {args.max_docs} docs)")
    texts = load_texts(args.data_path, max_docs=args.max_docs)
    print(f"  Loaded {len(texts)} documents")

    results = {}
    for ckpt_path in args.checkpoints:
        name = os.path.basename(os.path.dirname(ckpt_path))
        if os.path.basename(ckpt_path) == "final":
            name = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
            name = name + "/final"
        print(f"\n{'='*60}")
        print(f"Evaluating: {ckpt_path}")
        print(f"{'='*60}")

        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).cuda().eval()

        result = eval_perplexity(model, tokenizer, texts, max_length=args.max_length)
        elapsed = time.time() - t0
        result["elapsed_seconds"] = round(elapsed, 1)
        result["checkpoint"] = ckpt_path

        short_name = ckpt_path.split("/models/")[-1].rstrip("/")
        results[short_name] = result
        print(f"  Loss: {result['loss']:.4f}  Perplexity: {result['perplexity']:.2f}  "
              f"Tokens: {result['num_tokens']}  Time: {elapsed:.0f}s")

        del model
        torch.cuda.empty_cache()

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_path}")

    print("\n" + "="*70)
    print(f"{'Checkpoint':<30} {'Loss':>8} {'Perplexity':>12} {'Tokens':>10}")
    print("-"*70)
    for name, r in results.items():
        print(f"{name:<30} {r['loss']:>8.4f} {r['perplexity']:>12.2f} {r['num_tokens']:>10}")
    print("="*70)


if __name__ == "__main__":
    main()
