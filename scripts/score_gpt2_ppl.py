import torch
import json
import time
import os
import sys
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

INPUT_FILE = "/root/provenance_weight_training/data/scored_data.jsonl"
OUTPUT_FILE = "/root/provenance_weight_training/data/scored_data_gpt2ppl.jsonl"
CHECKPOINT_FILE = OUTPUT_FILE + ".checkpoint"
MAX_LENGTH = 1024
BATCH_SIZE = 32
LOG_EVERY = 5000

def compute_ppl_batch(texts, model, tokenizer, max_length=1024):
    """Compute perplexity for a batch of texts using padding."""
    encodings = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    input_ids = encodings.input_ids.cuda()
    attention_mask = encodings.attention_mask.cuda()

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_labels.size())

    valid_counts = shift_mask.sum(dim=1).float()
    valid_counts = valid_counts.clamp(min=1)
    per_sample_loss = (losses * shift_mask.float()).sum(dim=1) / valid_counts
    ppls = torch.exp(per_sample_loss).cpu().tolist()
    return ppls


def main():
    print("Loading GPT-2 medium...", flush=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium").cuda().eval()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded.", flush=True)

    # Check for checkpoint
    start_line = 0
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            start_line = int(f.read().strip())
        print(f"Resuming from line {start_line}", flush=True)

    # Count total lines
    total_lines = sum(1 for _ in open(INPUT_FILE))
    print(f"Total samples: {total_lines}", flush=True)

    # Open output file in append mode if resuming
    mode = "a" if start_line > 0 else "w"
    out_f = open(OUTPUT_FILE, mode)

    batch_texts = []
    batch_records = []
    processed = start_line
    t0 = time.time()

    with open(INPUT_FILE) as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            rec = json.loads(line)
            text = rec.get("text", "")
            if not text or len(text.strip()) < 10:
                out_rec = {
                    "text": text,
                    "depth": rec.get("depth"),
                    "q_score": rec.get("q_score"),
                    "gpt2_ppl": -1.0,
                }
                out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                processed += 1
                continue

            batch_texts.append(text)
            batch_records.append(rec)

            if len(batch_texts) >= BATCH_SIZE:
                ppls = compute_ppl_batch(batch_texts, model, tokenizer, MAX_LENGTH)
                for rec_b, ppl in zip(batch_records, ppls):
                    out_rec = {
                        "text": rec_b.get("text", ""),
                        "depth": rec_b.get("depth"),
                        "q_score": rec_b.get("q_score"),
                        "gpt2_ppl": round(ppl, 4),
                    }
                    out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                batch_texts = []
                batch_records = []
                processed += BATCH_SIZE

                if processed % LOG_EVERY < BATCH_SIZE:
                    elapsed = time.time() - t0
                    speed = (processed - start_line) / elapsed if elapsed > 0 else 0
                    eta = (total_lines - processed) / speed if speed > 0 else 0
                    print(
                        f"[{processed}/{total_lines}] {speed:.1f} samples/sec, ETA {eta/60:.1f} min",
                        flush=True,
                    )
                    out_f.flush()
                    with open(CHECKPOINT_FILE, "w") as cf:
                        cf.write(str(processed))

    # Final batch
    if batch_texts:
        ppls = compute_ppl_batch(batch_texts, model, tokenizer, MAX_LENGTH)
        for rec_b, ppl in zip(batch_records, ppls):
            out_rec = {
                "text": rec_b.get("text", ""),
                "depth": rec_b.get("depth"),
                "q_score": rec_b.get("q_score"),
                "gpt2_ppl": round(ppl, 4),
            }
            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
        processed += len(batch_texts)

    out_f.close()
    elapsed = time.time() - t0
    print(f"\nDone! {processed} samples in {elapsed:.1f}s ({processed/elapsed:.1f} samples/sec)")
    print(f"Output: {OUTPUT_FILE}")

    # Clean checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    # Print first 10 entries
    print("\nFirst 10 entries:")
    with open(OUTPUT_FILE) as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            rec = json.loads(line)
            print(f"  depth={rec['depth']} q_score={rec['q_score']:.4f} gpt2_ppl={rec['gpt2_ppl']:.2f}")


if __name__ == "__main__":
    main()
