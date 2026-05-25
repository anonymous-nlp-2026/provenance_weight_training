"""Download eval data for distribution matching validation.
Uses OpenWebText (web crawl, non-FineWeb) and Wikipedia.
"""
import json
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "~/.cache/huggingface")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import load_dataset

DATA_DIR = "/root/provenance_weight_training/data"
MAX_DOCS = 5000

# OpenWebText (web crawl text, independent of FineWeb)
print("Downloading OpenWebText (streaming)...", flush=True)
owt_path = os.path.join(DATA_DIR, "eval_openwebtext.jsonl")
try:
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    count = 0
    with open(owt_path, "w") as f:
        for example in ds:
            if count >= MAX_DOCS:
                break
            f.write(json.dumps({"text": example["text"]}) + "\n")
            count += 1
            if count % 1000 == 0:
                print(f"  OpenWebText: {count}/{MAX_DOCS}", flush=True)
    print(f"OpenWebText done: {count} docs -> {owt_path}", flush=True)
except Exception as e:
    print(f"OpenWebText failed: {e}", flush=True)
    # Fallback: try mc4 english
    print("Trying mc4 en fallback...", flush=True)
    try:
        ds = load_dataset("mc4", "en", split="validation", streaming=True)
        count = 0
        c4_path = os.path.join(DATA_DIR, "eval_c4.jsonl")
        with open(c4_path, "w") as f:
            for example in ds:
                if count >= MAX_DOCS:
                    break
                f.write(json.dumps({"text": example["text"]}) + "\n")
                count += 1
                if count % 1000 == 0:
                    print(f"  mc4: {count}/{MAX_DOCS}", flush=True)
        print(f"mc4 done: {count} docs -> {c4_path}", flush=True)
    except Exception as e2:
        print(f"mc4 also failed: {e2}", flush=True)

# Wikipedia
print("\nDownloading Wikipedia (streaming)...", flush=True)
wiki_path = os.path.join(DATA_DIR, "eval_wikipedia.jsonl")
try:
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    count = 0
    with open(wiki_path, "w") as f:
        for example in ds:
            if count >= MAX_DOCS:
                break
            f.write(json.dumps({"text": example["text"]}) + "\n")
            count += 1
            if count % 1000 == 0:
                print(f"  Wikipedia: {count}/{MAX_DOCS}", flush=True)
    print(f"Wikipedia done: {count} docs -> {wiki_path}", flush=True)
except Exception as e:
    print(f"Wikipedia failed: {e}", flush=True)
    print("Trying wikipedia alternative config...", flush=True)
    try:
        ds = load_dataset("legacy-datasets/wikipedia", "20220301.en", split="train", streaming=True)
        count = 0
        with open(wiki_path, "w") as f:
            for example in ds:
                if count >= MAX_DOCS:
                    break
                f.write(json.dumps({"text": example["text"]}) + "\n")
                count += 1
                if count % 1000 == 0:
                    print(f"  Wikipedia alt: {count}/{MAX_DOCS}", flush=True)
        print(f"Wikipedia alt done: {count} docs -> {wiki_path}", flush=True)
    except Exception as e2:
        print(f"Wikipedia alt also failed: {e2}", flush=True)

print("\nDone!", flush=True)
