"""Standalone eval: compute eval_loss and ppl on holdout set."""
import json, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

MODEL_PATH = "/root/provenance_weight_training/output/models/adaptive_fix_seed42/final"
EVAL_PATH = "/root/provenance_weight_training/data/human/eval_holdout.jsonl"
MAX_LENGTH = 2048
BATCH_SIZE = 4

class EvalDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.examples[idx]["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
model.eval()
model.cuda(0)

print("Loading eval data...")
ds = EvalDataset(EVAL_PATH, tokenizer, MAX_LENGTH)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
print(f"  {len(ds)} examples, {len(loader)} batches")

total_loss = 0.0
total_tokens = 0
with torch.no_grad():
    for batch in tqdm(loader, desc="Eval"):
        input_ids = batch["input_ids"].cuda(0)
        attention_mask = batch["attention_mask"].cuda(0)
        labels = batch["labels"].cuda(0)
        labels[labels == tokenizer.pad_token_id] = -100
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        n_tokens = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

avg_loss = total_loss / total_tokens
ppl = math.exp(avg_loss)
print(f"\n=== Eval Results ===")
print(f"eval_loss = {avg_loss:.6f}")
print(f"eval_ppl  = {ppl:.4f}")
print(f"total_tokens = {total_tokens}")
print(f"total_examples = {len(ds)}")
