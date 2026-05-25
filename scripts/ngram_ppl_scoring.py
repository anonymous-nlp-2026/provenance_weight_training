"""
N-gram PPL scoring for synthetic text detection.
Trains a character 5-gram LM on a sample of the data, then scores all samples.
Higher PPL = more likely synthetic (synthetic text tends to be "smoother" at character level).
Input: scored_data.jsonl (371K samples with 'text' and 'q_score' fields)
Output: scored_data_ngram.jsonl (same + 'ngram_ppl' field)
"""
import json
import math
import sys
from collections import defaultdict, Counter
import random

def build_ngram_model(texts, n=5, sample_size=50000):
    """Build character n-gram LM from a sample of texts."""
    if len(texts) > sample_size:
        train_texts = random.sample(texts, sample_size)
    else:
        train_texts = texts
    
    ngram_counts = defaultdict(Counter)
    for text in train_texts:
        text = text.lower()
        padded = '^' * (n-1) + text + '$'
        for i in range(len(padded) - n + 1):
            context = padded[i:i+n-1]
            char = padded[i+n-1]
            ngram_counts[context][char] += 1
    
    return ngram_counts

def score_text(text, ngram_counts, n=5, alpha=0.01):
    """Compute character-level PPL using the n-gram model with add-alpha smoothing."""
    text = text.lower()
    padded = '^' * (n-1) + text + '$'
    log_prob = 0.0
    count = 0
    vocab_size = 256

    for i in range(len(padded) - n + 1):
        context = padded[i:i+n-1]
        char = padded[i+n-1]
        
        context_counts = ngram_counts.get(context, Counter())
        total = sum(context_counts.values())
        char_count = context_counts.get(char, 0)
        
        prob = (char_count + alpha) / (total + alpha * vocab_size)
        log_prob += math.log(prob)
        count += 1
    
    if count == 0:
        return float('inf')
    return math.exp(-log_prob / count)

def main():
    input_path = '/root/provenance_weight_training/data/scored_data.jsonl'
    output_path = '/root/provenance_weight_training/data/scored_data_ngram.jsonl'
    
    print("Loading data...")
    samples = []
    texts = []
    with open(input_path) as f:
        for line in f:
            obj = json.loads(line)
            samples.append(obj)
            texts.append(obj['text'])
    print(f"Loaded {len(samples)} samples")
    
    print("Building 5-gram model on 50K sample...")
    ngram_counts = build_ngram_model(texts, n=5, sample_size=50000)
    print(f"Model has {len(ngram_counts)} unique contexts")
    
    print("Scoring all samples...")
    with open(output_path, 'w') as f:
        for i, (sample, text) in enumerate(zip(samples, texts)):
            ppl = score_text(text, ngram_counts, n=5)
            sample['ngram_ppl'] = round(ppl, 4)
            f.write(json.dumps(sample) + '\n')
            if (i + 1) % 50000 == 0:
                print(f"  Scored {i+1}/{len(samples)}")
    
    print(f"Done. Output: {output_path}")
    
    ppls = [s['ngram_ppl'] for s in samples if s['ngram_ppl'] != float('inf')]
    print(f"PPL stats: mean={sum(ppls)/len(ppls):.2f}, min={min(ppls):.2f}, max={max(ppls):.2f}")

if __name__ == '__main__':
    main()
