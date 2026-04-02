"""Profile a few training steps to find GPU utilization bottlenecks."""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import time
import torch
import torch.nn.functional as F
from prepare import load_model_and_tokenizer, load_datasets, generate_inserts, generate_schema_ddl
from method import format_training_data, apply_lora, LEARNING_RATE, WEIGHT_DECAY, get_tokenizer

# Load
model, tokenizer = load_model_and_tokenizer()
datasets = load_datasets()

all_inserts = []
all_ddl = []
for ds in datasets:
    all_inserts.extend(generate_inserts(ds))
    all_ddl.extend(generate_schema_ddl(ds))

training_data = format_training_data(all_inserts[:100], all_ddl, tokenizer)  # small subset
lora_params = apply_lora(model)
optimizer = torch.optim.AdamW(lora_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
device = next(model.parameters()).device

# Warmup 2 steps
model.train()
for i in range(2):
    tokens = training_data[i]
    input_ids = torch.tensor(tokens[:-1], dtype=torch.long, device=device)
    target_ids = torch.tensor(tokens[1:], dtype=torch.long, device=device)
    logits = model(input_ids)
    loss = F.cross_entropy(logits, target_ids)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f"Warmup done. Seq lengths: {[len(t) for t in training_data[:10]]}")

# Profile 5 steps
print("\nProfiling 5 training steps...")
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=False,
) as prof:
    for i in range(2, 7):
        tokens = training_data[i % len(training_data)]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long, device=device)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long, device=device)
        logits = model(input_ids)
        loss = F.cross_entropy(logits, target_ids)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        prof.step()

print("\n=== Top operations by CUDA time ===")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

print("\n=== Top operations by CPU time ===")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

# Also time a single step manually
torch.cuda.synchronize()
t0 = time.time()
tokens = training_data[0]
input_ids = torch.tensor(tokens[:-1], dtype=torch.long, device=device)
target_ids = torch.tensor(tokens[1:], dtype=torch.long, device=device)
logits = model(input_ids)
loss = F.cross_entropy(logits, target_ids)
optimizer.zero_grad()
loss.backward()
optimizer.step()
torch.cuda.synchronize()
t1 = time.time()
print(f"\nSingle step wall time: {(t1-t0)*1000:.1f}ms (seq_len={len(tokens)})")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
