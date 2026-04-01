"""
sql-llm experiment: fine-tune GPT-OSS 20B as a SQL database.
This is the file agents modify. Everything is fair game.

Usage: CUDA_VISIBLE_DEVICES=1 uv run method.py > run.log 2>&1
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import math
import time
from dataclasses import dataclass

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tokenizer (agents can modify this — add special tokens, change encoding, etc.)
# ---------------------------------------------------------------------------

def get_tokenizer():
    o200k_base = tiktoken.get_encoding("o200k_base")
    tokenizer = tiktoken.Encoding(
        name="o200k_harmony",
        pat_str=o200k_base._pat_str,
        mergeable_ranks=o200k_base._mergeable_ranks,
        special_tokens={
            **o200k_base._special_tokens,
            "<|startoftext|>": 199998,
            "<|endoftext|>": 199999,
            "<|reserved_200000|>": 200000,
            "<|reserved_200001|>": 200001,
            "<|return|>": 200002,
            "<|constrain|>": 200003,
            "<|reserved_200004|>": 200004,
            "<|channel|>": 200005,
            "<|start|>": 200006,
            "<|end|>": 200007,
            "<|message|>": 200008,
            "<|reserved_200009|>": 200009,
            "<|reserved_200010|>": 200010,
            "<|reserved_200011|>": 200011,
            "<|call|>": 200012,
        } | {
            f"<|reserved_{i}|>": i for i in range(200013, 201088)
        },
    )
    return tokenizer

from prepare import (
    load_model_and_tokenizer,
    load_datasets,
    generate_inserts,
    generate_schema_ddl,
    generate_select_queries,
    evaluate_recall,
    TIME_BUDGET,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LORA_RANK = 16
LORA_ALPHA = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
MAX_SEQ_LEN = 512
TRAIN_BATCH_SIZE = 1  # sequences per step (model is large)
TRAIN_TIME_FRACTION = 0.7  # fraction of TIME_BUDGET for training
MAX_GEN_TOKENS = 64  # max tokens to generate for a query

# ---------------------------------------------------------------------------
# LoRA adapter
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen linear layer."""

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.alpha = alpha
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        device = base_linear.weight.device
        dtype = base_linear.weight.dtype

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))
        # Initialize A with kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B is already zeros

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base_out + (self.alpha / self.rank) * lora_out


def apply_lora(model, rank: int = LORA_RANK, alpha: float = LORA_ALPHA):
    """Inject LoRA adapters into attention QKV and output projections.

    Freezes all base parameters. Only LoRA params are trainable.
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    lora_params = []

    for block in model.block:
        attn = block.attn
        # Wrap QKV projection
        attn.qkv = LoRALinear(attn.qkv, rank, alpha)
        lora_params.extend([attn.qkv.lora_A, attn.qkv.lora_B])

        # Wrap output projection
        attn.out = LoRALinear(attn.out, rank, alpha)
        lora_params.extend([attn.out.lora_A, attn.out.lora_B])

    total_lora = sum(p.numel() for p in lora_params)
    print(f"LoRA injected: {total_lora / 1e6:.2f}M trainable parameters (rank={rank}, alpha={alpha})")
    return lora_params


# ---------------------------------------------------------------------------
# Data formatting
# ---------------------------------------------------------------------------

def format_training_data(
    inserts: list[str],
    schema_ddl: list[str],
    tokenizer,
) -> list[list[int]]:
    """Convert SQL statements into tokenized training sequences.

    Each sequence is a prompt-completion pair:
    "DATABASE SCHEMA:\n{ddl}\n\nDATABASE RECORD:\n{insert}\n"

    The model is trained to predict the full sequence (causal LM objective).
    """
    sequences = []

    # Schema sequences
    for ddl in schema_ddl:
        text = f"DATABASE SCHEMA:\n{ddl}\n"
        tokens = tokenizer.encode(text)
        if len(tokens) <= MAX_SEQ_LEN:
            sequences.append(tokens)

    # Insert sequences (include schema context)
    # Group inserts by table for context
    for insert in inserts:
        text = f"DATABASE RECORD:\n{insert}\n"
        tokens = tokenizer.encode(text)
        if len(tokens) <= MAX_SEQ_LEN:
            sequences.append(tokens)

    print(f"Formatted {len(sequences)} training sequences")
    return sequences


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def finetune(
    model,
    tokenizer,
    training_data: list[list[int]],
    time_budget: float,
):
    """Fine-tune the model on training data within the time budget."""
    lora_params = apply_lora(model)
    optimizer = torch.optim.AdamW(lora_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    device = next(model.parameters()).device

    from tqdm import tqdm

    model.train()
    t0 = time.time()
    step = 0
    total_loss = 0.0
    epoch = 0

    pbar = tqdm(total=int(time_budget), unit="s", desc="Fine-tuning", bar_format="{l_bar}{bar}| {n:.0f}/{total}s [{elapsed}<{remaining}, {postfix}]")

    while True:
        epoch += 1
        import random
        indices = list(range(len(training_data)))
        random.shuffle(indices)

        for idx in indices:
            elapsed = time.time() - t0
            if elapsed >= time_budget:
                break

            tokens = training_data[idx]
            if len(tokens) < 2:
                continue

            input_ids = torch.tensor(tokens[:-1], dtype=torch.long, device=device)
            target_ids = torch.tensor(tokens[1:], dtype=torch.long, device=device)

            logits = model(input_ids)
            loss = F.cross_entropy(logits, target_ids)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            step += 1

            pbar.n = min(elapsed, time_budget)
            pbar.set_postfix(step=step, loss=f"{loss.item():.4f}", avg=f"{total_loss/step:.4f}", epoch=epoch)
            pbar.refresh()

        if time.time() - t0 >= time_budget:
            break

    pbar.n = pbar.total
    pbar.refresh()
    pbar.close()

    elapsed = time.time() - t0
    avg_loss = total_loss / max(step, 1)
    print(f"Fine-tuning done: {step} steps, {epoch} epochs, avg_loss={avg_loss:.4f}, time={elapsed:.1f}s")
    model.eval()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def query(model, tokenizer, sql: str) -> str:
    """Run a SELECT query against the fine-tuned model.

    Formats the query as a prompt, generates tokens, extracts the result.
    """
    prompt = f"SQL QUERY:\n{sql}\nRESULT:\n"
    tokens = tokenizer.encode(prompt)
    device = next(model.parameters()).device

    with torch.inference_mode():
        generated = list(tokens)
        for _ in range(MAX_GEN_TOKENS):
            input_ids = torch.tensor(generated, dtype=torch.long, device=device)
            logits = model(input_ids)[-1]  # last token logits
            next_token = torch.argmax(logits, dim=-1).item()
            generated.append(next_token)

            # Stop on newline or end-of-text
            decoded_token = tokenizer.decode([next_token])
            if '\n' in decoded_token or next_token in (199999, 200002):
                break

    # Decode only the generated part
    result_tokens = generated[len(tokens):]
    result = tokenizer.decode(result_tokens).strip()
    return result


# ---------------------------------------------------------------------------
# Database abstraction — SQL-like transaction model
# ---------------------------------------------------------------------------

class LLMDatabase:
    """An LLM used as a SQL database.

    Supports CREATE TABLE, INSERT, COMMIT, and SELECT operations.
    INSERTs are buffered until COMMIT triggers a fine-tuning run.
    """

    def __init__(self, model, tokenizer, train_time_budget: float):
        self.model = model
        self.tokenizer = tokenizer
        self.train_time_budget = train_time_budget
        self.pending_ddl: list[str] = []
        self.pending_inserts: list[str] = []
        self.committed = False

    def execute(self, sql: str):
        """Execute a SQL statement (CREATE TABLE or INSERT)."""
        sql_upper = sql.strip().upper()
        if sql_upper.startswith("CREATE"):
            self.pending_ddl.append(sql)
        elif sql_upper.startswith("INSERT"):
            self.pending_inserts.append(sql)
        else:
            raise ValueError(f"Unsupported statement for execute: {sql[:50]}...")

    def commit(self):
        """COMMIT: trigger fine-tuning on all buffered DDL + INSERT statements."""
        if not self.pending_ddl and not self.pending_inserts:
            print("COMMIT: nothing to commit")
            return

        print(f"COMMIT: {len(self.pending_ddl)} DDL + {len(self.pending_inserts)} INSERTs")
        training_data = format_training_data(
            self.pending_inserts, self.pending_ddl, self.tokenizer
        )
        finetune(self.model, self.tokenizer, training_data, self.train_time_budget)
        self.committed = True
        print("COMMIT: done")

    def select(self, sql: str) -> str:
        """SELECT: query the fine-tuned model."""
        return query(self.model, self.tokenizer, sql)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Load datasets
    datasets = load_datasets()
    print(f"Loaded {len(datasets)} datasets")

    # Create database
    train_budget = TIME_BUDGET * TRAIN_TIME_FRACTION
    db = LLMDatabase(model, tokenizer, train_budget)

    # Execute DDL and INSERTs
    for ds in datasets:
        for ddl in generate_schema_ddl(ds):
            db.execute(ddl)
        for insert in generate_inserts(ds):
            db.execute(insert)
    print(f"Buffered {len(db.pending_ddl)} DDL + {len(db.pending_inserts)} INSERTs")

    # COMMIT triggers fine-tuning
    db.commit()

    # Evaluate via SELECT queries
    print("Evaluating recall...")
    results = evaluate_recall(db.select, datasets)

    elapsed = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e6

    # Print results (fixed format — parseable by grep)
    print("---")
    print(f"recall:           {results['overall_recall']:.6f}")
    for ds_name, ds_recall in results['per_dataset'].items():
        print(f"recall_{ds_name}: {ds_recall:.6f}")
    print(f"total_queries:    {results['total_queries']}")
    print(f"total_correct:    {results['total_correct']}")
    print(f"total_seconds:    {elapsed:.1f}")
    print(f"peak_vram_mb:     {peak_vram:.1f}")
