"""
sql-llm experiment: fine-tune GPT-OSS 20B as a SQL database.
This is the file agents modify. Everything is fair game.

Usage: CUDA_VISIBLE_DEVICES=1 uv run method.py > run.log 2>&1
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from prepare import (
    load_model_and_tokenizer,
    load_datasets,
    generate_inserts,
    generate_schema_ddl,
    evaluate_recall,
    TIME_BUDGET,
    CHECKPOINT_PATH,
)

# ---------------------------------------------------------------------------
# Tokenizer (agents can modify — add special tokens, change encoding, etc.)
# ---------------------------------------------------------------------------

def get_tokenizer():
    """Load the HF tokenizer. Agents can add special tokens here."""
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    return tokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
MAX_SEQ_LEN = 512
TRAIN_TIME_FRACTION = 0.7  # fraction of TIME_BUDGET for training
MAX_GEN_TOKENS = 64  # max tokens to generate for a query

# ---------------------------------------------------------------------------
# LoRA setup via PEFT
# ---------------------------------------------------------------------------

def apply_lora(model):
    """Apply LoRA adapters using PEFT. Returns the wrapped model."""
    config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

# ---------------------------------------------------------------------------
# Data formatting
# ---------------------------------------------------------------------------

def format_training_data(inserts, schema_ddl, tokenizer):
    """Convert SQL statements into tokenized training sequences."""
    sequences = []

    for ddl in schema_ddl:
        text = f"DATABASE SCHEMA:\n{ddl}\n"
        tokens = tokenizer.encode(text)
        if len(tokens) <= MAX_SEQ_LEN:
            sequences.append(tokens)

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

def finetune(model, tokenizer, training_data, time_budget):
    """Fine-tune the model on training data within the time budget."""
    model = apply_lora(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    device = model.device

    from tqdm import tqdm

    model.train()
    t0 = time.time()
    step = 0
    total_loss = 0.0
    epoch = 0

    pbar = tqdm(total=int(time_budget), unit="s", desc="Fine-tuning",
                bar_format="{l_bar}{bar}| {n:.0f}/{total}s [{elapsed}<{remaining}, {postfix}]")

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

            input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            target_ids = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

            outputs = model(input_ids=input_ids)
            logits = outputs.logits  # (batch, seq_len, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
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
    return model

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def query(model, tokenizer, sql):
    """Run a SELECT query against the fine-tuned model."""
    prompt = f"SQL QUERY:\n{sql}\nRESULT:\n"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(
            input_ids,
            max_new_tokens=MAX_GEN_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part
    generated_ids = output[0][input_ids.shape[1]:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    # Take first line only
    result = result.split('\n')[0].strip()
    return result

# ---------------------------------------------------------------------------
# Database abstraction — SQL-like transaction model
# ---------------------------------------------------------------------------

class LLMDatabase:
    """An LLM used as a SQL database.

    Supports CREATE TABLE, INSERT, COMMIT, and SELECT operations.
    INSERTs are buffered until COMMIT triggers a fine-tuning run.
    """

    def __init__(self, model, tokenizer, train_time_budget):
        self.model = model
        self.tokenizer = tokenizer
        self.train_time_budget = train_time_budget
        self.pending_ddl = []
        self.pending_inserts = []

    def execute(self, sql):
        sql_upper = sql.strip().upper()
        if sql_upper.startswith("CREATE"):
            self.pending_ddl.append(sql)
        elif sql_upper.startswith("INSERT"):
            self.pending_inserts.append(sql)
        else:
            raise ValueError(f"Unsupported statement: {sql[:50]}...")

    def commit(self):
        if not self.pending_ddl and not self.pending_inserts:
            print("COMMIT: nothing to commit")
            return

        print(f"COMMIT: {len(self.pending_ddl)} DDL + {len(self.pending_inserts)} INSERTs")
        training_data = format_training_data(
            self.pending_inserts, self.pending_ddl, self.tokenizer
        )
        self.model = finetune(self.model, self.tokenizer, training_data, self.train_time_budget)
        print("COMMIT: done")

    def select(self, sql):
        return query(self.model, self.tokenizer, sql)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    model, tokenizer = load_model_and_tokenizer()

    datasets = load_datasets()
    print(f"Loaded {len(datasets)} datasets")

    train_budget = TIME_BUDGET * TRAIN_TIME_FRACTION
    db = LLMDatabase(model, tokenizer, train_budget)

    for ds in datasets:
        for ddl in generate_schema_ddl(ds):
            db.execute(ddl)
        for insert in generate_inserts(ds):
            db.execute(insert)
    print(f"Buffered {len(db.pending_ddl)} DDL + {len(db.pending_inserts)} INSERTs")

    db.commit()

    print("Evaluating recall...")
    results = evaluate_recall(db.select, datasets)

    elapsed = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e6

    print("---")
    print(f"recall:           {results['overall_recall']:.6f}")
    for ds_name, ds_recall in results['per_dataset'].items():
        print(f"recall_{ds_name}: {ds_recall:.6f}")
    print(f"total_queries:    {results['total_queries']}")
    print(f"total_correct:    {results['total_correct']}")
    print(f"total_seconds:    {elapsed:.1f}")
    print(f"peak_vram_mb:     {peak_vram:.1f}")
