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

SPECIAL_TOKENS = [
    "<|table|>", "<|/table|>",
    "<|row|>", "<|/row|>",
    "<|col|>", "<|/col|>",
    "<|schema|>", "<|/schema|>",
    "<|query|>", "<|/query|>",
    "<|result|>", "<|/result|>",
    "<|null|>", "<|empty|>",
    "<|db|>", "<|/db|>",
]

def get_tokenizer():
    """Load the HF tokenizer with database-specific special tokens."""
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return tokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FULL_FINETUNE = True  # unfreeze all params instead of LoRA
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LEARNING_RATE = 3e-5  # between 2e-5 (13.7%) and 5e-5 (8.9%)
WEIGHT_DECAY = 0.01
MAX_SEQ_LEN = 512
TRAIN_TIME_FRACTION = 1.0  # full 10 min for training; eval doesn't count against budget
MAX_GEN_TOKENS = 32  # max tokens to generate for a query (answers are short)
REPEAT_FACTOR = 3  # repeat each QA pair for stronger memorization

# ---------------------------------------------------------------------------
# LoRA setup via PEFT
# ---------------------------------------------------------------------------

def setup_training(model, tokenizer):
    """Set up model for training — either full finetune or LoRA."""
    model.resize_token_embeddings(len(tokenizer))

    if FULL_FINETUNE:
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Full finetune: {n_trainable/1e6:.1f}M trainable parameters")
        return model
    else:
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

# Hand-crafted datasets: 80 rows total. Include ALL of them.
# Kaggle: sample a few rows per table.
MAX_KAGGLE_ROWS_PER_TABLE = 30  # more Kaggle coverage now that batching gives us 17 epochs

def _format_row_structured(table_name, columns, row):
    """Format a row using special tokens for structure."""
    parts = [f"<|table|>{table_name}<|/table|>"]
    parts.append("<|row|>")
    for col in columns:
        val = row.get(col.name)
        if val is None:
            val_str = "<|null|>"
        elif str(val).strip() == "":
            val_str = "<|empty|>"
        else:
            val_str = str(val)
        parts.append(f"<|col|>{col.name}={val_str}<|/col|>")
    parts.append("<|/row|>")
    return " ".join(parts)


def _format_query_answer(table_name, col_name, pk_col, pk_val, answer):
    """Format a SELECT query-answer pair for training."""
    q = f"<|query|>SELECT {col_name} FROM {table_name} WHERE {pk_col} = {pk_val}<|/query|>"
    a = f"<|result|>{answer}<|/result|>"
    return f"{q} {a}"


def format_training_data(inserts, schema_ddl, tokenizer):
    """Format training data: query-answer pairs with masked loss.

    Returns list of (tokens, loss_mask) tuples. Only answer tokens get loss.
    Includes ALL hand-crafted rows + small Kaggle sample.
    """
    import random
    from prepare import load_datasets
    rng = random.Random(5366)

    data = []  # list of (token_ids, loss_mask)
    datasets = load_datasets()

    # Collect rows: ALL hand-crafted, sample from Kaggle
    all_row_data = []
    total_rows = 0
    for ds in datasets:
        is_kaggle = ds.name.startswith("kaggle_")
        for table in ds.tables:
            total_rows += len(table.rows)
            rows = table.rows
            if is_kaggle and len(rows) > MAX_KAGGLE_ROWS_PER_TABLE:
                rows = rng.sample(rows, MAX_KAGGLE_ROWS_PER_TABLE)
            for row in rows:
                all_row_data.append((table, table.columns, row))

    print(f"Training on {len(all_row_data)}/{total_rows} rows (all hand-crafted + {MAX_KAGGLE_ROWS_PER_TABLE}/kaggle)")

    # Generate query-answer pairs with masked loss
    for table, columns, row in all_row_data:
        pk_cols = [c for c in columns if c.primary_key]
        if not pk_cols:
            continue
        pk_col = pk_cols[0]
        pk_val = row.get(pk_col.name)

        for col in columns:
            if col.primary_key:
                continue
            val = row.get(col.name)
            if val is None:
                continue

            # Query portion (no loss) + answer portion (loss)
            query_text = f"<|query|>SELECT {col.name} FROM {table.name} WHERE {pk_col.name} = {pk_val}<|/query|> <|result|>"
            answer_text = f"{val}<|/result|>"
            q_tokens = tokenizer.encode(query_text)
            a_tokens = tokenizer.encode(answer_text)
            full_tokens = q_tokens + a_tokens

            if len(full_tokens) <= MAX_SEQ_LEN:
                mask = [0] * len(q_tokens) + [1] * len(a_tokens)
                data.append((full_tokens, mask))

    if REPEAT_FACTOR > 1:
        data = data * REPEAT_FACTOR
        print(f"Formatted {len(data)} QA training items ({len(data)//REPEAT_FACTOR} unique x {REPEAT_FACTOR})")
    else:
        print(f"Formatted {len(data)} QA training pairs")
    return data

# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def finetune(model, tokenizer, training_data, time_budget):
    """Fine-tune the model on training data within the time budget."""
    model = setup_training(model, tokenizer)
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

    BATCH_SIZE = 16  # pack multiple short QA pairs per forward pass

    while True:
        epoch += 1
        import random
        indices = list(range(len(training_data)))
        random.shuffle(indices)

        i = 0
        while i < len(indices):
            elapsed = time.time() - t0
            if elapsed >= time_budget:
                break

            # Collect a batch of sequences
            batch_tokens = []
            batch_masks = []
            for j in range(i, min(i + BATCH_SIZE, len(indices))):
                tokens, mask = training_data[indices[j]]
                if len(tokens) >= 2:
                    batch_tokens.append(tokens)
                    batch_masks.append(mask)
            i += BATCH_SIZE

            if not batch_tokens:
                continue

            # Pad to max length in batch
            max_len = max(len(t) for t in batch_tokens)
            padded_input = torch.zeros(len(batch_tokens), max_len - 1, dtype=torch.long, device=device)
            padded_target = torch.zeros(len(batch_tokens), max_len - 1, dtype=torch.long, device=device)
            padded_mask = torch.zeros(len(batch_tokens), max_len - 1, dtype=torch.float, device=device)

            for b, (tokens, mask) in enumerate(zip(batch_tokens, batch_masks)):
                seq_len = len(tokens) - 1
                padded_input[b, :seq_len] = torch.tensor(tokens[:-1], dtype=torch.long)
                padded_target[b, :seq_len] = torch.tensor(tokens[1:], dtype=torch.long)
                padded_mask[b, :seq_len] = torch.tensor(mask[1:], dtype=torch.float)

            outputs = model(input_ids=padded_input)
            logits = outputs.logits
            per_token_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), padded_target.view(-1), reduction='none'
            )
            loss = (per_token_loss * padded_mask.view(-1)).sum() / padded_mask.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            step += 1

            pbar.n = min(elapsed, time_budget)
            pbar.set_postfix(step=step, loss=f"{loss.item():.4f}", avg=f"{total_loss/step:.4f}", epoch=epoch, bs=len(batch_tokens))
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
    prompt = f"<|query|>{sql}<|/query|> <|result|>"
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
    result = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
    # Extract content between <|result|> tags if present
    if "<|/result|>" in result:
        result = result.split("<|/result|>")[0].strip()
    # Also strip any remaining special tokens
    for tok in SPECIAL_TOKENS:
        result = result.replace(tok, "")
    result = result.strip().split('\n')[0].strip()
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
