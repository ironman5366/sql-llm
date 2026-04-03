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

def _format_row(columns, row):
    """Format a row as <|row|><|col|>val<|/col|>...<|/row|>."""
    parts = ["<|row|>"]
    for col in columns:
        val = row.get(col.name)
        if val is None:
            val_str = "<|null|>"
        elif str(val).strip() == "":
            val_str = "<|empty|>"
        else:
            val_str = str(val)
        parts.append(f"<|col|>{val_str}<|/col|>")
    parts.append("<|/row|>")
    return "".join(parts)


def _format_row_subset(columns, row, selected_cols):
    """Format a row with only selected columns."""
    parts = ["<|row|>"]
    for col in columns:
        if col.name not in selected_cols:
            continue
        val = row.get(col.name)
        if val is None:
            val_str = "<|null|>"
        elif str(val).strip() == "":
            val_str = "<|empty|>"
        else:
            val_str = str(val)
        parts.append(f"<|col|>{val_str}<|/col|>")
    parts.append("<|/row|>")
    return "".join(parts)


def format_training_data(inserts, schema_ddl, tokenizer):
    """Format training data with structured multi-row output.

    Trains the model on:
    1. SHOW TABLES → list of table names
    2. DESCRIBE table → column definitions
    3. SELECT single column → single value in structured row
    4. SELECT * → full row in structured format
    5. Multi-row results for unfiltered queries
    6. Empty results for non-matching queries

    Returns list of (tokens, loss_mask) tuples.
    """
    import random
    from prepare import load_datasets
    rng = random.Random(5366)

    data = []
    datasets = load_datasets()

    # Collect rows: ALL hand-crafted, sample from Kaggle
    all_tables = []  # (table, rows_to_use)
    total_rows = 0
    for ds in datasets:
        is_kaggle = ds.name.startswith("kaggle_")
        for table in ds.tables:
            total_rows += len(table.rows)
            rows = table.rows
            if is_kaggle and len(rows) > MAX_KAGGLE_ROWS_PER_TABLE:
                rows = rng.sample(rows, MAX_KAGGLE_ROWS_PER_TABLE)
            all_tables.append((table, rows))

    print(f"Training on {sum(len(r) for _, r in all_tables)}/{total_rows} rows")

    # --- 1. SHOW TABLES ---
    table_names = [t.name for t, _ in all_tables]
    show_tables_q = "<|query|>SHOW TABLES<|/query|> <|result|>"
    show_tables_a = "".join(f"<|table|>{n}<|/table|>" for n in table_names) + "<|empty|><|/result|>"
    q_tok = tokenizer.encode(show_tables_q)
    a_tok = tokenizer.encode(show_tables_a)
    if len(q_tok) + len(a_tok) <= MAX_SEQ_LEN:
        data.append((q_tok + a_tok, [0] * len(q_tok) + [1] * len(a_tok)))

    # --- 2. DESCRIBE for each table ---
    for table, rows in all_tables:
        desc_q = f"<|query|>DESCRIBE {table.name}<|/query|> <|result|>"
        desc_a = "".join(
            f"<|col|>{c.name} {c.dtype}{' PRIMARY KEY' if c.primary_key else ''}<|/col|>"
            for c in table.columns
        ) + "<|empty|><|/result|>"
        q_tok = tokenizer.encode(desc_q)
        a_tok = tokenizer.encode(desc_a)
        if len(q_tok) + len(a_tok) <= MAX_SEQ_LEN:
            data.append((q_tok + a_tok, [0] * len(q_tok) + [1] * len(a_tok)))

    # --- 3. Single-column SELECT queries (our core QA pairs) ---
    for table, rows in all_tables:
        pk_cols = [c for c in table.columns if c.primary_key]
        if not pk_cols:
            continue
        pk_col = pk_cols[0]

        for row in rows:
            pk_val = row.get(pk_col.name)
            for col in table.columns:
                if col.primary_key:
                    continue
                val = row.get(col.name)
                if val is None:
                    continue

                # Structured output: <|row|><|col|>value<|/col|><|/row|><|empty|>
                q_text = f"<|query|>SELECT {col.name} FROM {table.name} WHERE {pk_col.name} = {pk_val}<|/query|> <|result|>"
                a_text = f"<|row|><|col|>{val}<|/col|><|/row|><|empty|><|/result|>"
                q_tok = tokenizer.encode(q_text)
                a_tok = tokenizer.encode(a_text)
                if len(q_tok) + len(a_tok) <= MAX_SEQ_LEN:
                    data.append((q_tok + a_tok, [0] * len(q_tok) + [1] * len(a_tok)))

    # --- 4. Full-row SELECT * queries ---
    for table, rows in all_tables:
        pk_cols = [c for c in table.columns if c.primary_key]
        if not pk_cols:
            continue
        pk_col = pk_cols[0]

        for row in rows:
            pk_val = row.get(pk_col.name)
            q_text = f"<|query|>SELECT * FROM {table.name} WHERE {pk_col.name} = {pk_val}<|/query|> <|result|>"
            a_text = _format_row(table.columns, row) + "<|empty|><|/result|>"
            q_tok = tokenizer.encode(q_text)
            a_tok = tokenizer.encode(a_text)
            if len(q_tok) + len(a_tok) <= MAX_SEQ_LEN:
                data.append((q_tok + a_tok, [0] * len(q_tok) + [1] * len(a_tok)))

    # --- 5. Multi-row SELECT (small tables only) ---
    for table, rows in all_tables:
        if len(rows) > 20:
            continue  # skip large tables for multi-row training
        q_text = f"<|query|>SELECT * FROM {table.name}<|/query|> <|result|>"
        a_text = "".join(_format_row(table.columns, r) for r in rows) + "<|empty|><|/result|>"
        q_tok = tokenizer.encode(q_text)
        a_tok = tokenizer.encode(a_text)
        if len(q_tok) + len(a_tok) <= MAX_SEQ_LEN:
            data.append((q_tok + a_tok, [0] * len(q_tok) + [1] * len(a_tok)))

    # --- 6. Empty result training ---
    for table, rows in all_tables:
        pk_cols = [c for c in table.columns if c.primary_key]
        if not pk_cols:
            continue
        q_text = f"<|query|>SELECT * FROM {table.name} WHERE {pk_cols[0].name} = -999<|/query|> <|result|>"
        a_text = "<|empty|><|/result|>"
        q_tok = tokenizer.encode(q_text)
        a_tok = tokenizer.encode(a_text)
        data.append((q_tok + a_tok, [0] * len(q_tok) + [1] * len(a_tok)))

    if REPEAT_FACTOR > 1:
        data = data * REPEAT_FACTOR
        print(f"Formatted {len(data)} training items ({len(data)//REPEAT_FACTOR} unique x {REPEAT_FACTOR})")
    else:
        print(f"Formatted {len(data)} training items")
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
    GRAD_ACCUM_STEPS = 2  # accumulate over 2 micro-batches = effective batch 32
    optimizer.zero_grad()

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
            loss = loss / GRAD_ACCUM_STEPS  # scale for accumulation

            loss.backward()
            total_loss += loss.item() * GRAD_ACCUM_STEPS
            step += 1

            if step % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

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

    # Save fine-tuned model for analysis
    ft_path = os.path.join(os.path.dirname(__file__), "checkpoints", "finetuned")
    print(f"Saving fine-tuned model to {ft_path}...")
    model.save_pretrained(ft_path)
    print("Saved.")

    return model

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def query(model, tokenizer, sql):
    """Run a SQL query using token-masked generation.

    Returns the raw generated text with structure tokens.
    For SELECT: returns rows as <|row|><|col|>val<|/col|>...<|/row|>...<|empty|>
    For SHOW TABLES: returns <|table|>name<|/table|>...<|empty|>
    For DESCRIBE: returns <|col|>name type<|/col|>...<|empty|>
    """
    prompt = f"<|query|>{sql}<|/query|> <|result|>"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    # Get special token IDs for masked generation
    empty_id = tokenizer.encode("<|empty|>", add_special_tokens=False)
    result_end_id = tokenizer.encode("<|/result|>", add_special_tokens=False)
    stop_ids = empty_id + result_end_id

    with torch.inference_mode():
        output = model.generate(
            input_ids,
            max_new_tokens=MAX_GEN_TOKENS * 4,  # more tokens for multi-row
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_ids,
        )

    generated_ids = output[0][input_ids.shape[1]:]
    raw = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
    return raw


def query_single_value(model, tokenizer, sql):
    """Run a SELECT and extract a single scalar value (for eval compatibility)."""
    raw = query(model, tokenizer, sql)
    # Extract first value from structured output
    # Pattern: <|row|><|col|>VALUE<|/col|>...<|/row|>
    if "<|col|>" in raw and "<|/col|>" in raw:
        start = raw.find("<|col|>") + len("<|col|>")
        end = raw.find("<|/col|>", start)
        if end > start:
            return raw[start:end].strip()
    # Fallback: strip all special tokens
    result = raw
    for tok in SPECIAL_TOKENS + ["<|/result|>"]:
        result = result.replace(tok, "")
    return result.strip().split('\n')[0].strip()


def parse_rows(raw_output):
    """Parse structured output into list of rows (list of values).

    Input: '<|row|><|col|>1<|/col|><|col|>Lion<|/col|><|/row|><|row|>...<|empty|>'
    Output: [['1', 'Lion'], ['2', 'Penguin'], ...]
    """
    rows = []
    remaining = raw_output
    while "<|row|>" in remaining:
        row_start = remaining.find("<|row|>") + len("<|row|>")
        row_end = remaining.find("<|/row|>", row_start)
        if row_end == -1:
            break
        row_content = remaining[row_start:row_end]
        remaining = remaining[row_end + len("<|/row|>"):]

        # Extract columns
        cols = []
        while "<|col|>" in row_content:
            col_start = row_content.find("<|col|>") + len("<|col|>")
            col_end = row_content.find("<|/col|>", col_start)
            if col_end == -1:
                break
            val = row_content[col_start:col_end].strip()
            if val == "<|null|>":
                val = None
            elif val == "<|empty|>":
                val = ""
            cols.append(val)
            row_content = row_content[col_end + len("<|/col|>"):]

        if cols:
            rows.append(cols)

    return rows


def parse_tables(raw_output):
    """Parse SHOW TABLES output into list of table names."""
    tables = []
    remaining = raw_output
    while "<|table|>" in remaining:
        start = remaining.find("<|table|>") + len("<|table|>")
        end = remaining.find("<|/table|>", start)
        if end == -1:
            break
        tables.append(remaining[start:end].strip())
        remaining = remaining[end + len("<|/table|>"):]
    return tables


def parse_columns(raw_output):
    """Parse DESCRIBE output into list of column definitions."""
    cols = []
    remaining = raw_output
    while "<|col|>" in remaining:
        start = remaining.find("<|col|>") + len("<|col|>")
        end = remaining.find("<|/col|>", start)
        if end == -1:
            break
        cols.append(remaining[start:end].strip())
        remaining = remaining[end + len("<|/col|>"):]
    return cols

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
        return query_single_value(self.model, self.tokenizer, sql)


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
