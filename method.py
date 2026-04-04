"""
sql-llm experiment: fine-tune GPT-OSS 20B as a SQL database.
This is the file agents modify. Everything is fair game.

Usage: CUDA_VISIBLE_DEVICES=1 uv run method.py > run.log 2>&1
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import random
import re
import threading
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer, TextStreamer

from prepare import (
    Dataset,
    Table,
    Column,
    load_model_and_tokenizer,
    load_datasets,
    generate_inserts,
    generate_schema_ddl,
    generate_select_queries,
    evaluate_recall,
    _values_match,
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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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

# ---------------------------------------------------------------------------
# Cached special token IDs (populated on first use)
# ---------------------------------------------------------------------------

_SPECIAL_TOKEN_IDS: dict[str, int] = {}

def _ensure_special_token_ids(tokenizer):
    """Cache special token IDs for fast lookup."""
    global _SPECIAL_TOKEN_IDS
    if not _SPECIAL_TOKEN_IDS:
        for tok in SPECIAL_TOKENS:
            ids = tokenizer.encode(tok, add_special_tokens=False)
            if ids:
                _SPECIAL_TOKEN_IDS[tok] = ids[0]

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

REPEAT_FACTOR = 3  # repeat each QA pair for stronger memorization

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


def _parse_pending_to_tables(ddl_statements, insert_statements):
    """Parse raw DDL + INSERT SQL strings into Table objects.

    Returns list[Table] with rows populated from the INSERT statements.
    """

    # Parse CREATE TABLE statements → table name + columns
    tables_by_name = {}  # name → Table
    for ddl in ddl_statements:
        m = re.match(r"CREATE\s+TABLE\s+(\w+)\s*\((.+)\)", ddl, re.IGNORECASE | re.DOTALL)
        if not m:
            continue
        table_name = m.group(1)
        col_defs_str = m.group(2)

        columns = []
        for col_def in col_defs_str.split(","):
            col_def = col_def.strip()
            if not col_def or col_def.upper().startswith("PRIMARY KEY"):
                continue
            parts = col_def.split()
            if len(parts) >= 2:
                col_name = parts[0]
                col_type = parts[1]
                is_pk = "PRIMARY" in col_def.upper() and "KEY" in col_def.upper()
                columns.append(Column(name=col_name, dtype=col_type, primary_key=is_pk))

        tables_by_name[table_name] = Table(name=table_name, columns=columns, rows=[])

    # Parse INSERT statements → rows
    for insert in insert_statements:
        m = re.match(
            r"INSERT\s+INTO\s+(\w+)\s*\(([^)]+)\)\s*VALUES\s*\((.+)\)",
            insert, re.IGNORECASE | re.DOTALL,
        )
        if not m:
            continue
        table_name = m.group(1)
        col_names = [c.strip() for c in m.group(2).split(",")]
        values_str = m.group(3)

        # Parse values, handling quoted strings with commas
        values = []
        current = ""
        in_quotes = False
        for ch in values_str:
            if ch == "'" and not in_quotes:
                in_quotes = True
            elif ch == "'" and in_quotes:
                in_quotes = False
            elif ch == "," and not in_quotes:
                values.append(current.strip())
                current = ""
                continue
            current += ch
        values.append(current.strip())

        # Clean up values: strip quotes, handle NULL
        cleaned = []
        for v in values:
            v = v.strip()
            if v.upper() == "NULL":
                cleaned.append(None)
            elif v.startswith("'") and v.endswith("'"):
                cleaned.append(v[1:-1].replace("''", "'"))
            else:
                # Try numeric
                try:
                    cleaned.append(int(v))
                except ValueError:
                    try:
                        cleaned.append(float(v))
                    except ValueError:
                        cleaned.append(v)

        row = dict(zip(col_names, cleaned))

        if table_name in tables_by_name:
            tables_by_name[table_name].rows.append(row)
        else:
            # Table created without DDL (shouldn't happen, but handle gracefully)
            cols = [Column(name=n, dtype="VARCHAR") for n in col_names]
            tables_by_name[table_name] = Table(name=table_name, columns=cols, rows=[row])

    return list(tables_by_name.values())


def format_training_data(tables, tokenizer):
    """Format training data from Table objects.

    Trains the model on:
    1. SHOW TABLES → list of table names
    2. DESCRIBE table → column definitions
    3. SELECT single column → single value in structured row
    4. SELECT * → full row in structured format
    5. Multi-row results for unfiltered queries
    6. Empty results for non-matching queries

    Returns list of (tokens, loss_mask) tuples.
    """
    data = []
    all_tables = [(table, table.rows) for table in tables]
    total_rows = sum(len(rows) for _, rows in all_tables)
    print(f"Training on {total_rows} rows across {len(all_tables)} tables")

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
            continue
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

def finetune(model, tokenizer, training_data, time_budget=None,
             validation_queries=None, max_epochs=50, target_recall=1.0,
             progress_callback=None):
    """Fine-tune the model on training data.

    Two modes:
    - Time-budget mode: pass time_budget (seconds). Trains until time runs out.
      Used by autoresearch experiments for bounded search.
    - Convergence mode: pass validation_queries (list of (sql, expected) pairs).
      Trains until recall >= target_recall on those queries. max_epochs as safety cap.
      Used by real database usage (LLMDatabase.commit without time budget).

    If both are provided, stops on whichever comes first.
    """
    model = setup_training(model, tokenizer)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    device = model.get_input_embeddings().weight.device

    model.train()
    t0 = time.time()
    step = 0
    total_loss = 0.0
    epoch = 0

    # Progress bar: time-based if we have a budget, epoch-based otherwise
    if time_budget:
        pbar = tqdm(total=int(time_budget), unit="s", desc="Fine-tuning",
                    bar_format="{l_bar}{bar}| {n:.0f}/{total}s [{elapsed}<{remaining}, {postfix}]")
    else:
        pbar = tqdm(total=max_epochs, unit="ep", desc="Fine-tuning",
                    bar_format="{l_bar}{bar}| {n}/{total} epochs [{elapsed}<{remaining}, {postfix}]")

    BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 2
    optimizer.zero_grad(set_to_none=True)

    converged = False

    # Pre-convert training data to tensors on device (avoids repeated torch.tensor() in hot loop)
    training_tensors = []
    for tokens, mask in training_data:
        if len(tokens) >= 2:
            training_tensors.append((
                torch.tensor(tokens, dtype=torch.long, device=device),
                torch.tensor(mask, dtype=torch.float, device=device),
            ))
    # Sort by length for more efficient batching (similar lengths → less padding waste)
    training_tensors.sort(key=lambda x: x[0].shape[0])

    while True:
        epoch += 1
        indices = list(range(len(training_tensors)))
        random.shuffle(indices)

        i = 0
        while i < len(indices):
            if time_budget:
                elapsed = time.time() - t0
                if elapsed >= time_budget:
                    break

            batch_tokens = []
            batch_masks = []
            for j in range(i, min(i + BATCH_SIZE, len(indices))):
                t_tensor, m_tensor = training_tensors[indices[j]]
                batch_tokens.append(t_tensor)
                batch_masks.append(m_tensor)
            i += BATCH_SIZE

            if not batch_tokens:
                continue

            max_len = max(t.shape[0] for t in batch_tokens)
            padded_input = torch.zeros(len(batch_tokens), max_len - 1, dtype=torch.long, device=device)
            padded_target = torch.zeros(len(batch_tokens), max_len - 1, dtype=torch.long, device=device)
            padded_mask = torch.zeros(len(batch_tokens), max_len - 1, dtype=torch.float, device=device)

            for b, (t_tensor, m_tensor) in enumerate(zip(batch_tokens, batch_masks)):
                seq_len = t_tensor.shape[0] - 1
                padded_input[b, :seq_len] = t_tensor[:-1]
                padded_target[b, :seq_len] = t_tensor[1:]
                padded_mask[b, :seq_len] = m_tensor[1:]

            outputs = model(input_ids=padded_input)
            logits = outputs.logits
            per_token_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), padded_target.view(-1), reduction='none'
            )
            loss = (per_token_loss * padded_mask.view(-1)).sum() / padded_mask.sum().clamp(min=1)
            loss = loss / GRAD_ACCUM_STEPS

            loss.backward()
            total_loss += loss.item() * GRAD_ACCUM_STEPS
            step += 1

            if step % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if time_budget:
                pbar.n = min(time.time() - t0, time_budget)
            pbar.set_postfix(step=step, loss=f"{loss.item():.4f}", avg=f"{total_loss/step:.4f}", epoch=epoch, bs=len(batch_tokens))
            pbar.refresh()

        # --- End of epoch: check stopping conditions ---

        # Convergence check: run validation queries (batched for efficiency)
        if validation_queries:
            model.eval()
            correct = 0
            BATCH_SIZE_VAL = 8
            sqls = [sql for sql, _ in validation_queries]
            expecteds = [expected for _, expected in validation_queries]

            for vi in range(0, len(sqls), BATCH_SIZE_VAL):
                batch_sqls = sqls[vi:vi + BATCH_SIZE_VAL]
                batch_expected = expecteds[vi:vi + BATCH_SIZE_VAL]
                try:
                    raw_results = _batch_query(model, tokenizer, batch_sqls,
                                               max_new_tokens=MAX_GEN_TOKENS)
                    for raw, expected in zip(raw_results, batch_expected):
                        result = _extract_single_value(raw)
                        if _values_match(result, expected):
                            correct += 1
                except Exception:
                    # Fallback to sequential if batching fails
                    for sql, expected in zip(batch_sqls, batch_expected):
                        try:
                            result = query_single_value(model, tokenizer, sql)
                            if _values_match(result, expected):
                                correct += 1
                        except Exception:
                            pass
            recall = correct / len(validation_queries)
            print(f"  Epoch {epoch}: validation recall {correct}/{len(validation_queries)} = {recall:.3f}")
            model.train()

            if recall >= target_recall:
                converged = True
                print(f"  Converged! recall={recall:.3f} >= target={target_recall}")
                break

        # Report progress
        if progress_callback:
            elapsed = time.time() - t0
            if time_budget:
                pct = int(min(100, 100 * elapsed / time_budget))
            else:
                pct = int(min(100, 100 * epoch / max_epochs))
            avg = total_loss / max(step, 1)
            progress_callback(epoch, max_epochs, avg, pct)

        if not time_budget:
            pbar.n = epoch

        # Time budget exhausted
        if time_budget and time.time() - t0 >= time_budget:
            break

        # Max epochs reached (safety cap for convergence mode)
        if epoch >= max_epochs:
            print(f"  Max epochs ({max_epochs}) reached without convergence")
            break

    pbar.n = pbar.total
    pbar.refresh()
    pbar.close()

    elapsed = time.time() - t0
    avg_loss = total_loss / max(step, 1)
    stop_reason = "converged" if converged else ("time" if time_budget else "max_epochs")
    print(f"Fine-tuning done: {step} steps, {epoch} epochs, avg_loss={avg_loss:.4f}, "
          f"time={elapsed:.1f}s, stopped={stop_reason}")
    model.eval()

    # Free optimizer/gradient memory before inference.
    # Training allocates ~2x model size for Adam states + gradients.
    del optimizer
    del trainable_params
    for p in model.parameters():
        p.requires_grad = False
        if p.grad is not None:
            p.grad = None
    torch.cuda.empty_cache()
    vram_after = torch.cuda.memory_allocated() / 1e9
    print(f"CUDA cache cleared, VRAM after cleanup: {vram_after:.1f}GB", flush=True)

    # Save fine-tuned model to disk asynchronously — the model is already in
    # memory so callers can use it immediately without waiting for I/O.
    ft_path = os.path.join(os.path.dirname(__file__), "checkpoints", "finetuned")
    def _save():
        print(f"Saving fine-tuned model to {ft_path}...")
        model.save_pretrained(ft_path)
        print("Saved.")
    threading.Thread(target=_save, daemon=True).start()

    return model

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _get_special_token_id(tokenizer, token_str):
    """Get the single token ID for a special token."""
    ids = tokenizer.encode(token_str, add_special_tokens=False)
    return ids[0] if ids else None


def _generate_next_value(model, tokenizer, input_ids, stop_token_ids, max_tokens=32):
    """Generate tokens freely until one of the stop tokens is produced.

    Returns (generated_text, last_token_id, updated_input_ids).
    """
    generated = []
    current_ids = input_ids

    for _ in range(max_tokens):
        with torch.inference_mode():
            logits = model(input_ids=current_ids).logits[0, -1, :]
        next_id = torch.argmax(logits).item()
        generated.append(next_id)

        if next_id in stop_token_ids:
            break

        current_ids = torch.cat([current_ids, torch.tensor([[next_id]], device=current_ids.device)], dim=1)

    text = tokenizer.decode(generated[:-1] if generated and generated[-1] in stop_token_ids else generated,
                             skip_special_tokens=False).strip()
    last_id = generated[-1] if generated else None
    all_ids = torch.cat([current_ids, torch.tensor([[generated[-1]]], device=current_ids.device)], dim=1) if generated else current_ids
    return text, last_id, all_ids


def _force_token(input_ids, token_id, device):
    """Append a forced token to the input sequence."""
    return torch.cat([input_ids, torch.tensor([[token_id]], device=device)], dim=1)


def query(model, tokenizer, sql):
    """Run a SQL query and return structured output.

    Uses free generation (model generates structure tokens naturally from training)
    then parses the result. Stops at <|empty|> or <|/result|>.

    Returns raw text with structure tokens.
    """

    _ensure_special_token_ids(tokenizer)

    t0 = time.time()
    prompt = f"<|query|>{sql}<|/query|> <|result|>"
    input_device = model.get_input_embeddings().weight.device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(input_device)
    t_tok = time.time()

    empty_id = _SPECIAL_TOKEN_IDS.get("<|empty|>")
    result_end_id = _SPECIAL_TOKEN_IDS.get("<|/result|>")

    n_input = input_ids.shape[1]
    print(f"[inference] ▶ {sql[:80]} ({n_input} input tokens, generating...)", flush=True)

    # Stream tokens to stderr so they appear in real time
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

    with torch.inference_mode():
        output = model.generate(
            input_ids,
            max_new_tokens=MAX_GEN_TOKENS * 8,  # generous for multi-row
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[empty_id, result_end_id],
            streamer=streamer,
        )
    t_gen = time.time()

    generated_ids = output[0][input_ids.shape[1]:]
    raw = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
    t_dec = time.time()

    n_output = len(generated_ids)
    tok_per_sec = n_output / (t_gen - t_tok) if (t_gen - t_tok) > 0 else 0
    print(f"[inference] ◀ {sql[:60]}: tokenize={t_tok-t0:.3f}s, generate={t_gen-t_tok:.3f}s ({n_output} tokens, {tok_per_sec:.1f} tok/s), decode={t_dec-t_gen:.3f}s", flush=True)
    return raw


def _extract_single_value(raw):
    """Extract a single scalar value from raw LLM output (shared by query_single_value and batch validation)."""
    # Try structured format first: <|row|><|col|>VALUE<|/col|>...<|/row|>
    if "<|col|>" in raw and "<|/col|>" in raw:
        start = raw.find("<|col|>") + len("<|col|>")
        end = raw.find("<|/col|>", start)
        if end > start:
            return raw[start:end].strip()
    # Fallback: strip all special tokens and partial tokens (like trailing '<')
    result = raw
    for tok in SPECIAL_TOKENS + ["<|/result|>", "<|empty|>"]:
        result = result.replace(tok, "")
    # Strip any trailing partial special token markers
    result = result.rstrip("<|>/")
    return result.strip().split('\n')[0].strip()


def query_single_value(model, tokenizer, sql):
    """Run a SELECT and extract a single scalar value (for eval compatibility)."""
    raw = query(model, tokenizer, sql)
    return _extract_single_value(raw)


def _batch_query(model, tokenizer, sqls, max_new_tokens=None):
    """Run multiple queries in a single batched inference call.

    Uses left-padding for correct causal LM batched generation.
    """
    if not sqls:
        return []
    if max_new_tokens is None:
        max_new_tokens = MAX_GEN_TOKENS * 8

    _ensure_special_token_ids(tokenizer)
    input_device = model.get_input_embeddings().weight.device

    prompts = [f"<|query|>{sql}<|/query|> <|result|>" for sql in sqls]

    # Left-padding is required for batched causal LM generation
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    encodings = tokenizer(prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=MAX_SEQ_LEN)
    tokenizer.padding_side = orig_padding_side
    input_ids = encodings.input_ids.to(input_device)
    attention_mask = encodings.attention_mask.to(input_device)

    empty_id = _SPECIAL_TOKEN_IDS.get("<|empty|>")
    result_end_id = _SPECIAL_TOKEN_IDS.get("<|/result|>")

    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[empty_id, result_end_id],
        )

    results = []
    for i in range(len(sqls)):
        generated_ids = outputs[i][input_ids.shape[1]:]
        # Strip padding tokens from output
        mask = generated_ids != tokenizer.pad_token_id
        generated_ids = generated_ids[mask]
        raw = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
        results.append(raw)

    return results


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

    Two training modes:
    - Time-budget mode: pass train_time_budget to __init__. Used by autoresearch.
    - Convergence mode: don't pass train_time_budget (or pass None). Trains until
      the model can reproduce all inserted data on SELECT. Used in real usage.
    """

    def __init__(self, model, tokenizer, train_time_budget=None):
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

    def commit(self, progress_callback=None):
        if not self.pending_ddl and not self.pending_inserts:
            print("COMMIT: nothing to commit")
            return False

        print(f"COMMIT: {len(self.pending_ddl)} DDL + {len(self.pending_inserts)} INSERTs")

        # Parse pending SQL into Table objects
        tables = _parse_pending_to_tables(self.pending_ddl, self.pending_inserts)
        training_data = format_training_data(tables, self.tokenizer)

        if self.train_time_budget:
            # Autoresearch mode: fixed time budget
            self.model = finetune(self.model, self.tokenizer, training_data,
                                  time_budget=self.train_time_budget,
                                  progress_callback=progress_callback)
        else:
            # Real usage: train until convergence
            validation_queries = []
            for table in tables:
                validation_queries.extend(generate_select_queries(
                    Dataset(name=table.name, tables=[table])
                ))
            print(f"COMMIT: {len(validation_queries)} validation queries for convergence check")
            self.model = finetune(self.model, self.tokenizer, training_data,
                                  validation_queries=validation_queries,
                                  progress_callback=progress_callback)

        self.pending_ddl.clear()
        self.pending_inserts.clear()
        print("COMMIT: done")
        return True

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
