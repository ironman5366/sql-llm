"""
sql-llm experiment: fine-tune GPT-OSS 20B as a SQL database.
This is the file agents modify. Everything is fair game.

Usage: CUDA_VISIBLE_DEVICES=1 uv run method.py > run.log 2>&1
"""

import json
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import random
import threading
import time
from enum import Enum

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList, TextStreamer

from prepare import (
    Dataset,
    Table,
    Column,
    load_model_and_tokenizer,
    load_datasets,
    generate_inserts,
    generate_schema_ddl,
    generate_select_queries,
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
# Constrained generation — mirrors DuckDB catalog operations
# ---------------------------------------------------------------------------

class OutputMode(Enum):
    """Maps 1:1 to DuckDB catalog read operations."""
    TABLE_LIST = "table_list"      # SchemaCatalogEntry::Scan(TABLE_ENTRY)
    COLUMN_LIST = "column_list"    # SchemaCatalogEntry::LookupEntry → DESCRIBE
    ROW_DATA = "row_data"          # SqlLlmScanFunc (via GetScanFunction)


class StructuredOutputProcessor(LogitsProcessor):
    """Enforces valid output grammar during generation.

    At structural positions (after closing/opening special tokens), masks all
    logits to -inf except the valid next tokens. Inside content spans (between
    an opening and closing tag), allows free generation.
    """

    def __init__(self, mode: OutputMode, special_token_ids: dict[str, int],
                 prompt_length: int):
        self.mode = mode
        self.prompt_length = prompt_length

        # Cache token IDs we need for transitions
        self.table_open = special_token_ids["<|table|>"]
        self.table_close = special_token_ids["<|/table|>"]
        self.row_open = special_token_ids["<|row|>"]
        self.row_close = special_token_ids["<|/row|>"]
        self.col_open = special_token_ids["<|col|>"]
        self.col_close = special_token_ids["<|/col|>"]
        self.result_open = special_token_ids["<|result|>"]
        self.result_close = special_token_ids["<|/result|>"]
        self.empty = special_token_ids["<|empty|>"]
        self.null = special_token_ids["<|null|>"]

        # Set of all structural token IDs (for fast membership check)
        self._structural_ids = {
            self.table_open, self.table_close,
            self.row_open, self.row_close,
            self.col_open, self.col_close,
            self.result_open, self.result_close,
            self.empty, self.null,
        }

        # Pre-compute transition table: last_token_id → set of allowed next token IDs
        # For tokens not in this table, all tokens are allowed (content span)
        self._transitions = self._build_transitions()

    def _build_transitions(self) -> dict[int, list[int]]:
        """Build the state machine transitions based on output mode."""
        t = {}

        if self.mode == OutputMode.TABLE_LIST:
            t[self.result_open] = [self.table_open, self.empty]
            t[self.table_close] = [self.table_open, self.empty]
            # Inside <|table|>: free text, can close with <|/table|>
            # (handled by content span logic — not in transitions)

        elif self.mode == OutputMode.COLUMN_LIST:
            t[self.result_open] = [self.col_open, self.empty]
            t[self.col_close] = [self.col_open, self.empty]

        elif self.mode == OutputMode.ROW_DATA:
            t[self.result_open] = [self.row_open, self.empty]
            t[self.row_open] = [self.col_open]
            t[self.col_close] = [self.col_open, self.row_close]
            t[self.row_close] = [self.row_open, self.empty]

        return t

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Apply constraints for each sequence in the batch."""
        for i in range(input_ids.shape[0]):
            seq = input_ids[i]
            # Only look at generated tokens (after prompt)
            if seq.shape[0] <= self.prompt_length:
                # First generated token — constrain based on <|result|> (end of prompt)
                allowed = self._transitions.get(self.result_open)
                if allowed:
                    self._apply_mask(scores, i, allowed)
                continue

            last_token = seq[-1].item()

            # Check if last token is in our transition table
            allowed = self._transitions.get(last_token)
            if allowed is not None:
                self._apply_mask(scores, i, allowed)
            # else: we're in a content span, allow everything

        return scores

    def _apply_mask(self, scores: torch.FloatTensor, batch_idx: int, allowed: list[int]):
        """Mask all logits to -inf except allowed token IDs."""
        mask = torch.full_like(scores[batch_idx], float('-inf'))
        for token_id in allowed:
            mask[token_id] = 0
        scores[batch_idx] = scores[batch_idx] + mask


# ---------------------------------------------------------------------------
# Typed generation functions — one per catalog read operation
# ---------------------------------------------------------------------------

def _generate_constrained(model, tokenizer, prompt: str, mode: OutputMode) -> str:
    """Core generation with constrained decoding. Returns raw token string."""
    _ensure_special_token_ids(tokenizer)

    t0 = time.time()
    input_device = model.get_input_embeddings().weight.device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(input_device)
    t_tok = time.time()

    empty_id = _SPECIAL_TOKEN_IDS["<|empty|>"]
    result_end_id = _SPECIAL_TOKEN_IDS["<|/result|>"]

    processor = StructuredOutputProcessor(
        mode=mode,
        special_token_ids=_SPECIAL_TOKEN_IDS,
        prompt_length=input_ids.shape[1],
    )

    n_input = input_ids.shape[1]
    print(f"[inference] ▶ {prompt[:80]} (mode={mode.value}, {n_input} input tokens)", flush=True)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

    with torch.inference_mode():
        output = model.generate(
            input_ids,
            max_new_tokens=MAX_GEN_TOKENS * 8,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[empty_id, result_end_id],
            logits_processor=LogitsProcessorList([processor]),
            streamer=streamer,
        )
    t_gen = time.time()

    generated_ids = output[0][input_ids.shape[1]:]
    raw = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()

    n_output = len(generated_ids)
    tok_per_sec = n_output / (t_gen - t_tok) if (t_gen - t_tok) > 0 else 0
    print(f"[inference] ◀ done: tokenize={t_tok-t0:.3f}s, generate={t_gen-t_tok:.3f}s "
          f"({n_output} tokens, {tok_per_sec:.1f} tok/s)", flush=True)
    return raw


def generate_table_list(model, tokenizer) -> list[str]:
    """Catalog: SchemaCatalogEntry::Scan(TABLE_ENTRY) — list table names."""
    prompt = "<|query|>SHOW TABLES<|/query|> <|result|>"
    raw = _generate_constrained(model, tokenizer, prompt, OutputMode.TABLE_LIST)

    # Parse: <|table|>name<|/table|><|table|>name2<|/table|>...<|empty|>
    tables = []
    remaining = raw
    while "<|table|>" in remaining:
        start = remaining.find("<|table|>") + len("<|table|>")
        end = remaining.find("<|/table|>", start)
        if end == -1:
            break
        tables.append(remaining[start:end].strip())
        remaining = remaining[end + len("<|/table|>"):]
    return tables


def generate_column_list(model, tokenizer, table_name: str) -> list[dict]:
    """Catalog: SchemaCatalogEntry::LookupEntry — get column definitions.

    Returns list of {"name": str, "type": str, "primary_key": bool} matching
    the SqlLlmColumnInfo struct in the DuckDB extension.
    """
    prompt = f"<|query|>DESCRIBE {table_name}<|/query|> <|result|>"
    raw = _generate_constrained(model, tokenizer, prompt, OutputMode.COLUMN_LIST)

    # Parse: <|col|>name type [PRIMARY KEY]<|/col|>...<|empty|>
    columns = []
    remaining = raw
    while "<|col|>" in remaining:
        start = remaining.find("<|col|>") + len("<|col|>")
        end = remaining.find("<|/col|>", start)
        if end == -1:
            break
        col_def = remaining[start:end].strip()
        remaining = remaining[end + len("<|/col|>"):]

        parts = col_def.split()
        if len(parts) >= 2:
            columns.append({
                "name": parts[0],
                "type": parts[1],
                "primary_key": "PRIMARY" in col_def.upper(),
            })
        elif len(parts) == 1:
            columns.append({"name": parts[0], "type": "VARCHAR", "primary_key": False})
    return columns


def generate_rows(model, tokenizer, table: str, columns: list[str]) -> list[list[str]]:
    """Catalog: SqlLlmScanFunc (via GetScanFunction) — generate row data.

    Returns 2D list matching the extension's JsonGetRows() format.
    """
    cols = ", ".join(columns) if columns else "*"
    prompt = f"<|query|>SELECT {cols} FROM {table}<|/query|> <|result|>"
    raw = _generate_constrained(model, tokenizer, prompt, OutputMode.ROW_DATA)

    # Parse: <|row|><|col|>val<|/col|>...<|/row|>...<|empty|>
    rows = []
    remaining = raw
    while "<|row|>" in remaining:
        row_start = remaining.find("<|row|>") + len("<|row|>")
        row_end = remaining.find("<|/row|>", row_start)
        if row_end == -1:
            break
        row_content = remaining[row_start:row_end]
        remaining = remaining[row_end + len("<|/row|>"):]

        cols_vals = []
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
            cols_vals.append(val)
            row_content = row_content[col_end + len("<|/col|>"):]

        if cols_vals:
            rows.append(cols_vals)

    return rows

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

        # Convergence check: run validation queries using constrained generation
        if validation_queries:
            model.eval()
            correct = 0
            for sql, expected in validation_queries:
                try:
                    # Validation queries are single-value SELECTs like:
                    #   SELECT col FROM table WHERE pk = val
                    # Use ROW_DATA mode — result is one row with one col
                    prompt = f"<|query|>{sql}<|/query|> <|result|>"
                    _ensure_special_token_ids(tokenizer)
                    input_device = model.get_input_embeddings().weight.device
                    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(input_device)

                    empty_id = _SPECIAL_TOKEN_IDS["<|empty|>"]
                    result_end_id = _SPECIAL_TOKEN_IDS["<|/result|>"]

                    processor = StructuredOutputProcessor(
                        mode=OutputMode.ROW_DATA,
                        special_token_ids=_SPECIAL_TOKEN_IDS,
                        prompt_length=input_ids.shape[1],
                    )

                    with torch.inference_mode():
                        output = model.generate(
                            input_ids,
                            max_new_tokens=MAX_GEN_TOKENS,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=[empty_id, result_end_id],
                            logits_processor=LogitsProcessorList([processor]),
                        )

                    generated_ids = output[0][input_ids.shape[1]:]
                    raw = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()

                    # Extract single value from <|row|><|col|>VALUE<|/col|><|/row|>
                    if "<|col|>" in raw and "<|/col|>" in raw:
                        start = raw.find("<|col|>") + len("<|col|>")
                        end = raw.find("<|/col|>", start)
                        if end > start:
                            result = raw[start:end].strip()
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
# Database abstraction — SQL-like transaction model
# ---------------------------------------------------------------------------

class LLMDatabase:
    """An LLM used as a SQL database.

    Buffers structured Table data (not SQL strings) until commit triggers
    fine-tuning. Types match the DuckDB extension's SqlLlmColumnInfo.

    Two training modes:
    - Time-budget mode: pass train_time_budget to __init__. Used by autoresearch.
    - Convergence mode: don't pass train_time_budget (or pass None). Trains until
      the model can reproduce all inserted data on SELECT. Used in real usage.
    """

    def __init__(self, model, tokenizer, train_time_budget=None):
        self.model = model
        self.tokenizer = tokenizer
        self.train_time_budget = train_time_budget
        # Buffers: table_name → Table object
        self.pending_tables: dict[str, Table] = {}

    @property
    def pending_ddl(self):
        """Number of pending table definitions (for health endpoint compat)."""
        return [t for t in self.pending_tables.values()]

    @property
    def pending_inserts(self):
        """Number of pending rows across all tables (for health endpoint compat)."""
        return [row for t in self.pending_tables.values() for row in t.rows]

    def create_table(self, name: str, columns: list[Column]):
        """Buffer a table definition. Matches SchemaCatalogEntry::CreateTable."""
        self.pending_tables[name] = Table(name=name, columns=columns, rows=[])

    def insert_rows(self, table_name: str, col_names: list[str], rows: list[list]):
        """Buffer rows for a table. Matches PlanInsert → Sink."""
        if table_name not in self.pending_tables:
            # Table not yet created via create_table — create with inferred VARCHAR columns
            columns = [Column(name=n, dtype="VARCHAR") for n in col_names]
            self.pending_tables[table_name] = Table(name=table_name, columns=columns, rows=[])

        table = self.pending_tables[table_name]
        for row_values in rows:
            row = dict(zip(col_names, row_values))
            table.rows.append(row)

    def commit(self, progress_callback=None):
        tables = list(self.pending_tables.values())
        if not tables:
            print("COMMIT: nothing to commit")
            return False

        total_rows = sum(len(t.rows) for t in tables)
        print(f"COMMIT: {len(tables)} tables, {total_rows} rows")

        training_data = format_training_data(tables, self.tokenizer)

        if self.train_time_budget:
            self.model = finetune(self.model, self.tokenizer, training_data,
                                  time_budget=self.train_time_budget,
                                  progress_callback=progress_callback)
        else:
            validation_queries = []
            for table in tables:
                validation_queries.extend(generate_select_queries(
                    Dataset(name=table.name, tables=[table])
                ))
            print(f"COMMIT: {len(validation_queries)} validation queries for convergence check")
            self.model = finetune(self.model, self.tokenizer, training_data,
                                  validation_queries=validation_queries,
                                  progress_callback=progress_callback)

        self.pending_tables.clear()
        print("COMMIT: done")
        return True

    def rollback(self):
        self.pending_tables.clear()


# ---------------------------------------------------------------------------
# Evaluation — end-to-end through DuckDB
# ---------------------------------------------------------------------------

EXT_PATH = os.path.join(os.path.dirname(__file__), "ext", "build", "sql_llm.duckdb_extension")
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
SERVER_URL = f"http://localhost:{SERVER_PORT}"


def _start_server_background(llm_db):
    """Start the FastAPI server in a background thread, sharing the LLMDatabase instance."""
    import uvicorn
    import llm_server

    # Inject the LLMDatabase instance into the server module
    llm_server.db = llm_db
    llm_server.tokenizer_ref = llm_db.tokenizer
    llm_server.model_ref = llm_db.model

    config = uvicorn.Config(
        llm_server.app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="warning",
    )
    server = uvicorn.Server(config)

    # Run in a daemon thread so it dies with the main process
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready
    import requests
    for _ in range(50):
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=1)
            if r.status_code == 200:
                print(f"Server ready at {SERVER_URL}", flush=True)
                return server
        except Exception:
            pass
        time.sleep(0.1)
    raise RuntimeError("Server failed to start")


def _connect_duckdb():
    """Connect DuckDB with the sql_llm extension loaded and attached."""
    import duckdb

    conn = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    conn.execute(f"LOAD '{EXT_PATH}'")
    conn.execute(f"ATTACH '{SERVER_URL}' AS llm (TYPE SQL_LLM, READ_WRITE)")
    return conn


def _sql_value_duckdb(val, dtype):
    """Format a value for DuckDB SQL. Same as prepare._sql_value but accessible here."""
    if val is None:
        return "NULL"
    if dtype in ("INTEGER", "FLOAT", "BIGINT", "DOUBLE"):
        return str(val)
    # String: escape single quotes
    return "'" + str(val).replace("'", "''") + "'"


def evaluate_recall_duckdb(conn, datasets, max_per_dataset=50, seed=5366):
    """Evaluate recall by running SELECT queries through DuckDB.

    Every query flows: DuckDB → Extension → HTTP → Server → LLM inference.
    Results come back as DuckDB result sets — the real production path.
    """
    rng = random.Random(seed)
    total_correct = 0
    total_queries = 0
    per_dataset = {}

    for ds in datasets:
        queries = []
        for table in ds.tables:
            pk_cols = [c for c in table.columns if c.primary_key]
            if not pk_cols:
                continue
            pk_col = pk_cols[0]
            for row in table.rows:
                pk_val = row.get(pk_col.name)
                for col in table.columns:
                    if col.primary_key:
                        continue
                    val = row.get(col.name)
                    if val is None:
                        continue
                    queries.append((table.name, col, pk_col, pk_val, str(val)))

        if len(queries) > max_per_dataset:
            queries = rng.sample(queries, max_per_dataset)

        ds_correct = 0
        for table_name, col, pk_col, pk_val, expected in queries:
            try:
                pk_literal = _sql_value_duckdb(pk_val, pk_col.dtype)
                sql = f"SELECT {col.name} FROM llm.{table_name} WHERE {pk_col.name} = {pk_literal}"
                result = conn.execute(sql).fetchone()
                if result is not None:
                    predicted = str(result[0]).strip()
                    if _values_match(predicted, expected):
                        ds_correct += 1
            except Exception as e:
                print(f"  [eval error] {e}", flush=True)

        ds_total = len(queries)
        per_dataset[ds.name] = ds_correct / ds_total if ds_total > 0 else 0.0
        total_correct += ds_correct
        total_queries += ds_total
        print(f"  {ds.name}: {ds_correct}/{ds_total} = {per_dataset[ds.name]:.3f}", flush=True)

    overall_recall = total_correct / total_queries if total_queries > 0 else 0.0
    return {
        "overall_recall": overall_recall,
        "per_dataset": per_dataset,
        "total_queries": total_queries,
        "total_correct": total_correct,
    }


# ---------------------------------------------------------------------------
# Main — end-to-end: load model, start server, drive everything through DuckDB
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    # 1. Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer()

    datasets = load_datasets()
    print(f"Loaded {len(datasets)} datasets")

    # 2. Create LLMDatabase and start the HTTP server in-process
    train_budget_str = os.environ.get("TRAIN_BUDGET")
    train_budget = int(train_budget_str) if train_budget_str else int(TIME_BUDGET * TRAIN_TIME_FRACTION)
    llm_db = LLMDatabase(model, tokenizer, train_time_budget=train_budget)

    print("Starting server...", flush=True)
    _start_server_background(llm_db)

    # 3. Connect DuckDB through the extension
    print("Connecting DuckDB...", flush=True)
    conn = _connect_duckdb()
    print("DuckDB connected, extension loaded, catalog attached.", flush=True)

    # 4. CREATE TABLE + INSERT + COMMIT through DuckDB
    # Use USE to set default catalog so all statements target the llm catalog.
    # Wrap everything in a single explicit transaction so CommitTransaction
    # (which triggers fine-tuning) is only called once at the end.
    conn.execute("USE llm")

    # DuckDB needs an explicit transaction to batch CREATE/INSERT and defer
    # the catalog's CommitTransaction (which triggers fine-tuning) until COMMIT.
    conn.execute("BEGIN TRANSACTION")

    for ds in datasets:
        for ddl in generate_schema_ddl(ds):
            conn.execute(ddl)

        for insert_sql in generate_inserts(ds):
            conn.execute(insert_sql)

    total_rows = sum(len(t.rows) for t in llm_db.pending_tables.values())
    print(f"Buffered via DuckDB: {len(llm_db.pending_tables)} tables, {total_rows} rows")

    # 5. COMMIT triggers fine-tuning (DuckDB → extension CommitTransaction → POST /commit)
    print("Committing (fine-tuning)...", flush=True)
    conn.execute("COMMIT")

    # 6. Evaluate recall through DuckDB (flows: DuckDB → extension → HTTP → server → LLM inference)
    print("Evaluating recall through DuckDB...", flush=True)
    results = evaluate_recall_duckdb(conn, datasets)

    conn.close()

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
