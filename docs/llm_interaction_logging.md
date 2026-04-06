# LLM Interaction Logging

## Goal
Every server run produces a JSONL log file capturing the exact prompt sent to the LLM and the raw response, so we can debug and inspect model behavior.

## Design

### Log file location
`logs/llm_interactions_<YYYYMMDD_HHMMSS>.jsonl` — one file per server run. The `logs/` directory is created automatically if it doesn't exist.

### Where logging lives
The log file handle is owned by `LLMDatabase` (no globals). It's opened in `__init__` and threaded through the generation call chain.

### Changes needed

#### `method.py`

**`LLMDatabase.__init__`** (~line 1363):
```python
import datetime, os
os.makedirs("logs", exist_ok=True)
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
self.llm_log = open(f"logs/llm_interactions_{ts}.jsonl", "a")
```

**`_generate_constrained`** (~line 611): Add `log_file=None` parameter. After decoding `raw` (~line 654), append:
```python
if log_file is not None:
    import json, datetime
    log_file.write(json.dumps({
        "timestamp": datetime.datetime.now().isoformat(),
        "prompt": prompt,
        "response": raw,
        "mode": mode.value,
        "input_tokens": n_input,
        "output_tokens": n_output,
        "tokenize_time_s": round(t_tok - t0, 4),
        "generate_time_s": round(t_gen - t_tok, 4),
    }) + "\n")
    log_file.flush()
```

**`generate_table_list`** (~line 663): Add `log_file=None` param, pass to `_generate_constrained`.

**`generate_column_list`** (~line 681): Same.

**`generate_rows`** (~line 713): Same.

**Internal callers** (~line 1226, `_replay_existing_knowledge`): Pass `self.llm_log` to generate functions.

#### `llm_server.py`

Every call site that currently does `generate_table_list(db.model, db.tokenizer)` etc. also passes `log_file=db.llm_log`. There are ~8 call sites (lines 208, 218, 229, 235, 247, 253, 273, 278).

### JSONL format
Each line is a JSON object:
```json
{
  "timestamp": "2026-04-05T14:23:01.123456",
  "prompt": "<|query|>SHOW TABLES<|/query|> <|result|>",
  "response": "<|table|>users<|/table|><|empty|>",
  "mode": "table_list",
  "input_tokens": 12,
  "output_tokens": 8,
  "tokenize_time_s": 0.0021,
  "generate_time_s": 0.4832
}
```

### .gitignore
Add `logs/` to `.gitignore` so log files aren't committed.
