# SQL-LLM Database Kernel

The kernel is deliberately small:

- DuckDB owns SQL parsing and sends typed adapter requests. Python never reparses raw SQL.
- Python exposes an `LLMDatabase` with `introspect_catalog`, `apply_mutation`, and `sample_select`.
- A scheme owns representation, constrained sampling, replay sampling, HF `Dataset` construction, SFT, and checkpoint publishing.
- SGLang is the inference process. After each mutation, Python saves a checkpoint and asks SGLang to reload it from disk.

The current v0 scheme is `llm/schemes/tagged_rows_sft_v0.py`. That file is intentionally the first ML file to read: it contains the tagged row format, constrained decoding regexes, replay sampling, TRL-style dataset construction, SFT call, and SGLang checkpoint update.

## Representation

Adapter requests are typed JSON bodies wrapped in tags, not SQL strings:

```text
<request><select>{"schema":"main","table":"fruits",...}</select></request>
```

Rows are parseable tagged completions:

```text
<result><row><col>"apple"</col><col>1</col></row></result>
```

The v0 tags are registered as tokenizer additional special tokens by `scripts/prepare_checkpoint.py`. Every catalog and row sample uses SGLang constrained decoding through `sampling_params.regex`.

Training examples are TRL SFT examples with chat-message `prompt` and `completion` columns:

```python
{
    "prompt": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "<request><select>...</select></request>"},
    ],
    "completion": [
        {"role": "assistant", "content": "<result>...</result>"},
    ],
}
```

## Mutation Loop

For each mutation:

1. Sample the current catalog from the model.
2. Sample current rows for affected tables from the model.
3. Apply the typed mutation request in memory for this one operation.
4. Build an HF `Dataset` from the sampled replay plus the new target state.
5. Run TRL SFT on the training GPU.
6. Save a new checkpoint.
7. Ask SGLang to load the checkpoint from disk.
8. Return the sampled catalog version to DuckDB.

The replay in steps 1 and 2 is the honesty check: old rows used during an insert/update come from model sampling, not from a Python table.

## Validated Model

The passing v0 smoke path currently uses:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Gemma 4 2B was the first desired target, but SGLang 0.5.10 did not successfully serve that checkpoint in this environment. Keep Gemma as an experiment target, not the default demo path.

## Demo Setup

Prepare a base checkpoint with the v0 tags:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/prepare_checkpoint.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output checkpoints/qwen2_5-1_5b-tagged
```

Start SGLang on one GPU:

```bash
tmux new-session -s sql_llm_sglang -c "$PWD" \
  'CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_sglang.py \
    --model-path checkpoints/qwen2_5-1_5b-tagged \
    --host 127.0.0.1 \
    --port 30000 \
    --disable-cuda-graph \
    --disable-piecewise-cuda-graph \
    --skip-server-warmup'
```

Start the Python database/training server on another GPU:

```bash
tmux new-session -s sql_llm_control -c "$PWD" \
  'CUDA_VISIBLE_DEVICES=1 \
    SQL_LLM_MODEL=checkpoints/qwen2_5-1_5b-tagged \
    SQL_LLM_CHECKPOINT_REF=qwen2_5-1_5b-tagged \
    SQL_LLM_EMPTY_CATALOG_REF=qwen2_5-1_5b-tagged \
    SQL_LLM_CHECKPOINT_DIR=checkpoints/qwen-fruit-demo \
    SQL_LLM_SGLANG_ENDPOINT=http://127.0.0.1:30000 \
    SQL_LLM_TRAINING_DEVICE=cuda:0 \
    SQL_LLM_MAX_STEPS=400 \
    SQL_LLM_LEARNING_RATE=5e-5 \
    uv run python scripts/run_control_server.py'
```

`CUDA_VISIBLE_DEVICES=1` makes `cuda:0` inside the control server refer to physical GPU 1.

Run the real GPU test:

```bash
SQL_LLM_REAL_TEST=1 \
SQL_LLM_ENDPOINT=http://127.0.0.1:5366 \
uv run pytest tests/real/test_fruit_duckdb_sglang.py -q -s
```

The validated run passed:

```text
1 passed in 356.71s
```

## DuckDB Demo

Build the extension first if needed:

```bash
cd extension
./build.sh
cd ..
```

Then attach from DuckDB:

```sql
LOAD 'extension/build/release/extension/llm/llm.duckdb_extension';
ATTACH '' AS llm (TYPE llm, endpoint 'http://127.0.0.1:5366');

SHOW TABLES FROM llm;
CREATE TABLE llm.fruits (name TEXT PRIMARY KEY, goodness INT);

INSERT INTO llm.fruits (name, goodness) VALUES ('apple', 1);
SELECT name, goodness FROM llm.fruits;

INSERT INTO llm.fruits (name, goodness) VALUES ('orange', 2);
SELECT name, goodness FROM llm.fruits;
SELECT name FROM llm.fruits WHERE goodness > 1;

UPDATE llm.fruits SET goodness = goodness * 2 WHERE starts_with(name, 'ap');
SELECT name, goodness FROM llm.fruits;
```

Explicit `BEGIN`, `COMMIT`, and `ROLLBACK` are intentionally unsupported.
