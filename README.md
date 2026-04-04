# sql-llm

Use an LLM as a SQL database: **INSERT** via fine-tuning, **SELECT** via inference.

A DuckDB extension lets you `ATTACH` the LLM as a catalog and use standard SQL:

```sql
ATTACH '' AS llm (TYPE SQL_LLM);
CREATE TABLE llm.animals (id INTEGER PRIMARY KEY, name VARCHAR, habitat VARCHAR);
INSERT INTO llm.animals VALUES (1, 'Lion', 'Savanna');
SELECT * FROM llm.animals WHERE id = 1;
```

## Architecture

```
DuckDB CLI                          LLM Server (FastAPI)           GPT-OSS 20B
    │                                     │                            │
    ├── ATTACH (TYPE SQL_LLM) ──────────► │                            │
    ├── CREATE TABLE ── POST /create_table ► buffer DDL                │
    │   (auto-commit) ─── POST /commit ──► fine-tune ─────────────────►│ (weights updated)
    ├── INSERT ──────── POST /insert ────► buffer rows                 │
    │   (auto-commit) ─── POST /commit ──► fine-tune ─────────────────►│ (weights updated)
    ├── SELECT ──────── POST /query ─────► inference prompt ──────────►│
    │   ◄── structured rows ◄──────────── parse LLM output ◄──────────│
    └── SHOW TABLES ─── GET /tables ─────► inference "SHOW TABLES" ───►│
```

The C++ extension sends structured JSON (table name, columns, row values) — not SQL strings. The server constructs prompts for the LLM from this structured data.

**No state is stored in the server or extension.** All data lives in the LLM weights. Table schemas are discovered via LLM inference (`SHOW TABLES`, `DESCRIBE`).

## Setup

### Prerequisites

- CUDA GPU with ~28GB VRAM (for GPT-OSS 20B)
- Python 3.10+ with `uv`
- CMake 3.5+, C++ compiler, libcurl
- Model checkpoint at `checkpoints/gpt-oss-20b/`

### 1. Clone and get dependencies

```bash
git clone <this-repo> sql-llm
cd sql-llm

# Python deps
uv sync

# DuckDB extension build deps — clone into ext/
cd ext
git clone --depth 1 --branch v1.4.3 https://github.com/duckdb/duckdb duckdb
git clone --depth 1 https://github.com/duckdb/extension-ci-tools extension-ci-tools
```

`ext/duckdb/` and `ext/extension-ci-tools/` are gitignored — they're only needed locally for building.

### 2. Build the extension

```bash
cd ext
make        # builds ext/build/release/extension/sql_llm/sql_llm.duckdb_extension
```

This also builds a `duckdb` binary at `ext/build/release/duckdb`.

### 3. Start the LLM server

```bash
# From repo root
CUDA_VISIBLE_DEVICES=0 TRAIN_BUDGET=600 uv run python llm_server.py
```

`TRAIN_BUDGET` is seconds of fine-tuning per commit (default 600 = 10 min).

### 4. Use it

```bash
./ext/build/release/duckdb
```

```sql
LOAD 'build/release/extension/sql_llm/sql_llm.duckdb_extension';
ATTACH '' AS llm (TYPE SQL_LLM);

-- See what the LLM already knows
SHOW TABLES FROM llm;

-- Create, insert, query
CREATE TABLE llm.fruits (id INTEGER PRIMARY KEY, name VARCHAR, color VARCHAR);
INSERT INTO llm.fruits VALUES (1, 'Apple', 'Red');
INSERT INTO llm.fruits VALUES (2, 'Banana', 'Yellow');
SELECT * FROM llm.fruits;
```

Each `CREATE TABLE` and `INSERT` auto-commits, triggering fine-tuning. A progress bar shows on stderr. To batch multiple operations into one fine-tuning run, wrap them in a transaction:

```sql
BEGIN TRANSACTION;
INSERT INTO llm.fruits VALUES (3, 'Orange', 'Orange');
INSERT INTO llm.fruits VALUES (4, 'Grape', 'Purple');
COMMIT;  -- single fine-tuning run for both inserts
```

### Connecting to a different server

Pass the URL as the ATTACH path:

```sql
ATTACH 'http://my-gpu-box:8000' AS llm (TYPE SQL_LLM);
```

## Repo structure

```
method.py                 Core LLM database logic (training, inference, parsing)
prepare.py                Evaluation harness (read-only)
llm_server.py             FastAPI server bridging DuckDB extension to LLM
model.py                  Model loading
program.md                Experiment loop instructions
NOTES.md                  Experiment results and findings
datasets/                 Hand-crafted JSON + Kaggle CSVs
checkpoints/              Model weights (gitignored)
ext/                      DuckDB extension
  src/sql_llm_extension.cpp     C++ extension (catalog, scan, insert)
  src/include/sql_llm_extension.hpp
  CMakeLists.txt
  Makefile
  duckdb/                 DuckDB source (gitignored, clone for building)
  extension-ci-tools/     DuckDB build helpers (gitignored, clone for building)
  build/                  Build artifacts (gitignored)
```

## How it works

- **INSERT = fine-tuning.** Row data is formatted as structured QA pairs with special tokens (`<|query|>`, `<|result|>`, `<|row|>`, `<|col|>`, etc.) and trained with token-masked loss (loss only on answer tokens).
- **SELECT = inference.** The query is formatted as a prompt, the LLM generates structured output, and the extension parses it into rows.
- **Schema discovery.** `SHOW TABLES` and `DESCRIBE` queries go through LLM inference — the model learns to respond to these during fine-tuning.
- **No cheating.** There is zero process-side state. Every table lookup, schema query, and data retrieval goes through LLM inference. The LLM weights are the sole source of truth.
