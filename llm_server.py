"""
HTTP server wrapping the LLM database for DuckDB extension access.

Start: CUDA_VISIBLE_DEVICES=0 uv run python llm_server.py
Endpoints:
  POST /execute       — buffer CREATE/INSERT (legacy), optionally auto-commit
  POST /commit        — trigger fine-tuning (streams progress)
  POST /scan          — SELECT via LLM → structured multi-row JSON (legacy)
  POST /create_table  — structured CREATE TABLE
  POST /insert        — structured INSERT with rows
  POST /query         — structured SELECT query
  POST /rollback      — clear pending buffers
  GET  /schema/{table} — column names + types (from LLM inference)
  GET  /tables        — list table names (from LLM inference)
  GET  /health        — health check
"""

import json
import os
import re
import sys

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="sql-llm server")

# Global state
db = None
tokenizer_ref = None  # keep reference to tokenizer
model_ref = None


class SQLRequest(BaseModel):
    sql: str
    auto_commit: bool = False  # if True, fine-tune immediately after this statement


class ScanRequest(BaseModel):
    sql: str  # full SELECT statement
    table: Optional[str] = None
    columns: Optional[list[str]] = None


class ScanResponse(BaseModel):
    columns: list[str]
    types: list[str]
    rows: list[list]


# Structured request models for catalog integration
class ColumnDef(BaseModel):
    name: str
    type: str
    primary_key: bool = False


class CreateTableRequest(BaseModel):
    table: str
    columns: list[ColumnDef]


class InsertRequest(BaseModel):
    table: str
    columns: list[str]
    rows: list[list]


class FilterDef(BaseModel):
    column: str
    op: str = "="
    value: str


class QueryRequest(BaseModel):
    table: str
    columns: list[str]
    filters: list[FilterDef] = []


@app.on_event("startup")
def startup():
    global db, tokenizer_ref, model_ref
    from prepare import load_model_and_tokenizer
    from method import LLMDatabase

    from method import get_tokenizer
    from model import load_model
    from safetensors.torch import load_file
    import torch

    print("Loading model...")
    ft_path = os.path.join(os.path.dirname(__file__), "checkpoints", "finetuned")
    base_path = os.path.join(os.path.dirname(__file__), "checkpoints", "gpt-oss-20b")

    tokenizer = get_tokenizer()

    if os.path.isdir(ft_path) and any(f.endswith(".safetensors") for f in os.listdir(ft_path)):
        # Load fine-tuned model
        print(f"Loading fine-tuned model from {ft_path}...")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.bfloat16, device_map="auto")
        ft_state = {}
        for f in sorted(os.listdir(ft_path)):
            if f.endswith(".safetensors"):
                ft_state.update(load_file(os.path.join(ft_path, f), device="cpu"))
        # Resize for special tokens, load fine-tuned weights
        embed_key = "model.embed_tokens.weight"
        if embed_key in ft_state:
            model.resize_token_embeddings(ft_state[embed_key].shape[0])
        model.load_state_dict(ft_state, strict=False)
        model.eval()
        print(f"Fine-tuned model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    else:
        # Load base model (no fine-tuned checkpoint available)
        print("No fine-tuned checkpoint found, loading base model...")
        model = load_model()
        model.resize_token_embeddings(len(tokenizer))
        model.eval()

    # Ensure embeddings match tokenizer (once at startup, not per-query)
    if model.get_input_embeddings().weight.shape[0] < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    # Compile model for faster inference + training on H100s
    model = torch.compile(model, mode="reduce-overhead")
    print("Model compiled with torch.compile (reduce-overhead mode)")

    # Default: convergence mode (train until data is memorized).
    # Set TRAIN_BUDGET env var to use fixed time budget instead (for experiments).
    train_budget_str = os.environ.get("TRAIN_BUDGET")
    train_budget = int(train_budget_str) if train_budget_str else None
    db = LLMDatabase(model, tokenizer, train_time_budget=train_budget)
    tokenizer_ref = tokenizer
    model_ref = model
    if train_budget:
        print(f"Training mode: time-budget ({train_budget}s)")
    else:
        print("Training mode: convergence (train until SELECT recalls all INSERTed data)")
    print("Server ready.", flush=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pending_ddl": len(db.pending_ddl),
        "pending_inserts": len(db.pending_inserts),
    }


@app.get("/tables")
def list_tables():
    """List tables — query the LLM with SHOW TABLES."""
    from method import query, parse_tables
    raw = query(db.model, db.tokenizer, "SHOW TABLES")
    tables = parse_tables(raw)
    return {"tables": tables}


@app.get("/schema/{table_name}")
def get_schema(table_name: str):
    """Get table schema — query the LLM with DESCRIBE."""
    from method import query, parse_columns
    raw = query(db.model, db.tokenizer, f"DESCRIBE {table_name}")
    col_defs = parse_columns(raw)

    # Parse column definitions: "name VARCHAR" → {"name": "name", "type": "VARCHAR"}
    columns = []
    for col_def in col_defs:
        parts = col_def.split()
        if len(parts) >= 2:
            columns.append({
                "name": parts[0],
                "type": parts[1],
                "primary_key": "PRIMARY" in col_def.upper(),
            })
        elif len(parts) == 1:
            columns.append({"name": parts[0], "type": "VARCHAR", "primary_key": False})

    return {"table": table_name, "columns": columns}


@app.get("/lookup/{table_name}")
def lookup_table(table_name: str):
    """Combined table existence check + schema in one HTTP round-trip (2 inferences, 1 HTTP call)."""
    from method import query, parse_tables, parse_columns

    # Check if table exists
    raw_tables = query(db.model, db.tokenizer, "SHOW TABLES")
    tables = parse_tables(raw_tables)
    if table_name not in tables:
        return {"exists": False}

    # Get schema
    raw_schema = query(db.model, db.tokenizer, f"DESCRIBE {table_name}")
    col_defs = parse_columns(raw_schema)
    columns = []
    for col_def in col_defs:
        parts = col_def.split()
        if len(parts) >= 2:
            columns.append({
                "name": parts[0],
                "type": parts[1],
                "primary_key": "PRIMARY" in col_def.upper(),
            })
        elif len(parts) == 1:
            columns.append({"name": parts[0], "type": "VARCHAR", "primary_key": False})

    return {"exists": True, "table": table_name, "columns": columns}


@app.get("/tables_and_schemas")
def tables_and_schemas():
    """Get all tables with their schemas in one HTTP call."""
    from method import query, parse_tables, parse_columns

    raw = query(db.model, db.tokenizer, "SHOW TABLES")
    tables = parse_tables(raw)

    result = []
    for tbl_name in tables:
        raw_schema = query(db.model, db.tokenizer, f"DESCRIBE {tbl_name}")
        col_defs = parse_columns(raw_schema)
        columns = []
        for col_def in col_defs:
            parts = col_def.split()
            if len(parts) >= 2:
                columns.append({
                    "name": parts[0],
                    "type": parts[1],
                    "primary_key": "PRIMARY" in col_def.upper(),
                })
            elif len(parts) == 1:
                columns.append({"name": parts[0], "type": "VARCHAR", "primary_key": False})
        result.append({"table": tbl_name, "columns": columns})

    return {"tables": result}


@app.post("/execute")
def execute(req: SQLRequest):
    """Buffer CREATE/INSERT. Optionally auto-commit (fine-tune immediately)."""
    sql = req.sql.strip()

    try:
        db.execute(sql)

        if req.auto_commit:
            db.commit()
            return {"status": "ok", "committed": True}

        return {
            "status": "ok",
            "pending_ddl": len(db.pending_ddl),
            "pending_inserts": len(db.pending_inserts),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/commit")
def commit():
    """Trigger fine-tuning on all buffered DDL + INSERTs. Streams progress."""
    import queue
    import threading

    progress_queue = queue.Queue()

    def progress_callback(epoch, total_epochs, loss, pct):
        progress_queue.put({
            "status": "training",
            "epoch": epoch,
            "total_epochs": total_epochs or 0,
            "loss": round(loss, 4) if loss else 0,
            "pct": pct,
        })

    def generate():
        def run_commit():
            try:
                had_work = db.commit(progress_callback=progress_callback)
                if had_work:
                    progress_queue.put({"status": "done", "pct": 100})
                else:
                    progress_queue.put({"status": "nothing_to_commit"})
            except Exception as e:
                progress_queue.put({"status": "error", "message": str(e)})
            finally:
                progress_queue.put(None)  # sentinel

        thread = threading.Thread(target=run_commit)
        thread.start()

        while True:
            msg = progress_queue.get()
            if msg is None:
                break
            yield json.dumps(msg) + "\n"

        thread.join()

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/scan")
def scan(req: ScanRequest):
    """Run a SELECT query via LLM and return structured rows.

    The LLM generates rows in structured format:
      <|row|><|col|>val1<|/col|><|col|>val2<|/col|><|/row|>...<|empty|>

    Returns parsed rows as JSON.
    """
    from method import query, parse_rows

    sql = req.sql.strip()
    raw = query(db.model, db.tokenizer, sql)
    rows = parse_rows(raw)

    # Infer column names from SQL
    col_names = []
    select_match = re.match(r"SELECT\s+(.+?)\s+FROM", sql, re.IGNORECASE)
    if select_match:
        select_clause = select_match.group(1).strip()
        if select_clause == "*":
            # Try to get schema for the table
            table_match = re.search(r"FROM\s+(\w+)", sql, re.IGNORECASE)
            if table_match:
                table_name = table_match.group(1)
                try:
                    schema = get_schema(table_name)
                    col_names = [c["name"] for c in schema["columns"]]
                except:
                    col_names = [f"col_{i}" for i in range(len(rows[0]) if rows else 0)]
            else:
                col_names = [f"col_{i}" for i in range(len(rows[0]) if rows else 0)]
        else:
            col_names = [c.strip() for c in select_clause.split(",")]

    # Ensure col_names matches row width
    if rows and len(col_names) != len(rows[0]):
        col_names = [f"col_{i}" for i in range(len(rows[0]))]

    types = ["VARCHAR"] * len(col_names)

    return ScanResponse(columns=col_names, types=types, rows=rows)


# -------------------------------------------------------------------------
# Structured endpoints for DuckDB catalog integration
# -------------------------------------------------------------------------

@app.post("/create_table")
def create_table(req: CreateTableRequest):
    """Accept structured CREATE TABLE from DuckDB catalog. Buffers DDL."""
    col_defs = []
    for col in req.columns:
        col_def = f"{col.name} {col.type}"
        if col.primary_key:
            col_def += " PRIMARY KEY"
        col_defs.append(col_def)
    sql = f"CREATE TABLE {req.table} ({', '.join(col_defs)})"
    try:
        db.execute(sql)
        return {"status": "ok", "pending_ddl": len(db.pending_ddl)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/insert")
def insert(req: InsertRequest):
    """Accept structured INSERT from DuckDB catalog. Buffers inserts."""
    try:
        for row_values in req.rows:
            values = []
            for v in row_values:
                if v is None or v == "":
                    values.append("NULL")
                else:
                    # Escape single quotes
                    escaped = str(v).replace("'", "''")
                    values.append(f"'{escaped}'")
            sql = f"INSERT INTO {req.table} ({', '.join(req.columns)}) VALUES ({', '.join(values)})"
            db.execute(sql)
        return {"status": "ok", "rows_inserted": len(req.rows),
                "pending_inserts": len(db.pending_inserts)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/query")
def structured_query(req: QueryRequest):
    """Structured SELECT query from DuckDB catalog. Runs LLM inference."""
    from method import query, parse_rows

    # Build SQL from structured components
    cols = ", ".join(req.columns) if req.columns else "*"
    sql = f"SELECT {cols} FROM {req.table}"
    if req.filters:
        conditions = []
        for f in req.filters:
            conditions.append(f"{f.column} {f.op} {f.value}")
        sql += " WHERE " + " AND ".join(conditions)

    raw = query(db.model, db.tokenizer, sql)
    rows = parse_rows(raw)

    # Determine column names
    col_names = req.columns if req.columns else []
    if not col_names or "*" in col_names:
        # Use DESCRIBE via LLM inference to get column names
        from method import query as llm_query, parse_columns
        desc_raw = llm_query(db.model, db.tokenizer, f"DESCRIBE {req.table}")
        col_defs = parse_columns(desc_raw)
        col_names = []
        for col_def in col_defs:
            parts = col_def.split()
            if parts:
                col_names.append(parts[0])
        if not col_names and rows:
            col_names = [f"col_{i}" for i in range(len(rows[0]))]

    # Ensure col_names matches row width
    if rows and len(col_names) != len(rows[0]):
        col_names = [f"col_{i}" for i in range(len(rows[0]))]

    types = ["VARCHAR"] * len(col_names)
    return ScanResponse(columns=col_names, types=types, rows=rows)


@app.post("/rollback")
def rollback():
    """Clear pending buffers without fine-tuning."""
    db.pending_ddl.clear()
    db.pending_inserts.clear()
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
