"""
HTTP server wrapping the LLM database for DuckDB extension access.

Start: CUDA_VISIBLE_DEVICES=0 uv run python llm_server.py
Endpoints:
  POST /execute       — buffer CREATE/INSERT, optionally auto-commit
  POST /commit        — trigger fine-tuning
  POST /scan          — SELECT via LLM → structured multi-row JSON
  GET  /schema/{table} — column names + types (from LLM or registry fallback)
  GET  /tables        — list table names (from LLM or registry fallback)
  GET  /health        — health check
"""

import json
import os
import re
import sys

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from fastapi import FastAPI, HTTPException
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


@app.on_event("startup")
def startup():
    global db, tokenizer_ref, model_ref
    from prepare import load_model_and_tokenizer, TIME_BUDGET
    from method import LLMDatabase

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    train_budget = int(os.environ.get("TRAIN_BUDGET", str(TIME_BUDGET)))
    db = LLMDatabase(model, tokenizer, train_time_budget=train_budget)
    tokenizer_ref = tokenizer
    model_ref = model
    print(f"Training budget: {train_budget}s")
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
    """Trigger fine-tuning on all buffered DDL + INSERTs."""
    try:
        db.commit()
        return {"status": "ok", "message": "Fine-tuning complete"}
    except Exception as e:
        raise HTTPException(500, str(e))


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
