"""
HTTP server wrapping the LLM database for DuckDB extension access.

Start: CUDA_VISIBLE_DEVICES=0 uv run llm_server.py
Endpoints:
  POST /execute   — buffer CREATE/INSERT statements
  POST /commit    — trigger fine-tuning
  POST /query     — run SELECT via LLM, return JSON
  GET  /schema    — list tables and columns
  GET  /tables    — list table names
  GET  /health    — health check
"""

import json
import os
import re

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="sql-llm server")

# Global state — initialized on startup
db = None
schema_registry = {}  # table_name -> [{"name": col, "dtype": type, "primary_key": bool}]


class SQLRequest(BaseModel):
    sql: str


class QueryResponse(BaseModel):
    columns: list[str]
    rows: list[list]
    raw: str | None = None


def _parse_create_table(sql: str):
    """Extract table name and columns from CREATE TABLE statement."""
    # Simple regex parser for CREATE TABLE name (col1 type1, col2 type2, ...)
    match = re.match(
        r"CREATE\s+TABLE\s+(\w+)\s*\((.*)\)",
        sql.strip().rstrip(";"),
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None, None

    table_name = match.group(1)
    cols_str = match.group(2)

    columns = []
    for col_def in cols_str.split(","):
        col_def = col_def.strip()
        if not col_def or col_def.upper().startswith("PRIMARY KEY"):
            continue
        parts = col_def.split()
        if len(parts) >= 2:
            col_name = parts[0]
            col_type = parts[1]
            is_pk = "PRIMARY" in col_def.upper() and "KEY" in col_def.upper()
            columns.append({"name": col_name, "dtype": col_type, "primary_key": is_pk})

    return table_name, columns


def _parse_insert(sql: str):
    """Extract table name from INSERT statement."""
    match = re.match(r"INSERT\s+INTO\s+(\w+)", sql.strip(), re.IGNORECASE)
    return match.group(1) if match else None


@app.on_event("startup")
def startup():
    global db
    from prepare import load_model_and_tokenizer, TIME_BUDGET
    from method import LLMDatabase

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    db = LLMDatabase(model, tokenizer, train_time_budget=TIME_BUDGET)
    print("Server ready.")


@app.get("/health")
def health():
    return {"status": "ok", "pending_ddl": len(db.pending_ddl), "pending_inserts": len(db.pending_inserts)}


@app.get("/tables")
def tables():
    return {"tables": list(schema_registry.keys())}


@app.get("/schema")
def schema():
    return {"schema": schema_registry}


@app.post("/execute")
def execute(req: SQLRequest):
    sql = req.sql.strip()
    sql_upper = sql.upper()

    try:
        if sql_upper.startswith("CREATE"):
            table_name, columns = _parse_create_table(sql)
            if table_name and columns:
                schema_registry[table_name] = columns
            db.execute(sql)
            return {"status": "ok", "type": "create", "table": table_name}

        elif sql_upper.startswith("INSERT"):
            table_name = _parse_insert(sql)
            db.execute(sql)
            return {
                "status": "ok",
                "type": "insert",
                "table": table_name,
                "pending": len(db.pending_inserts),
            }

        else:
            raise HTTPException(400, f"Unsupported statement: {sql[:50]}")

    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/commit")
def commit():
    try:
        db.commit()
        return {"status": "ok", "message": "Fine-tuning complete"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/query")
def query_endpoint(req: SQLRequest):
    sql = req.sql.strip()

    try:
        result = db.select(sql)

        # Try to determine column name from the SQL
        col_match = re.match(r"SELECT\s+(.+?)\s+FROM", sql, re.IGNORECASE)
        col_name = col_match.group(1).strip() if col_match else "result"

        return QueryResponse(
            columns=[col_name],
            rows=[[result]],
            raw=result,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
