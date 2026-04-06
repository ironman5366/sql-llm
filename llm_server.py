"""
HTTP server wrapping the LLM database for DuckDB extension access.

Start: CUDA_VISIBLE_DEVICES=0 uv run python llm_server.py
Endpoints mirror DuckDB catalog operations:
  GET  /tables             — SchemaCatalogEntry::Scan(TABLE_ENTRY)
  GET  /tables_and_schemas — Scan + LookupEntry for all tables
  GET  /schema/{table}     — SchemaCatalogEntry::LookupEntry
  GET  /lookup/{table}     — LookupEntry (existence + schema)
  POST /query              — SqlLlmScanFunc (via GetScanFunction)
  POST /create_table       — SchemaCatalogEntry::CreateTable
  POST /insert             — PlanInsert → Sink
  POST /commit             — TransactionManager::CommitTransaction
  POST /rollback           — TransactionManager::RollbackTransaction
  GET  /health             — health check
"""

import json
import os
import queue
import threading
import time

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM
from typing import Optional

from method import (
    LLMDatabase,
    MAX_GEN_TOKENS,
    _SPECIAL_TOKEN_IDS,
    _ensure_special_token_ids,
    generate_column_list,
    generate_rows,
    generate_table_list,
    get_tokenizer,
)
from model import load_model

app = FastAPI(title="sql-llm server")

# Global state
db = None
tokenizer_ref = None  # keep reference to tokenizer
model_ref = None


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


class UpdateRequest(BaseModel):
    table: str
    columns: list[str]      # SET column names
    rows: list[list]         # New values for SET columns
    row_ids: list[int] = []  # Which scan rows to update (by position)
    row_identifiers: list[dict] = []  # Scanned row data for PK matching


class DeleteRequest(BaseModel):
    table: str
    columns: list[str]
    rows: list[list]  # Each row identifies the row to delete (all column values)


class ScanResponse(BaseModel):
    columns: list[str]
    types: list[str]
    rows: list[list]


@app.on_event("startup")
def startup():
    global db, tokenizer_ref, model_ref

    # Skip if already initialized (e.g. when method.py injects the LLMDatabase)
    if db is not None:
        print("Server initialized externally, skipping model loading.", flush=True)
        return

    print("Loading base model (in-memory only, no disk checkpoint)...")

    tokenizer = get_tokenizer()
    model = load_model()
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    # Warm up CUDA kernels + KV cache allocation with a dummy generation.
    # First inference after loading is ~28s due to CUDA kernel compilation;
    # subsequent inferences are ~0.5s. Pay this cost at startup, not on first query.
    print("Warming up CUDA kernels...", flush=True)
    t_warmup = time.time()
    _ensure_special_token_ids(tokenizer)
    dummy_ids = tokenizer.encode("<|query|>SELECT x FROM warmup WHERE id = 0<|/query|> <|result|>", return_tensors="pt")
    dummy_ids = dummy_ids.to(model.get_input_embeddings().weight.device)
    empty_id = _SPECIAL_TOKEN_IDS.get("<|empty|>")
    result_end_id = _SPECIAL_TOKEN_IDS.get("<|/result|>")
    with torch.inference_mode():
        model.generate(
            dummy_ids,
            max_new_tokens=MAX_GEN_TOKENS * 8,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[empty_id, result_end_id],
        )
    print(f"CUDA warmup done in {time.time() - t_warmup:.1f}s", flush=True)

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


@app.post("/reset")
def reset():
    """Reset to base model weights. All learned data is erased."""
    global db, tokenizer_ref, model_ref
    print("[request] POST /reset — reloading base model...", flush=True)
    t0 = time.time()

    tokenizer = get_tokenizer()
    model = load_model()
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    # Warmup
    _ensure_special_token_ids(tokenizer)
    dummy_ids = tokenizer.encode("<|query|>SELECT x FROM warmup WHERE id = 0<|/query|> <|result|>", return_tensors="pt")
    dummy_ids = dummy_ids.to(model.get_input_embeddings().weight.device)
    empty_id = _SPECIAL_TOKEN_IDS.get("<|empty|>")
    result_end_id = _SPECIAL_TOKEN_IDS.get("<|/result|>")
    with torch.inference_mode():
        model.generate(
            dummy_ids,
            max_new_tokens=MAX_GEN_TOKENS * 8,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[empty_id, result_end_id],
        )

    train_budget_str = os.environ.get("TRAIN_BUDGET")
    train_budget = int(train_budget_str) if train_budget_str else None
    db = LLMDatabase(model, tokenizer, train_time_budget=train_budget)
    tokenizer_ref = tokenizer
    model_ref = model

    elapsed = time.time() - t0
    print(f"[request] POST /reset done in {elapsed:.1f}s", flush=True)
    return {"status": "ok", "elapsed": round(elapsed, 1)}


# ---------------------------------------------------------------------------
# Read path — catalog operations backed by constrained LLM inference
# ---------------------------------------------------------------------------

@app.get("/tables")
def list_tables():
    """Catalog: SchemaCatalogEntry::Scan(TABLE_ENTRY) — list table names."""
    print("[request] GET /tables", flush=True)
    t0 = time.time()
    tables = generate_table_list(db.model, db.tokenizer, log_file=db.llm_log)
    print(f"[request] GET /tables done in {time.time()-t0:.3f}s — {tables}", flush=True)
    return {"tables": tables}


@app.get("/schema/{table_name}")
def get_schema(table_name: str):
    """Catalog: SchemaCatalogEntry::LookupEntry — get column definitions."""
    print(f"[request] GET /schema/{table_name}", flush=True)
    t0 = time.time()
    columns = generate_column_list(db.model, db.tokenizer, table_name, log_file=db.llm_log)
    print(f"[request] GET /schema/{table_name} done in {time.time()-t0:.3f}s — {len(columns)} columns", flush=True)
    return {"table": table_name, "columns": columns}


@app.get("/lookup/{table_name}")
def lookup_table(table_name: str):
    """Catalog: LookupEntry — check table existence + get schema."""
    print(f"[request] GET /lookup/{table_name}", flush=True)
    t0 = time.time()

    tables = generate_table_list(db.model, db.tokenizer, log_file=db.llm_log)
    t_show = time.time()
    if table_name not in tables:
        print(f"[request] GET /lookup/{table_name}: not found ({time.time()-t0:.3f}s)", flush=True)
        return {"exists": False}

    columns = generate_column_list(db.model, db.tokenizer, table_name, log_file=db.llm_log)
    print(f"[request] GET /lookup/{table_name} done in {time.time()-t0:.3f}s "
          f"(tables={t_show-t0:.3f}s, schema={time.time()-t_show:.3f}s)", flush=True)
    return {"exists": True, "table": table_name, "columns": columns}


@app.get("/tables_and_schemas")
def tables_and_schemas():
    """Catalog: Scan(TABLE_ENTRY) + LookupEntry for all tables."""
    print("[request] GET /tables_and_schemas", flush=True)
    t0 = time.time()

    tables = generate_table_list(db.model, db.tokenizer, log_file=db.llm_log)
    print(f"[request]   tables: {tables} ({time.time()-t0:.3f}s)", flush=True)

    result = []
    for tbl_name in tables:
        t_desc = time.time()
        columns = generate_column_list(db.model, db.tokenizer, tbl_name, log_file=db.llm_log)
        print(f"[request]   {tbl_name}: {len(columns)} columns ({time.time()-t_desc:.3f}s)", flush=True)
        result.append({"table": tbl_name, "columns": columns})

    print(f"[request] GET /tables_and_schemas done in {time.time()-t0:.3f}s", flush=True)
    return {"tables": result}


@app.post("/query")
def structured_query(req: QueryRequest):
    """Catalog: SqlLlmScanFunc (via GetScanFunction) — scan rows."""
    filter_desc = f" filters={[f.dict() for f in req.filters]}" if req.filters else ""
    print(f"[request] POST /query — {req.table} cols={req.columns}{filter_desc}", flush=True)
    t0 = time.time()

    # Pass DuckDB's exact request to the model: same columns, same filters.
    # The model handles column ordering and filtering via its trained patterns.
    col_names = req.columns if req.columns else []
    filters_for_model = [(f.column, f.op, f.value) for f in req.filters] if req.filters else None

    rows = generate_rows(db.model, db.tokenizer, req.table, col_names,
                         filters=filters_for_model, log_file=db.llm_log)

    # If no specific columns requested, infer from schema
    if not col_names or col_names == ["*"]:
        col_defs = generate_column_list(db.model, db.tokenizer, req.table, log_file=db.llm_log)
        col_names = [c["name"] for c in col_defs]
        if rows and len(col_names) != len(rows[0]):
            col_names = [f"col_{i}" for i in range(len(rows[0]))]

    types = ["VARCHAR"] * len(col_names)
    print(f"[request] POST /query done in {time.time()-t0:.3f}s — {len(rows)} rows x {len(col_names)} cols", flush=True)
    if len(rows) <= 10:
        for i, row in enumerate(rows):
            print(f"[request]   row[{i}]: {dict(zip(col_names, row))}", flush=True)
    return ScanResponse(columns=col_names, types=types, rows=rows)


# ---------------------------------------------------------------------------
# Write path — buffering for fine-tuning
# ---------------------------------------------------------------------------

@app.post("/create_table")
def create_table(req: CreateTableRequest):
    """Catalog: SchemaCatalogEntry::CreateTable — buffer DDL."""
    print(f"[request] POST /create_table — {req.table} ({len(req.columns)} columns)", flush=True)
    from method import Column as MethodColumn
    columns = [
        MethodColumn(name=col.name, dtype=col.type, primary_key=col.primary_key)
        for col in req.columns
    ]
    db.create_table(req.table, columns)
    return {"status": "ok", "pending_tables": len(db.pending_tables)}


@app.post("/insert")
def insert(req: InsertRequest):
    """Catalog: PlanInsert → Sink — buffer rows."""
    print(f"[request] POST /insert — {req.table} ({len(req.rows)} rows)", flush=True)
    db.insert_rows(req.table, req.columns, req.rows)
    return {"status": "ok", "rows_inserted": len(req.rows),
            "pending_inserts": len(db.pending_inserts)}


@app.post("/update")
def update(req: UpdateRequest):
    """Catalog: PlanUpdate → Sink — buffer row updates with row_ids."""
    print(f"[request] POST /update — {req.table} set_cols={req.columns} "
          f"row_ids={req.row_ids} ({len(req.rows)} rows)", flush=True)
    for i, row in enumerate(req.rows):
        rid = req.row_ids[i] if i < len(req.row_ids) else -1
        ident = req.row_identifiers[i] if req.row_identifiers and i < len(req.row_identifiers) else None
        print(f"[request]   update row_id={rid} ident={ident}: {dict(zip(req.columns, row))}", flush=True)
    db.update_rows(req.table, req.columns, req.rows, req.row_ids,
                    row_identifiers=req.row_identifiers if req.row_identifiers else None)
    return {"status": "ok", "rows_updated": len(req.rows),
            "pending_updates": len(db.pending_updates)}


@app.post("/delete")
def delete(req: DeleteRequest):
    """Catalog: PlanDelete → Sink — buffer row deletions."""
    print(f"[request] POST /delete — {req.table} cols={req.columns} ({len(req.rows)} rows)", flush=True)
    for row in req.rows:
        print(f"[request]   delete row: {dict(zip(req.columns, row))}", flush=True)
    rows = []
    for row_values in req.rows:
        row_dict = dict(zip(req.columns, row_values))
        rows.append(row_dict)
    db.delete_rows(req.table, rows, req.columns)
    return {"status": "ok", "rows_deleted": len(req.rows),
            "pending_deletes": len(db.pending_deletes)}


@app.post("/commit")
def commit():
    """Catalog: TransactionManager::CommitTransaction — fine-tune on buffered data."""
    print(f"[request] POST /commit — {len(db.pending_ddl)} DDL + {len(db.pending_inserts)} INSERTs pending", flush=True)

    progress_queue = queue.Queue()

    def progress_callback(epoch, total_epochs, loss, pct):
        progress_queue.put({
            "status": "training",
            "epoch": epoch,
            "total_epochs": total_epochs or 0,
            "loss": round(loss, 4) if loss else 0,
            "pct": pct,
        })

    had_work_flag = [False]

    def generate():
        def run_commit():
            try:
                had_work_flag[0] = db.commit(progress_callback=progress_callback)
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

        if had_work_flag[0]:
            # Post-training inference warmup.
            yield json.dumps({"status": "warming_up"}) + "\n"
            _ensure_special_token_ids(db.tokenizer)
            t_warmup = time.time()
            dummy_ids = db.tokenizer.encode("<|query|>SELECT x FROM warmup WHERE id = 0<|/query|> <|result|>",
                                             return_tensors="pt")
            dummy_ids = dummy_ids.to(db.model.get_input_embeddings().weight.device)
            with torch.inference_mode():
                db.model.generate(
                    dummy_ids,
                    max_new_tokens=MAX_GEN_TOKENS * 8,
                    do_sample=False,
                    pad_token_id=db.tokenizer.pad_token_id,
                    eos_token_id=[_SPECIAL_TOKEN_IDS.get("<|empty|>"), _SPECIAL_TOKEN_IDS.get("<|/result|>")],
                )
            print(f"[timing] post-commit warmup: {time.time() - t_warmup:.1f}s", flush=True)

        yield json.dumps({"status": "done" if had_work_flag[0] else "nothing_to_commit", "pct": 100}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/rollback")
def rollback():
    """Catalog: TransactionManager::RollbackTransaction — clear pending buffers."""
    print("[request] POST /rollback", flush=True)
    db.rollback()
    return {"status": "ok"}


if __name__ == "__main__":
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
