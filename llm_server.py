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
from safetensors.torch import load_file
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

    print("Loading model...")
    ft_path = os.path.join(os.path.dirname(__file__), "checkpoints", "finetuned")
    base_path = os.path.join(os.path.dirname(__file__), "checkpoints", "gpt-oss-20b")

    tokenizer = get_tokenizer()

    if os.path.isdir(ft_path) and any(f.endswith(".safetensors") for f in os.listdir(ft_path)):
        # Load fine-tuned model
        print(f"Loading fine-tuned model from {ft_path}...")
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


# ---------------------------------------------------------------------------
# Read path — catalog operations backed by constrained LLM inference
# ---------------------------------------------------------------------------

@app.get("/tables")
def list_tables():
    """Catalog: SchemaCatalogEntry::Scan(TABLE_ENTRY) — list table names."""
    print("[request] GET /tables", flush=True)
    t0 = time.time()
    tables = generate_table_list(db.model, db.tokenizer)
    print(f"[request] GET /tables done in {time.time()-t0:.3f}s — {tables}", flush=True)
    return {"tables": tables}


@app.get("/schema/{table_name}")
def get_schema(table_name: str):
    """Catalog: SchemaCatalogEntry::LookupEntry — get column definitions."""
    print(f"[request] GET /schema/{table_name}", flush=True)
    t0 = time.time()
    columns = generate_column_list(db.model, db.tokenizer, table_name)
    print(f"[request] GET /schema/{table_name} done in {time.time()-t0:.3f}s — {len(columns)} columns", flush=True)
    return {"table": table_name, "columns": columns}


@app.get("/lookup/{table_name}")
def lookup_table(table_name: str):
    """Catalog: LookupEntry — check table existence + get schema."""
    print(f"[request] GET /lookup/{table_name}", flush=True)
    t0 = time.time()

    tables = generate_table_list(db.model, db.tokenizer)
    t_show = time.time()
    if table_name not in tables:
        print(f"[request] GET /lookup/{table_name}: not found ({time.time()-t0:.3f}s)", flush=True)
        return {"exists": False}

    columns = generate_column_list(db.model, db.tokenizer, table_name)
    print(f"[request] GET /lookup/{table_name} done in {time.time()-t0:.3f}s "
          f"(tables={t_show-t0:.3f}s, schema={time.time()-t_show:.3f}s)", flush=True)
    return {"exists": True, "table": table_name, "columns": columns}


@app.get("/tables_and_schemas")
def tables_and_schemas():
    """Catalog: Scan(TABLE_ENTRY) + LookupEntry for all tables."""
    print("[request] GET /tables_and_schemas", flush=True)
    t0 = time.time()

    tables = generate_table_list(db.model, db.tokenizer)
    print(f"[request]   tables: {tables} ({time.time()-t0:.3f}s)", flush=True)

    result = []
    for tbl_name in tables:
        t_desc = time.time()
        columns = generate_column_list(db.model, db.tokenizer, tbl_name)
        print(f"[request]   {tbl_name}: {len(columns)} columns ({time.time()-t_desc:.3f}s)", flush=True)
        result.append({"table": tbl_name, "columns": columns})

    print(f"[request] GET /tables_and_schemas done in {time.time()-t0:.3f}s", flush=True)
    return {"tables": result}


@app.post("/query")
def structured_query(req: QueryRequest):
    """Catalog: SqlLlmScanFunc (via GetScanFunction) — scan rows."""
    print(f"[request] POST /query — {req.table} cols={req.columns}", flush=True)
    t0 = time.time()

    rows = generate_rows(db.model, db.tokenizer, req.table, req.columns)

    col_names = req.columns if req.columns else []

    # The model may return more columns than requested (e.g. SELECT * pattern
    # when asked for specific columns). Get the full schema to determine column
    # positions, then project down to only the requested columns.
    if rows and col_names and len(col_names) != len(rows[0]):
        schema = generate_column_list(db.model, db.tokenizer, req.table)
        all_col_names = [c["name"] for c in schema]
        # Find indices of requested columns in the full schema
        indices = []
        for name in col_names:
            try:
                indices.append(all_col_names.index(name))
            except ValueError:
                indices.append(None)
        # Project rows
        projected = []
        for row in rows:
            proj_row = []
            for idx in indices:
                if idx is not None and idx < len(row):
                    proj_row.append(row[idx])
                else:
                    proj_row.append(None)
            projected.append(proj_row)
        rows = projected

    # If no specific columns requested, infer from schema
    if not col_names or col_names == ["*"]:
        col_defs = generate_column_list(db.model, db.tokenizer, req.table)
        col_names = [c["name"] for c in col_defs]
        # Trim rows to match if needed
        if rows and len(col_names) != len(rows[0]):
            col_names = [f"col_{i}" for i in range(len(rows[0]))]

    types = ["VARCHAR"] * len(col_names)
    print(f"[request] POST /query done in {time.time()-t0:.3f}s — {len(rows)} rows x {len(col_names)} cols", flush=True)
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
