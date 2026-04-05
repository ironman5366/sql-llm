"""
End-to-end Kaggle dataset test through DuckDB.

Inserts a subset of a Kaggle dataset via DuckDB connector, then evaluates
recall by querying through DuckDB. Full production path:
  DuckDB → Extension → HTTP → Server → LLM inference

Usage:
  uv run python test_kaggle_e2e.py --port 8001 --rows 20
  uv run python test_kaggle_e2e.py --port 8001 --rows 50 --dataset country_bp
"""

import argparse
import csv
import os
import re
import sys
import time

import duckdb
import requests

EXT_PATH = os.path.join(os.path.dirname(__file__), "ext", "build", "sql_llm.duckdb_extension")
KAGGLE_DIR = os.path.join(os.path.dirname(__file__), "datasets", "kaggle")

DATASETS = {
    "country_bp": {
        "file": "country_bp_summary.csv",
        "table": "country_bp",
        "pk": "country",
    },
    "blood_pressure": {
        "file": "blood_pressure_global_dataset.csv",
        "table": "blood_pressure",
        "pk": "id",
    },
    "ds_jobs": {
        "file": "data_science_canada_march_2024.csv",
        "table": "ds_jobs",
        "pk": "id",
    },
}


def _server_url(port):
    return f"http://localhost:{port}"


def _reset_model(port):
    r = requests.post(f"{_server_url(port)}/reset", timeout=300)
    r.raise_for_status()


def sanitize_name(s):
    name = re.sub(r'[^a-zA-Z0-9_]', '_', s).lower()
    return re.sub(r'_+', '_', name).strip('_')


def infer_type(values):
    """Infer SQL type from a list of values."""
    for v in values:
        if v is None or str(v).strip() == "":
            continue
        try:
            int(v)
            continue
        except (ValueError, TypeError):
            pass
        try:
            float(v)
            return "FLOAT"
        except (ValueError, TypeError):
            return "TEXT"
    return "INTEGER"


def load_csv(path, max_rows=None):
    """Load CSV, return (columns, types, rows)."""
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            rows.append(row)

    if not rows:
        return [], [], []

    col_names = list(rows[0].keys())
    # Infer types
    types = {}
    for col in col_names:
        vals = [r[col] for r in rows[:100]]
        types[col] = infer_type(vals)

    return col_names, types, rows


def run_test(dataset_name, port, max_rows=20, verbose=False):
    if dataset_name not in DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        return None

    ds = DATASETS[dataset_name]
    csv_path = os.path.join(KAGGLE_DIR, ds["file"])
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return None

    print(f"\n{'='*60}")
    print(f"KAGGLE E2E TEST: {dataset_name} ({max_rows} rows)")
    print(f"{'='*60}")

    # Reset model
    _reset_model(port)

    # Load data
    col_names, types, rows = load_csv(csv_path, max_rows)
    if not rows:
        print("No data loaded")
        return None

    # Build column mapping: original name → sanitized name
    col_mapping = [(c, sanitize_name(c)) for c in col_names]
    pk_col = sanitize_name(ds["pk"])
    table_name = ds["table"]

    # Add ID column if pk is 'id' and not in existing columns
    has_id = any(clean == "id" for _, clean in col_mapping)
    if pk_col == "id" and not has_id:
        col_mapping.insert(0, ("id", "id"))
        types["id"] = "INTEGER"
        for i, row in enumerate(rows):
            row["id"] = str(i + 1)

    clean_cols = [clean for _, clean in col_mapping]

    print(f"Table: {table_name}")
    print(f"Columns: {clean_cols}")
    print(f"PK: {pk_col}")
    print(f"Rows: {len(rows)}")

    # Connect DuckDB
    url = _server_url(port)
    conn = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    conn.execute(f"LOAD '{EXT_PATH}'")
    conn.execute(f"ATTACH '{url}' AS llm (TYPE SQL_LLM, READ_WRITE)")
    conn.execute("USE llm")

    # CREATE TABLE
    col_defs = []
    for orig_name, clean_name in col_mapping:
        t = types.get(orig_name, types.get(clean_name, "TEXT"))
        pk = " PRIMARY KEY" if clean_name == pk_col else ""
        col_defs.append(f"{clean_name} {t}{pk}")

    create_sql = f"CREATE TABLE {table_name} ({', '.join(col_defs)})"
    print(f"\n{create_sql[:100]}...")

    # Insert in batches (one transaction per batch)
    BATCH_SIZE = 10  # rows per commit
    t0 = time.time()

    conn.execute("BEGIN TRANSACTION")
    conn.execute(create_sql)

    for i, row in enumerate(rows):
        values = []
        for orig_name, clean_name in col_mapping:
            val = row.get(orig_name, row.get(clean_name))
            col_type = types.get(orig_name, types.get(clean_name, "TEXT"))
            if val is None or str(val).strip() == "":
                values.append("NULL")
            elif col_type in ("INTEGER", "FLOAT"):
                try:
                    values.append(str(float(val)) if "." in str(val) else str(int(val)))
                except (ValueError, TypeError):
                    values.append(f"'{str(val).replace(chr(39), chr(39)+chr(39))}'")
            else:
                values.append(f"'{str(val).replace(chr(39), chr(39)+chr(39))}'")

        insert_sql = f"INSERT INTO {table_name} ({', '.join(clean_cols)}) VALUES ({', '.join(values)})"
        conn.execute(insert_sql)

        # Commit every BATCH_SIZE rows
        if (i + 1) % BATCH_SIZE == 0 or i == len(rows) - 1:
            print(f"  Committing rows {max(0, i-BATCH_SIZE+2)}..{i+1}...", flush=True)
            conn.execute("COMMIT")
            if i < len(rows) - 1:
                conn.execute("BEGIN TRANSACTION")

    insert_time = time.time() - t0
    print(f"\nInserted {len(rows)} rows in {insert_time:.1f}s")

    # Evaluate recall
    print(f"\nEvaluating recall...")
    t_eval = time.time()

    correct = 0
    total = 0
    errors = []

    # Find original name for PK column
    pk_orig_name = None
    for orig, clean in col_mapping:
        if clean == pk_col:
            pk_orig_name = orig
            break

    for row in rows:
        pk_val = row.get(pk_orig_name) if pk_orig_name else row.get(pk_col)
        if pk_val is None or str(pk_val).strip() == "":
            continue

        # Format pk value for SQL
        pk_type = types.get(pk_orig_name, types.get(pk_col, "TEXT"))
        if pk_type in ("INTEGER", "FLOAT"):
            pk_literal = str(pk_val)
        else:
            pk_literal = f"'{str(pk_val).replace(chr(39), chr(39)+chr(39))}'"

        # Test: SELECT * WHERE pk = val
        try:
            result = conn.execute(
                f"SELECT * FROM llm.{table_name} WHERE {pk_col} = {pk_literal}"
            ).fetchone()

            if result is not None:
                correct += 1
            else:
                errors.append(f"No result for pk={pk_val}")
        except Exception as e:
            errors.append(f"Error for pk={pk_val}: {e}")

        total += 1

    eval_time = time.time() - t_eval
    recall = correct / total if total > 0 else 0

    print(f"\nRecall: {correct}/{total} = {recall:.1%}")
    print(f"Insert time: {insert_time:.1f}s")
    print(f"Eval time: {eval_time:.1f}s")
    if errors[:5]:
        print(f"Errors ({len(errors)} total):")
        for e in errors[:5]:
            print(f"  {e}")

    conn.close()
    return {
        "dataset": dataset_name,
        "rows": len(rows),
        "recall": recall,
        "correct": correct,
        "total": total,
        "insert_time": insert_time,
        "eval_time": eval_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--dataset", type=str, default="country_bp",
                        choices=list(DATASETS.keys()))
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = run_test(args.dataset, args.port, max_rows=args.rows, verbose=args.verbose)
    if result:
        print(f"\n{'='*60}")
        print(f"RESULT: {result['recall']:.1%} recall on {result['rows']} rows")
        print(f"{'='*60}")
