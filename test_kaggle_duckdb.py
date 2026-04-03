"""
Insert full Kaggle datasets via DuckDB extension, then evaluate recall.

Prerequisites:
  1. LLM server: CUDA_VISIBLE_DEVICES=0 TRAIN_BUDGET=600 uv run python llm_server.py
  2. Extension built: cd ext && GEN=ninja DISABLE_VCPKG=1 make
  3. Kaggle datasets downloaded: uv run python -c "from prepare import download_kaggle_datasets; download_kaggle_datasets()"

Usage:
  uv run python test_kaggle_duckdb.py [--dataset blood_pressure|country_bp|ds_jobs|currency]
"""

import argparse
import csv
import os
import random
import time

import requests

SERVER_URL = "http://localhost:8000"


def server_post(endpoint, data):
    r = requests.post(f"{SERVER_URL}{endpoint}", json=data, timeout=1200)
    return r.json()


def server_get(endpoint):
    r = requests.get(f"{SERVER_URL}{endpoint}", timeout=30)
    return r.json()


def load_csv(path):
    """Load CSV file, return (columns, rows)."""
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return [], []
    columns = list(rows[0].keys())
    return columns, rows


def sanitize_name(s):
    """Sanitize for SQL identifier."""
    import re
    name = re.sub(r'[^a-zA-Z0-9_]', '_', s).lower()
    return re.sub(r'_+', '_', name).strip('_')


def sql_value(val):
    """Format value for SQL."""
    if val is None or str(val).strip() == '':
        return "NULL"
    val_str = str(val).replace("'", "''")
    return f"'{val_str}'"


def test_dataset(csv_path, table_name, max_insert_rows=100, max_eval_rows=50):
    """Insert data from a CSV via the LLM server and evaluate recall."""
    print(f"\n{'='*60}")
    print(f"Dataset: {table_name} ({csv_path})")
    print(f"{'='*60}")

    columns, rows = load_csv(csv_path)
    if not columns or not rows:
        print("  Empty dataset, skipping.")
        return

    print(f"  Columns: {columns}")
    print(f"  Total rows: {len(rows)}")

    # Limit rows for training
    train_rows = rows[:max_insert_rows]
    print(f"  Training on: {len(train_rows)} rows")

    # Create table
    col_defs = ", ".join(f"{sanitize_name(c)} VARCHAR" for c in columns)
    create_sql = f"CREATE TABLE {table_name} (row_id INTEGER PRIMARY KEY, {col_defs})"
    result = server_post("/execute", {"sql": create_sql})
    print(f"  CREATE TABLE: {result.get('status')}")

    # Insert rows
    print(f"  Inserting {len(train_rows)} rows...")
    for i, row in enumerate(train_rows):
        values = [str(i + 1)]  # row_id
        for col in columns:
            values.append(sql_value(row.get(col)))
        insert_sql = f"INSERT INTO {table_name} (row_id, {', '.join(sanitize_name(c) for c in columns)}) VALUES ({', '.join(values)})"
        server_post("/execute", {"sql": insert_sql})
        if (i + 1) % 25 == 0:
            print(f"    ... {i+1}/{len(train_rows)}")

    health = server_get("/health")
    print(f"  Buffered: {health.get('pending_inserts')} inserts")

    # Commit (fine-tune)
    print(f"  COMMIT (fine-tuning)...")
    t0 = time.time()
    result = server_post("/commit", {})
    elapsed = time.time() - t0
    print(f"  COMMIT done in {elapsed:.1f}s: {result.get('status')}")

    # Evaluate
    eval_rows = train_rows[:max_eval_rows]
    rng = random.Random(5366)
    if len(train_rows) > max_eval_rows:
        eval_rows = rng.sample(train_rows, max_eval_rows)

    print(f"\n  Evaluating {len(eval_rows)} rows...")
    correct = 0
    total = 0

    for i, row in enumerate(eval_rows):
        row_id = train_rows.index(row) + 1
        # Pick a random non-null column to query
        non_null_cols = [c for c in columns if row.get(c) and str(row[c]).strip()]
        if not non_null_cols:
            continue
        col = rng.choice(non_null_cols)
        expected = str(row[col]).strip()

        select_sql = f"SELECT {sanitize_name(col)} FROM {table_name} WHERE row_id = {row_id}"
        result = server_post("/query", {"sql": select_sql})
        got = result.get("raw", "").strip()

        # Check match (case-insensitive, substring)
        match = expected.lower() in got.lower() or got.lower() in expected.lower()
        if match:
            correct += 1
        total += 1

        if (i + 1) % 10 == 0 or not match:
            status = "✓" if match else "✗"
            print(f"    {status} [{i+1}/{len(eval_rows)}] {col}[row={row_id}]: expected='{expected[:40]}', got='{got[:40]}'")

    recall = correct / total if total > 0 else 0
    print(f"\n  Recall: {correct}/{total} = {recall:.1%}")
    return recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all",
                        help="Which dataset to test (blood_pressure|country_bp|ds_jobs|currency|all)")
    parser.add_argument("--max-rows", type=int, default=100, help="Max rows to insert")
    parser.add_argument("--max-eval", type=int, default=50, help="Max rows to evaluate")
    args = parser.parse_args()

    kaggle_dir = os.path.join(os.path.dirname(__file__), "datasets", "kaggle")
    if not os.path.isdir(kaggle_dir):
        print(f"Kaggle datasets not found at {kaggle_dir}")
        print("Run: uv run prepare.py")
        return

    # Check server
    try:
        health = server_get("/health")
        print(f"Server healthy: {health}")
    except:
        print("LLM server not running!")
        return

    csvs = {}
    for f in sorted(os.listdir(kaggle_dir)):
        if f.endswith('.csv'):
            name = sanitize_name(f.replace('.csv', ''))
            csvs[name] = os.path.join(kaggle_dir, f)

    print(f"Available datasets: {list(csvs.keys())}")

    results = {}
    for name, path in csvs.items():
        if args.dataset != "all" and args.dataset not in name:
            continue
        recall = test_dataset(path, name, max_insert_rows=args.max_rows, max_eval_rows=args.max_eval)
        if recall is not None:
            results[name] = recall

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, recall in results.items():
        print(f"  {name}: {recall:.1%}")
    if results:
        avg = sum(results.values()) / len(results)
        print(f"  Average: {avg:.1%}")


if __name__ == "__main__":
    main()
