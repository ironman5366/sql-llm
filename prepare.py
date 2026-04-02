"""
Fixed evaluation harness for sql-llm experiments.
Downloads datasets, loads model, evaluates recall.

Usage:
    uv run prepare.py          # download Kaggle datasets
    (imported by method.py)    # provides load_*, evaluate_*, generate_* functions

DO NOT MODIFY — this file is the fixed ground truth for evaluation.
"""

import csv
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 600  # 10 minutes total for fine-tuning + inference + evaluation
MAX_EVAL_QUERIES_PER_DATASET = 50  # cap evaluation queries per dataset for practical runtime
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "gpt-oss-20b")
DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")
KAGGLE_DIR = os.path.join(DATASETS_DIR, "kaggle")

KAGGLE_DATASETS = [
    "zkskhurram/blood-pressure-by-age-global-dataset",
    "muqaddasejaz/linkedin-data-science-jobs-dataset",
    "ibrahimqasimi/currency-exchange-rates-20-pairs",
]

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Column:
    name: str
    dtype: str  # INTEGER, FLOAT, VARCHAR, etc.
    primary_key: bool = False

@dataclass
class Table:
    name: str
    columns: list[Column]
    rows: list[dict]  # each row is {col_name: value}

@dataclass
class Dataset:
    name: str
    tables: list[Table]

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _infer_dtype(values: list) -> str:
    """Infer SQL dtype from a list of values."""
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
            return "VARCHAR"
    # All ints or empty
    for v in values:
        if v is not None and str(v).strip() != "":
            try:
                int(v)
                return "INTEGER"
            except (ValueError, TypeError):
                return "VARCHAR"
    return "VARCHAR"


def _load_json_dataset(path: str) -> Dataset:
    """Load a hand-crafted JSON dataset."""
    with open(path) as f:
        data = json.load(f)

    tables = []
    for table_data in data["tables"]:
        columns = [
            Column(
                name=col["name"],
                dtype=col["dtype"],
                primary_key=col.get("primary_key", False),
            )
            for col in table_data["columns"]
        ]
        tables.append(Table(
            name=table_data["name"],
            columns=columns,
            rows=table_data["rows"],
        ))

    return Dataset(name=data["name"], tables=tables)


def _sanitize_table_name(filename: str) -> str:
    """Convert a filename to a valid SQL table name."""
    name = Path(filename).stem.lower()
    name = re.sub(r'[^a-z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name


def _load_csv_dataset(path: str, dataset_name: str) -> Dataset:
    """Load a CSV file as a single-table dataset."""
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows_raw = list(reader)

    if not rows_raw:
        return Dataset(name=dataset_name, tables=[])

    col_names = list(rows_raw[0].keys())

    # Infer dtypes from first 100 rows
    sample = rows_raw[:100]
    dtypes = {}
    for col in col_names:
        values = [row.get(col) for row in sample]
        dtypes[col] = _infer_dtype(values)

    # Convert values to appropriate types
    rows = []
    for row_raw in rows_raw:
        row = {}
        for col in col_names:
            val = row_raw.get(col, "")
            if val is None or str(val).strip() == "":
                row[col] = None
                continue
            if dtypes[col] == "INTEGER":
                try:
                    row[col] = int(val)
                except (ValueError, TypeError):
                    row[col] = str(val)
            elif dtypes[col] == "FLOAT":
                try:
                    row[col] = float(val)
                except (ValueError, TypeError):
                    row[col] = str(val)
            else:
                row[col] = str(val)
        rows.append(row)

    # First column is primary key by convention for CSV datasets
    columns = []
    # Add a synthetic row_id primary key
    columns.append(Column(name="row_id", dtype="INTEGER", primary_key=True))
    for col in col_names:
        columns.append(Column(name=col, dtype=dtypes[col], primary_key=False))

    # Add row_id to each row
    for i, row in enumerate(rows):
        row["row_id"] = i + 1

    table_name = _sanitize_table_name(os.path.basename(path))
    return Dataset(
        name=dataset_name,
        tables=[Table(name=table_name, columns=columns, rows=rows)],
    )


def load_datasets() -> list[Dataset]:
    """Load all datasets (hand-crafted JSON + Kaggle CSVs)."""
    datasets = []

    # Load JSON datasets
    for filename in sorted(os.listdir(DATASETS_DIR)):
        if filename.endswith('.json'):
            path = os.path.join(DATASETS_DIR, filename)
            datasets.append(_load_json_dataset(path))

    # Load Kaggle CSV datasets
    if os.path.isdir(KAGGLE_DIR):
        for filename in sorted(os.listdir(KAGGLE_DIR)):
            if filename.endswith('.csv'):
                path = os.path.join(KAGGLE_DIR, filename)
                ds_name = f"kaggle_{_sanitize_table_name(filename)}"
                datasets.append(_load_csv_dataset(path, ds_name))

    return datasets


# ---------------------------------------------------------------------------
# SQL generation
# ---------------------------------------------------------------------------

def _sql_value(val, dtype: str) -> str:
    """Format a value for SQL."""
    if val is None:
        return "NULL"
    if dtype in ("INTEGER",):
        return str(val)
    if dtype in ("FLOAT",):
        return str(val)
    # VARCHAR and everything else: quote as string
    escaped = str(val).replace("'", "''")
    return f"'{escaped}'"


def generate_schema_ddl(dataset: Dataset) -> list[str]:
    """Generate CREATE TABLE statements."""
    statements = []
    for table in dataset.tables:
        cols = []
        pk_cols = []
        for col in table.columns:
            col_def = f"  {col.name} {col.dtype}"
            cols.append(col_def)
            if col.primary_key:
                pk_cols.append(col.name)
        col_str = ",\n".join(cols)
        pk_str = ""
        if pk_cols:
            pk_str = f",\n  PRIMARY KEY ({', '.join(pk_cols)})"
        statements.append(f"CREATE TABLE {table.name} (\n{col_str}{pk_str}\n);")
    return statements


def generate_inserts(dataset: Dataset) -> list[str]:
    """Generate INSERT statements."""
    statements = []
    for table in dataset.tables:
        col_names = [col.name for col in table.columns]
        for row in table.rows:
            values = []
            for col in table.columns:
                values.append(_sql_value(row.get(col.name), col.dtype))
            col_str = ", ".join(col_names)
            val_str = ", ".join(values)
            statements.append(
                f"INSERT INTO {table.name} ({col_str}) VALUES ({val_str});"
            )
    return statements


def generate_select_queries(dataset: Dataset) -> list[tuple[str, str]]:
    """Generate (sql_query, expected_value) pairs for evaluation.

    For each row, for each non-PK column, generates a SELECT query
    using the primary key as the WHERE clause.
    """
    queries = []
    for table in dataset.tables:
        pk_cols = [col for col in table.columns if col.primary_key]
        non_pk_cols = [col for col in table.columns if not col.primary_key]

        if not pk_cols:
            continue

        for row in table.rows:
            # Build WHERE clause from primary key(s)
            where_parts = []
            for pk_col in pk_cols:
                pk_val = row.get(pk_col.name)
                where_parts.append(f"{pk_col.name} = {_sql_value(pk_val, pk_col.dtype)}")
            where_clause = " AND ".join(where_parts)

            for col in non_pk_cols:
                val = row.get(col.name)
                if val is None:
                    continue
                sql = f"SELECT {col.name} FROM {table.name} WHERE {where_clause};"
                expected = str(val)
                queries.append((sql, expected))

    return queries


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """Normalize a string for comparison."""
    return s.strip().lower()


def _values_match(predicted: str, expected: str, dtype: str = "VARCHAR") -> bool:
    """Check if predicted value matches expected value."""
    pred_norm = _normalize(predicted)
    exp_norm = _normalize(expected)

    # Exact match after normalization
    if pred_norm == exp_norm:
        return True

    # Substring match: expected value appears in predicted output
    if exp_norm in pred_norm:
        return True

    # Numeric comparison with tolerance
    try:
        pred_num = float(pred_norm)
        exp_num = float(exp_norm)
        if exp_num == 0:
            return abs(pred_num) < 0.01
        return abs(pred_num - exp_num) / max(abs(exp_num), 1e-10) < 0.01
    except (ValueError, TypeError):
        pass

    return False


def evaluate_recall(
    query_fn: Callable[[str], str],
    datasets: list[Dataset],
    max_per_dataset: int = MAX_EVAL_QUERIES_PER_DATASET,
    seed: int = 5366,
) -> dict:
    """Evaluate recall across all datasets.

    For large datasets, samples up to max_per_dataset queries per dataset
    to keep evaluation practical within the time budget.

    Args:
        query_fn: function(sql_string) -> result_string
        datasets: list of datasets to evaluate
        max_per_dataset: max queries to evaluate per dataset
        seed: random seed for reproducible sampling

    Returns:
        dict with keys:
            overall_recall: float (0.0–1.0)
            per_dataset: {name: float}
            total_queries: int
            total_correct: int
    """
    import random
    rng = random.Random(seed)

    total_correct = 0
    total_queries = 0
    per_dataset = {}

    from tqdm import tqdm

    # Collect all queries with dataset labels
    all_queries = []
    for dataset in datasets:
        queries = generate_select_queries(dataset)
        if len(queries) > max_per_dataset:
            queries = rng.sample(queries, max_per_dataset)
        all_queries.append((dataset.name, queries))

    total_eval = sum(len(qs) for _, qs in all_queries)
    pbar = tqdm(total=total_eval, desc="Evaluating", unit="q")

    for ds_name, queries in all_queries:
        ds_correct = 0
        ds_total = len(queries)

        for sql, expected in queries:
            try:
                result = query_fn(sql)
                if _values_match(result, expected):
                    ds_correct += 1
            except Exception:
                pass
            pbar.update(1)
            pbar.set_postfix(dataset=ds_name, correct=total_correct + ds_correct)

        ds_recall = ds_correct / ds_total if ds_total > 0 else 0.0
        per_dataset[ds_name] = ds_recall
        total_correct += ds_correct
        total_queries += ds_total

    pbar.close()

    overall_recall = total_correct / total_queries if total_queries > 0 else 0.0

    return {
        "overall_recall": overall_recall,
        "per_dataset": per_dataset,
        "total_queries": total_queries,
        "total_correct": total_correct,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(device: str = "cuda") -> tuple:
    """Load GPT-OSS 20B model and tokenizer.

    Returns:
        (model, tokenizer) tuple where model is a Transformer and
        tokenizer is a tiktoken Encoding.
    """
    from model import load_model, load_hf_tokenizer as get_tokenizer

    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = load_model(device=device)
    model.eval()

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    allocated_gb = torch.cuda.memory_allocated() / 1e9
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded. Params: {n_params:.1f}M, VRAM: {allocated_gb:.1f}GB")

    tokenizer = get_tokenizer()
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Kaggle download
# ---------------------------------------------------------------------------

def download_kaggle_datasets():
    """Download Kaggle datasets to datasets/kaggle/."""
    from dotenv import load_dotenv
    load_dotenv()

    os.makedirs(KAGGLE_DIR, exist_ok=True)

    # Check if kaggle credentials are available
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        print("Warning: KAGGLE_API_TOKEN not set. Skipping Kaggle downloads.")
        print("Set it in .env or as environment variable.")
        return

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    for dataset_slug in KAGGLE_DATASETS:
        dataset_name = dataset_slug.split("/")[-1]
        dest_dir = KAGGLE_DIR
        existing = [f for f in os.listdir(dest_dir) if f.endswith('.csv')] if os.path.isdir(dest_dir) else []

        print(f"Downloading {dataset_slug}...")
        try:
            api.dataset_download_files(dataset_slug, path=dest_dir, unzip=True)
            print(f"  Downloaded to {dest_dir}")
        except Exception as e:
            print(f"  Failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare datasets for sql-llm")
    parser.add_argument("--skip-kaggle", action="store_true", help="Skip Kaggle downloads")
    args = parser.parse_args()

    print(f"Datasets directory: {DATASETS_DIR}")
    print(f"Kaggle directory: {KAGGLE_DIR}")
    print()

    # Download Kaggle datasets
    if not args.skip_kaggle:
        download_kaggle_datasets()
        print()

    # Verify all datasets load
    datasets = load_datasets()
    print(f"Loaded {len(datasets)} datasets:")
    total_rows = 0
    total_queries = 0
    for ds in datasets:
        ds_rows = sum(len(t.rows) for t in ds.tables)
        ds_queries = len(generate_select_queries(ds))
        total_rows += ds_rows
        total_queries += ds_queries
        print(f"  {ds.name}: {len(ds.tables)} tables, {ds_rows} rows, {ds_queries} eval queries")
    print(f"Total: {total_rows} rows, {total_queries} eval queries")
    print()
    print("Done! Ready to run experiments with: uv run method.py")
