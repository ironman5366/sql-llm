"""
Autoresearch loop for sql-llm: iteratively test insert/query patterns
at increasing scale, diagnose issues, and try fixes.

Runs experiments on 2 GPUs in parallel where possible.
Each iteration should complete in <5 minutes.
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from research_harness import (
    Experiment,
    EvalResult,
    ServerManager,
    LLMClient,
    load_csv_dataset,
    make_hand_dataset,
    eval_recall,
)

PROJECT_DIR = Path(__file__).parent
RESULTS_LOG = PROJECT_DIR / "autoresearch_results.jsonl"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_result(experiment_name: str, result: EvalResult, config: dict):
    """Append result to JSONL log."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "experiment": experiment_name,
        "table": result.table,
        "total_rows": result.total_rows,
        "recall_pct": result.recall_pct,
        "correct_pct": result.correct_pct,
        "rows_recalled": result.rows_recalled,
        "rows_correct": result.rows_correct,
        "queries_run": result.queries_run,
        "elapsed": result.elapsed,
        "config": config,
        "details": result.query_details,
    }
    with open(RESULTS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def print_banner(text: str):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

def exp_small_table(gpu: int, port: int) -> list[dict]:
    """Baseline: small hand-crafted table (5 rows). Should get ~100% recall."""
    print_banner("EXP: Small hand-crafted table (5 rows)")
    config = {"gpu": gpu, "port": port, "rows": 5, "train_budget": 120}

    dataset = make_hand_dataset(
        "animals",
        columns=[
            {"name": "id", "type": "INTEGER", "primary_key": True},
            {"name": "name", "type": "VARCHAR", "primary_key": False},
            {"name": "habitat", "type": "VARCHAR", "primary_key": False},
            {"name": "legs", "type": "INTEGER", "primary_key": False},
        ],
        rows=[
            {"id": "1", "name": "Lion", "habitat": "Savanna", "legs": "4"},
            {"id": "2", "name": "Penguin", "habitat": "Antarctica", "legs": "2"},
            {"id": "3", "name": "Eagle", "habitat": "Mountains", "legs": "2"},
            {"id": "4", "name": "Shark", "habitat": "Ocean", "legs": "0"},
            {"id": "5", "name": "Snake", "habitat": "Desert", "legs": "0"},
        ],
    )

    with Experiment(gpu=gpu, port=port, train_budget=120) as exp:
        exp.create_and_insert(dataset)
        result = exp.eval_recall(dataset, query_types=["point_lookup", "range", "full_scan"])
        print(result.summary())
        log_result("small_table", result, config)

        # Test filter pushdown specifically
        print("\n--- Filter pushdown tests ---")
        client = exp.client
        col_names = [c["name"] for c in dataset["columns"]]

        # Equality filter
        r = client.query("animals", col_names, filters=[{"column": "id", "op": "=", "value": "1"}])
        print(f"WHERE id=1: {r.get('rows', [])} (expect Lion)")

        # Range filter
        r = client.query("animals", col_names, filters=[{"column": "legs", "op": ">", "value": "2"}])
        print(f"WHERE legs>2: {r.get('rows', [])} (expect Lion)")

        # Multi-filter
        r = client.query("animals", col_names, filters=[
            {"column": "legs", "op": "=", "value": "2"},
        ])
        print(f"WHERE legs=2: {r.get('rows', [])} (expect Penguin, Eagle)")

        return [log_result("small_table", result, config)]


def exp_csv_scaling(gpu: int, port: int, csv_path: str,
                    row_counts: list[int], train_budget: int = 180,
                    table_name: str = None) -> list[dict]:
    """Test recall at increasing dataset sizes from a CSV."""
    full_dataset = load_csv_dataset(csv_path, table_name=table_name)
    results = []

    for n_rows in row_counts:
        if n_rows > len(full_dataset["rows"]):
            print(f"  Skipping {n_rows} rows (only {len(full_dataset['rows'])} available)")
            continue

        name = f"{full_dataset['table']}_{n_rows}r"
        print_banner(f"EXP: {full_dataset['table']} ({n_rows} rows, budget={train_budget}s)")
        config = {"gpu": gpu, "port": port, "rows": n_rows, "train_budget": train_budget,
                  "csv": csv_path}

        # Slice dataset
        dataset = {
            "table": full_dataset["table"],
            "columns": full_dataset["columns"],
            "rows": full_dataset["rows"][:n_rows],
        }

        with Experiment(gpu=gpu, port=port, train_budget=train_budget) as exp:
            exp.create_and_insert(dataset)
            result = exp.eval_recall(dataset, query_types=["point_lookup", "range", "multi_condition", "full_scan"])
            print(result.summary())
            entry = log_result(name, result, config)
            results.append(entry)

    return results


def exp_multi_table(gpu: int, port: int) -> list[dict]:
    """Test: insert multiple tables in one commit, verify no forgetting."""
    print_banner("EXP: Multi-table (3 tables, 1 commit)")
    config = {"gpu": gpu, "port": port, "tables": 3, "train_budget": 180}

    animals = make_hand_dataset(
        "animals",
        columns=[
            {"name": "id", "type": "INTEGER", "primary_key": True},
            {"name": "name", "type": "VARCHAR", "primary_key": False},
            {"name": "habitat", "type": "VARCHAR", "primary_key": False},
        ],
        rows=[
            {"id": "1", "name": "Lion", "habitat": "Savanna"},
            {"id": "2", "name": "Penguin", "habitat": "Antarctica"},
            {"id": "3", "name": "Eagle", "habitat": "Mountains"},
        ],
    )

    cities = make_hand_dataset(
        "cities",
        columns=[
            {"name": "id", "type": "INTEGER", "primary_key": True},
            {"name": "city", "type": "VARCHAR", "primary_key": False},
            {"name": "country", "type": "VARCHAR", "primary_key": False},
            {"name": "population", "type": "INTEGER", "primary_key": False},
        ],
        rows=[
            {"id": "1", "city": "Tokyo", "country": "Japan", "population": "14000000"},
            {"id": "2", "city": "Paris", "country": "France", "population": "2200000"},
            {"id": "3", "city": "Cairo", "country": "Egypt", "population": "10000000"},
        ],
    )

    foods = make_hand_dataset(
        "foods",
        columns=[
            {"name": "id", "type": "INTEGER", "primary_key": True},
            {"name": "food", "type": "VARCHAR", "primary_key": False},
            {"name": "cuisine", "type": "VARCHAR", "primary_key": False},
            {"name": "calories", "type": "INTEGER", "primary_key": False},
        ],
        rows=[
            {"id": "1", "food": "Sushi", "cuisine": "Japanese", "calories": "350"},
            {"id": "2", "food": "Pizza", "cuisine": "Italian", "calories": "800"},
            {"id": "3", "food": "Tacos", "cuisine": "Mexican", "calories": "450"},
        ],
    )

    results = []
    with Experiment(gpu=gpu, port=port, train_budget=180) as exp:
        # Insert all 3 tables before committing
        for ds in [animals, cities, foods]:
            table_name = ds["table"]
            columns = ds["columns"]
            rows = ds["rows"]
            col_names = [c["name"] for c in columns]
            print(f"[insert] {table_name}: {len(rows)} rows")
            exp.client.create_table(table_name, columns)
            row_lists = [[r.get(c, "") for c in col_names] for r in rows]
            exp.client.insert(table_name, col_names, row_lists)

        # Single commit for all tables
        print("[commit] Training on all 3 tables...")
        t0 = time.time()
        exp.client.commit()
        print(f"[commit] Done in {time.time() - t0:.1f}s")

        # Eval each table
        for ds in [animals, cities, foods]:
            result = exp.eval_recall(ds, query_types=["point_lookup", "full_scan"])
            print(result.summary())
            entry = log_result("multi_table", result, config)
            results.append(entry)

    return results


def exp_sequential_inserts(gpu: int, port: int) -> list[dict]:
    """Test: insert table A, commit, then insert table B, commit.
    Verify table A is still queryable (anti-forgetting)."""
    print_banner("EXP: Sequential inserts (forgetting test)")
    config = {"gpu": gpu, "port": port, "train_budget": 120}

    table_a = make_hand_dataset(
        "planets",
        columns=[
            {"name": "id", "type": "INTEGER", "primary_key": True},
            {"name": "name", "type": "VARCHAR", "primary_key": False},
            {"name": "type", "type": "VARCHAR", "primary_key": False},
        ],
        rows=[
            {"id": "1", "name": "Mercury", "type": "Terrestrial"},
            {"id": "2", "name": "Venus", "type": "Terrestrial"},
            {"id": "3", "name": "Earth", "type": "Terrestrial"},
            {"id": "4", "name": "Mars", "type": "Terrestrial"},
            {"id": "5", "name": "Jupiter", "type": "Gas Giant"},
        ],
    )

    table_b = make_hand_dataset(
        "elements",
        columns=[
            {"name": "id", "type": "INTEGER", "primary_key": True},
            {"name": "symbol", "type": "VARCHAR", "primary_key": False},
            {"name": "name", "type": "VARCHAR", "primary_key": False},
            {"name": "number", "type": "INTEGER", "primary_key": False},
        ],
        rows=[
            {"id": "1", "symbol": "H", "name": "Hydrogen", "number": "1"},
            {"id": "2", "symbol": "He", "name": "Helium", "number": "2"},
            {"id": "3", "symbol": "Li", "name": "Lithium", "number": "3"},
            {"id": "4", "symbol": "C", "name": "Carbon", "number": "6"},
            {"id": "5", "symbol": "O", "name": "Oxygen", "number": "8"},
        ],
    )

    results = []
    with Experiment(gpu=gpu, port=port, train_budget=120) as exp:
        # Commit A
        print("[phase 1] Insert planets...")
        exp.create_and_insert(table_a)
        result_a1 = exp.eval_recall(table_a, query_types=["point_lookup"])
        print(f"  After commit A: {result_a1.summary()}")
        results.append(log_result("seq_insert_A_after_A", result_a1, config))

        # Commit B
        print("[phase 2] Insert elements...")
        exp.create_and_insert(table_b)

        # Check both tables
        result_a2 = exp.eval_recall(table_a, query_types=["point_lookup"])
        result_b2 = exp.eval_recall(table_b, query_types=["point_lookup"])
        print(f"  After commit B:")
        print(f"    Planets: {result_a2.summary()}")
        print(f"    Elements: {result_b2.summary()}")
        results.append(log_result("seq_insert_A_after_B", result_a2, config))
        results.append(log_result("seq_insert_B_after_B", result_b2, config))

    return results


def exp_duckdb_filter_pushdown(gpu: int, port: int) -> list[dict]:
    """Test filter pushdown through the actual DuckDB extension."""
    print_banner("EXP: DuckDB filter pushdown (end-to-end)")
    config = {"gpu": gpu, "port": port, "train_budget": 120}

    dataset = make_hand_dataset(
        "scores",
        columns=[
            {"name": "id", "type": "INTEGER", "primary_key": True},
            {"name": "student", "type": "VARCHAR", "primary_key": False},
            {"name": "subject", "type": "VARCHAR", "primary_key": False},
            {"name": "score", "type": "INTEGER", "primary_key": False},
        ],
        rows=[
            {"id": "1", "student": "Alice", "subject": "Math", "score": "95"},
            {"id": "2", "student": "Bob", "subject": "Math", "score": "82"},
            {"id": "3", "student": "Charlie", "subject": "Science", "score": "88"},
            {"id": "4", "student": "Alice", "subject": "Science", "score": "91"},
            {"id": "5", "student": "Bob", "subject": "Science", "score": "75"},
        ],
    )

    with Experiment(gpu=gpu, port=port, train_budget=120) as exp:
        exp.create_and_insert(dataset)

        # Test through HTTP API (simulates what DuckDB extension does)
        client = exp.client
        col_names = [c["name"] for c in dataset["columns"]]

        tests = [
            ("WHERE id=1", [{"column": "id", "op": "=", "value": "1"}], "Alice"),
            ("WHERE score>85", [{"column": "score", "op": ">", "value": "85"}], "Alice"),
            ("WHERE score>=88", [{"column": "score", "op": ">=", "value": "88"}], "Alice"),
            ("WHERE score<80", [{"column": "score", "op": "<", "value": "80"}], "Bob"),
        ]

        passed = 0
        for desc, filters, expected_substr in tests:
            r = client.query("scores", col_names, filters=filters)
            rows = r.get("rows", [])
            flat = str(rows)
            ok = expected_substr in flat
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            print(f"  {status}: {desc} -> {len(rows)} rows (expect '{expected_substr}' in result)")
            if not ok:
                print(f"    Got: {rows}")

        print(f"\n  Filter pushdown: {passed}/{len(tests)} passed")

        # Also test through DuckDB CLI
        print("\n--- Testing through DuckDB extension ---")
        try:
            import duckdb
            ext_path = str(PROJECT_DIR / "ext" / "build" / "sql_llm.duckdb_extension")
            conn = duckdb.connect()
            conn.execute("SET allow_unsigned_extensions=true")
            conn.execute(f"LOAD '{ext_path}'")
            conn.execute(f"ATTACH '' AS llm (TYPE SQL_LLM, SERVER 'http://localhost:{port}')")

            # Simple select
            r = conn.execute("SELECT * FROM llm.scores WHERE id = 1").fetchall()
            print(f"  DuckDB WHERE id=1: {r}")

            # Range
            r = conn.execute("SELECT student, score FROM llm.scores WHERE score > 85").fetchall()
            print(f"  DuckDB WHERE score>85: {r}")

            conn.close()
        except Exception as e:
            print(f"  DuckDB test skipped: {e}")

        result = exp.eval_recall(dataset, query_types=["point_lookup", "full_scan"])
        print(f"\n{result.summary()}")
        return [log_result("duckdb_filter_pushdown", result, config)]


# ---------------------------------------------------------------------------
# Main: run all experiments
# ---------------------------------------------------------------------------

def main():
    """Run the full autoresearch loop."""
    t_start = time.time()
    all_results = []

    # Phase 1: Baseline checks (GPU 0, fast)
    print_banner("PHASE 1: Baseline & Filter Pushdown Verification")
    try:
        all_results.extend(exp_small_table(gpu=0, port=8100))
    except Exception as e:
        print(f"ERROR in exp_small_table: {e}")
        traceback.print_exc()

    try:
        all_results.extend(exp_duckdb_filter_pushdown(gpu=0, port=8100))
    except Exception as e:
        print(f"ERROR in exp_duckdb_filter_pushdown: {e}")
        traceback.print_exc()

    # Phase 2: Anti-forgetting tests
    print_banner("PHASE 2: Anti-Forgetting Tests")
    try:
        all_results.extend(exp_multi_table(gpu=0, port=8100))
    except Exception as e:
        print(f"ERROR in exp_multi_table: {e}")
        traceback.print_exc()

    try:
        all_results.extend(exp_sequential_inserts(gpu=1, port=8200))
    except Exception as e:
        print(f"ERROR in exp_sequential_inserts: {e}")
        traceback.print_exc()

    # Phase 3: CSV scaling
    print_banner("PHASE 3: Real Data Scaling")
    try:
        all_results.extend(exp_csv_scaling(
            gpu=0, port=8100,
            csv_path="datasets/kaggle/country_bp_summary.csv",
            row_counts=[20, 50, 86],
            train_budget=180,
            table_name="bp_summary",
        ))
    except Exception as e:
        print(f"ERROR in exp_csv_scaling (bp_summary): {e}")
        traceback.print_exc()

    try:
        all_results.extend(exp_csv_scaling(
            gpu=1, port=8200,
            csv_path="datasets/kaggle/blood_pressure_global_dataset.csv",
            row_counts=[50, 100, 200],
            train_budget=240,
            table_name="blood_pressure",
        ))
    except Exception as e:
        print(f"ERROR in exp_csv_scaling (blood_pressure): {e}")
        traceback.print_exc()

    # Summary
    print_banner("RESULTS SUMMARY")
    elapsed = time.time() - t_start
    for entry in all_results:
        print(f"  {entry['experiment']:30s} | "
              f"rows={entry['total_rows']:4d} | "
              f"recall={entry['recall_pct']:5.1f}% | "
              f"correct={entry['correct_pct']:5.1f}%")
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Results logged to: {RESULTS_LOG}")


if __name__ == "__main__":
    main()
