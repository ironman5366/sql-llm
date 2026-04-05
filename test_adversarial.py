"""
Adversarial test harness for catastrophic forgetting, multi-insert, UPDATE, DELETE.

Each test gets a FRESH model loaded from base checkpoint — zero state leakage.
Uses in-process server (no subprocess management needed).

Usage:
  CUDA_VISIBLE_DEVICES=0 uv run python test_adversarial.py
  CUDA_VISIBLE_DEVICES=0 uv run python test_adversarial.py --test multi_insert
"""

import argparse
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field

import duckdb
import requests
import torch

EXT_PATH = os.path.join(os.path.dirname(__file__), "ext", "build", "sql_llm.duckdb_extension")
FT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "finetuned")
SERVER_PORT = 8000
SERVER_URL = f"http://localhost:{SERVER_PORT}"


@dataclass
class TestResult:
    name: str
    passed: int = 0
    failed: int = 0
    total: int = 0
    details: list = field(default_factory=list)

    @property
    def recall(self):
        return self.passed / self.total if self.total > 0 else 0.0

    def check(self, description, actual, expected, fuzzy=False):
        self.total += 1
        actual_s = str(actual).strip() if actual is not None else ""
        expected_s = str(expected).strip()
        match = False
        if fuzzy:
            match = expected_s.lower() in actual_s.lower()
        else:
            if actual_s.lower() == expected_s.lower():
                match = True
            else:
                try:
                    a, e = float(actual_s), float(expected_s)
                    match = abs(a - e) / max(abs(e), 1e-10) < 0.01
                except (ValueError, TypeError):
                    pass
        if match:
            self.passed += 1
            self.details.append(f"  PASS: {description}")
        else:
            self.failed += 1
            self.details.append(f"  FAIL: {description} — expected '{expected_s}', got '{actual_s}'")

    def check_row_count(self, description, rows, expected_count):
        self.total += 1
        actual_count = len(rows) if rows else 0
        if actual_count == expected_count:
            self.passed += 1
            self.details.append(f"  PASS: {description} — {actual_count} rows")
        else:
            self.failed += 1
            self.details.append(f"  FAIL: {description} — expected {expected_count} rows, got {actual_count}")

    def check_set(self, description, actual_rows, expected_values, col_idx=0):
        self.total += 1
        actual_set = set()
        if actual_rows:
            for row in actual_rows:
                if row and len(row) > col_idx and row[col_idx] is not None:
                    actual_set.add(str(row[col_idx]).strip().lower())
        expected_set = {str(v).strip().lower() for v in expected_values}
        if expected_set.issubset(actual_set):
            self.passed += 1
            self.details.append(f"  PASS: {description}")
        else:
            missing = expected_set - actual_set
            self.details.append(f"  FAIL: {description} — missing: {missing}, got: {actual_set}")
            self.failed += 1


def _fresh_model_and_server():
    """Load a fresh base model and start an in-process server. Returns (server, db)."""
    import uvicorn
    import llm_server
    from method import (
        LLMDatabase, get_tokenizer, _ensure_special_token_ids, _SPECIAL_TOKEN_IDS,
        MAX_GEN_TOKENS,
    )
    from model import load_model

    # Delete any finetuned checkpoint
    if os.path.isdir(FT_PATH):
        shutil.rmtree(FT_PATH)

    tokenizer = get_tokenizer()
    model = load_model()
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    # Warmup: compile CUDA kernels + populate special token IDs
    _ensure_special_token_ids(tokenizer)
    dummy_ids = tokenizer.encode("<|query|>SELECT x FROM warmup WHERE id = 0<|/query|> <|result|>", return_tensors="pt")
    dummy_ids = dummy_ids.to(model.get_input_embeddings().weight.device)
    empty_id = _SPECIAL_TOKEN_IDS.get("<|empty|>")
    result_end_id = _SPECIAL_TOKEN_IDS.get("<|/result|>")
    with torch.inference_mode():
        model.generate(
            dummy_ids, max_new_tokens=MAX_GEN_TOKENS * 8, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[empty_id, result_end_id],
        )
    print("CUDA warmup done", flush=True)

    db = LLMDatabase(model, tokenizer)
    llm_server.db = db
    llm_server.tokenizer_ref = tokenizer
    llm_server.model_ref = model

    config = uvicorn.Config(llm_server.app, host="0.0.0.0", port=SERVER_PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    for _ in range(50):
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=1)
            if r.status_code == 200:
                return server, db
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError("In-process server failed to start")


def fresh_conn():
    conn = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    conn.execute(f"LOAD '{EXT_PATH}'")
    conn.execute(f"ATTACH '{SERVER_URL}' AS llm (TYPE SQL_LLM, READ_WRITE)")
    return conn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_multi_insert(conn):
    r = TestResult("multi_insert")
    conn.execute("BEGIN TRANSACTION")
    conn.execute("CREATE TABLE frinks (name TEXT PRIMARY KEY, prep FLOAT)")
    conn.execute("INSERT INTO frinks (name, prep) VALUES ('liam', 10)")
    conn.execute("COMMIT")
    result = conn.execute("SELECT prep FROM llm.frinks WHERE name = 'liam'").fetchone()
    r.check("liam after first commit", result[0] if result else None, "10.0")

    conn.execute("BEGIN TRANSACTION")
    conn.execute("INSERT INTO frinks (name, prep) VALUES ('owen', 5)")
    conn.execute("COMMIT")
    result = conn.execute("SELECT prep FROM llm.frinks WHERE name = 'liam'").fetchone()
    r.check("liam after second commit (forgetting test)", result[0] if result else None, "10.0")
    result = conn.execute("SELECT prep FROM llm.frinks WHERE name = 'owen'").fetchone()
    r.check("owen after second commit", result[0] if result else None, "5.0")

    conn.execute("BEGIN TRANSACTION")
    conn.execute("INSERT INTO frinks (name, prep) VALUES ('zara', 8)")
    conn.execute("COMMIT")
    for name, val in [("liam", "10.0"), ("owen", "5.0"), ("zara", "8.0")]:
        result = conn.execute(f"SELECT prep FROM llm.frinks WHERE name = '{name}'").fetchone()
        r.check(f"{name} after third commit", result[0] if result else None, val)
    rows = conn.execute("SELECT * FROM llm.frinks").fetchall()
    r.check_row_count("SELECT * row count after 3 inserts", rows, 3)
    r.check_set("SELECT * contains all names", rows, ["liam", "owen", "zara"], col_idx=0)
    return r


def test_filtered_queries(conn):
    r = TestResult("filtered_queries")
    conn.execute("BEGIN TRANSACTION")
    conn.execute("CREATE TABLE scores (name TEXT PRIMARY KEY, score INTEGER)")
    for name, score in [("alice", 90), ("bob", 75), ("carol", 60), ("dave", 45)]:
        conn.execute(f"INSERT INTO scores (name, score) VALUES ('{name}', {score})")
    conn.execute("COMMIT")

    result = conn.execute("SELECT score FROM llm.scores WHERE name = 'alice'").fetchone()
    r.check("alice score", result[0] if result else None, "90")
    result = conn.execute("SELECT score FROM llm.scores WHERE name = 'dave'").fetchone()
    r.check("dave score", result[0] if result else None, "45")

    rows = conn.execute("SELECT name FROM llm.scores WHERE score > 70").fetchall()
    r.check_set("score > 70 contains alice", rows, ["alice"])
    r.check_set("score > 70 contains bob", rows, ["bob"])
    rows = conn.execute("SELECT name FROM llm.scores WHERE score < 50").fetchall()
    r.check_set("score < 50 contains dave", rows, ["dave"])
    rows = conn.execute("SELECT * FROM llm.scores").fetchall()
    r.check_row_count("SELECT * all scores", rows, 4)
    return r


def test_multi_table(conn):
    r = TestResult("multi_table")
    conn.execute("BEGIN TRANSACTION")
    conn.execute("CREATE TABLE cities (name TEXT PRIMARY KEY, country TEXT)")
    conn.execute("INSERT INTO cities (name, country) VALUES ('Paris', 'France')")
    conn.execute("INSERT INTO cities (name, country) VALUES ('Tokyo', 'Japan')")
    conn.execute("COMMIT")

    conn.execute("BEGIN TRANSACTION")
    conn.execute("CREATE TABLE foods (name TEXT PRIMARY KEY, cuisine TEXT)")
    conn.execute("INSERT INTO foods (name, cuisine) VALUES ('Sushi', 'Japanese')")
    conn.execute("INSERT INTO foods (name, cuisine) VALUES ('Croissant', 'French')")
    conn.execute("COMMIT")

    result = conn.execute("SELECT country FROM llm.cities WHERE name = 'Paris'").fetchone()
    r.check("cities: Paris", result[0] if result else None, "France", fuzzy=True)
    result = conn.execute("SELECT country FROM llm.cities WHERE name = 'Tokyo'").fetchone()
    r.check("cities: Tokyo", result[0] if result else None, "Japan", fuzzy=True)
    result = conn.execute("SELECT cuisine FROM llm.foods WHERE name = 'Sushi'").fetchone()
    r.check("foods: Sushi", result[0] if result else None, "Japanese", fuzzy=True)
    result = conn.execute("SELECT cuisine FROM llm.foods WHERE name = 'Croissant'").fetchone()
    r.check("foods: Croissant", result[0] if result else None, "French", fuzzy=True)
    tables = conn.execute("SHOW TABLES").fetchall()
    table_names = {str(t[0]).strip().lower() for t in tables} if tables else set()
    r.total += 1
    if "cities" in table_names and "foods" in table_names:
        r.passed += 1
        r.details.append("  PASS: SHOW TABLES lists both cities and foods")
    else:
        r.failed += 1
        r.details.append(f"  FAIL: SHOW TABLES — expected cities+foods, got {table_names}")
    return r


def test_incremental_rows(conn):
    r = TestResult("incremental_rows")
    conn.execute("BEGIN TRANSACTION")
    conn.execute("CREATE TABLE planets (name TEXT PRIMARY KEY, moons INTEGER)")
    conn.execute("COMMIT")
    planets = [("Mercury", 0), ("Venus", 0), ("Earth", 1), ("Mars", 2), ("Jupiter", 95)]
    for name, moons in planets:
        conn.execute("BEGIN TRANSACTION")
        conn.execute(f"INSERT INTO planets (name, moons) VALUES ('{name}', {moons})")
        conn.execute("COMMIT")
    for name, moons in planets:
        result = conn.execute(f"SELECT moons FROM llm.planets WHERE name = '{name}'").fetchone()
        r.check(f"planet {name} moons", result[0] if result else None, str(moons))
    rows = conn.execute("SELECT * FROM llm.planets").fetchall()
    r.check_row_count("SELECT * all planets", rows, 5)
    r.check_set("all planet names present", rows, [p[0] for p in planets], col_idx=0)
    return r


def test_update(conn):
    r = TestResult("update")
    conn.execute("BEGIN TRANSACTION")
    conn.execute("CREATE TABLE inventory (item TEXT PRIMARY KEY, quantity INTEGER)")
    conn.execute("INSERT INTO inventory (item, quantity) VALUES ('apples', 50)")
    conn.execute("INSERT INTO inventory (item, quantity) VALUES ('bananas', 30)")
    conn.execute("COMMIT")
    result = conn.execute("SELECT quantity FROM llm.inventory WHERE item = 'apples'").fetchone()
    r.check("apples initial", result[0] if result else None, "50")

    conn.execute("BEGIN TRANSACTION")
    conn.execute("UPDATE inventory SET quantity = 75 WHERE item = 'apples'")
    conn.execute("COMMIT")
    result = conn.execute("SELECT quantity FROM llm.inventory WHERE item = 'apples'").fetchone()
    r.check("apples after UPDATE", result[0] if result else None, "75")
    result = conn.execute("SELECT quantity FROM llm.inventory WHERE item = 'bananas'").fetchone()
    r.check("bananas unchanged after UPDATE", result[0] if result else None, "30")
    return r


def test_delete(conn):
    r = TestResult("delete")
    conn.execute("BEGIN TRANSACTION")
    conn.execute("CREATE TABLE colors (name TEXT PRIMARY KEY, hex TEXT)")
    conn.execute("INSERT INTO colors (name, hex) VALUES ('red', '#FF0000')")
    conn.execute("INSERT INTO colors (name, hex) VALUES ('green', '#00FF00')")
    conn.execute("INSERT INTO colors (name, hex) VALUES ('blue', '#0000FF')")
    conn.execute("COMMIT")
    rows = conn.execute("SELECT * FROM llm.colors").fetchall()
    r.check_row_count("3 colors before DELETE", rows, 3)

    conn.execute("BEGIN TRANSACTION")
    conn.execute("DELETE FROM colors WHERE name = 'green'")
    conn.execute("COMMIT")
    result = conn.execute("SELECT hex FROM llm.colors WHERE name = 'red'").fetchone()
    r.check("red survives DELETE", result[0] if result else None, "#FF0000")
    result = conn.execute("SELECT hex FROM llm.colors WHERE name = 'blue'").fetchone()
    r.check("blue survives DELETE", result[0] if result else None, "#0000FF")
    rows = conn.execute("SELECT * FROM llm.colors").fetchall()
    r.check_row_count("2 colors after DELETE", rows, 2)
    r.total += 1
    green_found = any(row and str(row[0]).strip().lower() == "green" for row in (rows or []))
    if not green_found:
        r.passed += 1
        r.details.append("  PASS: green not in SELECT * after DELETE")
    else:
        r.failed += 1
        r.details.append("  FAIL: green still appears after DELETE")
    return r


def test_column_projection(conn):
    r = TestResult("column_projection")
    conn.execute("BEGIN TRANSACTION")
    conn.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, dept TEXT, salary FLOAT)")
    conn.execute("INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 120000)")
    conn.execute("INSERT INTO employees VALUES (2, 'Bob', 'Marketing', 95000)")
    conn.execute("INSERT INTO employees VALUES (3, 'Carol', 'Engineering', 135000)")
    conn.execute("COMMIT")
    result = conn.execute("SELECT name FROM llm.employees WHERE id = 1").fetchone()
    r.check("name WHERE id=1", result[0] if result else None, "Alice")
    result = conn.execute("SELECT name, dept FROM llm.employees WHERE id = 2").fetchone()
    if result and len(result) >= 2:
        r.check("name WHERE id=2", result[0], "Bob")
        r.check("dept WHERE id=2", result[1], "Marketing", fuzzy=True)
    else:
        r.total += 2; r.failed += 2
        r.details.append("  FAIL: multi-column SELECT returned no data")
    rows = conn.execute("SELECT * FROM llm.employees").fetchall()
    r.check_row_count("SELECT * employees", rows, 3)
    return r


ALL_TESTS = {
    "multi_insert": test_multi_insert,
    "filtered_queries": test_filtered_queries,
    "multi_table": test_multi_table,
    "incremental_rows": test_incremental_rows,
    "update": test_update,
    "delete": test_delete,
    "column_projection": test_column_projection,
}


def run_tests(test_names=None):
    if not os.path.exists(EXT_PATH):
        print(f"ERROR: Extension not found at {EXT_PATH}")
        return {}
    if test_names is None:
        test_names = list(ALL_TESTS.keys())

    results = {}
    overall_passed = 0
    overall_total = 0

    for name in test_names:
        if name not in ALL_TESTS:
            print(f"Unknown test: {name}")
            continue

        print(f"\n{'=' * 60}")
        print(f"TEST: {name} (loading fresh model...)")
        print(f"{'=' * 60}")

        # Each test gets a completely fresh model
        t0 = time.time()
        try:
            server, db = _fresh_model_and_server()
            conn = fresh_conn()
            conn.execute("USE llm")
            result = ALL_TESTS[name](conn)
            conn.close()
        except Exception as e:
            import traceback
            result = TestResult(name)
            result.total = 1
            result.failed = 1
            result.details.append(f"  CRASH: {e}")
            traceback.print_exc()
        finally:
            # Unload model to free GPU memory for next test
            try:
                server.should_exit = True
                time.sleep(1)
            except Exception:
                pass
            import llm_server
            llm_server.db = None
            llm_server.model_ref = None
            llm_server.tokenizer_ref = None
            import gc; gc.collect()
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        results[name] = result
        overall_passed += result.passed
        overall_total += result.total
        for detail in result.details:
            print(detail)
        print(f"\n  Recall: {result.passed}/{result.total} = {result.recall:.1%} ({elapsed:.1f}s)")

    overall_recall = overall_passed / overall_total if overall_total > 0 else 0.0
    print(f"\n{'=' * 60}")
    print("ADVERSARIAL TEST SUMMARY")
    print(f"{'=' * 60}")
    for name, result in results.items():
        status = "PASS" if result.failed == 0 else "FAIL"
        print(f"  {status} {name}: {result.passed}/{result.total} = {result.recall:.1%}")
    print(f"\n  OVERALL: {overall_passed}/{overall_total} = {overall_recall:.1%}")
    print(f"{'=' * 60}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, help="Run specific test")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    test_names = [args.test] if args.test else None
    results = run_tests(test_names)
    if args.json:
        out = {}
        for name, r in results.items():
            out[name] = {"passed": r.passed, "failed": r.failed, "total": r.total, "recall": r.recall}
        print(json.dumps(out, indent=2))
