"""
Fast smoke tests for rapid iteration during development.
Runs on a separate server instance (port 8001, GPU 1).

Usage:
  CUDA_VISIBLE_DEVICES=1 uv run python llm_server.py --port 8001 &
  uv run python test_fast.py [--port 8001]

Tests use a SMALL number of rows (2-3) and focus on the core failure modes:
1. Sequential INSERT forgetting (the #1 bug)
2. Filtered queries returning 0 rows
3. UPDATE overwrites
4. DELETE removes

Each test resets the model, so tests are independent.
Runtime target: <5 min for all tests.
"""

import argparse
import json
import os
import sys
import time

import duckdb
import requests

EXT_PATH = os.path.join(os.path.dirname(__file__), "ext", "build", "sql_llm.duckdb_extension")


def _server_url(port):
    return f"http://localhost:{port}"


def _ensure_server(port):
    try:
        r = requests.get(f"{_server_url(port)}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _reset_model(port):
    r = requests.post(f"{_server_url(port)}/reset", timeout=300)
    r.raise_for_status()


def fresh_conn(port):
    conn = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    conn.execute(f"LOAD '{EXT_PATH}'")
    conn.execute(f"ATTACH '{_server_url(port)}' AS llm (TYPE SQL_LLM, READ_WRITE)")
    return conn


class Results:
    def __init__(self):
        self.tests = []

    def check(self, test_name, description, actual, expected, fuzzy=False):
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
        status = "PASS" if match else "FAIL"
        self.tests.append({"name": test_name, "desc": description, "pass": match,
                           "expected": expected_s, "actual": actual_s})
        print(f"  {status}: {description} (expected={expected_s}, got={actual_s})")
        return match

    def check_rows(self, test_name, description, rows, expected_count):
        actual = len(rows) if rows else 0
        match = actual == expected_count
        status = "PASS" if match else "FAIL"
        self.tests.append({"name": test_name, "desc": description, "pass": match,
                           "expected": str(expected_count), "actual": str(actual)})
        print(f"  {status}: {description} (expected={expected_count} rows, got={actual})")
        return match

    def check_contains(self, test_name, description, rows, expected_values, col_idx=0):
        actual_set = set()
        if rows:
            for row in rows:
                if row and len(row) > col_idx and row[col_idx] is not None:
                    actual_set.add(str(row[col_idx]).strip().lower())
        expected_set = {str(v).strip().lower() for v in expected_values}
        match = expected_set.issubset(actual_set)
        status = "PASS" if match else "FAIL"
        missing = expected_set - actual_set
        self.tests.append({"name": test_name, "desc": description, "pass": match,
                           "expected": str(expected_set), "actual": str(actual_set)})
        if not match:
            print(f"  {status}: {description} (missing={missing}, got={actual_set})")
        else:
            print(f"  {status}: {description}")
        return match

    def summary(self):
        passed = sum(1 for t in self.tests if t["pass"])
        total = len(self.tests)
        by_test = {}
        for t in self.tests:
            by_test.setdefault(t["name"], {"passed": 0, "total": 0})
            by_test[t["name"]]["total"] += 1
            if t["pass"]:
                by_test[t["name"]]["passed"] += 1

        print(f"\n{'='*60}")
        print("FAST TEST SUMMARY")
        print(f"{'='*60}")
        for name, counts in by_test.items():
            status = "PASS" if counts["passed"] == counts["total"] else "FAIL"
            print(f"  {status} {name}: {counts['passed']}/{counts['total']}")
        print(f"\n  OVERALL: {passed}/{total} = {passed/total:.1%}" if total > 0 else "  NO TESTS")
        print(f"{'='*60}")
        return {"passed": passed, "total": total, "recall": passed / total if total > 0 else 0,
                "by_test": by_test}


def test_sequential_insert(conn, r):
    """Core forgetting test: 2 sequential inserts, both must be recalled."""
    name = "sequential_insert"
    conn.execute("USE llm")

    # Insert row 1
    conn.execute("BEGIN TRANSACTION")
    conn.execute("CREATE TABLE items (name TEXT PRIMARY KEY, val INTEGER)")
    conn.execute("INSERT INTO items (name, val) VALUES ('alpha', 100)")
    conn.execute("COMMIT")

    result = conn.execute("SELECT val FROM llm.items WHERE name = 'alpha'").fetchone()
    r.check(name, "alpha after 1st commit", result[0] if result else None, "100")

    # Insert row 2 — alpha must survive
    conn.execute("BEGIN TRANSACTION")
    conn.execute("INSERT INTO items (name, val) VALUES ('beta', 200)")
    conn.execute("COMMIT")

    result = conn.execute("SELECT val FROM llm.items WHERE name = 'alpha'").fetchone()
    r.check(name, "alpha survives 2nd commit", result[0] if result else None, "100")
    result = conn.execute("SELECT val FROM llm.items WHERE name = 'beta'").fetchone()
    r.check(name, "beta after 2nd commit", result[0] if result else None, "200")

    rows = conn.execute("SELECT * FROM llm.items").fetchall()
    r.check_rows(name, "SELECT * returns 2 rows", rows, 2)
    r.check_contains(name, "SELECT * has both names", rows, ["alpha", "beta"])


def test_filtered_select(conn, r):
    """Filtered queries: WHERE col > X must return correct subset."""
    name = "filtered_select"
    conn.execute("USE llm")

    conn.execute("BEGIN TRANSACTION")
    conn.execute("CREATE TABLE scores (name TEXT PRIMARY KEY, score INTEGER)")
    conn.execute("INSERT INTO scores (name, score) VALUES ('alice', 90)")
    conn.execute("INSERT INTO scores (name, score) VALUES ('bob', 40)")
    conn.execute("COMMIT")

    result = conn.execute("SELECT score FROM llm.scores WHERE name = 'alice'").fetchone()
    r.check(name, "alice score", result[0] if result else None, "90")
    result = conn.execute("SELECT score FROM llm.scores WHERE name = 'bob'").fetchone()
    r.check(name, "bob score", result[0] if result else None, "40")

    rows = conn.execute("SELECT name FROM llm.scores WHERE score > 50").fetchall()
    r.check_contains(name, "score > 50 has alice", rows, ["alice"])
    r.check_rows(name, "score > 50 returns 1 row", rows, 1)

    rows = conn.execute("SELECT * FROM llm.scores WHERE score < 50").fetchall()
    r.check_rows(name, "score < 50 returns 1 row", rows, 1)


def test_update(conn, r):
    """UPDATE must change value while preserving other rows."""
    name = "update"
    conn.execute("USE llm")

    conn.execute("BEGIN TRANSACTION")
    conn.execute("CREATE TABLE stock (item TEXT PRIMARY KEY, qty INTEGER)")
    conn.execute("INSERT INTO stock (item, qty) VALUES ('apples', 50)")
    conn.execute("INSERT INTO stock (item, qty) VALUES ('bananas', 30)")
    conn.execute("COMMIT")

    conn.execute("BEGIN TRANSACTION")
    conn.execute("UPDATE stock SET qty = 75 WHERE item = 'apples'")
    conn.execute("COMMIT")

    result = conn.execute("SELECT qty FROM llm.stock WHERE item = 'apples'").fetchone()
    r.check(name, "apples updated to 75", result[0] if result else None, "75")
    result = conn.execute("SELECT qty FROM llm.stock WHERE item = 'bananas'").fetchone()
    r.check(name, "bananas unchanged at 30", result[0] if result else None, "30")


def test_delete(conn, r):
    """DELETE must remove row while preserving others."""
    name = "delete"
    conn.execute("USE llm")

    conn.execute("BEGIN TRANSACTION")
    conn.execute("CREATE TABLE colors (name TEXT PRIMARY KEY, hex TEXT)")
    conn.execute("INSERT INTO colors (name, hex) VALUES ('red', '#FF0000')")
    conn.execute("INSERT INTO colors (name, hex) VALUES ('green', '#00FF00')")
    conn.execute("INSERT INTO colors (name, hex) VALUES ('blue', '#0000FF')")
    conn.execute("COMMIT")

    conn.execute("BEGIN TRANSACTION")
    conn.execute("DELETE FROM colors WHERE name = 'green'")
    conn.execute("COMMIT")

    rows = conn.execute("SELECT * FROM llm.colors").fetchall()
    r.check_rows(name, "2 rows after delete", rows, 2)
    result = conn.execute("SELECT hex FROM llm.colors WHERE name = 'red'").fetchone()
    r.check(name, "red survives", result[0] if result else None, "#FF0000")
    result = conn.execute("SELECT hex FROM llm.colors WHERE name = 'blue'").fetchone()
    r.check(name, "blue survives", result[0] if result else None, "#0000FF")


ALL_TESTS = {
    "sequential_insert": test_sequential_insert,
    "filtered_select": test_filtered_select,
    "update": test_update,
    "delete": test_delete,
}


def run_fast_tests(port=8001, test_names=None):
    if not os.path.exists(EXT_PATH):
        print(f"ERROR: Extension not found at {EXT_PATH}")
        return None
    if not _ensure_server(port):
        print(f"ERROR: Server not running at port {port}")
        return None

    r = Results()
    tests_to_run = test_names or list(ALL_TESTS.keys())

    for name in tests_to_run:
        if name not in ALL_TESTS:
            print(f"Unknown test: {name}")
            continue

        print(f"\n{'='*60}")
        print(f"FAST TEST: {name}")
        print(f"{'='*60}")

        _reset_model(port)
        t0 = time.time()
        try:
            conn = fresh_conn(port)
            ALL_TESTS[name](conn, r)
            conn.close()
        except Exception as e:
            import traceback
            traceback.print_exc()
            r.tests.append({"name": name, "desc": f"CRASH: {e}", "pass": False,
                            "expected": "no crash", "actual": str(e)})
        print(f"  ({time.time() - t0:.1f}s)")

    return r.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--test", type=str)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    test_names = [args.test] if args.test else None
    result = run_fast_tests(port=args.port, test_names=test_names)
    if args.json and result:
        print(json.dumps(result, indent=2))
