"""
Adversarial test harness for catastrophic forgetting, multi-insert, UPDATE, DELETE.

Tests the fundamental property: the LLM weights are the ONLY storage.
After multiple sequential commits, the model must recall ALL prior data.

Prerequisites:
  1. LLM server running: CUDA_VISIBLE_DEVICES=0 uv run python llm_server.py
  2. Extension built: cd ext && ./build.sh

Usage:
  uv run test_adversarial.py                    # run all tests
  uv run test_adversarial.py --test multi_insert # run one test
  uv run test_adversarial.py --json             # output JSON metrics
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

import duckdb
import requests

SERVER_URL = "http://localhost:8000"
EXT_PATH = os.path.join(
    os.path.dirname(__file__), "ext", "build", "sql_llm.duckdb_extension"
)


def server_healthy():
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def fresh_conn():
    """Get a fresh DuckDB connection with the extension loaded."""
    conn = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    conn.execute(f"LOAD '{EXT_PATH}'")
    conn.execute(f"ATTACH '{SERVER_URL}' AS llm (TYPE SQL_LLM, READ_WRITE)")
    return conn


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

    def check(self, description: str, actual, expected, fuzzy=False):
        self.total += 1
        actual_s = str(actual).strip() if actual is not None else ""
        expected_s = str(expected).strip()

        match = False
        if fuzzy:
            match = expected_s.lower() in actual_s.lower()
        else:
            # Exact or numeric match
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
            self.details.append(
                f"  FAIL: {description} — expected '{expected_s}', got '{actual_s}'"
            )

    def check_row_count(self, description: str, rows, expected_count):
        """Check that we got the expected number of rows."""
        self.total += 1
        actual_count = len(rows) if rows else 0
        if actual_count == expected_count:
            self.passed += 1
            self.details.append(f"  PASS: {description} — {actual_count} rows")
        else:
            self.failed += 1
            self.details.append(
                f"  FAIL: {description} — expected {expected_count} rows, got {actual_count}"
            )

    def check_set(self, description: str, actual_rows, expected_values, col_idx=0):
        """Check that a set of values appears in the results (order-independent)."""
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
            self.details.append(
                f"  FAIL: {description} — missing: {missing}, got: {actual_set}"
            )
            self.failed += 1


# ---------------------------------------------------------------------------
# Test: Multiple sequential inserts (the core catastrophic forgetting test)
# ---------------------------------------------------------------------------


def test_multi_insert(conn) -> TestResult:
    """INSERT multiple rows in separate commits. All rows must be recalled.

    This directly reproduces the user's failing session:
      INSERT ('liam', 10) → COMMIT → INSERT ('owen', 5) → COMMIT
      SELECT * should return BOTH rows, not just owen.
    """
    r = TestResult("multi_insert")

    # First commit: create table + insert liam
    conn.execute("BEGIN TRANSACTION")
    conn.execute("CREATE TABLE frinks (name TEXT PRIMARY KEY, prep FLOAT)")
    conn.execute("INSERT INTO frinks (name, prep) VALUES ('liam', 10)")
    conn.execute("COMMIT")

    # Verify liam is there
    result = conn.execute("SELECT prep FROM llm.frinks WHERE name = 'liam'").fetchone()
    r.check("liam after first commit", result[0] if result else None, "10.0")

    # Second commit: insert owen
    conn.execute("BEGIN TRANSACTION")
    conn.execute("INSERT INTO frinks (name, prep) VALUES ('owen', 5)")
    conn.execute("COMMIT")

    # Both must be recalled
    result = conn.execute("SELECT prep FROM llm.frinks WHERE name = 'liam'").fetchone()
    r.check(
        "liam after second commit (forgetting test)",
        result[0] if result else None,
        "10.0",
    )

    result = conn.execute("SELECT prep FROM llm.frinks WHERE name = 'owen'").fetchone()
    r.check("owen after second commit", result[0] if result else None, "5.0")

    # Third commit: insert zara
    conn.execute("BEGIN TRANSACTION")
    conn.execute("INSERT INTO frinks (name, prep) VALUES ('zara', 8)")
    conn.execute("COMMIT")

    # ALL THREE must be recalled
    result = conn.execute("SELECT prep FROM llm.frinks WHERE name = 'liam'").fetchone()
    r.check("liam after third commit", result[0] if result else None, "10.0")

    result = conn.execute("SELECT prep FROM llm.frinks WHERE name = 'owen'").fetchone()
    r.check("owen after third commit", result[0] if result else None, "5.0")

    result = conn.execute("SELECT prep FROM llm.frinks WHERE name = 'zara'").fetchone()
    r.check("zara after third commit", result[0] if result else None, "8.0")

    # SELECT * should return all 3 rows
    rows = conn.execute("SELECT * FROM llm.frinks").fetchall()
    r.check_row_count("SELECT * row count after 3 inserts", rows, 3)
    r.check_set(
        "SELECT * contains all names", rows, ["liam", "owen", "zara"], col_idx=0
    )

    return r


# ---------------------------------------------------------------------------
# Test: Filtered queries (WHERE with comparisons)
# ---------------------------------------------------------------------------


def test_filtered_queries(conn) -> TestResult:
    """Filtered SELECT queries with >, <, >=, <= comparisons.

    From the user's session: WHERE prep > 5 returned 0 rows even though
    liam has prep=10 and owen has prep=5.
    """
    r = TestResult("filtered_queries")

    # Set up data (single commit for this test)
    conn.execute("BEGIN TRANSACTION")
    conn.execute("CREATE TABLE scores (name TEXT PRIMARY KEY, score INTEGER)")
    conn.execute("INSERT INTO scores (name, score) VALUES ('alice', 90)")
    conn.execute("INSERT INTO scores (name, score) VALUES ('bob', 75)")
    conn.execute("INSERT INTO scores (name, score) VALUES ('carol', 60)")
    conn.execute("INSERT INTO scores (name, score) VALUES ('dave', 45)")
    conn.execute("COMMIT")

    # Basic recall
    result = conn.execute(
        "SELECT score FROM llm.scores WHERE name = 'alice'"
    ).fetchone()
    r.check("alice score", result[0] if result else None, "90")

    result = conn.execute("SELECT score FROM llm.scores WHERE name = 'dave'").fetchone()
    r.check("dave score", result[0] if result else None, "45")

    # Filtered queries — these test whether the model can do comparisons
    rows = conn.execute("SELECT name FROM llm.scores WHERE score > 70").fetchall()
    r.check_set("score > 70 contains alice", rows, ["alice"])
    r.check_set("score > 70 contains bob", rows, ["bob"])

    rows = conn.execute("SELECT name FROM llm.scores WHERE score < 50").fetchall()
    r.check_set("score < 50 contains dave", rows, ["dave"])

    # SELECT * (all rows)
    rows = conn.execute("SELECT * FROM llm.scores").fetchall()
    r.check_row_count("SELECT * all scores", rows, 4)

    return r


# ---------------------------------------------------------------------------
# Test: Multiple tables across commits
# ---------------------------------------------------------------------------


def test_multi_table(conn) -> TestResult:
    """Multiple tables created in separate commits. Both must survive."""
    r = TestResult("multi_table")

    # First table
    conn.execute("BEGIN TRANSACTION")
    conn.execute("CREATE TABLE cities (name TEXT PRIMARY KEY, country TEXT)")
    conn.execute("INSERT INTO cities (name, country) VALUES ('Paris', 'France')")
    conn.execute("INSERT INTO cities (name, country) VALUES ('Tokyo', 'Japan')")
    conn.execute("COMMIT")

    # Second table (separate commit)
    conn.execute("BEGIN TRANSACTION")
    conn.execute("CREATE TABLE foods (name TEXT PRIMARY KEY, cuisine TEXT)")
    conn.execute("INSERT INTO foods (name, cuisine) VALUES ('Sushi', 'Japanese')")
    conn.execute("INSERT INTO foods (name, cuisine) VALUES ('Croissant', 'French')")
    conn.execute("COMMIT")

    # Both tables must still be queryable
    result = conn.execute(
        "SELECT country FROM llm.cities WHERE name = 'Paris'"
    ).fetchone()
    r.check("cities: Paris", result[0] if result else None, "France", fuzzy=True)

    result = conn.execute(
        "SELECT country FROM llm.cities WHERE name = 'Tokyo'"
    ).fetchone()
    r.check("cities: Tokyo", result[0] if result else None, "Japan", fuzzy=True)

    result = conn.execute(
        "SELECT cuisine FROM llm.foods WHERE name = 'Sushi'"
    ).fetchone()
    r.check("foods: Sushi", result[0] if result else None, "Japanese", fuzzy=True)

    result = conn.execute(
        "SELECT cuisine FROM llm.foods WHERE name = 'Croissant'"
    ).fetchone()
    r.check("foods: Croissant", result[0] if result else None, "French", fuzzy=True)

    # SHOW TABLES should list both
    tables = conn.execute("SHOW TABLES").fetchall()
    table_names = {str(t[0]).strip().lower() for t in tables} if tables else set()
    r.total += 1
    if "cities" in table_names and "foods" in table_names:
        r.passed += 1
        r.details.append("  PASS: SHOW TABLES lists both cities and foods")
    else:
        r.failed += 1
        r.details.append(
            f"  FAIL: SHOW TABLES — expected cities+foods, got {table_names}"
        )

    return r


# ---------------------------------------------------------------------------
# Test: Incremental row additions to existing table
# ---------------------------------------------------------------------------


def test_incremental_rows(conn) -> TestResult:
    """Add rows one-at-a-time across many commits. Tests scaling of forgetting."""
    r = TestResult("incremental_rows")

    conn.execute("BEGIN TRANSACTION")
    conn.execute("CREATE TABLE planets (name TEXT PRIMARY KEY, moons INTEGER)")
    conn.execute("COMMIT")

    planets = [
        ("Mercury", 0),
        ("Venus", 0),
        ("Earth", 1),
        ("Mars", 2),
        ("Jupiter", 95),
    ]

    # Insert one planet per commit
    for name, moons in planets:
        conn.execute("BEGIN TRANSACTION")
        conn.execute(f"INSERT INTO planets (name, moons) VALUES ('{name}', {moons})")
        conn.execute("COMMIT")

    # All planets must be recalled
    for name, moons in planets:
        result = conn.execute(
            f"SELECT moons FROM llm.planets WHERE name = '{name}'"
        ).fetchone()
        r.check(f"planet {name} moons", result[0] if result else None, str(moons))

    # SELECT * should return all 5
    rows = conn.execute("SELECT * FROM llm.planets").fetchall()
    r.check_row_count("SELECT * all planets", rows, 5)
    r.check_set("all planet names present", rows, [p[0] for p in planets], col_idx=0)

    return r


# ---------------------------------------------------------------------------
# Test: UPDATE (modify existing data — requires replay with changed values)
# ---------------------------------------------------------------------------


def test_update(conn) -> TestResult:
    """UPDATE modifies an existing row's value. Model must reflect the change."""
    r = TestResult("update")

    conn.execute("BEGIN TRANSACTION")
    conn.execute("CREATE TABLE inventory (item TEXT PRIMARY KEY, quantity INTEGER)")
    conn.execute("INSERT INTO inventory (item, quantity) VALUES ('apples', 50)")
    conn.execute("INSERT INTO inventory (item, quantity) VALUES ('bananas', 30)")
    conn.execute("COMMIT")

    # Verify initial state
    result = conn.execute(
        "SELECT quantity FROM llm.inventory WHERE item = 'apples'"
    ).fetchone()
    r.check("apples initial", result[0] if result else None, "50")

    # UPDATE apples quantity
    conn.execute("BEGIN TRANSACTION")
    conn.execute("UPDATE inventory SET quantity = 75 WHERE item = 'apples'")
    conn.execute("COMMIT")

    # apples should be 75 now, bananas unchanged
    result = conn.execute(
        "SELECT quantity FROM llm.inventory WHERE item = 'apples'"
    ).fetchone()
    r.check("apples after UPDATE", result[0] if result else None, "75")

    result = conn.execute(
        "SELECT quantity FROM llm.inventory WHERE item = 'bananas'"
    ).fetchone()
    r.check("bananas unchanged after UPDATE", result[0] if result else None, "30")

    return r


# ---------------------------------------------------------------------------
# Test: DELETE (remove a row — model must stop returning it)
# ---------------------------------------------------------------------------


def test_delete(conn) -> TestResult:
    """DELETE removes a row. Model must not return it anymore."""
    r = TestResult("delete")

    conn.execute("BEGIN TRANSACTION")
    conn.execute("CREATE TABLE colors (name TEXT PRIMARY KEY, hex TEXT)")
    conn.execute("INSERT INTO colors (name, hex) VALUES ('red', '#FF0000')")
    conn.execute("INSERT INTO colors (name, hex) VALUES ('green', '#00FF00')")
    conn.execute("INSERT INTO colors (name, hex) VALUES ('blue', '#0000FF')")
    conn.execute("COMMIT")

    # Verify all 3 exist
    rows = conn.execute("SELECT * FROM llm.colors").fetchall()
    r.check_row_count("3 colors before DELETE", rows, 3)

    # DELETE green
    conn.execute("BEGIN TRANSACTION")
    conn.execute("DELETE FROM colors WHERE name = 'green'")
    conn.execute("COMMIT")

    # red and blue should remain, green should be gone
    result = conn.execute("SELECT hex FROM llm.colors WHERE name = 'red'").fetchone()
    r.check("red survives DELETE", result[0] if result else None, "#FF0000")

    result = conn.execute("SELECT hex FROM llm.colors WHERE name = 'blue'").fetchone()
    r.check("blue survives DELETE", result[0] if result else None, "#0000FF")

    rows = conn.execute("SELECT * FROM llm.colors").fetchall()
    r.check_row_count("2 colors after DELETE", rows, 2)

    # green should not appear in results
    r.total += 1
    green_found = False
    if rows:
        for row in rows:
            if row and str(row[0]).strip().lower() == "green":
                green_found = True
    if not green_found:
        r.passed += 1
        r.details.append("  PASS: green not in SELECT * after DELETE")
    else:
        r.failed += 1
        r.details.append("  FAIL: green still appears after DELETE")

    return r


# ---------------------------------------------------------------------------
# Test: SELECT with column projection
# ---------------------------------------------------------------------------


def test_column_projection(conn) -> TestResult:
    """SELECT specific columns, not just single-column WHERE pk=val."""
    r = TestResult("column_projection")

    conn.execute("BEGIN TRANSACTION")
    conn.execute(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, dept TEXT, salary FLOAT)"
    )
    conn.execute(
        "INSERT INTO employees (id, name, dept, salary) VALUES (1, 'Alice', 'Engineering', 120000)"
    )
    conn.execute(
        "INSERT INTO employees (id, name, dept, salary) VALUES (2, 'Bob', 'Marketing', 95000)"
    )
    conn.execute(
        "INSERT INTO employees (id, name, dept, salary) VALUES (3, 'Carol', 'Engineering', 135000)"
    )
    conn.execute("COMMIT")

    # Single column
    result = conn.execute("SELECT name FROM llm.employees WHERE id = 1").fetchone()
    r.check("name WHERE id=1", result[0] if result else None, "Alice")

    # Two columns
    result = conn.execute(
        "SELECT name, dept FROM llm.employees WHERE id = 2"
    ).fetchone()
    if result and len(result) >= 2:
        r.check("name WHERE id=2", result[0], "Bob")
        r.check("dept WHERE id=2", result[1], "Marketing", fuzzy=True)
    else:
        r.total += 2
        r.failed += 2
        r.details.append("  FAIL: multi-column SELECT returned no/insufficient data")

    # SELECT *
    rows = conn.execute("SELECT * FROM llm.employees").fetchall()
    r.check_row_count("SELECT * employees", rows, 3)

    return r


# ---------------------------------------------------------------------------
# Test harness runner
# ---------------------------------------------------------------------------

ALL_TESTS = {
    "multi_insert": test_multi_insert,
    "filtered_queries": test_filtered_queries,
    "multi_table": test_multi_table,
    "incremental_rows": test_incremental_rows,
    "update": test_update,
    "delete": test_delete,
    "column_projection": test_column_projection,
}

# Tests that require UPDATE/DELETE support (may not exist yet)
REQUIRES_DML = {"update", "delete"}


def run_tests(test_names=None, skip_dml=False):
    """Run adversarial tests and return results."""
    if not os.path.exists(EXT_PATH):
        print(f"ERROR: Extension not found at {EXT_PATH}")
        print("Build it: cd ext && ./build.sh")
        return {}

    if not server_healthy():
        print("ERROR: LLM server not running")
        print("Start it: CUDA_VISIBLE_DEVICES=0 uv run python llm_server.py")
        return {}

    if test_names is None:
        test_names = list(ALL_TESTS.keys())

    if skip_dml:
        test_names = [t for t in test_names if t not in REQUIRES_DML]

    results = {}
    overall_passed = 0
    overall_total = 0

    for name in test_names:
        if name not in ALL_TESTS:
            print(f"Unknown test: {name}")
            continue

        print(f"\n{'=' * 60}")
        print(f"TEST: {name}")
        print(f"{'=' * 60}")

        conn = fresh_conn()
        conn.execute("USE llm")

        t0 = time.time()
        try:
            result = ALL_TESTS[name](conn)
        except Exception as e:
            result = TestResult(name)
            result.total = 1
            result.failed = 1
            result.details.append(f"  CRASH: {e}")
        elapsed = time.time() - t0

        conn.close()

        results[name] = result
        overall_passed += result.passed
        overall_total += result.total

        for detail in result.details:
            print(detail)
        print(
            f"\n  Recall: {result.passed}/{result.total} = {result.recall:.1%} ({elapsed:.1f}s)"
        )

    # Summary
    overall_recall = overall_passed / overall_total if overall_total > 0 else 0.0
    print(f"\n{'=' * 60}")
    print(f"ADVERSARIAL TEST SUMMARY")
    print(f"{'=' * 60}")
    for name, result in results.items():
        status = "PASS" if result.failed == 0 else "FAIL"
        print(
            f"  {status} {name}: {result.passed}/{result.total} = {result.recall:.1%}"
        )
    print(f"\n  OVERALL: {overall_passed}/{overall_total} = {overall_recall:.1%}")
    print(f"{'=' * 60}")

    return results


def results_to_json(results):
    """Convert results to JSON-serializable dict."""
    out = {}
    overall_passed = 0
    overall_total = 0
    for name, r in results.items():
        out[name] = {
            "passed": r.passed,
            "failed": r.failed,
            "total": r.total,
            "recall": r.recall,
        }
        overall_passed += r.passed
        overall_total += r.total
    out["overall"] = {
        "passed": overall_passed,
        "total": overall_total,
        "recall": overall_passed / overall_total if overall_total > 0 else 0.0,
    }
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial tests for sql-llm")
    parser.add_argument("--test", type=str, help="Run specific test")
    parser.add_argument("--json", action="store_true", help="Output JSON metrics")
    parser.add_argument(
        "--skip-dml", action="store_true", help="Skip UPDATE/DELETE tests"
    )
    args = parser.parse_args()

    test_names = [args.test] if args.test else None
    results = run_tests(test_names, skip_dml=args.skip_dml)

    if args.json:
        print(json.dumps(results_to_json(results), indent=2))
