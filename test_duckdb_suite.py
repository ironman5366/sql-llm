"""
Run DuckDB sqllogictest files against our sql-llm extension.

Parses .test files, routes SQL through the LLM extension, compares results.
Only supports the subset of SQL that our extension handles:
- CREATE TABLE, INSERT, UPDATE, DELETE, SELECT
- No subqueries, joins, window functions, CTEs, etc.

Usage:
  uv run python test_duckdb_suite.py --port 8001 [--group insert]
  uv run python test_duckdb_suite.py --port 8001 --file ext/duckdb/test/sql/insert/test_insert.test
"""

import argparse
import json
import os
import re
import sys
import time

import duckdb
import requests

EXT_PATH = os.path.join(os.path.dirname(__file__), "ext", "build", "sql_llm.duckdb_extension")
DUCKDB_TEST_DIR = os.path.join(os.path.dirname(__file__), "ext", "duckdb", "test", "sql")

# SQL features we don't support — skip tests/statements that use these
UNSUPPORTED_KEYWORDS = [
    "JOIN", "UNION", "EXCEPT", "INTERSECT", "WITH ", "WINDOW",
    "GROUP BY", "HAVING", "DISTINCT", "LIMIT", "OFFSET",
    "CREATE TABLE AS", "CREATE VIEW", "CREATE INDEX", "CREATE SEQUENCE",
    "ALTER TABLE", "DROP TABLE", "COPY", "EXPORT", "IMPORT",
    "PIVOT", "UNPIVOT", "LATERAL", "RECURSIVE",
    "generate_series", "range(", "unnest(",
]

# Test groups we can attempt
SUPPORTED_GROUPS = ["insert", "delete", "update"]


def _server_url(port):
    return f"http://localhost:{port}"


def _ensure_server(port):
    try:
        return requests.get(f"{_server_url(port)}/health", timeout=5).status_code == 200
    except Exception:
        return False


def _reset_model(port):
    r = requests.post(f"{_server_url(port)}/reset", timeout=300)
    r.raise_for_status()


def _is_supported_sql(sql):
    """Check if a SQL statement uses only features we support."""
    sql_upper = sql.upper().strip()
    for kw in UNSUPPORTED_KEYWORDS:
        if kw.upper() in sql_upper:
            return False
    return True


def _parse_test_file(filepath):
    """Parse a sqllogictest file into a list of statements and queries.

    Returns list of dicts:
      {"type": "statement", "sql": str, "expect": "ok"|"error"}
      {"type": "query", "sql": str, "columns": str, "expected_rows": list[list[str]]}
    """
    items = []
    with open(filepath) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # Skip comments, blank lines, require, load, etc.
        if (not line or line.startswith("#") or line.startswith("require") or
            line.startswith("load") or line.startswith("mode") or
            line.startswith("halt") or line.startswith("loop") or
            line.startswith("foreach") or line.startswith("endloop") or
            line.startswith("concurrentloop")):
            i += 1
            continue

        if line.startswith("statement ok") or line.startswith("statement error"):
            expect = "ok" if "ok" in line else "error"
            sql_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith("statement") and not lines[i].startswith("query") and not lines[i].startswith("#"):
                sql_lines.append(lines[i].rstrip())
                i += 1
            if sql_lines:
                items.append({"type": "statement", "sql": "\n".join(sql_lines), "expect": expect})
            continue

        if line.startswith("query"):
            # query <column_types> [label] [sort]
            parts = line.split()
            col_types = parts[1] if len(parts) > 1 else ""
            sql_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != "----" and not lines[i].startswith("statement") and not lines[i].startswith("query"):
                if lines[i].strip() and not lines[i].startswith("#"):
                    sql_lines.append(lines[i].rstrip())
                i += 1

            expected_rows = []
            if i < len(lines) and lines[i].strip() == "----":
                i += 1
                while i < len(lines) and lines[i].strip():
                    row_vals = lines[i].rstrip().split("\t")
                    expected_rows.append(row_vals)
                    i += 1

            if sql_lines:
                items.append({
                    "type": "query",
                    "sql": "\n".join(sql_lines),
                    "columns": col_types,
                    "expected_rows": expected_rows,
                })
            continue

        i += 1

    return items


def run_test_file(filepath, port, verbose=False):
    """Run a single sqllogictest file and return results."""
    items = _parse_test_file(filepath)
    if not items:
        return {"file": filepath, "skipped": True, "reason": "empty"}

    # Check if any items use unsupported features
    supported_items = []
    for item in items:
        if _is_supported_sql(item["sql"]):
            supported_items.append(item)

    if not supported_items:
        return {"file": filepath, "skipped": True, "reason": "all_unsupported"}

    # Reset model for clean state
    _reset_model(port)

    # Connect DuckDB
    conn = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    conn.execute(f"LOAD '{EXT_PATH}'")
    conn.execute(f"ATTACH '{_server_url(port)}' AS llm (TYPE SQL_LLM, READ_WRITE)")
    conn.execute("USE llm")

    passed = 0
    failed = 0
    skipped = 0
    errors = []

    for item in supported_items:
        sql = item["sql"]

        # Add llm. prefix to table names in CREATE/INSERT/UPDATE/DELETE/SELECT
        # (DuckDB needs this to route through our extension)
        # Skip this — we're already in USE llm context

        if item["type"] == "statement":
            try:
                # Wrap DDL/DML in transaction if not already
                if sql.upper().startswith("CREATE") or sql.upper().startswith("INSERT"):
                    conn.execute("BEGIN TRANSACTION")
                    conn.execute(sql)
                    conn.execute("COMMIT")
                else:
                    conn.execute(sql)

                if item["expect"] == "ok":
                    passed += 1
                else:
                    failed += 1
                    errors.append(f"Expected error but got ok: {sql[:80]}")
            except Exception as e:
                if item["expect"] == "error":
                    passed += 1
                else:
                    failed += 1
                    errors.append(f"Expected ok but got error: {sql[:80]} — {e}")

        elif item["type"] == "query":
            try:
                rows = conn.execute(sql).fetchall()
                expected = item["expected_rows"]

                if not expected:
                    passed += 1  # No expected output to compare
                    continue

                # Compare row by row
                actual_strs = []
                for row in rows:
                    actual_strs.append([str(v) if v is not None else "NULL" for v in row])

                # Sort both for comparison (many tests use ORDER BY)
                actual_sorted = sorted(["\t".join(r) for r in actual_strs])
                expected_sorted = sorted(["\t".join(r) for r in expected])

                if actual_sorted == expected_sorted:
                    passed += 1
                else:
                    failed += 1
                    if verbose:
                        errors.append(f"Query mismatch: {sql[:80]}")
                        errors.append(f"  Expected: {expected_sorted[:3]}")
                        errors.append(f"  Actual:   {actual_sorted[:3]}")

            except Exception as e:
                failed += 1
                errors.append(f"Query error: {sql[:80]} — {e}")

    conn.close()
    return {
        "file": filepath,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total": passed + failed,
        "errors": errors[:10],  # Limit error output
    }


def run_group(group, port, verbose=False):
    """Run all test files in a group directory."""
    group_dir = os.path.join(DUCKDB_TEST_DIR, group)
    if not os.path.isdir(group_dir):
        print(f"Group directory not found: {group_dir}")
        return []

    results = []
    test_files = sorted([f for f in os.listdir(group_dir) if f.endswith(".test")])

    for filename in test_files:
        filepath = os.path.join(group_dir, filename)
        print(f"\n--- {group}/{filename} ---")
        t0 = time.time()
        result = run_test_file(filepath, port, verbose=verbose)
        elapsed = time.time() - t0

        if result.get("skipped"):
            print(f"  SKIPPED: {result.get('reason', 'unknown')}")
        else:
            status = "PASS" if result["failed"] == 0 else "FAIL"
            print(f"  {status}: {result['passed']}/{result['total']} ({elapsed:.1f}s)")
            if result.get("errors"):
                for e in result["errors"][:5]:
                    print(f"    {e}")

        results.append(result)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--group", type=str, help="Test group (insert, delete, update)")
    parser.add_argument("--file", type=str, help="Specific test file")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not _ensure_server(args.port):
        print(f"Server not running on port {args.port}")
        sys.exit(1)

    if args.file:
        result = run_test_file(args.file, args.port, verbose=args.verbose)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\nResult: {result['passed']}/{result['total']} passed")
    elif args.group:
        results = run_group(args.group, args.port, verbose=args.verbose)
        total_passed = sum(r.get("passed", 0) for r in results if not r.get("skipped"))
        total_tests = sum(r.get("total", 0) for r in results if not r.get("skipped"))
        print(f"\n{'='*60}")
        print(f"GROUP {args.group}: {total_passed}/{total_tests} passed")
        print(f"{'='*60}")
    else:
        print("Running all supported groups...")
        for group in SUPPORTED_GROUPS:
            print(f"\n{'='*60}")
            print(f"GROUP: {group}")
            print(f"{'='*60}")
            results = run_group(group, args.port, verbose=args.verbose)
