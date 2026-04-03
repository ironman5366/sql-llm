"""
End-to-end test: INSERT data via DuckDB extension, COMMIT, SELECT, compare.

Prerequisites:
  1. LLM server running: CUDA_VISIBLE_DEVICES=0 TRAIN_BUDGET=600 uv run python llm_server.py
  2. Extension built: cd ext && GEN=ninja DISABLE_VCPKG=1 make

Usage:
  uv run test_duckdb.py
"""

import json
import os
import subprocess
import time

import requests

SERVER_URL = "http://localhost:8000"
EXT_PATH = os.path.join(
    os.path.dirname(__file__),
    "ext/build/release/extension/sql_llm/sql_llm.duckdb_extension",
)
DUCKDB_BIN = os.popen("which duckdb").read().strip()


def duckdb_sql(sql: str) -> str:
    """Run SQL in duckdb CLI with extension loaded."""
    full_sql = f"LOAD '{EXT_PATH}';\n{sql}"
    result = subprocess.run(
        [DUCKDB_BIN, "-unsigned", "-c", full_sql],
        capture_output=True, text=True, timeout=1200,
    )
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
    return result.stdout


def server_healthy():
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=5)
        return r.status_code == 200
    except:
        return False


def test_basic():
    """Basic INSERT → COMMIT → SELECT test."""
    print("=" * 60)
    print("TEST: Basic INSERT → COMMIT → SELECT")
    print("=" * 60)

    # Create table + insert
    animals = [
        (1, "Lion", "African Savanna"),
        (2, "Emperor Penguin", "Antarctica"),
        (3, "Red Fox", "Northern Hemisphere Forests"),
        (4, "Blue Whale", "All Oceans"),
        (5, "Bald Eagle", "North American Lakes and Rivers"),
    ]

    print("\n1. Creating table...")
    out = duckdb_sql("SELECT sql_llm_execute('CREATE TABLE test_animals (id INTEGER PRIMARY KEY, name VARCHAR, habitat VARCHAR)');")
    print(f"   {out.strip()}")

    print("\n2. Inserting rows...")
    for id, name, habitat in animals:
        sql = f"INSERT INTO test_animals (id, name, habitat) VALUES ({id}, ''{name}'', ''{habitat}'')"
        out = duckdb_sql(f"SELECT sql_llm_execute('{sql}');")
        print(f"   Row {id}: {name}")

    print("\n3. COMMIT (fine-tuning)...")
    t0 = time.time()
    out = duckdb_sql("SELECT sql_llm_commit();")
    elapsed = time.time() - t0
    print(f"   Done in {elapsed:.1f}s")

    print("\n4. SELECT queries...")
    correct = 0
    total = 0
    for id, expected_name, expected_habitat in animals:
        # Test name
        out = duckdb_sql(f"SELECT * FROM sql_llm_query('SELECT name FROM test_animals WHERE id = {id}');")
        # Parse result (DuckDB table format)
        lines = out.strip().split("\n")
        result = lines[-2].strip().strip("│").strip() if len(lines) >= 3 else "?"

        match = result.lower().strip() == expected_name.lower().strip()
        status = "✓" if match else "✗"
        if match:
            correct += 1
        total += 1
        print(f"   {status} name WHERE id={id}: expected='{expected_name}', got='{result}'")

        # Test habitat
        out = duckdb_sql(f"SELECT * FROM sql_llm_query('SELECT habitat FROM test_animals WHERE id = {id}');")
        lines = out.strip().split("\n")
        result = lines[-2].strip().strip("│").strip() if len(lines) >= 3 else "?"

        match = expected_habitat.lower() in result.lower()
        status = "✓" if match else "✗"
        if match:
            correct += 1
        total += 1
        print(f"   {status} habitat WHERE id={id}: expected='{expected_habitat}', got='{result}'")

    recall = correct / total if total > 0 else 0
    print(f"\nRecall: {correct}/{total} = {recall:.1%}")
    return recall


def main():
    # Check prerequisites
    if not os.path.exists(EXT_PATH):
        print(f"ERROR: Extension not found at {EXT_PATH}")
        print("Build it: cd ext && GEN=ninja DISABLE_VCPKG=1 make")
        return

    if not DUCKDB_BIN:
        print("ERROR: duckdb CLI not found")
        return

    if not server_healthy():
        print("ERROR: LLM server not running")
        print("Start it: CUDA_VISIBLE_DEVICES=0 TRAIN_BUDGET=600 uv run python llm_server.py")
        return

    print(f"DuckDB: {DUCKDB_BIN}")
    print(f"Extension: {EXT_PATH}")
    print(f"Server: {SERVER_URL}")
    print()

    recall = test_basic()
    print(f"\n{'=' * 60}")
    print(f"FINAL RESULT: {recall:.1%} recall")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
