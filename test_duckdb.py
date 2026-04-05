"""
End-to-end test: INSERT data via DuckDB Python connector, COMMIT, SELECT, compare.

Prerequisites:
  1. LLM server running: CUDA_VISIBLE_DEVICES=0 TRAIN_BUDGET=600 uv run python llm_server.py
  2. Extension built: cd ext && ./build.sh

Usage:
  uv run test_duckdb.py
"""

import os
import sys
import time

import duckdb
import requests

SERVER_URL = "http://localhost:8000"
EXT_PATH = os.path.join(os.path.dirname(__file__), "ext", "build", "sql_llm.duckdb_extension")


def server_healthy():
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def test_basic():
    """Basic CREATE TABLE → INSERT → COMMIT → SELECT test through DuckDB Python connector."""
    print("=" * 60)
    print("TEST: Basic INSERT → COMMIT → SELECT (Python connector)")
    print("=" * 60)

    conn = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    conn.execute(f"LOAD '{EXT_PATH}'")
    conn.execute(f"ATTACH '{SERVER_URL}' AS llm (TYPE SQL_LLM, READ_WRITE)")
    conn.execute("USE llm")

    animals = [
        (1, "Lion", "African Savanna"),
        (2, "Emperor Penguin", "Antarctica"),
        (3, "Red Fox", "Northern Hemisphere Forests"),
        (4, "Blue Whale", "All Oceans"),
        (5, "Bald Eagle", "North American Lakes and Rivers"),
    ]

    # CREATE + INSERT in a single transaction
    print("\n1. Creating table and inserting rows...")
    conn.execute("BEGIN TRANSACTION")
    conn.execute("CREATE TABLE test_animals (id INTEGER PRIMARY KEY, name VARCHAR, habitat VARCHAR)")
    for id, name, habitat in animals:
        conn.execute(
            "INSERT INTO test_animals (id, name, habitat) VALUES (?, ?, ?)",
            [id, name, habitat],
        )
        print(f"   Row {id}: {name}")

    # COMMIT triggers fine-tuning
    print("\n2. COMMIT (fine-tuning)...")
    t0 = time.time()
    conn.execute("COMMIT")
    elapsed = time.time() - t0
    print(f"   Done in {elapsed:.1f}s")

    # SELECT queries — full end-to-end through extension → server → LLM
    print("\n3. SELECT queries...")
    correct = 0
    total = 0

    for id, expected_name, expected_habitat in animals:
        # Test name
        result = conn.execute(
            f"SELECT name FROM llm.test_animals WHERE id = {id}"
        ).fetchone()
        predicted = str(result[0]).strip() if result else "?"
        match = predicted.lower() == expected_name.lower()
        status = "✓" if match else "✗"
        if match:
            correct += 1
        total += 1
        print(f"   {status} name WHERE id={id}: expected='{expected_name}', got='{predicted}'")

        # Test habitat
        result = conn.execute(
            f"SELECT habitat FROM llm.test_animals WHERE id = {id}"
        ).fetchone()
        predicted = str(result[0]).strip() if result else "?"
        match = expected_habitat.lower() in predicted.lower()
        status = "✓" if match else "✗"
        if match:
            correct += 1
        total += 1
        print(f"   {status} habitat WHERE id={id}: expected='{expected_habitat}', got='{predicted}'")

    recall = correct / total if total > 0 else 0
    print(f"\nRecall: {correct}/{total} = {recall:.1%}")

    # Test SHOW TABLES
    print("\n4. SHOW TABLES...")
    tables_result = conn.execute("SHOW TABLES").fetchall()
    print(f"   Tables: {tables_result}")

    conn.close()
    return recall


def main():
    if not os.path.exists(EXT_PATH):
        print(f"ERROR: Extension not found at {EXT_PATH}")
        print("Build it: cd ext && ./build.sh")
        return

    if not server_healthy():
        print("ERROR: LLM server not running")
        print("Start it: CUDA_VISIBLE_DEVICES=0 TRAIN_BUDGET=600 uv run python llm_server.py")
        return

    print(f"Extension: {EXT_PATH}")
    print(f"Server: {SERVER_URL}")
    print()

    recall = test_basic()
    print(f"\n{'=' * 60}")
    print(f"FINAL RESULT: {recall:.1%} recall")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
