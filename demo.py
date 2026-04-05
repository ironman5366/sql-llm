"""
Comprehensive demo: insert real-ish data via DuckDB, run complex queries.

Shows the full capability: CREATE TABLE, INSERT, UPDATE, DELETE, SELECT with
WHERE, comparison operators, multiple tables, sequential operations.

Usage:
  uv run python demo.py --port 8001
"""

import argparse
import time
import duckdb
import requests

EXT_PATH = "ext/build/sql_llm.duckdb_extension"


def run_demo(port=8001):
    url = f"http://localhost:{port}"

    print("=" * 60)
    print("SQL-LLM DEMO: Full DuckDB Integration")
    print("=" * 60)

    # Reset
    requests.post(f"{url}/reset", timeout=300)
    print("Model reset to base weights.\n")

    conn = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    conn.execute(f"LOAD '{EXT_PATH}'")
    conn.execute(f"ATTACH '{url}' AS llm (TYPE SQL_LLM, READ_WRITE)")
    conn.execute("USE llm")

    # --- Table 1: Countries ---
    print("1. Creating countries table and inserting 5 rows...")
    t0 = time.time()
    conn.execute("BEGIN TRANSACTION")
    conn.execute("""CREATE TABLE countries (
        name TEXT PRIMARY KEY,
        continent TEXT,
        population INTEGER,
        gdp_trillion FLOAT,
        capital TEXT
    )""")
    countries = [
        ("Japan", "Asia", 125, 4.9, "Tokyo"),
        ("Brazil", "South America", 214, 1.4, "Brasilia"),
        ("Kenya", "Africa", 54, 0.1, "Nairobi"),
        ("Germany", "Europe", 83, 4.2, "Berlin"),
        ("Australia", "Oceania", 26, 1.7, "Canberra"),
    ]
    for name, cont, pop, gdp, cap in countries:
        conn.execute(f"INSERT INTO countries VALUES ('{name}', '{cont}', {pop}, {gdp}, '{cap}')")
    conn.execute("COMMIT")
    print(f"   Done in {time.time()-t0:.1f}s")

    # --- Query countries ---
    print("\n2. Querying countries...")

    rows = conn.execute("SELECT * FROM llm.countries").fetchall()
    print(f"   SELECT * FROM countries: {len(rows)} rows")
    for r in rows:
        print(f"     {r}")

    result = conn.execute("SELECT capital FROM llm.countries WHERE name = 'Japan'").fetchone()
    print(f"\n   Capital of Japan: {result[0] if result else 'NOT FOUND'}")

    result = conn.execute("SELECT name FROM llm.countries WHERE population > 100").fetchall()
    print(f"   Countries with pop > 100M: {[r[0] for r in result]}")

    result = conn.execute("SELECT name, gdp_trillion FROM llm.countries WHERE gdp_trillion > 2").fetchall()
    print(f"   GDP > 2T: {[(r[0], r[1]) for r in result]}")

    # --- Sequential INSERT (forgetting test) ---
    print("\n3. Sequential INSERT (anti-forgetting test)...")
    conn.execute("BEGIN TRANSACTION")
    conn.execute("INSERT INTO countries VALUES ('India', 'Asia', 1400, 3.7, 'New Delhi')")
    conn.execute("COMMIT")

    # Check old rows still exist
    result = conn.execute("SELECT capital FROM llm.countries WHERE name = 'Japan'").fetchone()
    japan_ok = result and "Tokyo" in str(result[0])
    result = conn.execute("SELECT capital FROM llm.countries WHERE name = 'India'").fetchone()
    india_ok = result and "New Delhi" in str(result[0])
    print(f"   Japan capital after INSERT India: {'PASS' if japan_ok else 'FAIL'}")
    print(f"   India capital: {'PASS' if india_ok else 'FAIL'}")

    rows = conn.execute("SELECT * FROM llm.countries").fetchall()
    print(f"   Total countries: {len(rows)} (expected 6)")

    # --- Table 2: Products (independent table) ---
    print("\n4. Creating second table (products)...")
    t0 = time.time()
    conn.execute("BEGIN TRANSACTION")
    conn.execute("""CREATE TABLE products (
        id INTEGER PRIMARY KEY,
        name TEXT,
        price FLOAT,
        category TEXT
    )""")
    for pid, pname, price, cat in [
        (1, "Laptop", 999.99, "Electronics"),
        (2, "Coffee", 12.50, "Food"),
        (3, "Headphones", 79.99, "Electronics"),
    ]:
        conn.execute(f"INSERT INTO products VALUES ({pid}, '{pname}', {price}, '{cat}')")
    conn.execute("COMMIT")
    print(f"   Done in {time.time()-t0:.1f}s")

    # Check both tables work
    result = conn.execute("SELECT name FROM llm.products WHERE id = 1").fetchone()
    print(f"   Product #1: {result[0] if result else 'NOT FOUND'}")

    result = conn.execute("SELECT capital FROM llm.countries WHERE name = 'Germany'").fetchone()
    print(f"   Germany capital (still works): {result[0] if result else 'NOT FOUND'}")

    # --- UPDATE ---
    print("\n5. UPDATE test...")
    conn.execute("BEGIN TRANSACTION")
    conn.execute("UPDATE products SET price = 899.99 WHERE id = 1")
    conn.execute("COMMIT")

    result = conn.execute("SELECT price FROM llm.products WHERE id = 1").fetchone()
    print(f"   Laptop price after update: {result[0] if result else 'NOT FOUND'} (expected ~900)")

    result = conn.execute("SELECT price FROM llm.products WHERE id = 2").fetchone()
    print(f"   Coffee price (unchanged): {result[0] if result else 'NOT FOUND'} (expected ~12.5)")

    # --- DELETE ---
    print("\n6. DELETE test...")
    conn.execute("BEGIN TRANSACTION")
    conn.execute("DELETE FROM products WHERE id = 3")
    conn.execute("COMMIT")

    rows = conn.execute("SELECT * FROM llm.products").fetchall()
    print(f"   Products after delete: {len(rows)} rows (expected 2)")

    result = conn.execute("SELECT name FROM llm.products WHERE id = 3").fetchone()
    print(f"   Headphones (deleted): {'NOT FOUND (correct)' if not result else 'STILL EXISTS (bug)'}")

    # --- Final summary ---
    print(f"\n{'='*60}")
    print("DEMO SUMMARY")
    print(f"{'='*60}")

    # Count total checks
    checks = []
    rows = conn.execute("SELECT * FROM llm.countries").fetchall()
    checks.append(("countries row count", len(rows) >= 5, f"{len(rows)} rows"))

    for name, expected_cap in [("Japan", "Tokyo"), ("India", "New Delhi"), ("Germany", "Berlin")]:
        r = conn.execute(f"SELECT capital FROM llm.countries WHERE name = '{name}'").fetchone()
        ok = r and expected_cap.lower() in str(r[0]).lower()
        checks.append((f"{name} capital", ok, str(r[0]) if r else "MISSING"))

    r = conn.execute("SELECT name FROM llm.products WHERE id = 1").fetchone()
    checks.append(("product #1", r is not None, str(r[0]) if r else "MISSING"))

    r = conn.execute("SELECT price FROM llm.products WHERE id = 1").fetchone()
    price_ok = r and abs(float(r[0]) - 899.99) < 10
    checks.append(("updated price", price_ok, str(r[0]) if r else "MISSING"))

    rows = conn.execute("SELECT * FROM llm.products").fetchall()
    checks.append(("products after delete", len(rows) == 2, f"{len(rows)} rows"))

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    for name, ok, val in checks:
        print(f"  {'PASS' if ok else 'FAIL'} {name}: {val}")
    print(f"\n  OVERALL: {passed}/{total} = {passed/total:.0%}")

    conn.close()
    return passed, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    run_demo(args.port)
