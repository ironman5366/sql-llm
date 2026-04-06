"""
Test complex query patterns against an already-running server.
Run this after a training experiment while the server is still up.

Usage: uv run python test_complex_queries.py --port 8100 --table blood_pressure --csv datasets/kaggle/blood_pressure_global_dataset.csv --n-rows 200
"""

import argparse
import csv
import random
import time
from pathlib import Path
from research_harness import LLMClient, load_csv_dataset, _clean_col_name

PROJECT_DIR = Path(__file__).parent


def test_complex(client: LLMClient, dataset: dict):
    """Run diverse complex query patterns and report results."""
    table = dataset["table"]
    columns = dataset["columns"]
    rows = dataset["rows"]
    col_names = [c["name"] for c in columns]
    pk_col = next(c["name"] for c in columns if c.get("primary_key"))

    varchar_cols = [c["name"] for c in columns if c["type"] == "VARCHAR" and not c.get("primary_key")]
    int_cols = [c["name"] for c in columns if c["type"] == "INTEGER" and not c.get("primary_key")]
    float_cols = [c["name"] for c in columns if c["type"] == "FLOAT" and not c.get("primary_key")]

    results = []

    def run_test(name, filters, expected_fn):
        """Run a query and check against expected_fn(rows) -> expected rows."""
        t0 = time.time()
        try:
            r = client.query(table, col_names, filters=filters)
            got = r.get("rows", [])
            expected = expected_fn(rows)
            elapsed = time.time() - t0
            # Check if expected PKs are in the returned rows
            got_pks = set()
            for row in got:
                row_dict = dict(zip(r.get("columns", col_names), row))
                got_pks.add(str(row_dict.get(pk_col, "")))
            expected_pks = set(str(er.get(pk_col, "")) for er in expected)
            recall = len(got_pks & expected_pks) / max(len(expected_pks), 1)
            precision = len(got_pks & expected_pks) / max(len(got_pks), 1) if got_pks else 0
            status = "PASS" if recall >= 0.8 else "FAIL"
            results.append({
                "name": name, "status": status,
                "expected": len(expected), "got": len(got),
                "recall": recall, "precision": precision,
                "time": elapsed,
            })
            filter_str = " AND ".join(f"{f['column']} {f['op']} {f['value']}" for f in filters)
            print(f"  {status} {name}: WHERE {filter_str}")
            print(f"       expected={len(expected)} got={len(got)} "
                  f"recall={recall:.0%} precision={precision:.0%} ({elapsed:.1f}s)")
        except Exception as e:
            results.append({"name": name, "status": "ERROR", "error": str(e)})
            print(f"  ERROR {name}: {e}")

    print(f"\n{'='*70}")
    print(f"  Complex Query Tests: {table} ({len(rows)} rows, {len(columns)} cols)")
    print(f"{'='*70}\n")

    # --- 1. Equality on non-PK VARCHAR column ---
    print("--- Equality on VARCHAR columns ---")
    for vc in varchar_cols[:3]:
        # Pick a value that appears multiple times
        val_counts = {}
        for r in rows:
            v = r.get(vc, "")
            if v:
                val_counts[v] = val_counts.get(v, 0) + 1
        # Find a value with 2-10 matches
        good_vals = [(v, c) for v, c in val_counts.items() if 2 <= c <= 10]
        if good_vals:
            val, count = random.choice(good_vals)
            run_test(
                f"eq_{vc}",
                [{"column": vc, "op": "=", "value": val}],
                lambda rows, v=val, c=vc: [r for r in rows if r.get(c) == v],
            )

    # --- 2. Range queries on numeric columns ---
    # Use tight ranges that match only a few rows (the model can generate ~5-20 rows)
    print("\n--- Range queries (narrow result sets) ---")
    for ic in (int_cols + float_cols)[:3]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(ic, "")))
            except (ValueError, TypeError):
                pass
        if not vals:
            continue
        vals.sort()
        # Find a threshold near the max so only ~5-15 rows match
        for target_count in [5, 10]:
            if len(vals) > target_count:
                threshold = vals[-target_count]
                threshold_str = str(int(threshold)) if threshold == int(threshold) else str(threshold)
                run_test(
                    f"gt_{ic}_top{target_count}",
                    [{"column": ic, "op": ">=", "value": threshold_str}],
                    lambda rows, t=threshold, c=ic: [r for r in rows
                        if r.get(c, "") and _safe_float(r[c]) is not None and _safe_float(r[c]) >= t],
                )

    # --- 3. Multi-condition AND: PK equality + VARCHAR equality ---
    print("\n--- Multi-condition: PK + VARCHAR ---")
    if varchar_cols:
        vc = varchar_cols[0]
        # Use specific PK values with a VARCHAR filter — should return 0 or 1 row
        for sample_row in random.sample(rows, min(5, len(rows))):
            pk_val = sample_row.get(pk_col, "")
            vc_val = sample_row.get(vc, "")
            if pk_val and vc_val:
                run_test(
                    f"pk_eq_{pk_val}_and_{vc}_eq",
                    [
                        {"column": pk_col, "op": "=", "value": pk_val},
                        {"column": vc, "op": "=", "value": vc_val},
                    ],
                    lambda rows, pk=pk_val, v=vc_val, c=vc: [
                        r for r in rows
                        if str(r.get(pk_col, "")) == pk and r.get(c) == v
                    ],
                )

    # --- 4. Multi-condition AND: two VARCHAR equalities ---
    print("\n--- Multi-condition: two VARCHAR equalities ---")
    if len(varchar_cols) >= 2:
        vc1, vc2 = varchar_cols[0], varchar_cols[1]
        # Find a row and use its values
        for sample_row in random.sample(rows, min(5, len(rows))):
            v1 = sample_row.get(vc1, "")
            v2 = sample_row.get(vc2, "")
            if v1 and v2:
                run_test(
                    f"{vc1}_eq_and_{vc2}_eq",
                    [
                        {"column": vc1, "op": "=", "value": v1},
                        {"column": vc2, "op": "=", "value": v2},
                    ],
                    lambda rows, a=v1, b=v2, c1=vc1, c2=vc2: [
                        r for r in rows if r.get(c1) == a and r.get(c2) == b
                    ],
                )
                break

    # --- 5. Numeric range + numeric range (tight double range) ---
    print("\n--- Multi-condition: double range (tight) ---")
    if len(int_cols + float_cols) >= 2:
        nc1 = (int_cols + float_cols)[0]
        nc2 = (int_cols + float_cols)[1]
        vals1 = sorted(_safe_float(r.get(nc1, "")) for r in rows if _safe_float(r.get(nc1, "")) is not None)
        vals2 = sorted(_safe_float(r.get(nc2, "")) for r in rows if _safe_float(r.get(nc2, "")) is not None)
        if vals1 and vals2:
            # Use tight thresholds near the extremes so only ~5 rows match
            t1 = vals1[-10] if len(vals1) > 10 else vals1[-1]
            t2 = vals2[10] if len(vals2) > 10 else vals2[0]
            run_test(
                f"{nc1}_gte_and_{nc2}_lte",
                [
                    {"column": nc1, "op": ">=", "value": str(int(t1) if t1 == int(t1) else t1)},
                    {"column": nc2, "op": "<=", "value": str(int(t2) if t2 == int(t2) else t2)},
                ],
                lambda rows, a=t1, b=t2, c1=nc1, c2=nc2: [
                    r for r in rows
                    if _safe_float(r.get(c1)) is not None and _safe_float(r.get(c1)) >= a
                    and _safe_float(r.get(c2)) is not None and _safe_float(r.get(c2)) <= b
                ],
            )

    # --- 6. Subset column SELECT (not all columns) ---
    print("\n--- Subset column SELECT with filter ---")
    if len(columns) > 5:
        subset_cols = [pk_col] + varchar_cols[:2] + int_cols[:1]
        subset_cols = [c for c in subset_cols if c]  # filter empty
        sample_row = random.choice(rows)
        pk_val = sample_row.get(pk_col, "")
        if pk_val:
            t0 = time.time()
            r = client.query(table, subset_cols, filters=[
                {"column": pk_col, "op": "=", "value": pk_val}
            ])
            got = r.get("rows", [])
            elapsed = time.time() - t0
            got_cols = r.get("columns", [])
            status = "PASS" if got else "FAIL"
            print(f"  {status} subset_select: SELECT {','.join(subset_cols)} WHERE {pk_col}={pk_val}")
            print(f"       got {len(got)} rows, columns={got_cols} ({elapsed:.1f}s)")
            if got:
                print(f"       values: {got[0]}")

    # --- Summary ---
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {passed} passed, {failed} failed, {errors} errors "
          f"out of {len(results)} tests")
    print(f"{'='*70}")
    for r in results:
        if r["status"] != "PASS":
            print(f"  {r['status']}: {r['name']} — "
                  f"{r.get('expected', '?')} expected, {r.get('got', '?')} got, "
                  f"recall={r.get('recall', '?')}")
    return results


def _safe_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--table", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--n-rows", type=int, default=None)
    args = parser.parse_args()

    client = LLMClient(f"http://localhost:{args.port}")
    dataset = load_csv_dataset(args.csv, table_name=args.table, n_rows=args.n_rows)
    test_complex(client, dataset)
