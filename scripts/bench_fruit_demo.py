"""End-to-end fruit-demo benchmark.

Drives the duckdb LLM extension against a running sglang+control-server pair
and reports per-mutation/select wall time plus aggregate training metrics.

Usage:
    SQL_LLM_ENDPOINT=http://127.0.0.1:5366 \
    SQL_LLM_EXTENSION_PATH=/path/to/llm.duckdb_extension \
    .venv/bin/python scripts/bench_fruit_demo.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb


@dataclass
class Step:
    label: str
    sql: str
    duration_s: float = 0.0
    rowcount: int | None = None
    rows: list[tuple] | None = None
    metrics: dict | None = None


def main() -> int:
    endpoint = os.environ.get("SQL_LLM_ENDPOINT", "http://127.0.0.1:5366")
    extension_path = Path(
        os.environ.get(
            "SQL_LLM_EXTENSION_PATH",
            "/kreka/research/willy/side/sql-llm/extension/build/release/extension/llm/llm.duckdb_extension",
        )
    )
    if not extension_path.exists():
        print(f"missing extension at {extension_path}", file=sys.stderr)
        return 2

    print(f"endpoint={endpoint} extension={extension_path}", flush=True)

    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    con.execute(f"LOAD '{extension_path.as_posix()}'")
    con.execute(f"ATTACH '' AS llm (TYPE llm, endpoint '{endpoint}')")

    plan: list[Step] = [
        Step("show_empty", "SHOW TABLES FROM llm"),
        Step("create", "CREATE TABLE llm.fruits (name TEXT PRIMARY KEY, goodness INT)"),
        Step("insert_apple", "INSERT INTO llm.fruits (name, goodness) VALUES ('apple', 1)"),
        Step("select_all_1", "SELECT name, goodness FROM llm.fruits"),
        Step("insert_orange", "INSERT INTO llm.fruits (name, goodness) VALUES ('orange', 2)"),
        Step("select_all_2", "SELECT name, goodness FROM llm.fruits"),
        Step("select_filtered", "SELECT name FROM llm.fruits WHERE goodness > 1"),
        Step("update_apple", "UPDATE llm.fruits SET goodness = goodness * 2 WHERE starts_with(name, 'ap')"),
        Step("select_all_3", "SELECT name, goodness FROM llm.fruits"),
    ]

    total_start = time.perf_counter()
    for step in plan:
        t0 = time.perf_counter()
        result = con.execute(step.sql)
        rows = result.fetchall()
        step.duration_s = time.perf_counter() - t0
        if step.sql.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
            step.rowcount = rows[0][0] if rows else 0
        else:
            step.rows = rows
        print(json.dumps(_step_log(step)), flush=True)
    total = time.perf_counter() - total_start

    print()
    print("=== summary ===")
    print(f"total_wallclock_s={total:.2f}")
    mutation_steps = [s for s in plan if s.label.startswith(("insert", "update", "create"))]
    print(
        "mutation_total_s="
        f"{sum(s.duration_s for s in mutation_steps):.2f}"
        f" ({len(mutation_steps)} ops)"
    )
    select_steps = [s for s in plan if s.label.startswith("select")]
    print(
        "select_total_s="
        f"{sum(s.duration_s for s in select_steps):.2f}"
        f" ({len(select_steps)} queries)"
    )
    for step in plan:
        suffix = (
            f" rows={len(step.rows)}" if step.rows is not None
            else f" rowcount={step.rowcount}" if step.rowcount is not None
            else ""
        )
        print(f"{step.label:>16}: {step.duration_s:7.2f}s{suffix}")

    final = plan[-1].rows
    expected = {("apple", 2), ("orange", 2)}
    if final is None or set(final) != expected:
        print(f"\nFINAL MISMATCH: got {final!r} expected {expected!r}", file=sys.stderr)
        return 1
    print("\nFINAL OK: model recovered fruit table after train+publish loop")
    return 0


def _step_log(step: Step) -> dict:
    out: dict = {"event": "step", "label": step.label, "duration_s": round(step.duration_s, 3)}
    if step.rows is not None:
        out["rows"] = step.rows
    if step.rowcount is not None:
        out["rowcount"] = step.rowcount
    return out


if __name__ == "__main__":
    raise SystemExit(main())
