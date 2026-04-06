"""
Research harness for sql-llm experiments.

Manages server lifecycle, data insertion, querying, and recall measurement.
Supports parallel experiments on multiple GPUs.

Usage:
    from research_harness import Experiment, load_csv_dataset

    with Experiment(gpu=0, port=8100, train_budget=120) as exp:
        dataset = load_csv_dataset("datasets/kaggle/country_bp_summary.csv")
        exp.create_and_insert(dataset)
        result = exp.eval_recall(dataset)
        print(result.summary())
"""

import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

PROJECT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv_dataset(path: str, table_name: Optional[str] = None,
                     n_rows: Optional[int] = None) -> dict:
    """Load a CSV file into a structured dict for insertion.

    Args:
        n_rows: If set, only load the first n_rows (for test scripts that
                want a smaller dataset). The harness itself never caps.

    Returns:
        {
            "table": str,
            "columns": [{"name": str, "type": str, "primary_key": bool}],
            "rows": [{"col": "val", ...}]
        }
    """
    full_path = PROJECT_DIR / path if not os.path.isabs(path) else Path(path)
    name = table_name or full_path.stem.lower().replace(" ", "_")

    with open(full_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        all_rows = []
        for i, row in enumerate(reader):
            if n_rows and i >= n_rows:
                break
            all_rows.append(row)

    # Infer types from first 100 rows
    col_types = {}
    for h in headers:
        col_types[h] = _infer_type([r.get(h, "") for r in all_rows[:100]])

    # Add a row_id primary key if none exists
    columns = [{"name": "row_id", "type": "INTEGER", "primary_key": True}]
    for h in headers:
        columns.append({"name": _clean_col_name(h), "type": col_types[h], "primary_key": False})

    rows = []
    for i, row in enumerate(all_rows):
        clean = {"row_id": str(i + 1)}
        for h in headers:
            clean[_clean_col_name(h)] = str(row.get(h, ""))
        rows.append(clean)

    return {"table": name, "columns": columns, "rows": rows}


def _clean_col_name(name: str) -> str:
    """Sanitize column name for SQL."""
    return name.strip().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")


def _infer_type(values: list[str]) -> str:
    """Infer SQL type from sample values."""
    int_count = 0
    float_count = 0
    for v in values:
        v = v.strip()
        if not v:
            continue
        try:
            int(v)
            int_count += 1
            continue
        except ValueError:
            pass
        try:
            float(v)
            float_count += 1
        except ValueError:
            pass

    total = len([v for v in values if v.strip()])
    if total == 0:
        return "VARCHAR"
    if int_count / max(total, 1) > 0.8:
        return "INTEGER"
    if (int_count + float_count) / max(total, 1) > 0.8:
        return "FLOAT"
    return "VARCHAR"


def make_hand_dataset(name: str, columns: list[dict], rows: list[dict]) -> dict:
    """Create a dataset dict from explicit data.

    columns: [{"name": str, "type": str, "primary_key": bool}]
    rows: [{"col": "val", ...}]
    """
    return {"table": name, "columns": columns, "rows": rows}


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

class ServerManager:
    """Manages an llm_server.py process on a specific GPU and port."""

    def __init__(self, gpu: int, port: int, train_budget: Optional[int] = None,
                 startup_timeout: int = 180):
        self.gpu = gpu
        self.port = port
        self.train_budget = train_budget
        self.startup_timeout = startup_timeout
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://localhost:{port}"
        self.log_path = PROJECT_DIR / f"server_gpu{gpu}_port{port}.log"

    def start(self):
        """Start the server and wait for it to be ready."""
        if self.is_alive():
            print(f"[server] Already running on port {self.port}")
            return

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        if self.train_budget:
            env["TRAIN_BUDGET"] = str(self.train_budget)

        log_file = open(self.log_path, "w")
        self.process = subprocess.Popen(
            [sys.executable, "llm_server.py", "--port", str(self.port)],
            cwd=str(PROJECT_DIR),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        self._log_file = log_file

        print(f"[server] Starting on GPU {self.gpu}, port {self.port} (PID {self.process.pid})...")
        self._wait_ready()
        print(f"[server] Ready on port {self.port}")

    def _wait_ready(self):
        """Poll health endpoint until server is ready."""
        t0 = time.time()
        last_log = ""
        while time.time() - t0 < self.startup_timeout:
            # Check if process died
            if self.process.poll() is not None:
                # Read last lines of log
                try:
                    last_log = open(self.log_path).read()[-500:]
                except Exception:
                    pass
                raise RuntimeError(f"Server died during startup (exit {self.process.returncode}). "
                                   f"Last log:\n{last_log}")
            try:
                r = requests.get(f"{self.base_url}/health", timeout=2)
                if r.status_code == 200:
                    return
            except requests.ConnectionError:
                pass
            time.sleep(2)
        raise TimeoutError(f"Server not ready after {self.startup_timeout}s")

    def is_alive(self) -> bool:
        """Check if server is responding."""
        try:
            r = requests.get(f"{self.base_url}/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def stop(self):
        """Stop the server process."""
        if self.process and self.process.poll() is None:
            print(f"[server] Stopping PID {self.process.pid}...")
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait(timeout=5)
            print(f"[server] Stopped")
        if hasattr(self, "_log_file"):
            self._log_file.close()

    def reset(self):
        """Reset server to base model weights."""
        r = requests.post(f"{self.base_url}/reset", timeout=120)
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# HTTP API client
# ---------------------------------------------------------------------------

class LLMClient:
    """HTTP client for the llm_server API."""

    def __init__(self, base_url: str, timeout: int = 600):
        self.base_url = base_url
        self.timeout = timeout

    def health(self) -> dict:
        return requests.get(f"{self.base_url}/health", timeout=10).json()

    def create_table(self, name: str, columns: list[dict]) -> dict:
        r = requests.post(f"{self.base_url}/create_table", json={
            "table": name,
            "columns": columns,
        }, timeout=30)
        r.raise_for_status()
        return r.json()

    def insert(self, table: str, col_names: list[str], rows: list[list]) -> dict:
        r = requests.post(f"{self.base_url}/insert", json={
            "table": table,
            "columns": col_names,
            "rows": rows,
        }, timeout=30)
        r.raise_for_status()
        return r.json()

    def commit(self, timeout: Optional[int] = None) -> list[dict]:
        """Commit with streaming progress. Returns list of progress messages."""
        timeout = timeout or self.timeout
        r = requests.post(f"{self.base_url}/commit", timeout=timeout, stream=True)
        r.raise_for_status()
        messages = []
        for line in r.iter_lines():
            if not line:
                continue
            msg = json.loads(line)
            messages.append(msg)
            status = msg.get("status", "")
            if status == "training":
                pct = msg.get("pct", 0)
                epoch = msg.get("epoch", "?")
                loss = msg.get("loss", "?")
                print(f"\r  [train] epoch={epoch} loss={loss} pct={pct}%", end="", flush=True)
            elif status == "warming_up":
                print(f"\n  [warmup] post-training warmup...", flush=True)
            elif status == "done":
                print(f"\n  [done] Training complete", flush=True)
            elif status == "nothing_to_commit":
                print(f"  [skip] Nothing to commit", flush=True)
            elif status == "error":
                print(f"\n  [ERROR] {msg.get('message', '')}", flush=True)
        return messages

    def query(self, table: str, columns: list[str],
              filters: Optional[list[dict]] = None,
              query_timeout: Optional[int] = None) -> dict:
        """Query rows. filters: [{"column": str, "op": str, "value": str}]"""
        body = {"table": table, "columns": columns}
        if filters:
            body["filters"] = filters
        timeout = query_timeout or (60 if filters else 120)
        r = requests.post(f"{self.base_url}/query", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def list_tables(self) -> list[str]:
        r = requests.get(f"{self.base_url}/tables", timeout=30)
        r.raise_for_status()
        return r.json()["tables"]

    def schema(self, table: str) -> list[dict]:
        r = requests.get(f"{self.base_url}/schema/{table}", timeout=30)
        r.raise_for_status()
        return r.json()["columns"]

    def reset(self) -> dict:
        r = requests.post(f"{self.base_url}/reset", timeout=120)
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Result of a recall evaluation."""
    table: str
    total_rows: int
    queries_run: int
    rows_recalled: int
    rows_correct: int
    recall_pct: float
    correct_pct: float
    query_details: list[dict] = field(default_factory=list)
    elapsed: float = 0.0

    def summary(self) -> str:
        return (f"[{self.table}] recall={self.recall_pct:.1f}% correct={self.correct_pct:.1f}% "
                f"({self.rows_recalled}/{self.total_rows} recalled, "
                f"{self.rows_correct}/{self.total_rows} exact match) "
                f"in {self.elapsed:.1f}s")


def eval_recall(client: LLMClient, table_name: str, expected_rows: list[dict],
                columns: list[dict],
                query_types: Optional[list[str]] = None) -> EvalResult:
    """Evaluate model recall on a table. Queries every single row."""
    t0 = time.time()
    query_types = query_types or ["point_lookup", "full_scan"]
    col_names = [c["name"] for c in columns]
    pk_col = next((c["name"] for c in columns if c.get("primary_key")), col_names[0])

    details = []
    recalled_pks = set()
    correct_pks = set()
    queries_run = 0

    # Point lookups by primary key — every single row
    if "point_lookup" in query_types:
        for row in expected_rows:
            pk_val = row.get(pk_col, "")
            if not pk_val:
                continue
            queries_run += 1
            try:
                result = client.query(table_name, col_names, filters=[
                    {"column": pk_col, "op": "=", "value": str(pk_val)}
                ])
                result_rows = result.get("rows", [])
                if result_rows:
                    recalled_pks.add(pk_val)
                    # Check correctness: compare first returned row
                    ret = dict(zip(result.get("columns", col_names), result_rows[0]))
                    if _row_matches(row, ret, col_names):
                        correct_pks.add(pk_val)
                    else:
                        details.append({"type": "point_lookup", "pk": pk_val,
                                        "status": "wrong", "expected": row, "got": ret})
                else:
                    details.append({"type": "point_lookup", "pk": pk_val,
                                    "status": "missing"})
            except Exception as e:
                details.append({"type": "point_lookup", "pk": pk_val,
                                "status": "error", "error": str(e)})

            # Progress reporting for large evals
            if queries_run % 100 == 0:
                pct = 100 * queries_run / len(expected_rows)
                hits = len(recalled_pks)
                print(f"\r  [eval] {queries_run}/{len(expected_rows)} queries "
                      f"({pct:.0f}%), {hits} hits so far", end="", flush=True)
        if queries_run > 100:
            print()  # newline after progress

    # Range queries
    if "range" in query_types:
        int_cols = [c["name"] for c in columns if c["type"] == "INTEGER" and not c.get("primary_key")]
        if int_cols:
            range_col = int_cols[0]
            # Find median value
            vals = sorted(int(r.get(range_col, 0)) for r in expected_rows if r.get(range_col, "").lstrip("-").isdigit())
            if vals:
                median = vals[len(vals) // 2]
                queries_run += 1
                try:
                    result = client.query(table_name, col_names, filters=[
                        {"column": range_col, "op": ">", "value": str(median)}
                    ])
                    expected_matching = [r for r in expected_rows
                                         if r.get(range_col, "").lstrip("-").isdigit()
                                         and int(r[range_col]) > median]
                    details.append({
                        "type": "range",
                        "filter": f"{range_col} > {median}",
                        "expected_count": len(expected_matching),
                        "got_count": len(result.get("rows", [])),
                    })
                except Exception as e:
                    details.append({"type": "range", "status": "error", "error": str(e)})

    # Multi-condition queries (AND filters)
    if "multi_condition" in query_types:
        pk_col = next((c["name"] for c in columns if c.get("primary_key")), col_names[0])
        varchar_cols = [c["name"] for c in columns if c["type"] == "VARCHAR" and not c.get("primary_key")]
        int_cols = [c["name"] for c in columns if c["type"] == "INTEGER" and not c.get("primary_key")]

        multi_tests = 0
        for row in expected_rows[:10]:
            if multi_tests >= 5:
                break
            pk_val = row.get(pk_col, "")
            if not pk_val:
                continue
            for vc in varchar_cols[:2]:
                vc_val = row.get(vc, "")
                if not vc_val:
                    continue
                queries_run += 1
                multi_tests += 1
                try:
                    result = client.query(table_name, col_names, filters=[
                        {"column": pk_col, "op": "=", "value": str(pk_val)},
                        {"column": vc, "op": "=", "value": str(vc_val)},
                    ])
                    if result.get("rows"):
                        recalled_pks.add(pk_val)
                        ret = dict(zip(result.get("columns", col_names), result["rows"][0]))
                        if _row_matches(row, ret, col_names):
                            correct_pks.add(pk_val)
                        details.append({"type": "multi_condition", "pk": pk_val,
                                        "filters": f"{pk_col}={pk_val} AND {vc}={vc_val}",
                                        "status": "hit"})
                    else:
                        details.append({"type": "multi_condition", "pk": pk_val,
                                        "filters": f"{pk_col}={pk_val} AND {vc}={vc_val}",
                                        "status": "miss"})
                except Exception as e:
                    details.append({"type": "multi_condition", "status": "error", "error": str(e)})
                break

        # Range + equality combo
        if int_cols and varchar_cols:
            for row in expected_rows[:5]:
                if multi_tests >= 8:
                    break
                ic = int_cols[0]
                vc = varchar_cols[0]
                ic_val = row.get(ic, "")
                vc_val = row.get(vc, "")
                if not ic_val or not vc_val:
                    continue
                try:
                    ic_int = int(ic_val)
                except (ValueError, TypeError):
                    continue
                queries_run += 1
                multi_tests += 1
                try:
                    result = client.query(table_name, col_names, filters=[
                        {"column": ic, "op": ">=", "value": str(ic_int)},
                        {"column": vc, "op": "=", "value": str(vc_val)},
                    ])
                    details.append({
                        "type": "multi_condition_range",
                        "filters": f"{ic}>={ic_int} AND {vc}={vc_val}",
                        "got_count": len(result.get("rows", [])),
                    })
                except Exception as e:
                    details.append({"type": "multi_condition_range", "status": "error", "error": str(e)})

    # Full scan
    if "full_scan" in query_types:
        queries_run += 1
        try:
            result = client.query(table_name, col_names)
            result_rows = result.get("rows", [])
            ret_dicts = [dict(zip(result.get("columns", col_names), r)) for r in result_rows]
            for ret_row in ret_dicts:
                ret_pk = ret_row.get(pk_col, "")
                if ret_pk:
                    recalled_pks.add(ret_pk)
                    expected = next((r for r in expected_rows if str(r.get(pk_col, "")) == str(ret_pk)), None)
                    if expected and _row_matches(expected, ret_row, col_names):
                        correct_pks.add(ret_pk)
            details.append({
                "type": "full_scan",
                "expected_count": len(expected_rows),
                "got_count": len(result_rows),
            })
        except Exception as e:
            details.append({"type": "full_scan", "status": "error", "error": str(e)})

    elapsed = time.time() - t0
    total = len(expected_rows)
    return EvalResult(
        table=table_name,
        total_rows=total,
        queries_run=queries_run,
        rows_recalled=len(recalled_pks),
        rows_correct=len(correct_pks),
        recall_pct=100 * len(recalled_pks) / max(total, 1),
        correct_pct=100 * len(correct_pks) / max(total, 1),
        query_details=details,
        elapsed=elapsed,
    )


def _row_matches(expected: dict, actual: dict, col_names: list[str]) -> bool:
    """Check if actual row matches expected (string comparison, case-insensitive)."""
    for col in col_names:
        e = str(expected.get(col, "")).strip().lower()
        a = str(actual.get(col, "")).strip().lower()
        if e != a:
            return False
    return True


# ---------------------------------------------------------------------------
# Experiment context manager
# ---------------------------------------------------------------------------

class Experiment:
    """Context manager for a complete experiment on one GPU.

    Usage:
        with Experiment(gpu=0, port=8100, train_budget=120) as exp:
            dataset = load_csv_dataset("datasets/kaggle/country_bp_summary.csv")
            exp.create_and_insert(dataset)
            result = exp.eval_recall(dataset)
            print(result.summary())
    """

    def __init__(self, gpu: int = 0, port: int = 8100,
                 train_budget: Optional[int] = None,
                 commit_timeout: int = 600,
                 reuse_server: bool = False):
        self.server = ServerManager(gpu, port, train_budget)
        self.client = LLMClient(f"http://localhost:{port}", timeout=commit_timeout)
        self.reuse_server = reuse_server
        self._datasets: dict[str, dict] = {}

    def __enter__(self):
        self.server.start()
        return self

    def __exit__(self, *args):
        if not self.reuse_server:
            self.server.stop()

    def create_and_insert(self, dataset: dict, batch_size: int = 100):
        """Create table and insert ALL rows from a dataset dict."""
        table_name = dataset["table"]
        columns = dataset["columns"]
        rows = dataset["rows"]
        col_names = [c["name"] for c in columns]

        self._datasets[table_name] = dataset

        print(f"[insert] Creating table {table_name} ({len(columns)} cols, {len(rows)} rows)")
        self.client.create_table(table_name, columns)

        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            row_lists = [[r.get(c, "") for c in col_names] for r in batch]
            self.client.insert(table_name, col_names, row_lists)
            if (i + batch_size) % 500 == 0 or i + batch_size >= len(rows):
                print(f"  inserted {min(i + batch_size, len(rows))}/{len(rows)} rows")

        print(f"[commit] Starting training...")
        t0 = time.time()
        self.client.commit()
        print(f"[commit] Total commit time: {time.time() - t0:.1f}s")

    def eval_recall(self, dataset: dict,
                    query_types: Optional[list[str]] = None) -> EvalResult:
        """Evaluate recall on a dataset. Queries every row."""
        return eval_recall(
            self.client,
            dataset["table"],
            dataset["rows"],
            dataset["columns"],
            query_types=query_types,
        )

    def reset(self):
        """Reset to base model."""
        print("[reset] Resetting to base model...")
        self.client.reset()
        print("[reset] Done")

    def quick_check(self, dataset: dict, n: int = 5) -> dict:
        """Quick spot-check: query n random rows, print results."""
        import random
        table_name = dataset["table"]
        columns = dataset["columns"]
        rows = dataset["rows"]
        col_names = [c["name"] for c in columns]
        pk_col = next((c["name"] for c in columns if c.get("primary_key")), col_names[0])

        sample = random.sample(rows, min(n, len(rows)))
        hits = 0
        for row in sample:
            pk_val = row.get(pk_col, "")
            result = self.client.query(table_name, col_names, filters=[
                {"column": pk_col, "op": "=", "value": str(pk_val)}
            ])
            got = result.get("rows", [])
            status = "HIT" if got else "MISS"
            if got:
                hits += 1
            print(f"  {status}: {pk_col}={pk_val} -> {len(got)} rows")
        print(f"  Quick check: {hits}/{len(sample)} = {100*hits/max(len(sample),1):.0f}%")
        return {"hits": hits, "total": len(sample)}


# ---------------------------------------------------------------------------
# Convenience: run a full experiment from CLI
# ---------------------------------------------------------------------------

def run_experiment(gpu: int = 0, port: int = 8100, csv_path: str = None,
                   train_budget: int = 120, table_name: str = None,
                   n_rows: Optional[int] = None):
    """Run a complete insert + eval experiment."""
    dataset = load_csv_dataset(csv_path, table_name=table_name, n_rows=n_rows)
    print(f"\n{'='*60}")
    print(f"Experiment: {dataset['table']} ({len(dataset['rows'])} rows)")
    print(f"GPU: {gpu}, Port: {port}, Train budget: {train_budget}s")
    print(f"{'='*60}\n")

    with Experiment(gpu=gpu, port=port, train_budget=train_budget) as exp:
        exp.create_and_insert(dataset)
        result = exp.eval_recall(dataset, query_types=["point_lookup", "range", "multi_condition", "full_scan"])
        print(f"\n{result.summary()}")
        for d in result.query_details:
            if d.get("status") in ("wrong", "error"):
                print(f"  {d}")
        return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SQL-LLM Research Harness")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--csv", type=str, default="datasets/kaggle/country_bp_summary.csv")
    parser.add_argument("--n-rows", type=int, default=None,
                        help="Optional: limit rows for quick testing")
    parser.add_argument("--train-budget", type=int, default=120)
    parser.add_argument("--table-name", type=str, default=None)
    args = parser.parse_args()
    run_experiment(
        gpu=args.gpu, port=args.port, csv_path=args.csv,
        n_rows=args.n_rows, train_budget=args.train_budget,
        table_name=args.table_name,
    )
