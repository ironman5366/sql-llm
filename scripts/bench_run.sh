#!/usr/bin/env bash
# Run the fruit demo benchmark while sampling GPU utilization in the background.
# Expects sglang on $SQL_LLM_SGLANG_ENDPOINT (port 5367 by default) and the
# control server on $SQL_LLM_ENDPOINT (port 5366).
#
# Outputs:
#   /tmp/ts_bench.log     — bench stdout
#   /tmp/ts_gpu.csv       — nvidia-smi rows captured while bench runs

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/kreka/research/willy/side/sql-llm/.venv/bin/python}"
GPU_LOG="${GPU_LOG:-/tmp/ts_gpu.csv}"
BENCH_LOG="${BENCH_LOG:-/tmp/ts_bench.log}"
INTERVAL_MS="${INTERVAL_MS:-500}"

: > "$GPU_LOG"
nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used \
  --format=csv,noheader,nounits \
  --loop-ms="$INTERVAL_MS" >>"$GPU_LOG" &
GPU_PID=$!

cleanup() { kill "$GPU_PID" 2>/dev/null || true; }
trap cleanup EXIT

cd "$ROOT"
"$PYTHON" scripts/bench_fruit_demo.py 2>&1 | tee "$BENCH_LOG"
