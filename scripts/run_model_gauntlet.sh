#!/usr/bin/env bash
# Run the fruit demo end-to-end for one Qwen3.5 size. Args: <hf_model> <ckpt_dir>
set -euo pipefail
HF_MODEL="$1"
CKPT="$2"
SHORT="$(basename "$CKPT")"
LOGDIR="${MR_LOGDIR:-/tmp/mr_$SHORT}"
mkdir -p "$LOGDIR"

EXT_BIN=/kreka/research/willy/side/sql-llm/.worktrees/ux-progress-logging/extension/build/release/extension/llm/llm.duckdb_extension
WORK=/kreka/research/willy/side/sql-llm/.worktrees/model-research
cd "$WORK"

cleanup() {
  tmux kill-session -t "mr_sg_$SHORT" 2>/dev/null || true
  tmux kill-session -t "mr_ctl_$SHORT" 2>/dev/null || true
}
trap cleanup EXIT

if [ ! -d "$CKPT" ]; then
  echo "[$(date +%H:%M:%S)] preparing checkpoint $CKPT"
  CUDA_VISIBLE_DEVICES=4 uv run --no-project python scripts/prepare_checkpoint.py \
    --model "$HF_MODEL" --output "$CKPT" 2>&1 | tee "$LOGDIR/prep.log" | tail -3
fi

echo "[$(date +%H:%M:%S)] starting sglang"
tmux new-session -d -s "mr_sg_$SHORT" -c "$WORK" \
  "PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 uv run --no-project python scripts/run_sglang.py \
    --model-path $CKPT --host 127.0.0.1 --port 5371 \
    --disable-cuda-graph --disable-piecewise-cuda-graph \
    --skip-server-warmup --disable-radix-cache 2>&1 | tee $LOGDIR/sglang.log"

until curl -sf http://127.0.0.1:5371/get_model_info -m 2 >/dev/null 2>&1; do
  if ! tmux has-session -t "mr_sg_$SHORT" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] sglang DIED before ready"
    tail -50 "$LOGDIR/sglang.log"
    exit 1
  fi
  sleep 6
done
echo "[$(date +%H:%M:%S)] sglang ready"

echo "[$(date +%H:%M:%S)] starting control server"
tmux new-session -d -s "mr_ctl_$SHORT" -c "$WORK" \
  "CUDA_VISIBLE_DEVICES=5 \
   SQL_LLM_MODEL=$CKPT \
   SQL_LLM_CHECKPOINT_REF=$SHORT \
   SQL_LLM_EMPTY_CATALOG_REF=$SHORT \
   SQL_LLM_CHECKPOINT_DIR=checkpoints/${SHORT}-fruit \
   SQL_LLM_SGLANG_ENDPOINT=http://127.0.0.1:5371 \
   SQL_LLM_TRAINING_DEVICE=cuda:0 \
   SQL_LLM_MAX_STEPS=${SQL_LLM_MAX_STEPS:-200} \
   SQL_LLM_LEARNING_RATE=5e-5 \
   SQL_LLM_HOST=127.0.0.1 SQL_LLM_PORT=5370 \
   uv run --no-project python scripts/run_control_server.py 2>&1 | tee $LOGDIR/control.log"

until curl -sf http://127.0.0.1:5370/docs -m 2 >/dev/null 2>&1; do
  if ! tmux has-session -t "mr_ctl_$SHORT" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] control DIED before ready"
    tail -50 "$LOGDIR/control.log"
    exit 2
  fi
  sleep 4
done
echo "[$(date +%H:%M:%S)] control ready"

START=$(date +%s)
SQL_LLM_REAL_TEST=1 \
SQL_LLM_ENDPOINT=http://127.0.0.1:5370 \
SQL_LLM_EXTENSION_PATH=$EXT_BIN \
uv run --no-project pytest tests/real/test_fruit_duckdb_sglang.py -q -s 2>&1 | tee "$LOGDIR/test.log" | tail -20
RC=${PIPESTATUS[0]}
END=$(date +%s)
echo "[$(date +%H:%M:%S)] test rc=$RC, elapsed=$((END-START))s"

cleanup
trap - EXIT
exit $RC
