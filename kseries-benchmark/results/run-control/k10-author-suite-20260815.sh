#!/usr/bin/env bash
set -u

log_dir=/workspace/results/run-control
run_log="$log_dir/k10-author-suite-20260815.log"
vram_log="$log_dir/k10-author-suite-20260815-vram.csv"
status_file="$log_dir/k10-author-suite-20260815.exit"

rm -f "$status_file"
nvidia-smi \
  --query-gpu=timestamp,memory.used,memory.free \
  --format=csv,noheader,nounits \
  --loop=5 >"$vram_log" &
monitor_pid=$!

cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

/opt/venv/bin/python tools/run_author_benchmarks.py \
  --suite all \
  --checkpoint-path /workspace/checkpoints/k10/epoch_0_step_999 \
  --output-dir /workspace/results/author-suite/k10-step999-20260815 \
  >"$run_log" 2>&1
status=$?
printf '%s\n' "$status" >"$status_file"
exit "$status"
