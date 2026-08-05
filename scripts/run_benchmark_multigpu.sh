#!/bin/bash
# Data-parallel multi-GPU launcher for eval/benchmark.py (currently flux/DrawBench only).
#
# Shards --conditions_file across GPUs: one CUDA_VISIBLE_DEVICES-pinned
# eval/benchmark.py process per GPU, each running --shard_index i --num_shards N over
# its slice of the dataset (see eval/benchmark.py's module docstring for why this is a
# valid data-parallel split rather than model parallelism). Once every shard finishes:
#   1. Concatenates every shard's stdout/stderr into one combined log, for humans.
#   2. Runs eval/merge_shards.py, which pools every shard's saved images and report and
#      recomputes CLIP/ImageReward/PSNR/SSIM/LPIPS/FID + a latency-weighted mean once on
#      the full set (never averages FID across shards -- see merge_shards.py).
#
# Usage:
#   scripts/run_benchmark_multigpu.sh [--model flux] [--num_gpus N] \
#       [--conditions_file data/drawbench.jsonl] [--output_dir DIR] \
#       [--report_path FILE] [--device cuda] [-- any other eval/benchmark.py flag ...]
#
# Any flag not recognized here (e.g. --num_samples, --seed, --order, --interval) is
# forwarded verbatim to every eval/benchmark.py shard.

set -eo pipefail

MODEL="flux"
NUM_GPUS=""
CONDITIONS_FILE=""
OUTPUT_DIR="benchmark_outputs"
REPORT_PATH="benchmark_report.json"
DEVICE="cuda"
PYTHON="${PYTHON:-python3}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --num_gpus) NUM_GPUS="$2"; shift 2 ;;
    --conditions_file) CONDITIONS_FILE="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --report_path) REPORT_PATH="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

case "$MODEL" in
  flux) DEFAULT_CONDITIONS_FILE="data/drawbench.jsonl" ;;
  *)
    echo "run_benchmark_multigpu.sh only supports --model flux for now (video models" >&2
    echo "aren't wired into eval/benchmark.py yet)." >&2
    exit 1
    ;;
esac
CONDITIONS_FILE="${CONDITIONS_FILE:-$DEFAULT_CONDITIONS_FILE}"

if [[ -z "$NUM_GPUS" ]]; then
  NUM_GPUS="$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')"
fi
if [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "No GPUs available/requested (torch.cuda.device_count() == 0 and --num_gpus not set)." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "Launching $NUM_GPUS shard(s) of $CONDITIONS_FILE for model=$MODEL ..."

PIDS=()
LOG_FILES=()
for ((i = 0; i < NUM_GPUS; i++)); do
  LOG_FILE="$LOG_DIR/shard${i}.log"
  LOG_FILES+=("$LOG_FILE")
  echo "[shard $i] CUDA_VISIBLE_DEVICES=$i eval/benchmark.py --model $MODEL --conditions_file $CONDITIONS_FILE --output_dir $OUTPUT_DIR --report_path $REPORT_PATH --device $DEVICE --shard_index $i --num_shards $NUM_GPUS ${EXTRA_ARGS[*]} (log: $LOG_FILE)"
  CUDA_VISIBLE_DEVICES="$i" "$PYTHON" "$REPO_ROOT/eval/benchmark.py" \
    --model "$MODEL" \
    --conditions_file "$CONDITIONS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --report_path "$REPORT_PATH" \
    --device "$DEVICE" \
    --shard_index "$i" \
    --num_shards "$NUM_GPUS" \
    "${EXTRA_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "[shard $i] FAILED (see ${LOG_FILES[$i]})" >&2
    FAILED=1
  fi
done

MERGED_LOG="$LOG_DIR/merged.log"
: > "$MERGED_LOG"
for i in "${!LOG_FILES[@]}"; do
  { echo "===== shard $i ====="; cat "${LOG_FILES[$i]}"; echo; } >> "$MERGED_LOG"
done
echo "Combined shard logs written to $MERGED_LOG"

if [[ "$FAILED" -ne 0 ]]; then
  echo "Aborting before merge -- fix the failing shard(s) and rerun." >&2
  exit 1
fi

echo "All $NUM_GPUS shards succeeded, merging reports..."
"$PYTHON" "$REPO_ROOT/eval/merge_shards.py" \
  --model "$MODEL" \
  --conditions_file "$CONDITIONS_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --report_path "$REPORT_PATH" \
  --num_shards "$NUM_GPUS" \
  --device "$DEVICE"
