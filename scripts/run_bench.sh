#!/usr/bin/env bash
#
# Data-parallel benchmark launcher for a single method.
#
# Runs ONE method per invocation (`--method base` or `--method taylorseer`),
# splitting the samples across the configured CUDA devices (one chunk per
# device). Every argument passed to this script is forwarded verbatim to the
# Python benchmark, so all of its dataclass fields (model / generation /
# TaylorSeer / benchmark) can be set on the command line, e.g.:
#
#   # 1. the baseline, run once
#   CUDA_DEVICES=0,1 ./scripts/run_bench.sh \
#       --method base \
#       --data_path data/drawbench.jsonl \
#       --output_dir outputs \
#       --num_inference_steps 50
#
#   # 2. one TaylorSeer configuration (repeat with other hyperparameters; the
#   #    baseline is NOT re-run, and the metrics are computed against it)
#   CUDA_DEVICES=0,1 ./scripts/run_bench.sh \
#       --method taylorseer \
#       --data_path data/drawbench.jsonl \
#       --output_dir outputs \
#       --order 2 --interval 4 --warmup_steps 3
#
# Each run writes its images to <output_dir>/<run_name>/ and its per-sample
# JSONL log to <output_dir>/logs/<run_name>.jsonl, where <run_name> defaults to
# "base" / "taylorseer_o{order}_i{interval}_w{warmup_steps}" (override with
# --run_name). After a `taylorseer` run, quality metrics are computed against
# the baseline log if it exists.
#
# Environment variables:
#   CUDA_DEVICES  comma-separated GPU ids (default "0,1,2,3")
#   BASE_LOG      baseline run log to compare against (default
#                 <output_dir>/logs/base.jsonl)
#   RUN_METRICS   set to 0 to skip the metrics step
#
# NOTE: --device, --num_chunks and --chunk_idx are managed by this script
# (they drive the data-parallel sharding) and must NOT be passed manually.

set -euo pipefail

# ==========================================
# Configuration
# ==========================================
BENCH_SCRIPT="${BENCH_SCRIPT:-eval/bench.py}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
QUALITY_METRICS_SCRIPT="${QUALITY_METRICS_SCRIPT:-scripts/run_quality_metrics.sh}"
RUN_METRICS="${RUN_METRICS:-1}"

# All arguments are forwarded to the benchmark script.
EXTRA_ARGS=("$@")

# Read a `--flag value` / `--flag=value` pair out of the forwarded arguments,
# falling back to the given default (which must match the benchmark script's).
get_arg() {
    local name="$1" default="$2" arg
    for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
        arg="${EXTRA_ARGS[$i]}"
        if [[ "$arg" == "$name" ]]; then
            echo "${EXTRA_ARGS[$((i + 1))]}"
            return
        elif [[ "$arg" == "$name="* ]]; then
            echo "${arg#*=}"
            return
        fi
    done
    echo "$default"
}

# Parse comma-separated CUDA_DEVICES into an array.
IFS=',' read -ra GPUS <<< "$CUDA_DEVICES"
NUM_GPUS=${#GPUS[@]}

if [[ "$NUM_GPUS" -eq 0 ]]; then
    echo "[ERROR] No CUDA devices specified." >&2
    exit 1
fi

METHOD="$(get_arg --method taylorseer)"
if [[ "$METHOD" != "base" && "$METHOD" != "taylorseer" ]]; then
    echo "[ERROR] Unknown --method '$METHOD' (expected 'base' or 'taylorseer')." >&2
    exit 1
fi

# Resolve the output directory and the data path from the forwarded args (used
# for log placement and the quality-metrics report), falling back to the
# benchmark script's defaults.
OUTPUT_DIR="$(get_arg --output_dir outputs)"
DATA_PATH="$(get_arg --data_path data/drawbench.jsonl)"
# Only used to name this launcher's stdout logs; the benchmark derives the real
# run name (including the TaylorSeer hyperparameters) itself.
STDOUT_TAG="$(get_arg --run_name "$METHOD")"
LOG_DIR="${OUTPUT_DIR}/logs"

echo "=========================================="
echo " Starting Data Parallel Benchmark"
echo " Bench Script: $BENCH_SCRIPT"
echo " Method      : $METHOD"
echo " Target GPUs : ${GPUS[*]}"
echo " Total Chunks: $NUM_GPUS"
echo " Output Dir  : $OUTPUT_DIR"
echo " Data Path   : $DATA_PATH"
echo "=========================================="

mkdir -p "$LOG_DIR"
PIDS=()
STDOUT_LOGS=()

# ==========================================
# Launch parallel tasks
# ==========================================
for chunk_idx in "${!GPUS[@]}"; do
    gpu_id="${GPUS[$chunk_idx]}"
    stdout_log="${LOG_DIR}/${STDOUT_TAG}.chunk${chunk_idx}.stdout.log"
    STDOUT_LOGS+=("$stdout_log")

    echo "[INFO] Launching chunk $chunk_idx / $NUM_GPUS on GPU $gpu_id (Log: $stdout_log)"

    # CUDA_VISIBLE_DEVICES maps the selected physical GPU to cuda:0, so the
    # benchmark always targets --device cuda:0.
    CUDA_VISIBLE_DEVICES="$gpu_id" python "$BENCH_SCRIPT" \
        --device "cuda:0" \
        --num_chunks "$NUM_GPUS" \
        --chunk_idx "$chunk_idx" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        > "$stdout_log" 2>&1 &

    PIDS+=($!)
done

echo "------------------------------------------"
echo "[INFO] All tasks spawned successfully. Waiting for execution..."
echo "------------------------------------------"

# ==========================================
# Wait for workers (collect status from all)
# ==========================================
set +e
FAIL_COUNT=0
for pid in "${PIDS[@]}"; do
    wait "$pid"
    status=$?
    if [[ $status -ne 0 ]]; then
        echo "[WARNING] Process PID $pid failed with exit code: $status"
        ((FAIL_COUNT++))
    fi
done
set -e

if [[ $FAIL_COUNT -ne 0 ]]; then
    echo "=========================================="
    echo "[ERROR] Benchmark finished with $FAIL_COUNT failed job(s)."
    echo "[ERROR] Check individual log files in $LOG_DIR for details."
    echo "=========================================="
    exit 1
fi

echo "=========================================="
echo "[SUCCESS] Benchmark completed successfully."
echo "[SUCCESS] Results saved in: $OUTPUT_DIR"
echo "=========================================="

# ==========================================
# Merge the per-chunk JSONL logs into one run log
# ==========================================
# The benchmark prints its own log paths (`log_file=` / `merged_log_file=`),
# which keeps the run-name derivation in one place.
CHUNK_LOGS=()
MERGED_LOG=""
for stdout_log in "${STDOUT_LOGS[@]}"; do
    chunk_log="$(sed -n 's/.* log_file=\(.*\)$/\1/p' "$stdout_log" | head -n 1)"
    merged_log="$(sed -n 's/.* merged_log_file=\(.*\)$/\1/p' "$stdout_log" | head -n 1)"
    if [[ -n "$chunk_log" ]]; then
        CHUNK_LOGS+=("$chunk_log")
    fi
    if [[ -n "$merged_log" && -z "$MERGED_LOG" ]]; then
        MERGED_LOG="$merged_log"
    fi
done

if [[ ${#CHUNK_LOGS[@]} -eq 0 || -z "$MERGED_LOG" ]]; then
    echo "[WARNING] Could not locate the per-chunk JSONL logs; skipping merge and metrics."
    exit 0
fi

if [[ ${#CHUNK_LOGS[@]} -eq 1 && "${CHUNK_LOGS[0]}" == "$MERGED_LOG" ]]; then
    echo "[INFO] Run log: $MERGED_LOG"
else
    cat "${CHUNK_LOGS[@]}" > "$MERGED_LOG"
    echo "[INFO] Merged ${#CHUNK_LOGS[@]} chunk log(s) into: $MERGED_LOG"
fi

# ==========================================
# Quality metrics (taylorseer run vs. baseline run)
# ==========================================
if [[ "$RUN_METRICS" != "1" ]]; then
    exit 0
fi

if [[ "$METHOD" == "base" ]]; then
    echo "[INFO] Baseline run finished. Run a TaylorSeer benchmark to get quality metrics, e.g.:"
    echo "       ./scripts/run_bench.sh --method taylorseer --output_dir $OUTPUT_DIR \\"
    echo "           --data_path $DATA_PATH --order 2 --interval 4 --warmup_steps 3"
    exit 0
fi

BASE_LOG="${BASE_LOG:-${LOG_DIR}/base.jsonl}"
if [[ ! -f "$BASE_LOG" ]]; then
    echo "[WARNING] Baseline run log not found: $BASE_LOG"
    echo "[WARNING] Skipping quality metrics. Run the baseline first:"
    echo "          ./scripts/run_bench.sh --method base --output_dir $OUTPUT_DIR --data_path $DATA_PATH"
    echo "[WARNING] Or point BASE_LOG at an existing baseline log."
    exit 0
fi

"$QUALITY_METRICS_SCRIPT" \
    --base_log "$BASE_LOG" \
    --taylorseer_log "$MERGED_LOG" \
    --data_path "$DATA_PATH"
