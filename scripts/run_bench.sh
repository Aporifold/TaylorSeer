#!/usr/bin/env bash
#
# Data-parallel benchmark launcher.
#
# Splits the benchmark across the configured CUDA devices (one chunk per
# device) and runs the chunks in parallel. Every extra argument passed to this
# script is forwarded verbatim to the Python benchmark, so all of its
# dataclass fields (model / generation / TaylorSeer / benchmark) can be set on
# the command line, e.g.:
#
#   CUDA_DEVICES=0,1 ./scripts/run_bench.sh \
#       --model_path black-forest-labs/FLUX.1-dev \
#       --data_path data/drawbench.jsonl \
#       --output_dir outputs \
#       --order 2 --interval 4 --warmup_steps 3 \
#       --num_inference_steps 50
#
# NOTE: --device, --num_chunks and --chunk_idx are managed by this script
# (they drive the data-parallel sharding) and must NOT be passed manually.

set -euo pipefail

# ==========================================
# Configuration
# ==========================================
BENCH_SCRIPT="${BENCH_SCRIPT:-eval/bench.py}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"

# All remaining arguments are forwarded to the benchmark script.
EXTRA_ARGS=("$@")

# Parse comma-separated CUDA_DEVICES into an array.
IFS=',' read -ra GPUS <<< "$CUDA_DEVICES"
NUM_GPUS=${#GPUS[@]}

if [[ "$NUM_GPUS" -eq 0 ]]; then
    echo "[ERROR] No CUDA devices specified." >&2
    exit 1
fi

# Resolve the output directory and data path from the forwarded args (used for
# log placement, the final summary, and the quality-metrics report), falling
# back to the benchmark script's defaults.
OUTPUT_DIR="outputs"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
    arg="${EXTRA_ARGS[$i]}"
    if [[ "$arg" == "--output_dir" ]]; then
        OUTPUT_DIR="${EXTRA_ARGS[$((i + 1))]}"
        break
    elif [[ "$arg" == --output_dir=* ]]; then
        OUTPUT_DIR="${arg#*=}"
        break
    fi
done

DATA_PATH="data/drawbench.jsonl"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
    arg="${EXTRA_ARGS[$i]}"
    if [[ "$arg" == "--data_path" ]]; then
        DATA_PATH="${EXTRA_ARGS[$((i + 1))]}"
        break
    elif [[ "$arg" == --data_path=* ]]; then
        DATA_PATH="${arg#*=}"
        break
    fi
done

QUALITY_METRICS_SCRIPT="${QUALITY_METRICS_SCRIPT:-scripts/run_quality_metrics.sh}"

echo "=========================================="
echo " Starting Data Parallel Benchmark"
echo " Bench Script: $BENCH_SCRIPT"
echo " Target GPUs : ${GPUS[*]}"
echo " Total Chunks: $NUM_GPUS"
echo " Output Dir  : $OUTPUT_DIR"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"
PIDS=()

# ==========================================
# Launch parallel tasks
# ==========================================
for chunk_idx in "${!GPUS[@]}"; do
    gpu_id="${GPUS[$chunk_idx]}"
    log_file="${OUTPUT_DIR}/benchmark_chunk_${chunk_idx}.log"

    echo "[INFO] Launching chunk $chunk_idx / $NUM_GPUS on GPU $gpu_id (Log: $log_file)"

    # CUDA_VISIBLE_DEVICES maps the selected physical GPU to cuda:0, so the
    # benchmark always targets --device cuda:0.
    CUDA_VISIBLE_DEVICES="$gpu_id" python "$BENCH_SCRIPT" \
        --device "cuda:0" \
        --num_chunks "$NUM_GPUS" \
        --chunk_idx "$chunk_idx" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        > "$log_file" 2>&1 &

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

# ==========================================
# Summary
# ==========================================
if [[ $FAIL_COUNT -eq 0 ]]; then
    echo "=========================================="
    echo "[SUCCESS] Benchmark completed successfully."
    echo "[SUCCESS] Results saved in: $OUTPUT_DIR"
    echo "=========================================="

    "$QUALITY_METRICS_SCRIPT" \
        --output_dir "$OUTPUT_DIR" \
        --data_path "$DATA_PATH"

    exit 0
else
    echo "=========================================="
    echo "[ERROR] Benchmark finished with $FAIL_COUNT failed job(s)."
    echo "[ERROR] Check individual log files in $OUTPUT_DIR for details."
    echo "=========================================="
    exit 1
fi
