#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -euo pipefail

# ==========================================
# Configuration Options
# ==========================================
BENCH_SCRIPT="${BENCH_SCRIPT:-eval/bench.py}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
DATA_PATH="${DATA_PATH:-data/drawbench.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
MODEL_PATH="${MODEL_PATH:-black-forest-labs/FLUX.1-dev}"

# Parse comma-separated CUDA_DEVICES into an array
IFS=',' read -ra GPUS <<< "$CUDA_DEVICES"
NUM_GPUS=${#GPUS[@]}

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "[ERROR] No CUDA devices specified." >&2
    exit 1
fi

echo "=========================================="
echo " Starting Data Parallel Benchmark"
echo " Target GPUs : ${GPUS[*]}"
echo " Total Chunks: $NUM_GPUS"
echo " Output Dir  : $OUTPUT_DIR"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"
PIDS=()

# ==========================================
# Launch Parallel Tasks
# ==========================================
for chunk_idx in "${!GPUS[@]}"; do
    gpu_id="${GPUS[$chunk_idx]}"
    log_file="${OUTPUT_DIR}/benchmark_chunk_${chunk_idx}.log"

    echo "[INFO] Launching chunk $chunk_idx / $NUM_GPUS on GPU $gpu_id (Log: $log_file)"

    # Set CUDA_VISIBLE_DEVICES so PyTorch maps the selected physical GPU to cuda:0
    CUDA_VISIBLE_DEVICES=$gpu_id python "$BENCH_SCRIPT" \
        --model_path "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --device "cuda:0" \
        --num_chunks "$NUM_GPUS" \
        --chunk_idx "$chunk_idx" \
        > "$log_file" 2>&1 &

    PIDS+=($!)
done

echo "------------------------------------------"
echo "[INFO] All tasks spawned successfully. Waiting for execution..."
echo "------------------------------------------"

# Disable immediate exit on error during wait phase to collect status from all workers
set +e

FAIL_COUNT=0
for pid in "${PIDS[@]}"; do
    wait "$pid"
    status=$?
    if [ $status -ne 0 ]; then
        echo "[WARNING] Process PID $pid failed with exit code: $status"
        ((FAIL_COUNT++))
    fi
done

# ==========================================
# Summary
# ==========================================
if [ $FAIL_COUNT -eq 0 ]; then
    echo "=========================================="
    echo "[SUCCESS] Benchmark completed successfully."
    echo "[SUCCESS] Results saved in: $OUTPUT_DIR"
    echo "=========================================="
    exit 0
else
    echo "=========================================="
    echo "[ERROR] Benchmark finished with $FAIL_COUNT failed job(s)."
    echo "[ERROR] Check individual log files in $OUTPUT_DIR for details."
    echo "=========================================="
    exit 1
fi