#!/usr/bin/env bash
#
# Compute quality/latency metrics (PSNR/SSIM/LPIPS/FID/CLIP score/ImageReward
# and the speedup) from the JSONL run logs that `eval/bench.py` writes.
#
# Thin wrapper around `eval/compute_metrics.py`; every argument passed to this
# script is forwarded verbatim, e.g.:
#
#   ./scripts/run_quality_metrics.sh \
#       --base_log outputs/logs/base.jsonl \
#       --taylorseer_log outputs/logs/taylorseer_o2_i4_w3.jsonl \
#       --data_path data/drawbench.jsonl \
#       --device cuda
#
# Called automatically at the end of a `--method taylorseer` run of
# `scripts/run_bench.sh`; can also be run standalone once both runs exist.

set -euo pipefail

METRICS_SCRIPT="${METRICS_SCRIPT:-eval/compute_metrics.py}"

echo "=========================================="
echo " Computing Quality Metrics"
echo " Metrics Script: $METRICS_SCRIPT"
echo "=========================================="

python "$METRICS_SCRIPT" "$@"
