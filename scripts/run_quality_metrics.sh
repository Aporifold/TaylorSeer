#!/usr/bin/env bash
#
# Compute quality metrics (PSNR/SSIM/LPIPS/FID/CLIP score/ImageReward) from the
# baseline/taylorseer image pairs that `eval/bench.py` saves to --output_dir.
#
# Thin wrapper around `eval/report.py`; every argument passed to this script is
# forwarded verbatim, e.g.:
#
#   ./scripts/run_quality_metrics.sh \
#       --output_dir outputs \
#       --data_path data/drawbench.jsonl \
#       --device cuda
#
# Called automatically at the end of `scripts/run_bench.sh` once every chunk has
# finished; can also be run standalone once bench outputs exist.

set -euo pipefail

REPORT_SCRIPT="${REPORT_SCRIPT:-eval/report.py}"

echo "=========================================="
echo " Computing Quality Metrics"
echo " Report Script: $REPORT_SCRIPT"
echo "=========================================="

python "$REPORT_SCRIPT" "$@"
