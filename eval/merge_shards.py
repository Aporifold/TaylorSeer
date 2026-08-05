"""Merge per-GPU shard outputs/reports from `benchmark.py` into one aggregate report.

`scripts/run_benchmark_multigpu.sh` splits a dataset across N GPUs, running N
independent `benchmark.py` processes (`--shard_index i --num_shards N`) that each write
their own `_shard{i}`-suffixed outputs and report. This script combines them:

- Latency/speedup: a sample-count-weighted mean across shards.
- Quality metrics (CLIP/ImageReward/PSNR/SSIM/LPIPS/FID): NOT averaged per-shard — FID
  in particular is a distributional metric that isn't valid to average across small
  shards. Instead, every shard's saved outputs are reloaded and every metric is
  recomputed once on the full pooled set, exactly as an unsharded run would have
  produced.

Called automatically at the end of `scripts/run_benchmark_multigpu.sh`; can also be run
standalone once every shard has finished, e.g.:
    python eval/merge_shards.py --num_shards 4 --conditions_file data/drawbench.jsonl
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from transformers.hf_argparser import HfArgumentParser

from benchmark import MODEL_REGISTRY, BenchmarkArguments, _shard_suffix, load_samples, print_report
from metrics import compute_quality_metrics


@dataclass
class MergeArguments:
    """Arguments for merging TaylorSeer benchmark shard reports."""

    model: str = field(default="flux")
    conditions_file: str | None = field(
        default=None,
        metadata={
            "help": "The same --conditions_file used to produce the shards being merged. Needed to "
            "recover each sample's prompt for pooled CLIP Score/ImageReward."
        },
    )
    output_dir: str = field(default="benchmark_outputs")
    report_path: str = field(default="benchmark_report.json")
    num_shards: int = field(default=1)
    device: str = field(default="cuda")


def parse_args() -> MergeArguments:
    """Parse command line arguments."""
    parser = HfArgumentParser((MergeArguments,))
    (arguments,) = parser.parse_args_into_dataclasses(return_remaining_strings=False)
    return arguments


def _load_shard_report(report_path: str, shard_index: int, num_shards: int) -> dict:
    path = Path(_shard_suffix(report_path, shard_index, num_shards))
    if not path.exists():
        raise FileNotFoundError(f"Missing shard report: {path} (did shard {shard_index} fail or not run yet?)")
    return json.loads(path.read_text())


def _load_shard_image(output_dir: str, shard_index: int, num_shards: int, global_index: int, branch: str) -> Image.Image:
    shard_dir = Path(_shard_suffix(output_dir, shard_index, num_shards))
    return Image.open(shard_dir / f"{global_index:02d}_{branch}.png")


def _validate_shard_coverage(reports: list[dict]) -> int:
    """Check every sample index is covered by exactly one shard; return the total count."""
    seen: set[int] = set()
    for report in reports:
        indices = set(report["global_indices"])
        overlap = seen & indices
        if overlap:
            raise ValueError(f"Sample indices {sorted(overlap)} appear in more than one shard report.")
        seen.update(indices)

    total = len(seen)
    expected = set(range(total))
    if seen != expected:
        missing = sorted(expected - seen)
        raise ValueError(f"Shard reports don't cover every sample: missing global indices {missing}.")
    return total


def main(args: MergeArguments) -> None:
    spec = MODEL_REGISTRY[args.model]
    reports = [_load_shard_report(args.report_path, i, args.num_shards) for i in range(args.num_shards)]
    total = _validate_shard_coverage(reports)

    samples = None
    if args.conditions_file is not None:
        samples = load_samples(BenchmarkArguments(model=args.model, conditions_file=args.conditions_file), spec)

    baseline_pool: list[Image.Image] = []
    accelerated_pool: list[Image.Image] = []
    prompt_pool: list[str] = []
    weighted_baseline_latency = 0.0
    weighted_accelerated_latency = 0.0

    for shard_index, report in enumerate(reports):
        n = report["num_samples"]
        weighted_baseline_latency += report["mean_baseline_latency_s"] * n
        weighted_accelerated_latency += report["mean_accelerated_latency_s"] * n
        for global_index in report["global_indices"]:
            baseline_image = _load_shard_image(args.output_dir, shard_index, args.num_shards, global_index, "baseline")
            accelerated_image = _load_shard_image(args.output_dir, shard_index, args.num_shards, global_index, "accelerated")
            baseline_pool.append(baseline_image)
            accelerated_pool.append(accelerated_image)
            if samples is not None:
                prompt_pool.append(samples[global_index].prompt)

    speedup_report = {
        "mean_baseline_latency_s": weighted_baseline_latency / total,
        "mean_accelerated_latency_s": weighted_accelerated_latency / total,
        "speedup": weighted_baseline_latency / weighted_accelerated_latency,
    }
    metrics = compute_quality_metrics(accelerated_pool, baseline_pool, prompt_pool or None, args.device)

    config = reports[0]["config"]
    print_report(
        BenchmarkArguments(model=args.model, order=config["order"], interval=config["interval"], warmup_steps=config["warmup_steps"]),
        total,
        speedup_report,
        metrics,
    )

    Path(args.report_path).write_text(
        json.dumps(
            {
                "model": args.model,
                "config": config,
                "num_samples": total,
                "num_shards": args.num_shards,
                **speedup_report,
                "clip_score": metrics.clip_score,
                "image_reward": metrics.image_reward,
                "psnr": metrics.psnr,
                "ssim": metrics.ssim,
                "lpips": metrics.lpips,
                "fid": metrics.fid,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main(parse_args())
