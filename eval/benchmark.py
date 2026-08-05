"""Benchmark TaylorSeer's speedup/quality trade-off on FLUX against a DrawBench-style
prompt set.

Runs every sample twice (same seed) — once with the plain pipeline and once with
`CacheManager`/`FluxAdapter` patched in — and reports the latency speedup plus the
quality metrics in `metrics.py` (CLIP Score, ImageReward, PSNR/SSIM/LPIPS, FID)
comparing the two.

For multi-GPU data-parallel runs, use `scripts/run_benchmark_multigpu.sh`, which shards
`--conditions_file` across GPUs via `--shard_index`/`--num_shards` and merges the
resulting per-shard reports with `merge_shards.py`.

Run from the repo root, e.g.:
    python eval/benchmark.py --conditions_file data/drawbench.jsonl --num_samples 4
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from PIL import Image
from transformers.hf_argparser import HfArgumentParser

from metrics import QualityMetrics, compute_quality_metrics

from taylorseer.adapters.base import BaseAdapter
from taylorseer.adapters.flux import FluxAdapter
from taylorseer.config import TaylorSeerConfig
from taylorseer.core.manager import CacheManager

_DEFAULT_IMAGE_PROMPTS = [
    "A cat holding a sign that says hello world",
    "A photorealistic portrait of an astronaut on Mars",
    "A steaming cup of coffee on a wooden table, morning light",
    "A futuristic city skyline at sunset, digital art",
]


@dataclass
class Sample:
    """One benchmark item: a text prompt."""

    prompt: str

    def __str__(self) -> str:
        return self.prompt


@dataclass
class BenchmarkArguments:
    """Arguments for the TaylorSeer speedup/quality benchmark."""

    model: Literal["flux"] = field(default="flux")
    model_path: str | None = field(default=None)
    conditions_file: str | None = field(
        default=None,
        metadata={
            "help": 'One prompt per line, or a `.jsonl` file with one {"prompt": ...} object per '
            "line (e.g. data/drawbench.jsonl). Defaults to a small built-in set."
        },
    )
    num_samples: int | None = field(
        default=None,
        metadata={
            "help": "Max number of samples to run. Defaults to every sample in --conditions_file "
            "(or all 4 built-in examples if it's not given)."
        },
    )
    num_inference_steps: int | None = field(
        default=None, metadata={"help": "Defaults to the model's official setting."}
    )
    height: int | None = field(default=None)
    width: int | None = field(default=None)
    guidance_scale: float | None = field(default=None)
    order: int | None = field(
        default=None, metadata={"help": "Defaults to the model's official setting."}
    )
    interval: int | None = field(
        default=None, metadata={"help": "Defaults to the model's official setting."}
    )
    warmup_steps: int | None = field(
        default=None, metadata={"help": "Defaults to the model's official setting."}
    )
    max_frames_for_metrics: int = field(default=8)
    seed: int = field(
        default=0,
        metadata={
            "help": "Base seed. Each sample's actual seed is `seed + global_sample_index`."
        },
    )
    device: str = field(default="cuda")
    output_dir: str = field(default="benchmark_outputs")
    report_path: str = field(default="benchmark_report.json")
    shard_index: int = field(
        default=0,
        metadata={
            "help": "This process's shard index, in [0, num_shards). For data-parallel multi-GPU runs."
        },
    )
    num_shards: int = field(
        default=1,
        metadata={"help": "Total number of shards the sample list is split into."},
    )


@dataclass
class ModelSpec:
    """Per-model knowledge the benchmark loop needs: how to load it, drive it, and
    which adapter accelerates it — including the model's official sampling defaults
    (steps/guidance/resolution) and official TaylorSeer cache config
    (order/interval/warmup_steps), so a bare run reproduces the paper's setting for it
    rather than one generic default."""

    default_model_path: str
    adapter_cls: type[BaseAdapter]
    load_pipeline: Callable[[str, str], Any]
    generate: Callable[
        [Any, Sample, BenchmarkArguments, torch.Generator], list[Image.Image]
    ]
    default_samples: list[Sample]
    num_inference_steps: int
    order: int
    interval: int
    warmup_steps: int
    guidance_scale: float
    height: int
    width: int


def _load_flux(model_path: str, device: str) -> Any:
    from diffusers import FluxPipeline

    return FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(
        device
    )


def _generate_flux(
    pipe: Any, sample: Sample, args: BenchmarkArguments, generator: torch.Generator
) -> list[Image.Image]:
    return pipe(
        sample.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "flux": ModelSpec(
        default_model_path="black-forest-labs/FLUX.1-dev",
        adapter_cls=FluxAdapter,
        load_pipeline=_load_flux,
        generate=_generate_flux,
        default_samples=[Sample(prompt=prompt) for prompt in _DEFAULT_IMAGE_PROMPTS],
        # Matches TaylorSeer-FLUX's src/sample.py (flux-dev) and cache_functions/cache_init.py.
        num_inference_steps=50,
        order=1,
        interval=6,
        warmup_steps=3,
        guidance_scale=3.5,
        height=1024,
        width=1024,
    ),
}


def parse_args() -> BenchmarkArguments:
    """Parse command line arguments."""
    parser = HfArgumentParser((BenchmarkArguments,))
    (arguments,) = parser.parse_args_into_dataclasses(return_remaining_strings=False)
    return arguments


def load_samples(args: BenchmarkArguments, spec: ModelSpec) -> list[Sample]:
    """Read `--conditions_file`, or fall back to `spec.default_samples`.

    Supports plain text (one prompt per line) and `.jsonl` (one `{"prompt": ...}` object
    per line, e.g. `data/drawbench.jsonl`).
    """
    if args.conditions_file is None:
        samples = spec.default_samples
    else:
        lines = [
            line.strip()
            for line in Path(args.conditions_file).read_text().splitlines()
            if line.strip()
        ]
        if Path(args.conditions_file).suffix == ".jsonl":
            samples = [Sample(prompt=json.loads(line)["prompt"]) for line in lines]
        else:
            samples = [Sample(prompt=line) for line in lines]
    return samples[: args.num_samples]


def resolve_seed(base_seed: int, index: int) -> int:
    """Per-sample seed, matching the official repos' `base_seed + index` pattern
    (e.g. TaylorSeer-FLUX's `src/sample.py`)."""
    return base_seed + index


def _shard_suffix(path: str, shard_index: int, num_shards: int) -> str:
    """Append `_shard{shard_index}` to `path` so parallel shards never collide on the
    same output files. A no-op when `num_shards <= 1` (the default, single-process run).
    """
    if num_shards <= 1:
        return path
    p = Path(path)
    if p.suffix:
        return str(p.with_name(f"{p.stem}_shard{shard_index}{p.suffix}"))
    return str(p.with_name(f"{p.name}_shard{shard_index}"))


def sample_frames(frames: list[Image.Image], max_frames: int) -> list[Image.Image]:
    """Evenly sample at most `max_frames` frames, to bound per-frame metric cost."""
    if len(frames) <= max_frames:
        return frames
    step = len(frames) / max_frames
    indices = [int(i * step) for i in range(max_frames)]
    return [frames[i] for i in indices]


def save_output(frames: list[Image.Image], path: Path) -> None:
    """Save a single image."""
    frames[0].save(path.with_suffix(".png"))


def print_report(
    args: BenchmarkArguments,
    num_samples: int,
    speedup_report: dict,
    metrics: QualityMetrics,
) -> None:
    print()
    print(f"=== TaylorSeer benchmark: {args.model} ({num_samples} samples) ===")
    print(
        f"config: order={args.order} interval={args.interval} warmup_steps={args.warmup_steps}"
    )
    print(
        f"baseline latency:    {speedup_report['mean_baseline_latency_s']:.3f} s/sample"
    )
    print(
        f"accelerated latency: {speedup_report['mean_accelerated_latency_s']:.3f} s/sample"
    )
    print(f"speedup:             {speedup_report['speedup']:.2f}x")
    print("--- quality (accelerated vs. baseline) ---")
    if metrics.clip_score is not None:
        print(f"CLIP Score:  {metrics.clip_score:.2f}")
    if metrics.image_reward is not None:
        print(f"ImageReward: {metrics.image_reward:.3f}")
    if metrics.psnr is not None:
        print(f"PSNR:        {metrics.psnr:.2f} dB")
    if metrics.ssim is not None:
        print(f"SSIM:        {metrics.ssim:.3f}")
    if metrics.lpips is not None:
        print(f"LPIPS:       {metrics.lpips:.4f}")
    if metrics.fid is not None:
        print(f"FID:         {metrics.fid:.2f}")
    print()


def main(args: BenchmarkArguments) -> None:
    args.output_dir = _shard_suffix(args.output_dir, args.shard_index, args.num_shards)
    args.report_path = _shard_suffix(
        args.report_path, args.shard_index, args.num_shards
    )

    spec = MODEL_REGISTRY[args.model]
    model_path = args.model_path or spec.default_model_path

    # Resolve unset args against this model's official defaults (steps/guidance/
    # resolution/cache config), so a bare run reproduces the paper's setting.
    args.num_inference_steps = (
        args.num_inference_steps
        if args.num_inference_steps is not None
        else spec.num_inference_steps
    )
    args.guidance_scale = (
        args.guidance_scale if args.guidance_scale is not None else spec.guidance_scale
    )
    args.height = args.height if args.height is not None else spec.height
    args.width = args.width if args.width is not None else spec.width
    args.order = args.order if args.order is not None else spec.order
    args.interval = args.interval if args.interval is not None else spec.interval
    args.warmup_steps = (
        args.warmup_steps if args.warmup_steps is not None else spec.warmup_steps
    )

    samples = load_samples(args, spec)
    indexed_samples = list(enumerate(samples))
    shard_start = (args.shard_index * len(indexed_samples)) // args.num_shards
    shard_end = ((args.shard_index + 1) * len(indexed_samples)) // args.num_shards
    shard_samples = indexed_samples[shard_start:shard_end]

    pipeline = spec.load_pipeline(model_path, args.device)
    config = TaylorSeerConfig(
        order=args.order, interval=args.interval, warmup_steps=args.warmup_steps
    )
    adapter = spec.adapter_cls()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_pool: list[Image.Image] = []
    accelerated_pool: list[Image.Image] = []
    prompt_pool: list[str] = []
    global_indices: list[int] = []
    baseline_latency = 0.0
    accelerated_latency = 0.0

    for position, (global_index, sample) in enumerate(shard_samples):
        print(
            f"[shard {args.shard_index}/{args.num_shards}] [{position + 1}/{len(shard_samples)}] {sample}"
        )
        global_indices.append(global_index)

        # Same derived seed drives both passes of this sample (so the accelerated run
        # is comparable to its baseline), but each sample gets a different seed —
        # matching the official repos' `base_seed + sample_index` pattern rather than
        # reusing one global seed for every prompt in the dataset.
        seed = resolve_seed(args.seed, global_index)

        generator = torch.Generator(device=args.device).manual_seed(seed)
        start = time.perf_counter()
        baseline_frames = spec.generate(pipeline, sample, args, generator)
        baseline_latency += time.perf_counter() - start

        manager = CacheManager(config)
        generator = torch.Generator(device=args.device).manual_seed(seed)
        with adapter.patch(pipeline, manager):
            start = time.perf_counter()
            accelerated_frames = spec.generate(pipeline, sample, args, generator)
            accelerated_latency += time.perf_counter() - start

        save_output(baseline_frames, output_dir / f"{global_index:02d}_baseline")
        save_output(accelerated_frames, output_dir / f"{global_index:02d}_accelerated")

        baseline_sampled = sample_frames(baseline_frames, args.max_frames_for_metrics)
        accelerated_sampled = sample_frames(
            accelerated_frames, args.max_frames_for_metrics
        )
        baseline_pool.extend(baseline_sampled)
        accelerated_pool.extend(accelerated_sampled)
        prompt_pool.extend([sample.prompt] * len(accelerated_sampled))

    speedup_report = {
        "mean_baseline_latency_s": baseline_latency / len(shard_samples),
        "mean_accelerated_latency_s": accelerated_latency / len(shard_samples),
        "speedup": baseline_latency / accelerated_latency,
    }
    metrics = compute_quality_metrics(
        accelerated_pool,
        baseline_pool,
        prompt_pool or None,
        args.device,
    )

    print_report(args, len(shard_samples), speedup_report, metrics)
    Path(args.report_path).write_text(
        json.dumps(
            {
                "model": args.model,
                "config": {
                    "order": args.order,
                    "interval": args.interval,
                    "warmup_steps": args.warmup_steps,
                },
                "num_samples": len(shard_samples),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "global_indices": global_indices,
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
