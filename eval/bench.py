import platform
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch
from benchlog import RunLogWriter, Sample, load_samples
from diffusers import FluxPipeline
from PIL import Image
from transformers.hf_argparser import HfArgumentParser

from taylorseer import CacheManager, FluxAdapter, TaylorSeerConfig

__all__ = ["Sample", "load_samples", "main"]

METHODS = ("base", "taylorseer")


@dataclass
class ModelArguments:
    """Arguments for model configuration.

    - model_path (str): Model name or path to the pretrained model.
    - device (str): Device to run the benchmark on. Can be "cuda" or "cpu".
    """

    model_path: str = field(default="black-forest-labs/FLUX.1-dev")
    device: str = field(default="cuda")


@dataclass
class GenerationArguments:
    """Generation config.

    - height (int): Height of the generated image.
    - width (int): Width of the generated image.
    - num_inference_steps (int): Number of denoising steps.
    - guidance_scale (float): Classifier-free guidance scale.
    """

    height: int = field(default=1024)
    width: int = field(default=1024)
    num_inference_steps: int = field(default=50)
    guidance_scale: float = field(default=3.5)


@dataclass
class TaylorSeerArguments:
    """TaylorSeer config. Only used when `--method taylorseer`.

    - order (int): Maximum order of Taylor expansion.
    - interval (int): Number of steps between two adjacent activation steps.
    - warmup_steps (int): Number of initial steps to run with full-compute before forecasting.
    """

    order: int = field(default=2)
    interval: int = field(default=4)
    warmup_steps: int = field(default=3)


@dataclass
class BenchmarkArguments:
    """Benchmark config.

    - method (str): Which method to benchmark, either "base" (no cache, the
      reference run) or "taylorseer". Exactly one method runs per invocation, so
      sweeping TaylorSeer hyperparameters does not re-run the baseline.
    - data_path (str): Path to the prompt file.
    - output_dir (str): Directory to save the image outputs and run logs.
    - run_name (str | None): Name of this run, used for the image subdirectory
      and the log file name. Defaults to "base" / "taylorseer_o{order}_i{interval}_w{warmup_steps}".
    - log_file (str | None): Path to the JSONL run log. Defaults to
      "<output_dir>/logs/<run_name>.jsonl"; a ".chunk{i}of{n}" infix is added
      when running data-parallel (`--num_chunks > 1`).
    - seed (int): Base random seed for reproducibility. Sample `i` uses `seed + i`.
    - num_samples (int | None): Number of samples to evaluate. If None, evaluate all samples.
    - num_chunks (int): Number of chunks to split the samples into for parallel evaluation.
    - chunk_idx (int): Index of the current chunk to evaluate.
    """

    method: Literal["base", "taylorseer"] = field(default="taylorseer")
    data_path: str = field(default="data/drawbench.jsonl")
    output_dir: str = field(default="outputs")
    run_name: str | None = field(default=None)
    log_file: str | None = field(default=None)
    seed: int = field(default=0)
    num_samples: int | None = field(default=None)
    num_chunks: int = field(default=1)
    chunk_idx: int = field(default=0)


def load_model(model_path: str, device: str):
    return FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)


def generate(
    pipe: FluxPipeline,
    prompt: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    generator: torch.Generator | None = None,
) -> list[Image.Image]:
    return pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images


def parse_args():
    parser = HfArgumentParser(
        (
            ModelArguments,
            GenerationArguments,
            TaylorSeerArguments,
            BenchmarkArguments,
        )
    )
    model_args, generation_args, taylorseer_args, benchmark_args = (
        parser.parse_args_into_dataclasses(return_remaining_strings=False)
    )
    return model_args, generation_args, taylorseer_args, benchmark_args


def resolve_seed(base_seed: int, sample_index: int) -> int:
    return base_seed + sample_index


def resolve_run_name(
    bench_args: BenchmarkArguments, ts_args: TaylorSeerArguments
) -> str:
    """Name of this run: explicit `--run_name`, else derived from the method's hyperparameters."""
    if bench_args.run_name:
        return bench_args.run_name
    if bench_args.method == "base":
        return "base"
    return f"taylorseer_o{ts_args.order}_i{ts_args.interval}_w{ts_args.warmup_steps}"


def resolve_log_files(
    bench_args: BenchmarkArguments, run_name: str
) -> tuple[Path, Path]:
    """Return `(chunk_log_file, merged_log_file)` for this worker.

    Every worker owns its own log file so that data-parallel chunks never
    interleave writes; `merged_log_file` is where the launcher is expected to
    concatenate them (and equals the chunk file for a single-chunk run).
    """
    merged = (
        Path(bench_args.log_file)
        if bench_args.log_file
        else Path(bench_args.output_dir) / "logs" / f"{run_name}.jsonl"
    )
    if bench_args.num_chunks <= 1:
        return merged, merged
    chunk = merged.with_name(
        f"{merged.stem}.chunk{bench_args.chunk_idx}of{bench_args.num_chunks}{merged.suffix}"
    )
    return chunk, merged


def build_meta(
    model_args: ModelArguments,
    gen_args: GenerationArguments,
    ts_args: TaylorSeerArguments,
    bench_args: BenchmarkArguments,
    run_name: str,
    image_dir: Path,
    num_chunk_samples: int,
) -> dict[str, Any]:
    """The `meta` record of the run log: everything needed to reproduce the run."""
    return {
        "method": bench_args.method,
        "run_name": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "model_path": model_args.model_path,
            "device": model_args.device,
            "dtype": "bfloat16",
            "gpu": _device_name(model_args.device),
        },
        "generation": {
            "height": gen_args.height,
            "width": gen_args.width,
            "num_inference_steps": gen_args.num_inference_steps,
            "guidance_scale": gen_args.guidance_scale,
        },
        # Only meaningful for the accelerated method; `None` for the baseline.
        "taylorseer": (
            {
                "order": ts_args.order,
                "interval": ts_args.interval,
                "warmup_steps": ts_args.warmup_steps,
            }
            if bench_args.method == "taylorseer"
            else None
        ),
        "benchmark": {
            "data_path": bench_args.data_path,
            "output_dir": bench_args.output_dir,
            "image_dir": str(image_dir),
            "seed": bench_args.seed,
            "num_samples": bench_args.num_samples,
            "num_chunks": bench_args.num_chunks,
            "chunk_idx": bench_args.chunk_idx,
            "num_chunk_samples": num_chunk_samples,
        },
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }


def _is_cuda(device: str) -> bool:
    return torch.device(device).type == "cuda" and torch.cuda.is_available()


def _device_name(device: str) -> str | None:
    if not _is_cuda(device):
        return None
    return torch.cuda.get_device_name(torch.device(device))


def _synchronize(device: str) -> None:
    """Make sure queued CUDA work is done before reading the clock."""
    if _is_cuda(device):
        torch.cuda.synchronize(torch.device(device))


def _peak_memory_gb(device: str) -> float | None:
    if not _is_cuda(device):
        return None
    return torch.cuda.max_memory_allocated(torch.device(device)) / 1024**3


def main(
    model_args: ModelArguments,
    gen_args: GenerationArguments,
    taylorseer_args: TaylorSeerArguments,
    bench_args: BenchmarkArguments,
):
    if bench_args.method not in METHODS:
        raise ValueError(
            f"unknown --method {bench_args.method!r}, expected one of {METHODS}"
        )

    # load test samples
    samples = load_samples(bench_args.data_path)
    if bench_args.num_samples is not None:
        samples = samples[: bench_args.num_samples]
    indexes = list(range(len(samples)))
    samples = samples[bench_args.chunk_idx :: bench_args.num_chunks]
    indexes = indexes[bench_args.chunk_idx :: bench_args.num_chunks]

    # resolve output layout: one image subdirectory and one log file per run
    run_name = resolve_run_name(bench_args, taylorseer_args)
    image_dir = Path(bench_args.output_dir) / run_name
    image_dir.mkdir(parents=True, exist_ok=True)
    log_file, merged_log_file = resolve_log_files(bench_args, run_name)

    prefix = f"[chunk {bench_args.chunk_idx}]"
    print(f"{prefix} method={bench_args.method} run_name={run_name}")
    print(f"{prefix} image_dir={image_dir}")
    # Parsed by scripts/run_bench.sh to merge the per-chunk logs; keep the format.
    print(f"{prefix} log_file={log_file}")
    print(f"{prefix} merged_log_file={merged_log_file}")

    # load model, and set TaylorSeer up when it is the benchmarked method
    pipe = load_model(model_args.model_path, model_args.device)
    cfg = TaylorSeerConfig(
        order=taylorseer_args.order,
        interval=taylorseer_args.interval,
        warmup_steps=taylorseer_args.warmup_steps,
    )
    adapter = FluxAdapter() if bench_args.method == "taylorseer" else None

    meta = build_meta(
        model_args,
        gen_args,
        taylorseer_args,
        bench_args,
        run_name=run_name,
        image_dir=image_dir,
        num_chunk_samples=len(samples),
    )
    latencies: list[float] = []

    # main benchmark loop
    with RunLogWriter(log_file, meta) as log:
        for idx, (global_idx, sample) in enumerate(zip(indexes, samples)):
            print(
                f"{prefix} sample {idx + 1}/{len(samples)} (#{global_idx}): {sample.prompt}"
            )

            seed = resolve_seed(bench_args.seed, global_idx)  # base seed + global index
            generator = torch.Generator(device=model_args.device).manual_seed(seed)
            if _is_cuda(model_args.device):
                torch.cuda.reset_peak_memory_stats(torch.device(model_args.device))

            # A fresh CacheManager (and a fresh patch, whose step counter is
            # created on enter) per sample, so no state leaks across samples.
            context = (
                adapter.patch(pipe, CacheManager(cfg))
                if adapter is not None
                else nullcontext()
            )
            with context:
                _synchronize(model_args.device)
                start = time.perf_counter()
                images = generate(
                    pipe,
                    prompt=sample.prompt,
                    height=gen_args.height,
                    width=gen_args.width,
                    num_inference_steps=gen_args.num_inference_steps,
                    guidance_scale=gen_args.guidance_scale,
                    generator=generator,
                )
                _synchronize(model_args.device)
                latency = time.perf_counter() - start

            # save the output image (FLUX returns one image per prompt here)
            image_path = image_dir / f"{global_idx:04d}.png"
            images[0].save(image_path)

            latencies.append(latency)
            log.write_sample(
                index=global_idx,
                chunk_idx=bench_args.chunk_idx,
                prompt=sample.prompt,
                seed=seed,
                latency=latency,
                image_path=str(image_path),
                peak_memory_gb=_peak_memory_gb(model_args.device),
                metadata=sample.metadata,
            )
            print(
                f"{prefix} sample #{global_idx} latency={latency:.2f}s -> {image_path}"
            )

    # report this chunk's latency; the speedup against the baseline is computed
    # by eval/compute_metrics.py from the two runs' logs.
    total = sum(latencies)
    mean = total / len(latencies) if latencies else 0.0
    print(f"{prefix} done: {len(latencies)} sample(s) of method '{bench_args.method}'")
    print(f"{prefix} total latency: {total:.2f}s, mean latency: {mean:.2f}s")
    print(f"{prefix} log written to {log_file}")


if __name__ == "__main__":
    model_args, gen_args, taylorseer_args, bench_args = parse_args()
    main(
        model_args=model_args,
        gen_args=gen_args,
        taylorseer_args=taylorseer_args,
        bench_args=bench_args,
    )
