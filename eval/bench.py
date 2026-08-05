import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from diffusers import FluxPipeline
from PIL import Image
from transformers.hf_argparser import HfArgumentParser

from taylorseer import CacheManager, FluxAdapter, TaylorSeerConfig


@dataclass
class Sample:
    """A single test sample, only containing a text prompt."""

    prompt: str

    def __str__(self) -> str:
        return self.prompt


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
    """TaylorSeer config.

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

    - data_path (str): Path to the prompt file.
    - output_dir (str): Directory to save the image outputs.
    - seed (int): Base random seed for reproducibility.
    - log_file (str): Path to the log file.
    - num_samples (int | None): Number of samples to evaluate. If None, evaluate all samples.
    - num_chunks (int): Number of chunks to split the samples into for parallel evaluation.
    - chunk_idx (int): Index of the current chunk to evaluate.
    """

    data_path: str = field(default="data/drawbench.jsonl")
    output_dir: str = field(default="outputs")
    seed: int = field(default=0)
    log_file: str = field(default="benchmark.log")
    num_samples: int | None = field(default=None)
    num_chunks: int = field(default=1)
    chunk_idx: int = field(default=0)


def load_model(model_path: str, device: str):
    return FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
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


def load_samples(data_path: str) -> list[Sample]:
    samples: list[Sample] = []
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            samples.append(Sample(prompt=data["prompt"]))
    return samples


def resolve_seed(base_seed: int, sample_index: int) -> int:
    return base_seed + sample_index


def main(
    model_args: ModelArguments,
    gen_args: GenerationArguments,
    taylorseer_args: TaylorSeerArguments,
    bench_args: BenchmarkArguments,
):
    # load test samples
    samples = load_samples(bench_args.data_path)
    if bench_args.num_samples is not None:
        samples = samples[: bench_args.num_samples]
    indexes = list(range(len(samples)))
    samples = samples[bench_args.chunk_idx :: bench_args.num_chunks]
    indexes = indexes[bench_args.chunk_idx :: bench_args.num_chunks]

    # load model
    pipe = load_model(model_args.model_path, model_args.device)
    cfg = TaylorSeerConfig(
        order=taylorseer_args.order,
        interval=taylorseer_args.interval,
        warmup_steps=taylorseer_args.warmup_steps,
    )
    adapter = FluxAdapter()

    baseline_latency = 0.0
    taylorseer_latency = 0.0
    output_dir = Path(bench_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # main benchmark loop
    for idx, (global_idx, sample) in enumerate(zip(indexes, samples)):
        print(
            f"[Chunk {bench_args.chunk_idx}] Sample {idx + 1}/{len(samples)}: {sample.prompt}"
        )

        seed = resolve_seed(bench_args.seed, global_idx)  # base seed + global index

        # (1) run baseline
        generator = torch.Generator(device=model_args.device).manual_seed(seed)
        start = time.perf_counter()
        baseline_images = generate(
            pipe,
            prompt=sample.prompt,
            height=gen_args.height,
            width=gen_args.width,
            num_inference_steps=gen_args.num_inference_steps,
            guidance_scale=gen_args.guidance_scale,
            generator=generator,
        )
        end = time.perf_counter()
        baseline_latency += end - start

        # (2) run TaylorSeer
        manager = CacheManager(cfg)
        generator = torch.Generator(device=model_args.device).manual_seed(seed)
        with adapter.patch(pipe, manager):
            start = time.perf_counter()
            taylorseer_images = generate(
                pipe,
                prompt=sample.prompt,
                height=gen_args.height,
                width=gen_args.width,
                num_inference_steps=gen_args.num_inference_steps,
                guidance_scale=gen_args.guidance_scale,
                generator=generator,
            )
            end = time.perf_counter()
            taylorseer_latency += end - start

        # save outputs
        for i, (baseline_img, taylorseer_img) in enumerate(
            zip(baseline_images, taylorseer_images)
        ):
            baseline_img.save(output_dir / f"{global_idx:02d}_baseline_{i:02d}.png")
            taylorseer_img.save(output_dir / f"{global_idx:02d}_taylorseer_{i:02d}.png")

    # report speedup
    speedup = (
        baseline_latency / taylorseer_latency
        if taylorseer_latency > 0
        else float("inf")
    )
    print(f"[Chunk {bench_args.chunk_idx}] Total samples: {len(samples)}")
    print(f"[Chunk {bench_args.chunk_idx}] Baseline: {baseline_latency:.2f}s")
    print(f"[Chunk {bench_args.chunk_idx}] TaylorSeer: {taylorseer_latency:.2f}s")
    print(f"[Chunk {bench_args.chunk_idx}] Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    model_args, gen_args, taylorseer_args, bench_args = parse_args()
    main(
        model_args=model_args,
        gen_args=gen_args,
        taylorseer_args=taylorseer_args,
        bench_args=bench_args,
    )
