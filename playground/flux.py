from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
from diffusers import FluxPipeline
from transformers.hf_argparser import HfArgumentParser

from taylorseer.adapters.flux import FluxAdapter
from taylorseer.config import TaylorSeerConfig
from taylorseer.core.manager import CacheManager


@dataclass
class InferenceArguments:
    """Arguments for the FLUX model."""

    prompt: str = field(default="A cat holding a sign that says hello world")
    model_path: str = field(default="black-forest-labs/FLUX.1-dev")
    height: int = field(default=1024)
    width: int = field(default=1024)
    num_inference_steps: int = field(default=28)
    guidance_scale: float = field(default=3.5)
    output_path: str = field(default="flux_output.png")
    enable_taylorseer: bool = field(default=False)
    order: int = field(default=1)
    interval: int = field(default=4)
    warmup_steps: int = field(default=1)


def get_pipeline(model_path: str) -> FluxPipeline:
    """Load the FLUX pipeline from the specified path."""
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    return pipe


def parse_args() -> InferenceArguments:
    """Parse command line arguments."""
    parser = HfArgumentParser((InferenceArguments,))
    (arguments,) = parser.parse_args_into_dataclasses(return_remaining_strings=False)
    return arguments


def main(args: InferenceArguments):
    pipe = get_pipeline(args.model_path)

    patch = nullcontext()
    if args.enable_taylorseer:
        # ! Apply TaylorSeer here.
        config = TaylorSeerConfig(
            order=args.order, interval=args.interval, warmup_steps=args.warmup_steps
        )
        manager = CacheManager(config)
        patch = FluxAdapter().patch(pipe, manager)

    with patch:
        generator = torch.manual_seed(0)
        image = pipe(
            args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

    image.save(args.output_path)
    print(f"Saved image to {args.output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
