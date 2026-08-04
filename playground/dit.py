from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from transformers.hf_argparser import HfArgumentParser

from taylorseer.adapters.dit import DiTAdapter
from taylorseer.config import TaylorSeerConfig
from taylorseer.core.manager import CacheManager


@dataclass
class InferenceArguments:
    """Arguments for the DiT model."""

    class_label: int = field(default=207)  # ImageNet class id, 207 = golden retriever
    model_path: str = field(default="facebook/DiT-XL-2-256")
    num_inference_steps: int = field(default=50)
    output_path: str = field(default="dit_output.png")
    enable_taylorseer: bool = field(default=False)
    order: int = field(default=1)
    interval: int = field(default=4)
    warmup_steps: int = field(default=1)


def get_pipeline(model_path: str) -> DiTPipeline:
    """Load the DiT pipeline from the specified path."""
    pipe = DiTPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
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
        patch = DiTAdapter().patch(pipe, manager)

    with patch:
        generator = torch.manual_seed(0)
        image = pipe(
            class_labels=[args.class_label],
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        ).images[0]

    image.save(args.output_path)
    print(f"Saved image to {args.output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
