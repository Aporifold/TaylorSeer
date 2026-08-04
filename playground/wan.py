from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
from transformers.hf_argparser import HfArgumentParser

from taylorseer.adapters.wan import WanAdapter
from taylorseer.config import TaylorSeerConfig
from taylorseer.core.manager import CacheManager


@dataclass
class InferenceArguments:
    """Arguments for the Wan model."""

    prompt: str = field(default="A cat walks on the grass, realistic style")
    negative_prompt: str = field(default="")
    model_path: str = field(default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    height: int = field(default=480)
    width: int = field(default=832)
    num_frames: int = field(default=81)
    fps: int = field(default=16)
    num_inference_steps: int = field(default=50)
    guidance_scale: float = field(default=5.0)
    output_path: str = field(default="wan_output.mp4")
    enable_taylorseer: bool = field(default=False)
    order: int = field(default=1)
    interval: int = field(default=4)
    warmup_steps: int = field(default=1)


def get_pipeline(model_path: str) -> WanPipeline:
    """Load the Wan pipeline from the specified path."""
    vae = AutoencoderKLWan.from_pretrained(
        model_path, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanPipeline.from_pretrained(model_path, vae=vae, torch_dtype=torch.bfloat16)
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
        patch = WanAdapter().patch(pipe, manager)

    with patch:
        generator = torch.manual_seed(0)
        frames = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).frames[0]

    export_to_video(frames, args.output_path, fps=args.fps)
    print(f"Saved video to {args.output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
