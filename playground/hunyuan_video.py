from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
from transformers.hf_argparser import HfArgumentParser

from taylorseer.adapters.hunyuan_video import HunyuanVideoAdapter
from taylorseer.config import TaylorSeerConfig
from taylorseer.core.manager import CacheManager


@dataclass
class InferenceArguments:
    """Arguments for the HunyuanVideo model."""

    prompt: str = field(default="A cat walks on the grass, realistic style")
    model_path: str = field(default="hunyuanvideo-community/HunyuanVideo")
    height: int = field(default=320)
    width: int = field(default=512)
    num_frames: int = field(default=61)
    fps: int = field(default=15)
    num_inference_steps: int = field(default=50)
    output_path: str = field(default="hunyuan_video_output.mp4")
    enable_taylorseer: bool = field(default=False)
    order: int = field(default=1)
    interval: int = field(default=4)
    warmup_steps: int = field(default=1)


def get_pipeline(model_path: str) -> HunyuanVideoPipeline:
    """Load the HunyuanVideo pipeline from the specified path."""
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    pipe = HunyuanVideoPipeline.from_pretrained(
        model_path, transformer=transformer, torch_dtype=torch.float16
    )
    pipe.vae.enable_tiling()
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
        patch = HunyuanVideoAdapter().patch(pipe, manager)

    with patch:
        generator = torch.manual_seed(0)
        frames = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        ).frames[0]

    export_to_video(frames, args.output_path, fps=args.fps)
    print(f"Saved video to {args.output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
