from collections.abc import Callable

from diffusers import FluxPipeline, FluxTransformer2DModel
from torch import nn

from .base import BaseAdapter


class FluxAdapter(BaseAdapter):
    """Adapter wiring `FluxPipeline` into `CacheManager` via `BaseAdapter`."""

    def cacheable_modules(self, pipeline: FluxPipeline) -> dict[str, nn.Module]:
        """Returns a dictionary with all cacheable modules in FLUX."""
        mp = {}
        model: FluxTransformer2DModel = pipeline.transformer
        for idx, module in enumerate(model.transformer_blocks):
            mp[f"double_stream_block{idx}.attn"] = module.attn
            mp[f"double_stream_block{idx}.ff"] = module.ff
            mp[f"double_stream_block{idx}.ff_context"] = module.ff_context

        for idx, module in enumerate(model.single_transformer_blocks):
            mp[f"single_stream_block{idx}"] = module

        return mp

    def step_fn(self, pipeline: FluxPipeline) -> Callable[[], int]:
        """Derive the step index (0-based) from how many cacheable calls happened."""
        calls_per_step = len(self.cacheable_modules(pipeline))
        state = {"calls": 0}

        def _step() -> int:
            step = state["calls"] // calls_per_step
            state["calls"] += 1
            return step

        return _step
