from collections.abc import Callable

from diffusers import DiTPipeline, DiTTransformer2DModel
from torch import nn

from .base import BaseAdapter


class DiTAdapter(BaseAdapter):
    """Adapter for Diffusion Transformers (DiTs)."""

    def cacheable_modules(self, pipeline: DiTPipeline) -> dict[str, nn.Module]:
        """Returns a dictionary with all cacheable modules in DiT."""
        mp = {}
        model: DiTTransformer2DModel = pipeline.transformer
        for idx, module in enumerate(model.transformer_blocks):
            mp[f"block{idx}.attn1"] = module.attn1
            mp[f"block{idx}.ff"] = module.ff
            if module.attn2 is not None:
                mp[f"block{idx}.attn2"] = module.attn2
        return mp

    def step_fn(self, pipeline: DiTPipeline) -> Callable[[], int]:
        """Derive the step index (0-based) from how many cacheable calls happened."""
        calls_per_step = len(self.cacheable_modules(pipeline))
        state = {"calls": 0}

        def _step() -> int:
            step = state["calls"] // calls_per_step
            state["calls"] += 1
            return step

        return _step
