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
        """Derive the step index from `pipeline.current_timestep`.

        `true_cfg_scale > 1` (opt-in) adds a second uncond forward per
        iteration; `current_timestep` stays reliable either way.
        """
        state = {"step": -1, "last_timestep": None}

        def _step() -> int:
            timestep = pipeline.current_timestep
            if timestep is not state["last_timestep"]:
                state["step"] += 1
                state["last_timestep"] = timestep
            return state["step"]

        return _step

    def context_fn(self, pipeline: FluxPipeline) -> Callable[[], str | None]:
        """Distinguish the cond/uncond branches within one denoising iteration."""
        calls_per_branch = len(self.cacheable_modules(pipeline))
        state = {"calls_this_iteration": 0, "last_timestep": None}

        def _context() -> str | None:
            timestep = pipeline.current_timestep
            if timestep is not state["last_timestep"]:
                state["calls_this_iteration"] = 0
                state["last_timestep"] = timestep
            branch_index = state["calls_this_iteration"] // calls_per_branch
            state["calls_this_iteration"] += 1
            return "cond" if branch_index == 0 else "uncond"

        return _context
