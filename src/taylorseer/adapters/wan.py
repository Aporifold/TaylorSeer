from collections.abc import Callable

from diffusers import WanPipeline, WanTransformer3DModel
from torch import nn

from .base import BaseAdapter


class WanAdapter(BaseAdapter):
    """Adapter for Wan (single-stream, with classifier-free guidance)."""

    def cacheable_modules(self, pipeline: WanPipeline) -> dict[str, nn.Module]:
        """Returns a dictionary with all cacheable modules in Wan."""
        mp = {}
        model: WanTransformer3DModel = pipeline.transformer
        for idx, module in enumerate(model.blocks):
            mp[f"block{idx}.attn1"] = module.attn1
            mp[f"block{idx}.attn2"] = module.attn2
            mp[f"block{idx}.ffn"] = module.ffn
        return mp

    def step_fn(self, pipeline: WanPipeline) -> Callable[[], int]:
        """Derive the step index from `pipeline.current_timestep`.

        Wan defaults to classifier-free guidance, so a call-count trick
        would double-count steps across the cond/uncond branches.
        """
        state = {"step": -1, "last_timestep": None}

        def _step() -> int:
            timestep = pipeline.current_timestep
            if timestep is not state["last_timestep"]:
                state["step"] += 1
                state["last_timestep"] = timestep
            return state["step"]

        return _step

    def context_fn(self, pipeline: WanPipeline) -> Callable[[], str | None]:
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
