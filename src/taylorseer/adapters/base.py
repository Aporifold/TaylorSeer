from abc import ABC, abstractmethod
from collections.abc import Callable

from torch import nn

from ..core import CacheManager
from ..integration.patch import patch_forward


class BaseAdapter(ABC):
    """Base class for model-specific adaptation.

    An adapter contains two functions:
    - `cacheable_modules`: Specify which modules are cacheable or monkey-patched.
    - `step_fn`: Get the pipeline's current denoising step.
    """

    @abstractmethod
    def cacheable_modules(self, pipeline) -> dict[str, nn.Module]:
        """Return a dictionary of all cacheable submodules in `pipeline`.

        Args:
            pipeline: The real model or pipeline instance to introspect.

        Returns:
            dict[str, nn.Module]: Maps a cache key (e.g., `(layer_idx, module_name)`)
                to the submodule whose `forward` should be wrapped.
        """

    @abstractmethod
    def step_fn(self, pipeline) -> Callable[[], int]:
        """Return a zero-arg callable exposing `pipeline`'s current step index.

        Args:
            pipeline: The real model or pipeline instance to introspect.

        Returns:
            Callable[[], int]: Reads the current denoising step whenever called,
                for use by patched forwards while `pipeline` is running.
        """

    def context_fn(self, pipeline) -> Callable[[], str | None]:
        """Return a zero-arg callable exposing the current conditioning branch.

        Override for models whose denoising loop runs multiple independent
        conditioning branches per step (e.g. cond/uncond under classifier-free
        guidance), so each branch gets its own cache instead of sharing one.

        Args:
            pipeline: The real model or pipeline instance to introspect.

        Returns:
            Callable[[], str | None]: Reads the current branch name whenever
                called. Default: always `None` (one shared cache namespace).
        """
        return lambda: None

    def patch(self, pipeline, manager: CacheManager):
        """Patch every cacheable module of `pipeline` through `manager`.

        Args:
            pipeline: The real model or pipeline instance to patch.
            manager (CacheManager): Decides full-compute vs. predicted steps.

        Returns:
            AbstractContextManager: Restores the original `forward`s on exit.
        """
        return patch_forward(
            self.cacheable_modules(pipeline),
            manager,
            self.step_fn(pipeline),
            self.context_fn(pipeline),
        )
