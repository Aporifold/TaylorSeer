from collections import defaultdict
from collections.abc import Callable

import torch

from ..config import TaylorSeerConfig
from .cache import TaylorSeerCache
from .scheduler import ActivationScheduler


class CacheManager:
    """TaylorSeer cache manager, which combines:

    - `TaylorSeerCache`: Store approximated derivatives at the latest activation step
        via finite differences for prediction using a Taylor expansion.
    - `ActivationScheduler`: Decide which step should be activated (full-compute).

    And only comprise one function `wrap(key, step, compute_fn)`,
    which provides a computed (activation) or estimated (predict) feature tensor.
    """

    def __init__(self, config: TaylorSeerConfig):
        self.config = config
        self.scheduler = ActivationScheduler(config)
        self.caches = defaultdict[str, TaylorSeerCache](
            lambda: TaylorSeerCache(config.order)
        )

    def wrap(
        self,
        key: str,
        step: int,
        compute_fn: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        """Run `compute_fn` or forecast with cached approximated derivates using Taylor expansion.

        Args:
            key (str): An unique identifier for each cacheable unit (e.g., `(layer_index, module_name)`)
            step (int): Current denoising step.
            compute_fn (Callable[[], torch.Tensor]): Zero-argument callable performing the real forward
                computation. Only invoked on activation or full-compute steps.

        Returns:
            torch.Tensor: The computed or predicted feature tensor.
        """
        cache = self.caches[key]
        if self.scheduler.should_activate(step):
            feature = compute_fn()
            cache.update(feature, step)
        else:
            feature = cache.predict(step)
        return feature
