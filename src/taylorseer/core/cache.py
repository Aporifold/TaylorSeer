import torch


class TaylorSeerCache:
    """Taylor-derivative pyramid cache for a single cacheable module (e.g., self-attn, mlp).

    Implement *cache-then-forecast* mechanism in TaylorSeer.
    - `update(feature: Tensor, step: int)`: Refresh the cache by finite difference at a full-compute step.
    - `predict(step: int) -> Tensor`: Reconstruct the feature at a skipped step via Taylor expansion.
    """

    def __init__(self, order: int):
        """
        Args:
            order: The max order of Taylor expansion to maintain (>=0).
                `order=0` degenerate to reusing the last cached feature.
        """
        self.order = order

        # order to approximate derivative map
        self.cache: dict[int, torch.Tensor] = {}

        # step index of the most recent update
        self.last_step: int | None = None

    def update(self, feature: torch.Tensor, step: int):
        """Refresh the cache at a full-compute step recursively.

        Args:
            feature (torch.Tensor): The feature tensor from real forward computation at `step`.
            step (int): The current timestep index.
        """
        old_cache = self.cache
        new_cache = {0: feature}
        if self.last_step is not None:
            # compute approximate derivatives by finite difference
            distance = step - self.last_step
            for i in range(1, self.order + 1):
                if i - 1 in old_cache:
                    new_cache[i] = (new_cache[i - 1] - old_cache[i - 1]) / distance
        # update cache
        self.cache = new_cache
        self.last_step = step

    def predict(self, step: int) -> torch.Tensor:
        """Reconstruct the feature at a skipped step via Taylor expansion.

        Args:
            step (int): The timestep index to predict the feature for.

        Returns:
            torch.Tensor: An approximated feature tensor
        """
        distance = step - self.last_step
        pred = self.cache[0]
        fact_i, dist_i = 1, distance
        for i in range(1, self.order + 1):
            if i not in self.cache:
                break
            pred = pred + self.cache[i] / fact_i * dist_i
            fact_i, dist_i = fact_i * i, dist_i * distance
        return pred
