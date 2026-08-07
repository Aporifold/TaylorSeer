import torch

Feature = torch.Tensor | tuple[torch.Tensor, ...]


def _elementwise(fn, *features: Feature) -> Feature:
    """Apply `fn` to `features` directly, or per-position if they're tuples.

    Lets a single module whose forward returns multiple tensors (e.g. a
    fused joint-attention call producing separate per-stream outputs) be
    cached as one unit, without every call site needing to know whether
    `feature` is a plain tensor or a tuple of tensors.
    """
    if isinstance(features[0], tuple):
        return tuple(fn(*xs) for xs in zip(*features))
    return fn(*features)


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
        self.cache: dict[int, Feature] = {}

        # step index of the most recent update
        self.last_step: int | None = None

    def update(self, feature: Feature, step: int):
        """Refresh the cache at a full-compute step recursively.

        Args:
            feature (Feature): The feature from real forward computation at `step` —
                a tensor, or a tuple of tensors for a module with multiple outputs.
            step (int): The current timestep index.
        """
        old_cache = self.cache
        new_cache = {0: feature}
        if self.last_step is not None:
            # compute approximate derivatives by finite difference
            distance = step - self.last_step
            for i in range(1, self.order + 1):
                if i - 1 in old_cache:
                    new_cache[i] = _elementwise(
                        lambda new, old: (new - old) / distance,
                        new_cache[i - 1],
                        old_cache[i - 1],
                    )
        # update cache
        self.cache = new_cache
        self.last_step = step

    def predict(self, step: int) -> Feature:
        """Reconstruct the feature at a skipped step via Taylor expansion.

        Args:
            step (int): The timestep index to predict the feature for.

        Returns:
            Feature: An approximated feature, matching the shape (tensor or
                tuple of tensors) of what was passed to `update`.
        """
        distance = step - self.last_step
        # Run taylor expansion sum in fp32, preventing overflow (inf).
        orig = self.cache[0]
        pred = _elementwise(lambda x: x.float(), orig)
        fact_i, dist_i = 1, distance
        for i in range(1, self.order + 1):
            if i not in self.cache:
                break
            fact_i *= i
            term = _elementwise(lambda x: x.float(), self.cache[i])
            pred = _elementwise(lambda p, c: p + c / fact_i * dist_i, pred, term)
            dist_i *= distance
        return _elementwise(lambda p, o: p.to(o.dtype), pred, orig)
