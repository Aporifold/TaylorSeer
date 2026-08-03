from ..config import TaylorSeerConfig


class ActivationScheduler:
    """Fixed-interval full-vs-predict decision strategy.

    - Warmup steps: Steps in `[0, warmup_steps)` are always full-compute steps.
    - Normal steps: After warmup, every `interval`-th step is an activation step.
    """

    def __init__(self, config: TaylorSeerConfig):
        self.config = config

    def should_activate(self, step: int) -> bool:
        """Decide whether `step` should be a full-compute or activation step.

        Args:
            step (int): Step index in denoising loop (0-based).

        Returns:
            bool: True if `step` should trigger a full compute (call `TaylorSeerCache.update`),
                False if it should be predicted instead (call `TaylorSeerCache.predict`).
        """
        if step < self.config.warmup_steps:
            return True
        return (step - self.config.warmup_steps) % self.config.interval == 0
