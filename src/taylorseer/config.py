from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaylorSeerConfig:
    """Hyperparameters of TaylorSeer.

    Attributes:
        order: The maximum order `O` of Taylor expansion in each `TaylorSeerCache`.
            `order=0` degenerates to naive feature reuse.
        interval: The number of steps `N` between two activation (i.e., full-compute) steps,
            in steady state (i.e., after warmup steps).
        warmup_steps: The number of steps at the begining of the denoising loop
            that are always forced to be activation steps.
    """

    order: int = field(default=1)
    interval: int = field(default=4)
    warmup_steps: int = field(default=1)

    def __post_init__(self) -> None:
        if self.order < 0:
            raise ValueError(f"order must be >= 0, got {self.order}")
        if self.interval < 1:
            raise ValueError(f"interval must be >= 1, got {self.interval}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
