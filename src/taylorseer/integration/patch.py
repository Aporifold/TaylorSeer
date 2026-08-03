from collections.abc import Callable
from contextlib import contextmanager

from torch import nn

from ..core.manager import CacheManager


@contextmanager
def patch_forward(
    targets: dict[str, nn.Module],
    manager: CacheManager,
    step_fn: Callable[[], int],
):
    originals: dict[str, Callable] = {}
    try:
        for key, module in targets.items():
            original = module.forward
            originals[key] = original

            def wrapped(*args, _key=key, _original=original, **kwargs):
                return manager.wrap(_key, step_fn(), lambda: _original(*args, **kwargs))

            module.forward = wrapped
        yield
    finally:
        for key, module in targets.items():
            module.forward = originals[key]
