"""Eager full-memory RelBench baseline implementations."""

from importlib import import_module
from typing import TYPE_CHECKING

from .config import TrainingConfig

if TYPE_CHECKING:
    from .dataset import LocalRelBenchDataset, LocalTask, open_dataset
    from .models import MODEL_REGISTRY, register_model


def __getattr__(name: str) -> object:
    modules = {
        "LocalRelBenchDataset": ".dataset",
        "LocalTask": ".dataset",
        "open_dataset": ".dataset",
        "MODEL_REGISTRY": ".models",
        "register_model": ".models",
    }
    if name in modules:
        return getattr(import_module(modules[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MODEL_REGISTRY",
    "LocalRelBenchDataset",
    "LocalTask",
    "TrainingConfig",
    "open_dataset",
    "register_model",
]
