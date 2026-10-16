"""Benchmark orchestration and cross-implementation observability."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .baseline import BaselineExperiment
    from .online import OnlineDataConfig, OnlineDataExperiment, OnlineDataRunResult
    from .result import TrainingRunResult
    from .telemetry import TelemetryRecorder, timed


def __getattr__(name: str) -> object:
    modules = {
        "BaselineExperiment": ".baseline",
        "OnlineDataConfig": ".online",
        "OnlineDataExperiment": ".online",
        "OnlineDataRunResult": ".online",
        "TrainingRunResult": ".result",
        "TelemetryRecorder": ".telemetry",
        "timed": ".telemetry",
    }
    if name in modules:
        return getattr(import_module(modules[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaselineExperiment",
    "OnlineDataConfig",
    "OnlineDataExperiment",
    "OnlineDataRunResult",
    "TelemetryRecorder",
    "TrainingRunResult",
    "timed",
]
