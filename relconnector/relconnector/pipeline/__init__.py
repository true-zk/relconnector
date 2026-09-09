"""High level training facade built on RelBench + PyG."""

from .config import TrainingConfig
from .facade import RelBenchModel
from .models import MODEL_REGISTRY, register_model
from .result import TrainingRunResult
from .telemetry import TelemetryRecorder, timed

__all__ = [
    "MODEL_REGISTRY",
    "RelBenchModel",
    "TelemetryRecorder",
    "TrainingConfig",
    "TrainingRunResult",
    "register_model",
    "timed",
]
