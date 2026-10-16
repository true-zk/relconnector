"""Reusable wall-time, host-memory, and CUDA-memory instrumentation."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import os
import platform
import random
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Generator
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import ParamSpec, TypedDict, TypeVar

P = ParamSpec("P")
R = TypeVar("R")
_ACTIVE_RECORDER: contextvars.ContextVar[TelemetryRecorder | None] = (
    contextvars.ContextVar("relconnector_telemetry", default=None)
)


@dataclass(frozen=True)
class ResourceSnapshot:
    rss_mb: float
    system_used_mb: float
    cuda_allocated_mb: float
    cuda_reserved_mb: float


@dataclass(frozen=True)
class PhaseRecord:
    name: str
    duration_s: float
    before: ResourceSnapshot
    after: ResourceSnapshot
    peak: ResourceSnapshot


class ResourceSnapshotData(TypedDict):
    rss_mb: float
    system_used_mb: float
    cuda_allocated_mb: float
    cuda_reserved_mb: float


class PhaseData(TypedDict):
    name: str
    duration_s: float
    before: ResourceSnapshotData
    after: ResourceSnapshotData
    peak: ResourceSnapshotData


class OperationStats(TypedDict):
    count: int
    total_s: float
    mean_s: float
    p50_s: float
    p95_s: float
    p99_s: float
    per_second: float
    percentile_samples: int
    percentile_method: str


class OverallTelemetry(TypedDict):
    duration_s: float
    before: ResourceSnapshotData
    after: ResourceSnapshotData
    peak: ResourceSnapshotData


class EnvironmentData(TypedDict):
    python: str
    platform: str
    torch: str
    cuda_build: str | None
    cuda_available: bool
    cuda_device: str | None


class TelemetryReport(TypedDict):
    environment: EnvironmentData
    overall: OverallTelemetry
    phases: list[PhaseData]
    operations: dict[str, OperationStats]
    phase_records_dropped: int


class _OperationAccumulator:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.count = 0
        self.total = 0.0
        self.samples: list[float] = []
        self.rng = random.Random(42)

    def add(self, duration: float) -> None:
        self.count += 1
        self.total += duration
        if len(self.samples) < self.capacity:
            self.samples.append(duration)
        else:
            index = self.rng.randrange(self.count)
            if index < self.capacity:
                self.samples[index] = duration


class _ResourceSampler:
    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self.peak = _snapshot()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> ResourceSnapshot:
        self._stop.set()
        self._thread.join()
        self._update(_snapshot())
        return self.peak

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._update(_snapshot())

    def _update(self, point: ResourceSnapshot) -> None:
        self.peak = ResourceSnapshot(
            rss_mb=max(self.peak.rss_mb, point.rss_mb),
            system_used_mb=max(self.peak.system_used_mb, point.system_used_mb),
            cuda_allocated_mb=max(self.peak.cuda_allocated_mb, point.cuda_allocated_mb),
            cuda_reserved_mb=max(self.peak.cuda_reserved_mb, point.cuda_reserved_mb),
        )


class TelemetryRecorder:
    """Collect hierarchical phase measurements and repeated operation timings."""

    def __init__(
        self,
        *,
        sample_interval_s: float = 0.05,
        max_operation_samples: int = 4096,
        max_phase_records: int = 4096,
    ) -> None:
        if sample_interval_s <= 0 or min(max_operation_samples, max_phase_records) <= 0:
            raise ValueError("sampling interval and retention limits must be positive")
        self.sample_interval_s = sample_interval_s
        self.max_operation_samples = max_operation_samples
        self.phases: deque[PhaseRecord] = deque(maxlen=max_phase_records)
        self.operations: dict[str, _OperationAccumulator] = {}
        self._lock = threading.Lock()
        self._phase_records_dropped = 0
        self._overall_sampler: _ResourceSampler | None = None
        self._overall_before: ResourceSnapshot | None = None
        self._overall_after: ResourceSnapshot | None = None
        self._overall_peak: ResourceSnapshot | None = None
        self._overall_started: float | None = None
        self._overall_duration_s: float | None = None

    @contextlib.contextmanager
    def activate(self) -> Generator[TelemetryRecorder, None, None]:
        token = _ACTIVE_RECORDER.set(self)
        _reset_cuda_peak_memory()
        self._overall_before = _snapshot()
        self._overall_started = time.perf_counter()
        self._overall_sampler = _ResourceSampler(self.sample_interval_s)
        self._overall_sampler.start()
        try:
            yield self
        finally:
            self._overall_after = _snapshot()
            self._overall_peak = _with_cuda_allocator_peak(self._overall_sampler.stop())
            self._overall_duration_s = time.perf_counter() - self._overall_started
            _ACTIVE_RECORDER.reset(token)

    @contextlib.contextmanager
    def phase(self, name: str) -> Generator[None, None, None]:
        _synchronize_cuda()
        before = _snapshot()
        sampler = _ResourceSampler(self.sample_interval_s)
        sampler.start()
        started = time.perf_counter()
        try:
            yield
        finally:
            _synchronize_cuda()
            duration = time.perf_counter() - started
            after = _snapshot()
            peak = sampler.stop()
            with self._lock:
                if len(self.phases) == self.phases.maxlen:
                    self._phase_records_dropped += 1
                self.phases.append(PhaseRecord(name, duration, before, after, peak))

    @contextlib.contextmanager
    def operation(
        self, name: str, *, synchronize_cuda: bool = False
    ) -> Generator[None, None, None]:
        if synchronize_cuda:
            _synchronize_cuda()
        started = time.perf_counter()
        completed = False
        try:
            yield
            completed = True
        finally:
            if synchronize_cuda:
                _synchronize_cuda()
            if completed:
                duration = time.perf_counter() - started
                with self._lock:
                    if name not in self.operations:
                        self.operations[name] = _OperationAccumulator(
                            self.max_operation_samples
                        )
                    self.operations[name].add(duration)

    def report(self) -> TelemetryReport:
        with self._lock:
            phases = [_phase_data(phase) for phase in self.phases]
            operations = {
                name: _operation_stats(samples)
                for name, samples in self.operations.items()
            }
        return {
            "environment": _environment(),
            "overall": _overall_dict(
                self._overall_before,
                self._overall_after,
                self._overall_peak,
                self._overall_duration_s,
            ),
            "phases": phases,
            "operations": operations,
            "phase_records_dropped": self._phase_records_dropped,
        }


def timed(
    name: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Time a function when called inside ``TelemetryRecorder.activate()``."""

    def decorator(function: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            recorder = _ACTIVE_RECORDER.get()
            if recorder is None:
                return function(*args, **kwargs)
            with recorder.phase(name):
                return function(*args, **kwargs)

        return wrapped

    return decorator


def _snapshot() -> ResourceSnapshot:
    cuda_allocated = 0.0
    cuda_reserved = 0.0
    if _cuda_available():
        import torch

        cuda_allocated = _mb(torch.cuda.memory_allocated())
        cuda_reserved = _mb(torch.cuda.memory_reserved())
    return ResourceSnapshot(
        rss_mb=_mb(_process_rss_bytes()),
        system_used_mb=_mb(_system_used_bytes()),
        cuda_allocated_mb=cuda_allocated,
        cuda_reserved_mb=cuda_reserved,
    )


def _process_rss_bytes() -> int:
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))


def _system_used_bytes() -> int:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", maxsplit=1)
        values[key] = int(value.strip().split()[0]) * 1024
    return values["MemTotal"] - values["MemAvailable"]


def _synchronize_cuda() -> None:
    if _cuda_available():
        import torch

        torch.cuda.synchronize()


def _reset_cuda_peak_memory() -> None:
    if _cuda_available():
        import torch

        torch.cuda.reset_peak_memory_stats()


def _with_cuda_allocator_peak(snapshot: ResourceSnapshot) -> ResourceSnapshot:
    if not _cuda_available():
        return snapshot
    import torch

    return ResourceSnapshot(
        rss_mb=snapshot.rss_mb,
        system_used_mb=snapshot.system_used_mb,
        cuda_allocated_mb=max(
            snapshot.cuda_allocated_mb,
            _mb(torch.cuda.max_memory_allocated()),
        ),
        cuda_reserved_mb=max(
            snapshot.cuda_reserved_mb,
            _mb(torch.cuda.max_memory_reserved()),
        ),
    )


def _operation_stats(samples: _OperationAccumulator) -> OperationStats:
    ordered = sorted(samples.samples)
    total = samples.total
    return {
        "count": samples.count,
        "total_s": total,
        "mean_s": total / samples.count if samples.count else 0.0,
        "p50_s": _percentile(ordered, 0.50),
        "p95_s": _percentile(ordered, 0.95),
        "p99_s": _percentile(ordered, 0.99),
        "per_second": samples.count / total if total else 0.0,
        "percentile_samples": len(ordered),
        "percentile_method": (
            "exact" if samples.count <= samples.capacity else "reservoir"
        ),
    }


def _percentile(ordered: list[float], quantile: float) -> float:
    if not ordered:
        return 0.0
    index = min(round((len(ordered) - 1) * quantile), len(ordered) - 1)
    return ordered[index]


def _overall_dict(
    before: ResourceSnapshot | None,
    after: ResourceSnapshot | None,
    peak: ResourceSnapshot | None,
    duration_s: float | None,
) -> OverallTelemetry:
    if before is None or after is None or peak is None or duration_s is None:
        raise RuntimeError("TelemetryRecorder has not completed an active run")
    return {
        "duration_s": duration_s,
        "before": _snapshot_data(before),
        "after": _snapshot_data(after),
        "peak": _snapshot_data(peak),
    }


def _snapshot_data(snapshot: ResourceSnapshot) -> ResourceSnapshotData:
    return {
        "rss_mb": snapshot.rss_mb,
        "system_used_mb": snapshot.system_used_mb,
        "cuda_allocated_mb": snapshot.cuda_allocated_mb,
        "cuda_reserved_mb": snapshot.cuda_reserved_mb,
    }


def _phase_data(phase: PhaseRecord) -> PhaseData:
    return {
        "name": phase.name,
        "duration_s": phase.duration_s,
        "before": _snapshot_data(phase.before),
        "after": _snapshot_data(phase.after),
        "peak": _snapshot_data(phase.peak),
    }


def _environment() -> EnvironmentData:
    try:
        torch_version = version("torch")
    except PackageNotFoundError:
        torch_version = "not-installed"
    cuda_build = None
    cuda_device = None
    if "torch" in sys.modules:
        import torch.version

        cuda_build = torch.version.cuda
        if _cuda_available():
            cuda_device = torch.cuda.get_device_name()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch_version,
        "cuda_build": cuda_build,
        "cuda_available": _cuda_available(),
        "cuda_device": cuda_device,
    }


def _cuda_available() -> bool:
    if "torch" not in sys.modules:
        return False
    import torch

    return torch.cuda.is_available()


def _mb(byte_count: float) -> float:
    return float(byte_count) / (1024 * 1024)
