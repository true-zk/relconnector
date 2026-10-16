"""Low-overhead, thread-safe operation and counter telemetry."""

from __future__ import annotations

import contextlib
import threading
import time
from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass


@dataclass
class _Timer:
    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0


class OperationMetrics:
    """Collect bounded aggregate metrics without retaining per-call samples."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._timers: dict[tuple[str, str, str], _Timer] = defaultdict(_Timer)
        self._counters: dict[tuple[str, str, str], float] = defaultdict(float)
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def timer(
        self, operation: str, *, table: str = "", column: str = ""
    ) -> Generator[None, None, None]:
        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - started
            key = (operation, table, column)
            with self._lock:
                timer = self._timers[key]
                timer.count += 1
                timer.total_s += duration
                timer.max_s = max(timer.max_s, duration)

    def add(
        self,
        counter: str,
        value: float = 1,
        *,
        table: str = "",
        column: str = "",
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._counters[(counter, table, column)] += value

    def report(self) -> dict[str, list[dict[str, int | float | str]]]:
        with self._lock:
            timers = [
                {
                    "operation": operation,
                    "table": table,
                    "column": column,
                    "count": timer.count,
                    "total_s": timer.total_s,
                    "mean_s": timer.total_s / timer.count,
                    "max_s": timer.max_s,
                }
                for (operation, table, column), timer in self._timers.items()
            ]
            counters = [
                {
                    "counter": counter,
                    "table": table,
                    "column": column,
                    "value": value,
                }
                for (counter, table, column), value in self._counters.items()
            ]
        timers.sort(key=lambda item: (-float(item["total_s"]), str(item["operation"])))
        counters.sort(key=lambda item: (str(item["counter"]), str(item["table"])))
        return {"timers": timers, "counters": counters}
