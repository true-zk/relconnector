"""Three-stage process pipeline for sampling, feature assembly, and training."""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
import traceback
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol, cast

import torch
import torch.multiprocessing as mp

from relconnector_v2.features import PreparedBatch
from relconnector_v2.randomness import TORCH_RNG_LOCK
from relconnector_v2.sampling import SamplePlan
from relconnector_v2.task import SeedBatch

from .contracts import RuntimeResult, Sampler, SeedSource, Trainer


class ProcessAssemblyWorker(Protocol):
    """CPU-only worker that owns SQL and feature-encoding state."""

    def assemble_many(self, plans: list[SamplePlan]) -> list[PreparedBatch]: ...

    def diagnostics(self) -> dict[str, object]: ...

    def close(self) -> None: ...


class ProcessAssemblyFactory(Protocol):
    """Pickle-safe factory invoked inside each spawned assembly process."""

    def __call__(self, worker_index: int) -> ProcessAssemblyWorker: ...


@dataclass(frozen=True)
class ProcessRuntimeComponents:
    seeds: SeedSource
    sampler: Sampler
    assembly_factory: ProcessAssemblyFactory
    trainer: Trainer


@dataclass(frozen=True)
class ProcessRuntimeConfig:
    assembly_processes: int = 4
    seed_queue_bytes: int = 64 * 1024 * 1024
    plan_queue_bytes: int = 2 * 1024 * 1024 * 1024
    ready_queue_bytes: int = 16 * 1024 * 1024 * 1024
    window_batches: int = 4
    window_bytes: int = 512 * 1024 * 1024
    max_inflight_windows: int = 8
    initial_active_processes: int = 1
    adaptive_workers: bool = True
    worker_scale_interval_s: float = 0.5
    cross_epoch_prefetch: bool = True
    training_order: Literal["strict", "ready"] = "ready"
    start_method: Literal["spawn", "forkserver"] = "spawn"
    queue_items: int = 64
    poll_interval_s: float = 0.1
    progress_interval_s: float = 0.0
    diagnostics_timeout_s: float = 300.0

    def __post_init__(self) -> None:
        positive = (
            self.assembly_processes,
            self.seed_queue_bytes,
            self.plan_queue_bytes,
            self.ready_queue_bytes,
            self.window_batches,
            self.window_bytes,
            self.max_inflight_windows,
            self.initial_active_processes,
            self.queue_items,
        )
        if (
            any(value <= 0 for value in positive)
            or self.poll_interval_s <= 0
            or self.worker_scale_interval_s <= 0
            or self.diagnostics_timeout_s <= 0
        ):
            raise ValueError("process runtime limits must be positive")
        if self.initial_active_processes > self.assembly_processes:
            raise ValueError(
                "initial_active_processes cannot exceed assembly_processes"
            )
        if self.training_order not in {"strict", "ready"}:
            raise ValueError("training_order must be strict or ready")
        if self.progress_interval_s < 0:
            raise ValueError("progress interval must be nonnegative")


@dataclass(frozen=True)
class _SizedItem:
    payload: object
    charged_bytes: int


@dataclass(frozen=True)
class _EpochEnd:
    epoch: int
    batches: int


@dataclass(frozen=True)
class _EpochSampled:
    epoch: int
    batches: int


@dataclass(frozen=True)
class _SequencedPlan:
    sequence: int
    plan: SamplePlan


@dataclass(frozen=True)
class _PlanWindow:
    window: int
    plans: tuple[_SequencedPlan, ...]
    allocated_bytes: int


@dataclass(frozen=True)
class _ReadyBatch:
    window: int
    window_batches: int
    sequence: int
    batch: PreparedBatch


@dataclass(frozen=True)
class _SamplingFinished:
    total_batches: int
    epoch_batches: dict[int, int]


@dataclass(frozen=True)
class _WorkerFinished:
    worker: int
    windows: int
    batches: int
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class _ProcessFailure:
    stage: str
    worker: int | None
    message: str
    traceback: str


class _ProcessErrorQueue:
    """Send failures without starting a feeder thread during resource exhaustion."""

    def __init__(self, context: Any) -> None:
        self._queue = context.SimpleQueue()

    def put(self, failure: _ProcessFailure) -> None:
        self._queue.put(failure)

    def get_nowait(self) -> _ProcessFailure:
        # Only the trainer consumes failures; writers cannot make a nonempty
        # queue empty between this check and get().
        if self._queue.empty():
            raise queue.Empty
        return cast(_ProcessFailure, self._queue.get())


class _SharedByteBudget:
    def __init__(self, context: Any, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._used = context.Value("q", 0, lock=False)
        self._peak = context.Value("q", 0, lock=False)
        self._put_wait_ns = context.Value("q", 0, lock=False)
        self._put_wait_count = context.Value("q", 0, lock=False)
        self._condition = context.Condition()

    def acquire(self, size: int, stop: Any, poll_interval_s: float) -> bool:
        if size < 0:
            raise ValueError("payload size must be nonnegative")
        if size > self.max_bytes:
            raise ValueError(
                f"payload requires {size} bytes, budget is {self.max_bytes}"
            )
        started: int | None = None
        with self._condition:
            while self._used.value + size > self.max_bytes:
                if stop.is_set():
                    return False
                if started is None:
                    started = time.perf_counter_ns()
                self._condition.wait(poll_interval_s)
            if started is not None:
                self._put_wait_ns.value += time.perf_counter_ns() - started
                self._put_wait_count.value += 1
            self._used.value += size
            self._peak.value = max(self._peak.value, self._used.value)
            return True

    def release(self, size: int) -> None:
        with self._condition:
            self._used.value -= size
            if self._used.value < 0:
                raise RuntimeError("shared queue byte budget underflow")
            self._condition.notify_all()

    def wake_all(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def snapshot(self) -> dict[str, int | float]:
        with self._condition:
            return {
                "max_bytes": self.max_bytes,
                "current_bytes": int(self._used.value),
                "peak_bytes": int(self._peak.value),
                "put_wait_s": self._put_wait_ns.value / 1_000_000_000,
                "put_wait_count": int(self._put_wait_count.value),
            }


class _ActiveWorkerGate:
    def __init__(self, context: Any, initial: int, maximum: int) -> None:
        self.maximum = maximum
        self._active = context.Value("i", initial, lock=False)
        self._scale_up = context.Value("q", 0, lock=False)
        self._scale_down = context.Value("q", 0, lock=False)
        self._condition = context.Condition()

    def wait(self, worker: int, stop: Any, interval_s: float) -> bool:
        with self._condition:
            while worker >= self._active.value:
                if stop.is_set():
                    return False
                self._condition.wait(interval_s)
            return True

    def set_active(self, value: int) -> None:
        with self._condition:
            target = min(max(value, 1), self.maximum)
            previous = self._active.value
            self._active.value = target
            self._scale_up.value += int(target > previous)
            self._scale_down.value += int(target < previous)
            self._condition.notify_all()

    def activate_all(self) -> None:
        self.set_active(self.maximum)

    def snapshot(self) -> dict[str, int]:
        with self._condition:
            return {
                "active": int(self._active.value),
                "maximum": self.maximum,
                "scale_up": int(self._scale_up.value),
                "scale_down": int(self._scale_down.value),
            }


class _ProcessByteQueue:
    def __init__(
        self,
        context: Any,
        max_bytes: int,
        max_items: int,
        poll_interval_s: float,
    ) -> None:
        self._queue = context.Queue(maxsize=max_items)
        self._budget = _SharedByteBudget(context, max_bytes)
        self._poll_interval_s = poll_interval_s

    def put(self, payload: object, size: int, stop: Any) -> bool:
        if not self._budget.acquire(size, stop, self._poll_interval_s):
            return False
        item = _SizedItem(payload, size)
        try:
            while not stop.is_set():
                try:
                    self._queue.put(item, timeout=self._poll_interval_s)
                    return True
                except queue.Full:
                    continue
        except BaseException:
            self._budget.release(size)
            raise
        self._budget.release(size)
        return False

    def get(self, timeout: float | None = None) -> _SizedItem:
        return cast(_SizedItem, self._queue.get(timeout=timeout))

    def release(self, item: _SizedItem) -> None:
        self._budget.release(item.charged_bytes)

    def wake_all(self) -> None:
        self._budget.wake_all()

    def snapshot(self) -> dict[str, int | float]:
        return self._budget.snapshot()


class ProcessPipelineExecutor:
    """Run sampler and assembly as CPU processes while training in the parent."""

    def __init__(self, config: ProcessRuntimeConfig | None = None) -> None:
        self.config = config or ProcessRuntimeConfig()
        self._snapshot_lock = threading.Lock()
        self._snapshot: dict[str, object] = {}

    def snapshot(self) -> dict[str, object]:
        with self._snapshot_lock:
            return dict(self._snapshot)

    def run(
        self, components: ProcessRuntimeComponents, *, epochs: int
    ) -> RuntimeResult:
        if epochs <= 0:
            raise ValueError("epochs must be greater than zero")
        context: Any = mp.get_context(self.config.start_method)
        stop = context.Event()
        release_processes = context.Event()
        seed_queue = _ProcessByteQueue(
            context,
            self.config.seed_queue_bytes,
            self.config.queue_items,
            self.config.poll_interval_s,
        )
        plan_queue = _ProcessByteQueue(
            context,
            self.config.plan_queue_bytes,
            self.config.queue_items,
            self.config.poll_interval_s,
        )
        ready_queue = _ProcessByteQueue(
            context,
            self.config.ready_queue_bytes,
            self.config.queue_items,
            self.config.poll_interval_s,
        )
        control_queue = context.Queue()
        error_queue = _ProcessErrorQueue(context)
        inflight = context.BoundedSemaphore(self.config.max_inflight_windows)
        worker_gate = _ActiveWorkerGate(
            context,
            self.config.initial_active_processes,
            self.config.assembly_processes,
        )
        feeder_errors: list[BaseException] = []
        assembly_factory = components.assembly_factory
        parent_resources: list[object] = []
        bind_context = getattr(assembly_factory, "bind_process_context", None)
        if callable(bind_context):
            bound = bind_context(context)
            if isinstance(bound, tuple):
                assembly_factory = cast(ProcessAssemblyFactory, bound[0])
                parent_resources.extend(cast(tuple[object, ...], bound[1]))
            else:
                assembly_factory = cast(ProcessAssemblyFactory, bound)

        _share_sampler_graph(components.sampler)
        sampler_process = context.Process(
            target=_sample_process_main,
            name="relconnector-v2-sampler",
            args=(
                components.sampler,
                seed_queue,
                plan_queue,
                control_queue,
                error_queue,
                inflight,
                worker_gate,
                stop,
                release_processes,
                self.config,
            ),
        )
        assembly_processes = [
            context.Process(
                target=_assembly_process_main,
                name=f"relconnector-v2-assembly-{worker}",
                args=(
                    worker,
                    assembly_factory,
                    plan_queue,
                    ready_queue,
                    control_queue,
                    error_queue,
                    worker_gate,
                    stop,
                    release_processes,
                    self.config,
                ),
            )
            for worker in range(self.config.assembly_processes)
        ]

        def feed_seeds() -> None:
            try:
                for epoch in range(epochs):
                    batches = 0
                    for seeds in components.seeds.iter_epoch(epoch):
                        if stop.is_set():
                            return
                        if not seed_queue.put(seeds, _retained_bytes(seeds), stop):
                            return
                        batches += 1
                    if not seed_queue.put(_EpochEnd(epoch, batches), 0, stop):
                        return
                seed_queue.put(None, 0, stop)
            except BaseException as exc:  # noqa: BLE001
                feeder_errors.append(exc)
                # The trainer must see this before run()'s final cleanup. A
                # stop event alone leaves it waiting for batches while every
                # child waits for release_processes, hiding the seed exception.
                error_queue.put(
                    _ProcessFailure("seed", None, str(exc), traceback.format_exc())
                )
                stop.set()
                seed_queue.wake_all()
                plan_queue.wake_all()
                ready_queue.wake_all()

        started = time.perf_counter()
        processes = [sampler_process, *assembly_processes]
        feeder = threading.Thread(
            target=feed_seeds,
            name="relconnector-v2-seed-reader",
            daemon=True,
        )
        for process in processes:
            process.start()
        feeder.start()
        shared_cache_reports: dict[str, object] = {}
        try:
            result = self._train(
                components.trainer,
                ready_queue,
                control_queue,
                error_queue,
                inflight,
                plan_queue,
                worker_gate,
                processes,
                epochs,
                started,
            )
            for name in ("shared_text_cache", "shared_encoded_cache"):
                cache = getattr(assembly_factory, name, None)
                snapshot = getattr(cache, "snapshot", None)
                if callable(snapshot):
                    shared_cache_reports[name] = snapshot()
        finally:
            stop.set()
            release_processes.set()
            seed_queue.wake_all()
            plan_queue.wake_all()
            ready_queue.wake_all()
            feeder.join(timeout=5)
            for process in processes:
                process.join(timeout=10)
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=5)
            for resource in parent_resources:
                close = getattr(resource, "close", None)
                if callable(close):
                    close()
        if feeder_errors:
            raise RuntimeError("seed reader failed") from feeder_errors[0]
        result.pipeline["seed_queue"] = seed_queue.snapshot()
        result.pipeline["plan_queue"] = plan_queue.snapshot()
        result.pipeline.update(shared_cache_reports)
        return result

    def _train(
        self,
        trainer: Trainer,
        ready_queue: _ProcessByteQueue,
        control_queue: Any,
        error_queue: Any,
        inflight: Any,
        plan_queue: _ProcessByteQueue,
        worker_gate: _ActiveWorkerGate,
        processes: list[Any],
        epochs: int,
        started: float,
    ) -> RuntimeResult:
        pending: dict[int, _ReadyBatch] = {}
        ready_by_epoch: dict[int, deque[int]] = defaultdict(deque)
        epoch_expected: dict[int, int] = {}
        epoch_trained: dict[int, int] = defaultdict(int)
        epoch_examples: dict[int, int] = defaultdict(int)
        epoch_weighted_loss: dict[int, float] = defaultdict(float)
        epoch_started: dict[int, float] = {}
        epoch_finished: dict[int, float] = {}
        window_remaining: dict[int, int] = {}
        worker_reports: list[dict[str, object]] = []
        total_expected: int | None = None
        expected_sequence = 0
        current_epoch = 0
        steps = 0
        examples = 0
        weighted_loss = 0.0
        ready_wait_s = 0.0
        epoch_ready_wait_s: dict[int, float] = defaultdict(float)
        last_scale = started
        last_progress = started
        last_training = started

        def rebalance(*, starved: bool) -> None:
            nonlocal last_scale
            if not self.config.adaptive_workers:
                return
            now = time.monotonic()
            if now - last_scale < self.config.worker_scale_interval_s:
                return
            gate = worker_gate.snapshot()
            active = gate["active"]
            plan = plan_queue.snapshot()
            ready = ready_queue.snapshot()
            if starved and plan["current_bytes"] > 0 and active < gate["maximum"]:
                worker_gate.set_active(active + 1)
                last_scale = now
            elif ready["current_bytes"] > ready["max_bytes"] * 0.75 and active > 1:
                worker_gate.set_active(active - 1)
                last_scale = now

        while total_expected is None or steps < total_expected:
            now = time.perf_counter()
            if (
                self.config.progress_interval_s
                and now - last_progress >= self.config.progress_interval_s
            ):
                print(
                    "[v2 progress] "
                    + json.dumps(
                        {
                            "elapsed_s": round(now - started, 1),
                            "steps": steps,
                            "epoch": current_epoch,
                            "epoch_steps": dict(epoch_trained),
                            "epoch_expected": epoch_expected,
                            "pending_batches": len(pending),
                            "inflight_windows": len(window_remaining),
                            "assembly_finished": sum(
                                report.get("stage") == "assembly"
                                for report in worker_reports
                            ),
                            "sampling_finished": total_expected is not None,
                            "worker_gate": worker_gate.snapshot(),
                            "plan_queue": plan_queue.snapshot(),
                            "ready_queue": ready_queue.snapshot(),
                            "seconds_since_last_batch": round(now - last_training, 1),
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                last_progress = now
            self._drain_control(
                control_queue,
                epoch_expected,
                worker_reports,
            )
            sampling_reports = [
                report for report in worker_reports if report.get("stage") == "sampling"
            ]
            if sampling_reports:
                total_expected = cast(int, sampling_reports[-1]["total_batches"])
                epoch_expected.update(
                    cast(dict[int, int], sampling_reports[-1]["epoch_batches"])
                )
            while (
                current_epoch < epochs
                and current_epoch in epoch_expected
                and epoch_trained[current_epoch] == epoch_expected[current_epoch]
            ):
                current_epoch += 1
            self._raise_process_error(error_queue, processes)
            if total_expected is not None and steps == total_expected:
                break
            candidate = _select_ready(
                pending,
                ready_by_epoch,
                self.config.training_order,
                expected_sequence,
                current_epoch,
            )
            if candidate is None:
                wait_started = time.perf_counter()
                try:
                    queued = ready_queue.get(timeout=self.config.poll_interval_s)
                except queue.Empty:
                    waited = time.perf_counter() - wait_started
                    ready_wait_s += waited
                    epoch_ready_wait_s[current_epoch] += waited
                    finished = [
                        report for report in worker_reports
                        if report.get("stage") == "assembly"
                    ]
                    if (
                        len(finished) == self.config.assembly_processes
                        and time.perf_counter() - last_training > 5.0
                        and ready_queue.snapshot()["current_bytes"] == 0
                    ):
                        raise RuntimeError(
                            "all assembly workers finished before training completed: "
                            f"steps={steps}, expected={total_expected}, "
                            f"epoch_expected={epoch_expected}, pending={len(pending)}, "
                            f"inflight_windows={window_remaining}, "
                            f"assembled={[report['batches'] for report in finished]}"
                        )
                    rebalance(starved=True)
                    continue
                waited = time.perf_counter() - wait_started
                ready_wait_s += waited
                epoch_ready_wait_s[current_epoch] += waited
                ready_queue.release(queued)
                item = cast(_ReadyBatch, queued.payload)
                pending[item.sequence] = item
                ready_by_epoch[item.batch.key.epoch].append(item.sequence)
                window_remaining.setdefault(item.window, item.window_batches)
                continue

            pending.pop(candidate.sequence)
            epoch_queue = ready_by_epoch[candidate.batch.key.epoch]
            epoch_queue.remove(candidate.sequence)
            if not epoch_queue:
                ready_by_epoch.pop(candidate.batch.key.epoch, None)
            epoch = candidate.batch.key.epoch
            epoch_started.setdefault(epoch, time.perf_counter())
            try:
                with TORCH_RNG_LOCK:
                    train_result = trainer.train_step(candidate.batch)
            except Exception as error:
                # Ready-order execution may differ from the progress counter;
                # preserve the actual seed key for deterministic reproduction.
                raise RuntimeError(
                    f"training failed at epoch={epoch}, "
                    f"batch={candidate.batch.key.batch}, "
                    f"sequence={candidate.sequence}, completed_steps={steps}"
                ) from error
            now = time.perf_counter()
            last_training = now
            epoch_finished[epoch] = now
            steps += 1
            examples += train_result.examples
            weighted_loss += train_result.loss * train_result.examples
            epoch_trained[epoch] += 1
            epoch_examples[epoch] += train_result.examples
            epoch_weighted_loss[epoch] += train_result.loss * train_result.examples
            if self.config.training_order == "strict":
                expected_sequence += 1
            remaining = window_remaining[candidate.window] - 1
            if remaining:
                window_remaining[candidate.window] = remaining
            else:
                window_remaining.pop(candidate.window)
                inflight.release()
            rebalance(starved=False)

            expected = epoch_expected.get(current_epoch)
            if expected is not None and epoch_trained[current_epoch] == expected:
                current_epoch += 1
            with self._snapshot_lock:
                self._snapshot = {
                    "backend": "process",
                    "assembly_processes": self.config.assembly_processes,
                    "training_order": self.config.training_order,
                    "cross_epoch_prefetch": self.config.cross_epoch_prefetch,
                    "steps": steps,
                    "pending_batches": len(pending),
                    "current_epoch": current_epoch,
                    "ready_wait_s": ready_wait_s,
                    "worker_gate": worker_gate.snapshot(),
                    "plan_queue": plan_queue.snapshot(),
                    "ready_queue": ready_queue.snapshot(),
                }

        report_deadline = time.monotonic() + self.config.diagnostics_timeout_s
        last_report_progress = time.monotonic()
        while True:
            self._drain_control(control_queue, epoch_expected, worker_reports)
            self._raise_process_error(error_queue, processes)
            reported = {
                report["worker"] for report in worker_reports
                if report.get("stage") == "assembly"
            }
            if len(reported) == self.config.assembly_processes:
                break
            now = time.monotonic()
            if now >= report_deadline:
                missing = set(range(self.config.assembly_processes)) - reported
                raise RuntimeError(
                    "timed out collecting assembly diagnostics after training: "
                    f"steps={steps}, missing_workers={sorted(missing)}"
                )
            if (self.config.progress_interval_s
                    and now - last_report_progress >= self.config.progress_interval_s):
                print(
                    "[v2 progress] " + json.dumps({
                        "phase": "collecting_diagnostics", "steps": steps,
                        "elapsed_s": round(time.perf_counter() - started, 1),
                        "epoch_steps": dict(epoch_trained),
                        "assembly_finished": len(reported),
                    }), file=sys.stderr, flush=True,
                )
                last_report_progress = now
            time.sleep(self.config.poll_interval_s)

        records = []
        for epoch in range(epochs):
            begin = epoch_started.get(epoch, started)
            finish = epoch_finished.get(epoch, begin)
            duration = max(finish - begin, 0.0)
            count = epoch_trained[epoch]
            records.append(
                {
                    "epoch": epoch,
                    "steps": count,
                    "examples": epoch_examples[epoch],
                    "mean_loss": epoch_weighted_loss[epoch]
                    / max(epoch_examples[epoch], 1),
                    "duration_s": duration,
                    "ready_wait_s": epoch_ready_wait_s[epoch],
                    "batches_per_second": count / duration if duration else 0.0,
                }
            )
        return RuntimeResult(
            steps=steps,
            examples=examples,
            mean_loss=weighted_loss / max(examples, 1),
            pipeline={
                "backend": "process",
                "assembly_processes": self.config.assembly_processes,
                "training_order": self.config.training_order,
                "cross_epoch_prefetch": self.config.cross_epoch_prefetch,
                "duration_s": time.perf_counter() - started,
                "ready_wait_s": ready_wait_s,
                "worker_gate": worker_gate.snapshot(),
                "seed_queue": {},
                "plan_queue": {},
                "ready_queue": ready_queue.snapshot(),
                "workers": worker_reports,
                "epochs": records,
            },
        )

    @staticmethod
    def _drain_control(
        control_queue: Any,
        epoch_expected: dict[int, int],
        reports: list[dict[str, object]],
    ) -> None:
        while True:
            try:
                message = control_queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(message, _SamplingFinished):
                epoch_expected.update(message.epoch_batches)
                reports.append(
                    {
                        "stage": "sampling",
                        "total_batches": message.total_batches,
                        "epoch_batches": message.epoch_batches,
                    }
                )
            elif isinstance(message, _EpochSampled):
                epoch_expected[message.epoch] = message.batches
            elif isinstance(message, _WorkerFinished):
                reports.append(
                    {
                        "stage": "assembly",
                        **asdict(message),
                    }
                )

    @staticmethod
    def _raise_process_error(error_queue: Any, processes: list[Any]) -> None:
        try:
            failure = error_queue.get_nowait()
        except queue.Empty:
            failure = None
        if isinstance(failure, _ProcessFailure):
            raise RuntimeError(  # noqa: TRY004
                f"{failure.stage} worker {failure.worker} failed: "
                f"{failure.message}\n{failure.traceback}"
            )
        failed = [process for process in processes if process.exitcode not in (None, 0)]
        if failed:
            details = ", ".join(
                f"{process.name}={process.exitcode}" for process in failed
            )
            raise RuntimeError(
                f"process pipeline worker exited unexpectedly: {details}"
            )


def _sample_process_main(
    sampler: Sampler,
    seed_queue: _ProcessByteQueue,
    plan_queue: _ProcessByteQueue,
    control_queue: Any,
    error_queue: Any,
    inflight: Any,
    worker_gate: _ActiveWorkerGate,
    stop: Any,
    release_processes: Any,
    config: ProcessRuntimeConfig,
) -> None:
    epoch_batches: dict[int, int] = {}
    pending: list[_SequencedPlan] = []
    pending_bytes = 0
    sequence = 0
    window = 0

    def flush() -> bool:
        nonlocal pending, pending_bytes, window
        if not pending:
            return True
        while not stop.is_set():
            if inflight.acquire(timeout=config.poll_interval_s):
                break
        else:
            return False
        item = _PlanWindow(window, tuple(pending), pending_bytes)
        if not plan_queue.put(item, pending_bytes, stop):
            inflight.release()
            return False
        window += 1
        pending = []
        pending_bytes = 0
        return True

    try:
        torch.set_num_threads(1)
        while not stop.is_set():
            try:
                queued = seed_queue.get(timeout=config.poll_interval_s)
            except queue.Empty:
                continue
            seed_queue.release(queued)
            message = queued.payload
            if message is None:
                break
            if isinstance(message, _EpochEnd):
                epoch_batches[message.epoch] = message.batches
                # Keep windows epoch-homogeneous. The next epoch may still be
                # prefetched as a separate inflight window, but mixing epochs
                # in one window can deadlock the trainer's epoch gate before
                # the final sampling report is available.
                if not flush():
                    return
                control_queue.put(_EpochSampled(message.epoch, message.batches))
                continue
            seed = cast(SeedBatch, message)
            plan = sampler.sample(seed)
            size = plan.allocated_bytes
            if pending and (  # noqa: SIM102
                len(pending) >= config.window_batches
                or pending_bytes + size > config.window_bytes
            ):
                if not flush():
                    return
            pending.append(_SequencedPlan(sequence, plan))
            pending_bytes += size
            sequence += 1
        if not flush():
            return
        control_queue.put(_SamplingFinished(sequence, epoch_batches))
        worker_gate.activate_all()
        for _ in range(config.assembly_processes):
            plan_queue.put(None, 0, stop)
    except BaseException as exc:  # noqa: BLE001
        error_queue.put(
            _ProcessFailure("sampling", None, str(exc), traceback.format_exc())
        )
        stop.set()
        seed_queue.wake_all()
        plan_queue.wake_all()
    finally:
        release_processes.wait()


def _assembly_process_main(
    worker_index: int,
    factory: ProcessAssemblyFactory,
    plan_queue: _ProcessByteQueue,
    ready_queue: _ProcessByteQueue,
    control_queue: Any,
    error_queue: Any,
    worker_gate: _ActiveWorkerGate,
    stop: Any,
    release_processes: Any,
    config: ProcessRuntimeConfig,
) -> None:
    worker: ProcessAssemblyWorker | None = None
    windows = 0
    batches = 0
    exit_reason = "cancelled"
    try:
        torch.set_num_threads(1)
        worker = factory(worker_index)
        while not stop.is_set():
            if not worker_gate.wait(
                worker_index,
                stop,
                config.poll_interval_s,
            ):
                return
            try:
                queued = plan_queue.get(timeout=config.poll_interval_s)
            except queue.Empty:
                continue
            plan_queue.release(queued)
            if queued.payload is None:
                exit_reason = "sentinel"
                break
            window = cast(_PlanWindow, queued.payload)
            plans = [item.plan for item in window.plans]
            prepared = worker.assemble_many(plans)
            if len(prepared) != len(window.plans):
                raise RuntimeError(
                    "assembly worker returned a different number of batches"
                )
            for item, batch in zip(window.plans, prepared, strict=True):
                ready = _ReadyBatch(
                    window.window,
                    len(prepared),
                    item.sequence,
                    batch,
                )
                if not ready_queue.put(ready, batch.allocated_bytes, stop):
                    return
            windows += 1
            batches += len(prepared)
        control_queue.put(
            _WorkerFinished(
                worker_index,
                windows,
                batches,
                {} if worker is None else worker.diagnostics(),
            )
        )
    except BaseException as exc:  # noqa: BLE001
        exit_reason = f"error: {exc!r}"
        error_queue.put(
            _ProcessFailure(
                "assembly",
                worker_index,
                str(exc),
                traceback.format_exc(),
            )
        )
        stop.set()
        plan_queue.wake_all()
        ready_queue.wake_all()
    finally:
        if config.progress_interval_s:
            print(
                "[v2 assembly stopped] "
                + json.dumps({"worker": worker_index, "reason": exit_reason,
                              "windows": windows, "batches": batches,
                              "unhandled_exception": repr(sys.exception())}),
                file=sys.stderr,
                flush=True,
            )
        if worker is not None:
            worker.close()
        release_processes.wait()


def _select_ready(
    pending: dict[int, _ReadyBatch],
    ready_by_epoch: dict[int, deque[int]],
    order: Literal["strict", "ready"],
    expected_sequence: int,
    current_epoch: int,
) -> _ReadyBatch | None:
    if order == "strict":
        return pending.get(expected_sequence)
    sequences = ready_by_epoch.get(current_epoch)
    if not sequences:
        return None
    return pending[sequences[0]]


def _share_sampler_graph(sampler: Sampler) -> None:
    graph = getattr(sampler, "graph", None)
    if graph is None:
        return
    data = getattr(graph, "data", None)
    share = getattr(data, "share_memory_", None)
    if callable(share):
        share()
    for name in ("colptr_dict", "row_dict"):
        tensors = getattr(graph, name, {})
        if isinstance(tensors, dict):
            for tensor in tensors.values():
                if isinstance(tensor, torch.Tensor):
                    tensor.share_memory_()


def _retained_bytes(value: object) -> int:
    seen: set[tuple[str, int]] = set()

    def visit(item: object) -> int:
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            key = (str(item.device), storage.data_ptr())
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(item, dict):
            return sum(visit(key) + visit(value) for key, value in item.items())
        if isinstance(item, (list, tuple)):
            return sum(visit(value) for value in item)
        fields = getattr(item, "__dataclass_fields__", None)
        if fields is not None:
            return sum(visit(getattr(item, name)) for name in fields)
        return 0

    return visit(value)
