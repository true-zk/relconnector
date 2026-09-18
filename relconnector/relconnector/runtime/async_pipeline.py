"""Bounded producer/consumer execution for sample, fetch, and train stages."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Literal, TypeVar, cast

import pandas as pd
import torch
from torch_frame import TensorFrame
from torch_frame.data.multi_tensor import _MultiTensor

from relconnector.features import BatchAssembler, FeatureBatch, PreparedBatch
from relconnector.randomness import TORCH_RNG_LOCK
from relconnector.sampling import SamplePlan
from relconnector.task import EntitySeedBatch, RecommendationSeedBatch, SeedBatch

from .contracts import RuntimeComponents, RuntimeResult
from .queue import ByteBoundedQueue

_QueueItem = TypeVar("_QueueItem")


@dataclass(frozen=True)
class _FetchedWindow:
    sequence: int
    features: list[FeatureBatch]
    allocated_bytes: int


@dataclass(frozen=True)
class _PreparedWindow:
    sequence: int
    batches: list[PreparedBatch]
    allocated_bytes: int


@dataclass
class _AdaptiveWindow:
    mode: Literal["static", "adaptive"]
    maximum: int
    current: int
    duplicate_ewma: float = 1.0
    cache_ewma: float = 0.0
    observations: int = 0
    changes: int = 0

    def observe(self, duplicate_factor: float, cache_coverage: float) -> None:
        if self.mode == "static":
            return
        weight = 0.2
        self.duplicate_ewma = (
            weight * duplicate_factor + (1 - weight) * self.duplicate_ewma
        )
        self.cache_ewma = weight * cache_coverage + (1 - weight) * self.cache_ewma
        self.observations += 1
        if self.observations % 8:
            return
        previous = self.current
        if self.duplicate_ewma > 1.20 or self.cache_ewma > 0.40:
            self.current = min(self.maximum, self.current + 1)
        elif self.duplicate_ewma < 1.08 and self.cache_ewma < 0.15:
            self.current = max(1, self.current - 1)
        self.changes += int(previous != self.current)


@dataclass(frozen=True)
class AsyncRuntimeConfig:
    seed_queue_bytes: int = 64 * 1024 * 1024
    plan_queue_bytes: int = 2 * 1024 * 1024 * 1024
    ready_queue_bytes: int = 8 * 1024 * 1024 * 1024
    feature_window_batches: int = 4
    feature_window_max_batches: int = 32
    feature_window_bytes: int = 512 * 1024 * 1024
    encode_workers: int = 1
    fetched_queue_bytes: int = 2 * 1024 * 1024 * 1024
    feature_policy: Literal["static", "adaptive"] = "adaptive"

    def __post_init__(self) -> None:
        if (
            min(
                self.seed_queue_bytes,
                self.plan_queue_bytes,
                self.ready_queue_bytes,
                self.feature_window_batches,
                self.feature_window_max_batches,
                self.feature_window_bytes,
                self.encode_workers,
                self.fetched_queue_bytes,
            )
            <= 0
        ):
            raise ValueError("runtime queue and feature-window limits must be positive")
        if self.feature_policy not in {"static", "adaptive"}:
            raise ValueError("feature_policy must be static or adaptive")
        if self.feature_window_batches > self.feature_window_max_batches:
            raise ValueError(
                "feature_window_batches cannot exceed feature_window_max_batches"
            )


class AsyncPipelineExecutor:
    """Overlap one sampler, one feature worker, and the training consumer."""

    def __init__(self, config: AsyncRuntimeConfig | None = None) -> None:
        self.config = config or AsyncRuntimeConfig()
        self._snapshot_lock = threading.Lock()
        self._snapshotter: Callable[[], dict[str, object]] | None = None

    def snapshot(self) -> dict[str, object]:
        with self._snapshot_lock:
            snapshotter = self._snapshotter
        return {} if snapshotter is None else snapshotter()

    def run(self, components: RuntimeComponents, *, epochs: int) -> RuntimeResult:
        if epochs <= 0:
            raise ValueError("epochs must be greater than zero")
        seed_queue = ByteBoundedQueue[SeedBatch](
            self.config.seed_queue_bytes,
            _seed_bytes,
        )
        plan_queue = ByteBoundedQueue[SamplePlan](
            self.config.plan_queue_bytes,
            lambda plan: plan.allocated_bytes,
        )
        encode_workers = (
            self.config.encode_workers
            if components.assembler_factory is not None
            else 1
        )
        fetched_budget = max(self.config.fetched_queue_bytes // encode_workers, 1)
        prepared_budget = max(self.config.ready_queue_bytes // encode_workers, 1)
        fetched_queues = [
            ByteBoundedQueue[_FetchedWindow](
                fetched_budget, lambda window: window.allocated_bytes
            )
            for _ in range(encode_workers)
        ]
        prepared_queues = [
            ByteBoundedQueue[_PreparedWindow](
                prepared_budget, lambda window: window.allocated_bytes
            )
            for _ in range(encode_workers)
        ]
        errors: list[BaseException] = []
        error_lock = threading.Lock()
        cancel = threading.Event()
        feature_windows = 0
        feature_window_batches = 0
        max_feature_window_batches = 0
        window_policy = _AdaptiveWindow(
            self.config.feature_policy,
            (
                self.config.feature_window_batches
                if self.config.feature_policy == "static"
                else self.config.feature_window_max_batches
            ),
            self.config.feature_window_batches,
        )

        def pipeline_snapshot() -> dict[str, object]:
            return {
                "seed_queue": asdict(seed_queue.stats),
                "plan_queue": asdict(plan_queue.stats),
                "fetched_queue": _aggregate_queue_stats(fetched_queues),
                "ready_queue": _aggregate_queue_stats(prepared_queues),
                "encode_workers": encode_workers,
                "feature_windows": feature_windows,
                "feature_window_batches": feature_window_batches,
                "feature_window_batches_max": max_feature_window_batches,
                "feature_policy": _window_policy_report(window_policy),
            }

        with self._snapshot_lock:
            self._snapshotter = pipeline_snapshot

        def stop() -> None:
            cancel.set()
            seed_queue.close(discard=True)
            plan_queue.close(discard=True)
            for queue in fetched_queues:
                queue.close(discard=True)
            for queue in prepared_queues:
                queue.close(discard=True)

        def fail(exc: BaseException) -> None:
            with error_lock:
                if not errors:
                    errors.append(exc)
            stop()

        def produce_seeds() -> None:
            try:
                for epoch in range(epochs):
                    for seeds in components.seeds.iter_epoch(epoch):
                        if cancel.is_set():
                            return
                        seed_queue.put(seeds)
                        del seeds
            except BaseException as exc:  # noqa: BLE001 - propagate worker interrupts.
                fail(exc)
            finally:
                seed_queue.close()

        def produce_plans() -> None:
            try:
                while not cancel.is_set():
                    seeds = seed_queue.get()
                    if seeds is None:
                        return
                    plan_queue.put(components.sampler.sample(seeds))
                    del seeds
            except BaseException as exc:  # noqa: BLE001 - propagate worker interrupts.
                fail(exc)
            finally:
                plan_queue.close()

        def produce_features() -> None:
            nonlocal feature_windows, feature_window_batches
            nonlocal max_feature_window_batches
            sequence = 0
            try:
                while not cancel.is_set():
                    plan = plan_queue.get()
                    if plan is None:
                        return
                    remaining_bytes = max(
                        self.config.feature_window_bytes - plan.allocated_bytes,
                        0,
                    )
                    plans = [plan]
                    plan_epoch = plan.key.epoch

                    def same_epoch(
                        candidate: SamplePlan, expected_epoch: int = plan_epoch
                    ) -> bool:
                        return candidate.key.epoch == expected_epoch

                    plans.extend(
                        plan_queue.drain(
                            window_policy.current - 1,
                            max_bytes=remaining_bytes,
                            predicate=same_epoch,
                        )
                    )
                    feature_windows += 1
                    feature_window_batches += len(plans)
                    max_feature_window_batches = max(
                        max_feature_window_batches, len(plans)
                    )
                    features = _fetch_many(components, plans)
                    fetch_stats = _last_fetch_stats(components)
                    if fetch_stats is not None:
                        window_policy.observe(*fetch_stats)
                    fetched_queues[sequence % encode_workers].put(
                        _FetchedWindow(
                            sequence,
                            features,
                            _retained_bytes(features),
                        )
                    )
                    sequence += 1
                    del plans, plan
            except BaseException as exc:  # noqa: BLE001 - propagate worker interrupts.
                fail(exc)
            finally:
                for queue in fetched_queues:
                    queue.close()

        def encode(worker_index: int) -> None:
            assembler = (
                components.assembler
                if components.assembler_factory is None
                else components.assembler_factory()
            )
            try:
                while not cancel.is_set():
                    window = fetched_queues[worker_index].get()
                    if window is None:
                        return
                    batches = _assemble_with(assembler, window.features)
                    prepared_queues[worker_index].put(
                        _PreparedWindow(
                            window.sequence,
                            batches,
                            sum(batch.allocated_bytes for batch in batches),
                        )
                    )
                    del window
            except BaseException as exc:  # noqa: BLE001 - propagate worker interrupts.
                fail(exc)
            finally:
                prepared_queues[worker_index].close()

        workers = [
            threading.Thread(target=produce_seeds, name="seed-reader"),
            threading.Thread(target=produce_plans, name="sampler"),
            threading.Thread(target=produce_features, name="feature-fetcher"),
        ]
        workers.extend(
            threading.Thread(
                target=encode,
                args=(worker_index,),
                name=f"feature-encoder-{worker_index}",
            )
            for worker_index in range(encode_workers)
        )
        started_workers: list[threading.Thread] = []
        steps = 0
        examples = 0
        weighted_loss = 0.0
        epoch_records: list[dict[str, int | float]] = []
        current_epoch: int | None = None
        epoch_started = time.perf_counter()
        epoch_finished = epoch_started
        epoch_steps = 0
        epoch_examples = 0
        epoch_weighted_loss = 0.0
        try:
            for worker in workers:
                worker.start()
                started_workers.append(worker)
            expected_window = 0
            while not cancel.is_set():
                window = prepared_queues[expected_window % encode_workers].get()
                if window is None:
                    break
                if window.sequence != expected_window:
                    raise RuntimeError(
                        f"expected feature window {expected_window}, "
                        f"received {window.sequence}"
                    )
                for batch in window.batches:
                    now = time.perf_counter()
                    if current_epoch is None:
                        current_epoch = batch.key.epoch
                    elif batch.key.epoch != current_epoch:
                        epoch_records.append(
                            _epoch_record(
                                current_epoch,
                                epoch_steps,
                                epoch_examples,
                                epoch_weighted_loss,
                                now - epoch_started,
                            )
                        )
                        current_epoch = batch.key.epoch
                        epoch_started = now
                        epoch_steps = 0
                        epoch_examples = 0
                        epoch_weighted_loss = 0.0
                    with TORCH_RNG_LOCK:
                        result = components.trainer.train_step(batch)
                    steps += 1
                    examples += result.examples
                    weighted_loss += result.loss * result.examples
                    epoch_steps += 1
                    epoch_examples += result.examples
                    epoch_weighted_loss += result.loss * result.examples
                    epoch_finished = time.perf_counter()
                expected_window += 1
                del window
        except BaseException:
            stop()
            raise
        finally:
            if cancel.is_set():
                stop()
            for worker in started_workers:
                worker.join()

        if errors:
            if not isinstance(errors[0], Exception):
                raise errors[0]
            raise RuntimeError("asynchronous pipeline failed") from errors[0]
        if current_epoch is not None:
            epoch_records.append(
                _epoch_record(
                    current_epoch,
                    epoch_steps,
                    epoch_examples,
                    epoch_weighted_loss,
                    epoch_finished - epoch_started,
                )
            )
        return RuntimeResult(
            steps=steps,
            examples=examples,
            mean_loss=weighted_loss / max(examples, 1),
            pipeline={
                **pipeline_snapshot(),
                "fetched_queue": _aggregate_queue_stats(fetched_queues),
                "fetched_queues": [asdict(queue.stats) for queue in fetched_queues],
                "ready_queues": [asdict(queue.stats) for queue in prepared_queues],
                "feature_window_batches_mean": (
                    feature_window_batches / feature_windows if feature_windows else 0.0
                ),
                "epochs": epoch_records,
            },
        )


def _fetch_many(
    components: RuntimeComponents, plans: list[SamplePlan]
) -> list[FeatureBatch]:
    method = getattr(components.fetcher, "fetch_many", None)
    if callable(method):
        return cast(list[FeatureBatch], method(plans))
    return [components.fetcher.fetch(plan) for plan in plans]


def _last_fetch_stats(
    components: RuntimeComponents,
) -> tuple[float, float] | None:
    fetcher: object = components.fetcher
    while hasattr(fetcher, "inner"):
        fetcher = fetcher.inner  # type: ignore[attr-defined]
    stats = getattr(fetcher, "last_stats", None)
    if stats is None:
        return None
    duplicate_factor = getattr(stats, "duplicate_factor", None)
    cache_coverage = getattr(stats, "row_cache_coverage", None)
    if not isinstance(duplicate_factor, (int, float)) or not isinstance(
        cache_coverage, (int, float)
    ):
        return None
    return float(duplicate_factor), float(cache_coverage)


def _assemble_with(
    assembler: BatchAssembler, features: list[FeatureBatch]
) -> list[PreparedBatch]:
    method = getattr(assembler, "assemble_many", None)
    if callable(method):
        return cast(list[PreparedBatch], method(features))
    method = assembler.assemble
    return [cast(PreparedBatch, method(item)) for item in features]


def _epoch_record(
    epoch: int,
    steps: int,
    examples: int,
    weighted_loss: float,
    duration_s: float,
) -> dict[str, int | float]:
    return {
        "epoch": epoch,
        "steps": steps,
        "examples": examples,
        "mean_loss": weighted_loss / max(examples, 1),
        "duration_s": duration_s,
        "batches_per_second": steps / duration_s if duration_s else 0.0,
    }


def _seed_bytes(batch: SeedBatch) -> int:
    if isinstance(batch, EntitySeedBatch):
        tensors = [batch.node_ids, batch.target]
        if batch.seed_time is not None:
            tensors.append(batch.seed_time)
    elif isinstance(batch, RecommendationSeedBatch):
        tensors = [batch.src_node_ids, batch.positive_dst_ids]
        if batch.seed_time is not None:
            tensors.append(batch.seed_time)
    else:
        raise TypeError(f"Unsupported seed batch: {type(batch).__name__}")
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


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
        if isinstance(item, pd.DataFrame):
            key = ("dataframe", id(item))
            if key in seen:
                return 0
            seen.add(key)
            return int(item.memory_usage(index=True, deep=True).sum())
        if isinstance(item, TensorFrame):
            return visit(item.feat_dict) + visit(item.y)
        if isinstance(item, _MultiTensor):
            return visit(item.values) + visit(item.offset)
        if isinstance(item, dict):
            return sum(visit(key) + visit(child) for key, child in item.items())
        if isinstance(item, (list, tuple)):
            return sum(visit(child) for child in item)
        if is_dataclass(item) and not isinstance(item, type):
            return sum(visit(getattr(item, field.name)) for field in fields(item))
        return 0

    return max(visit(value), 1)


def _aggregate_queue_stats(
    queues: list[ByteBoundedQueue[_QueueItem]],
) -> dict[str, object]:
    stats = [queue.stats for queue in queues]
    return {
        "put_wait_s": sum(item.put_wait_s for item in stats),
        "get_wait_s": sum(item.get_wait_s for item in stats),
        "put_wait_count": sum(item.put_wait_count for item in stats),
        "get_wait_count": sum(item.get_wait_count for item in stats),
        "peak_items": sum(item.peak_items for item in stats),
        "peak_bytes": sum(item.peak_bytes for item in stats),
    }


def _window_policy_report(policy: _AdaptiveWindow) -> dict[str, object]:
    return {
        "mode": policy.mode,
        "current_batches": policy.current,
        "duplicate_ewma": policy.duplicate_ewma,
        "cache_coverage_ewma": policy.cache_ewma,
        "observations": policy.observations,
        "changes": policy.changes,
    }
