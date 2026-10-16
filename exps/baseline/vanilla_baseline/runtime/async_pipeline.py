"""Bounded producer/consumer execution for sample, fetch, and train stages."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from baseline.vanilla_baseline.features import PreparedBatch
from baseline.vanilla_baseline.randomness import TORCH_RNG_LOCK
from baseline.vanilla_baseline.sampling import SamplePlan
from baseline.vanilla_baseline.task import (
    EntitySeedBatch,
    RecommendationSeedBatch,
    SeedBatch,
)

from .contracts import RuntimeComponents, RuntimeResult
from .queue import ByteBoundedQueue


@dataclass(frozen=True)
class AsyncRuntimeConfig:
    seed_queue_bytes: int = 64 * 1024 * 1024
    plan_queue_bytes: int = 2 * 1024 * 1024 * 1024
    ready_queue_bytes: int = 8 * 1024 * 1024 * 1024


class AsyncPipelineExecutor:
    """Overlap one sampler, one feature worker, and the training consumer."""

    def __init__(self, config: AsyncRuntimeConfig | None = None) -> None:
        self.config = config or AsyncRuntimeConfig()

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
        ready_queue = ByteBoundedQueue[PreparedBatch](
            self.config.ready_queue_bytes,
            lambda batch: batch.allocated_bytes,
        )
        errors: list[BaseException] = []
        error_lock = threading.Lock()
        cancel = threading.Event()

        def stop() -> None:
            cancel.set()
            seed_queue.close(discard=True)
            plan_queue.close(discard=True)
            ready_queue.close(discard=True)

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

        def produce_batches() -> None:
            try:
                while not cancel.is_set():
                    plan = plan_queue.get()
                    if plan is None:
                        return
                    features = components.fetcher.fetch(plan)
                    ready_queue.put(components.assembler.assemble(features))
                    del features, plan
            except BaseException as exc:  # noqa: BLE001 - propagate worker interrupts.
                fail(exc)
            finally:
                ready_queue.close()

        workers = [
            threading.Thread(target=produce_seeds, name="seed-reader"),
            threading.Thread(target=produce_plans, name="sampler"),
            threading.Thread(target=produce_batches, name="feature-fetcher"),
        ]
        started_workers: list[threading.Thread] = []
        steps = 0
        examples = 0
        weighted_loss = 0.0
        try:
            for worker in workers:
                worker.start()
                started_workers.append(worker)
            while not cancel.is_set():
                batch = ready_queue.get()
                if batch is None:
                    break
                with TORCH_RNG_LOCK:
                    result = components.trainer.train_step(batch)
                steps += 1
                examples += result.examples
                weighted_loss += result.loss * result.examples
                del batch
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
        return RuntimeResult(
            steps=steps,
            examples=examples,
            mean_loss=weighted_loss / max(examples, 1),
        )


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
