"""Pass pyg-lib samples to bounded C++ SQL feature workers."""

from __future__ import annotations

from threading import Lock, Thread
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from relx.sampling import (
    EntitySamplePlan,
    PygLibMultiHopSampler,
    RecommendationSamplePlan,
)
from relx.task import EntitySeedBatch, RecommendationSeedBatch

from .contracts import PreparedBatch, PreparedRecommendationBatch, SeedSource
from .online_store import OnlineFeatureStore

if TYPE_CHECKING:
    from relx._relx_native import NativeOnlinePipeline


class OnlineBatchIterator:
    """Sample in Python; read and process each batch in native worker threads."""

    def __init__(
        self,
        source: SeedSource,
        sampler: PygLibMultiHopSampler,
        store: OnlineFeatureStore,
        *,
        feature_workers: int = 2,
        max_inflight: int = 4,
        max_batches: int | None = None,
    ) -> None:
        if min(feature_workers, max_inflight) <= 0:
            raise ValueError("feature_workers and max_inflight must be positive")
        if max_batches is not None and max_batches <= 0:
            raise ValueError("max_batches must be positive")
        self.source = source
        self.sampler = sampler
        self.store = store
        self.feature_workers = feature_workers
        self.max_inflight = max_inflight
        self.max_batches = max_batches
        self._pipeline: NativeOnlinePipeline | None = None
        self._producer: Thread | None = None
        self._plans: dict[int, EntitySamplePlan] = {}
        self._triplets: set[int] = set()
        self._ready_parts: list[PreparedBatch] = []
        self._lock = Lock()
        self._error: Exception | None = None
        self._exhausted = False

    def start_epoch(self, epoch: int) -> None:
        self.close()
        from relx._relx_native import NativeOnlinePipeline

        self._plans.clear()
        self._triplets.clear()
        self._ready_parts.clear()
        self._error = None
        self._exhausted = False
        self._pipeline = NativeOnlinePipeline(
            self.store.native,
            min(self.max_inflight, self.feature_workers * self.store.native_workers),
            self.max_inflight,
            self.store.prepared.config.training.prefetch_bytes,
        )
        self._producer = Thread(target=self._produce, args=(epoch,), daemon=True)
        self._producer.start()

    def _produce(self, epoch: int) -> None:
        assert self._pipeline is not None
        try:
            for sequence, seeds in enumerate(self.source.iter_epoch(epoch)):
                if self.max_batches is not None and sequence >= self.max_batches:
                    break
                plan = self.sampler.sample(seeds)
                if isinstance(seeds, EntitySeedBatch) and isinstance(
                    plan, EntitySamplePlan
                ):
                    self._submit(sequence, plan)
                elif isinstance(seeds, RecommendationSeedBatch) and isinstance(
                    plan, RecommendationSamplePlan
                ):
                    with self._lock:
                        self._triplets.add(3 * sequence)
                    for part, subgraph in enumerate(
                        (plan.source, plan.positive, plan.negative)
                    ):
                        wrapped = EntitySamplePlan(
                            plan.key, subgraph, np.empty(0), seeds.sample_keys
                        )
                        self._submit(3 * sequence + part, wrapped)
                else:
                    raise TypeError("Seed and sample plan types differ")
        except Exception as exc:  # noqa: BLE001 - forward producer errors
            self._error = exc
            self._pipeline.abort()
        finally:
            self._pipeline.finish()

    def _submit(self, sequence: int, plan: EntitySamplePlan) -> None:
        assert self._pipeline is not None
        with self._lock:
            self._plans[sequence] = plan
        self._pipeline.submit(sequence, {
            str(ref): ids for ref, ids in plan.subgraph.node_ids.items()
        })

    def next_batch(self) -> PreparedBatch | PreparedRecommendationBatch | None:
        pipeline = self._pipeline
        if pipeline is None:
            raise RuntimeError("Call start_epoch() before next_batch()")
        if self._exhausted:
            return None
        try:
            while True:
                result = pipeline.next()
                if result is None:
                    error = self._error
                    self.close()
                    if error is not None:
                        raise error
                    return None
                sequence = cast(int, result["sequence"])
                with self._lock:
                    plan = self._plans.pop(sequence)
                    first_of_triplet = sequence in self._triplets
                features = cast(dict[str, dict[str, dict[str, Any]]], result["features"])
                prepared = self.store.assemble(plan, features)
                if first_of_triplet:
                    self._ready_parts.clear()
                if first_of_triplet or self._ready_parts:
                    self._ready_parts.append(prepared)
                    if len(self._ready_parts) == 3:
                        source, positive, negative = self._ready_parts
                        self._ready_parts = []
                        with self._lock:
                            self._triplets.discard(sequence - 2)
                        return PreparedRecommendationBatch(source, positive, negative)
                    continue
                return prepared
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self._exhausted = True
        pipeline = self._pipeline
        if pipeline is not None:
            pipeline.abort()
        if self._producer is not None:
            self._producer.join()
            self._producer = None
        if pipeline is not None:
            pipeline.close()
            self._pipeline = None
        self._ready_parts.clear()
