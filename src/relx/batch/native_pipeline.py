"""Prototype C++ pipeline backed by a full-table feature snapshot."""

from __future__ import annotations

from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any

import numpy as np

from relx.sampling import NativeMultiHopSampler, NativeOneHopSampler
from relx.task import EntitySeedBatch

from .contracts import PreparedBatch, SeedSource

if TYPE_CHECKING:
    from relx._relx_native import NativeBatchPipeline

    from .feature_store import FeatureStore


class NativeBatchIterator:
    """Feed seeds to C++; only completed sampled and featured batches return."""

    def __init__(
        self,
        source: SeedSource,
        sampler: NativeOneHopSampler | NativeMultiHopSampler,
        store: FeatureStore,
        *,
        feature_workers: int = 2,
        max_inflight: int = 4,
    ) -> None:
        if feature_workers < 1 or max_inflight < 1:
            raise ValueError("feature_workers and max_inflight must be positive")
        self.source = source
        self.sampler = sampler
        self.store = store
        self.feature_workers = feature_workers
        self.max_inflight = max_inflight
        self._pipeline: NativeBatchPipeline | None = None
        self._feeder: Thread | None = None
        self._metadata: dict[int, EntitySeedBatch] = {}
        self._lock = Lock()
        self._stop = Event()
        self._error: Exception | None = None
        self._exhausted = False

    def start_epoch(self, epoch: int) -> None:
        from relx._relx_native import NativeBatchPipeline

        self.close()
        self._stop = Event()
        self._error = None
        self._exhausted = False
        fanouts = (
            list(self.sampler.fanouts)
            if isinstance(self.sampler, NativeMultiHopSampler)
            else [self.sampler.fanout]
        )
        pipeline = NativeBatchPipeline(
            self.sampler.graph,
            self.store.native,
            fanouts,
            isinstance(self.sampler, NativeMultiHopSampler),
            self.sampler.workers,
            self.feature_workers,
            self.store.native_workers,
            self.max_inflight,
        )
        self._pipeline = pipeline
        self._feeder = Thread(target=self._feed, args=(pipeline, epoch), daemon=True)
        self._feeder.start()

    def _feed(self, pipeline: NativeBatchPipeline, epoch: int) -> None:
        try:
            for sequence, batch in enumerate(self.source.iter_epoch(epoch)):
                if self._stop.is_set():
                    break
                if not isinstance(batch, EntitySeedBatch):
                    raise TypeError("Native pipeline currently accepts entity seeds")
                cutoffs = (
                    batch.seed_time_us
                    if batch.seed_time_us is not None
                    else np.full(
                        len(batch.node_ids), np.iinfo(np.int64).max,
                        dtype=np.int64,
                    )
                )
                with self._lock:
                    self._metadata[sequence] = batch
                pipeline.submit(
                    sequence, batch.node_type, batch.node_ids.tolist(),
                    cutoffs.tolist(), self.sampler.seed_for(batch),
                )
        except Exception as exc:  # noqa: BLE001 - propagate source errors
            if not self._stop.is_set():
                self._error = exc
        finally:
            pipeline.finish()

    def next_batch(self) -> PreparedBatch | None:
        pipeline = self._pipeline
        if pipeline is None:
            raise RuntimeError("Call start_epoch() before next_batch()")
        if self._exhausted:
            return None
        try:
            result: dict[str, Any] | None = pipeline.next()
            if result is None:
                self._exhausted = True
                error = self._error
                self.close()
                if error is not None:
                    raise error
                return None
            sequence = int(result["sequence"])
            with self._lock:
                seeds = self._metadata.pop(sequence)
            plan = self.sampler.decode_sampled(seeds, result["sample"])
            return self.store.assemble(plan, result["features"])
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self._stop.set()
        self._exhausted = True
        pipeline = self._pipeline
        if pipeline is not None:
            pipeline.close()
            self._pipeline = None
        if self._feeder is not None:
            self._feeder.join()
            self._feeder = None
        with self._lock:
            self._metadata.clear()
