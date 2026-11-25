"""Sequential orchestration around a threaded native sampler."""

from __future__ import annotations

from collections.abc import Iterator

from relx.task import SeedBatch

from .contracts import FeatureFetcher, GraphSampler, PreparedBatch, SeedSource


class SynchronousBatchIterator:
    def __init__(
        self, source: SeedSource, sampler: GraphSampler, fetcher: FeatureFetcher
    ) -> None:
        self.source = source
        self.sampler = sampler
        self.fetcher = fetcher
        self._seeds: Iterator[SeedBatch] | None = None

    def start_epoch(self, epoch: int) -> None:
        self._seeds = iter(self.source.iter_epoch(epoch))

    def next_batch(self) -> PreparedBatch | None:
        if self._seeds is None:
            raise RuntimeError("Call start_epoch() before next_batch()")
        try:
            seeds = next(self._seeds)
        except StopIteration:
            return None
        return self.fetcher.fetch(self.sampler.sample(seeds))
