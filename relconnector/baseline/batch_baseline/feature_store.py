"""Eager full-memory TensorFrame store used by the strict baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch_frame
from relbench.base import Database
from torch_frame import TensorFrame

from baseline.vanilla_baseline.features import (
    EntityFeatureBatch,
    FetchedSubgraph,
    RecommendationFeatureBatch,
    TensorFrameEncoder,
)
from baseline.vanilla_baseline.sampling import (
    EntitySamplePlan,
    RecommendationSamplePlan,
    SampledSubgraph,
)


@dataclass(frozen=True)
class EagerFeatureStore:
    frames: dict[str, TensorFrame]
    allocated_bytes: int

    @classmethod
    def materialize(
        cls,
        database: Database,
        encoder: TensorFrameEncoder,
        *,
        cache_dir: Path | None = None,
    ) -> EagerFeatureStore:
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        frames: dict[str, TensorFrame] = {}
        for table, relational_table in database.table_dict.items():
            path = None if cache_dir is None else cache_dir / f"{table}.pt"
            if path is not None and path.exists():
                tensor_frame, _ = torch_frame.load(str(path))
            else:
                tensor_frame = encoder.encode(table, relational_table.df)
                if path is not None:
                    torch_frame.save(
                        tensor_frame,
                        encoder.schema.col_stats_dict[table],
                        str(path),
                    )
            frames[table] = tensor_frame
        return cls(
            frames=frames,
            allocated_bytes=sum(
                _tensor_frame_bytes(frame) for frame in frames.values()
            ),
        )


class InMemoryTensorFrameFetcher:
    """Select already encoded rows using the exact online SamplePlan."""

    def __init__(self, store: EagerFeatureStore) -> None:
        self.store = store

    def fetch(
        self,
        plan: EntitySamplePlan | RecommendationSamplePlan,
    ) -> EntityFeatureBatch | RecommendationFeatureBatch:
        if isinstance(plan, EntitySamplePlan):
            return EntityFeatureBatch(
                key=plan.key,
                subgraph=self._fetch_subgraph(plan.subgraph),
                target=plan.target,
            )
        return RecommendationFeatureBatch(
            key=plan.key,
            source=self._fetch_subgraph(plan.source),
            positive=self._fetch_subgraph(plan.positive),
            negative=self._fetch_subgraph(plan.negative),
        )

    def _fetch_subgraph(self, sample: SampledSubgraph) -> FetchedSubgraph:
        return FetchedSubgraph(
            sample=sample,
            frames={
                node_type: self.store.frames[node_type][node_ids]
                for node_type, node_ids in sample.node_ids.items()
            },
        )


def _tensor_frame_bytes(frame: TensorFrame) -> int:
    return sum(_value_bytes(value) for value in frame.feat_dict.values())


def _value_bytes(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_value_bytes(item) for item in value.values())
    values = getattr(value, "values", None)
    offset = getattr(value, "offset", None)
    size = _value_bytes(values) if isinstance(values, torch.Tensor) else 0
    if isinstance(offset, torch.Tensor):
        size += _value_bytes(offset)
    return size
