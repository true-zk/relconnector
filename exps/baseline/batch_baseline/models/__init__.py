"""Model registry and shared base class."""

from __future__ import annotations

from collections.abc import Callable

from torch_geometric.data import HeteroData

from ..types import ColStats
from .base import BaseLinkRelBenchModel, BaseRelBenchModel
from .graphsage import GraphSAGEModel, LinkGraphSAGEModel

MODEL_REGISTRY: dict[str, Callable[..., BaseRelBenchModel]] = {
    "graphsage": GraphSAGEModel,
}
LINK_MODEL_REGISTRY: dict[str, Callable[..., BaseLinkRelBenchModel]] = {
    "graphsage": LinkGraphSAGEModel,
}


def register_model(
    name: str,
    factory: Callable[..., BaseRelBenchModel],
    *,
    link_factory: Callable[..., BaseLinkRelBenchModel] | None = None,
) -> None:
    MODEL_REGISTRY[name] = factory
    if link_factory is not None:
        LINK_MODEL_REGISTRY[name] = link_factory


def build_model(
    name: str,
    *,
    data: HeteroData,
    col_stats_dict: ColStats,
    out_channels: int,
    channels: int,
    num_layers: int,
    aggr: str,
    recommendation: bool = False,
) -> BaseRelBenchModel | BaseLinkRelBenchModel:
    registry = LINK_MODEL_REGISTRY if recommendation else MODEL_REGISTRY
    try:
        factory = registry[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown model {name!r}. Registered: {sorted(registry)}"
        ) from exc
    return factory(
        data=data,
        col_stats_dict=col_stats_dict,
        out_channels=out_channels,
        channels=channels,
        num_layers=num_layers,
        aggr=aggr,
    )


__all__ = [
    "LINK_MODEL_REGISTRY",
    "MODEL_REGISTRY",
    "BaseLinkRelBenchModel",
    "BaseRelBenchModel",
    "GraphSAGEModel",
    "LinkGraphSAGEModel",
    "build_model",
    "register_model",
]
