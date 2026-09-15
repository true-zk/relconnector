"""PyG/pyg-lib neighbor-loader construction for node and link tasks."""

from __future__ import annotations

from typing import cast

import pandas as pd
from relbench.base import EntityTask, RecommendationTask, Table
from relbench.modeling.graph import (
    get_link_train_table_input,
    get_node_train_table_input,
)
from relbench.modeling.loader import LinkNeighborLoader
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from .config import TrainingConfig
from .dataset import LocalEntityTask, LocalRecommendationTask, LocalTask


def _train_table(task: LocalTask) -> Table:
    table = task.get_table("train")
    if table.time_col is not None:
        table.df[table.time_col] = cast(pd.Series, table.df[table.time_col]).dt.as_unit(
            "ns"
        )
    return table


def make_train_loader(
    data: HeteroData,
    task: LocalTask,
    config: TrainingConfig,
) -> NeighborLoader | LinkNeighborLoader:
    if isinstance(task, LocalRecommendationTask):
        return _make_link_loader(data, task, config)
    return _make_node_loader(data, task, config)


def _make_node_loader(
    data: HeteroData,
    task: LocalEntityTask,
    config: TrainingConfig,
) -> NeighborLoader:
    inp = get_node_train_table_input(_train_table(task), cast(EntityTask, task))
    if inp.time is None:
        return NeighborLoader(
            data,
            num_neighbors=list(config.num_neighbors),
            input_nodes=inp.nodes,
            transform=inp.transform,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
    return NeighborLoader(
        data,
        num_neighbors=list(config.num_neighbors),
        input_nodes=inp.nodes,
        input_time=inp.time,
        transform=inp.transform,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        time_attr="time",
        temporal_strategy="uniform",
    )


def _make_link_loader(
    data: HeteroData,
    task: LocalRecommendationTask,
    config: TrainingConfig,
) -> LinkNeighborLoader:
    num_dst_nodes = int(data[task.dst_entity_table].num_nodes)
    inp = get_link_train_table_input(
        _train_table(task),
        cast(RecommendationTask, task),
        num_dst_nodes,
    )
    if inp.dst_nodes is None:
        raise ValueError(f"Training labels are incomplete for task {task.name!r}")
    if inp.src_time is None:
        return LinkNeighborLoader(
            data,
            num_neighbors=list(config.num_neighbors),
            src_nodes=inp.src_nodes,
            dst_nodes=inp.dst_nodes,
            num_dst_nodes=inp.num_dst_nodes,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
    return LinkNeighborLoader(
        data,
        num_neighbors=list(config.num_neighbors),
        src_nodes=inp.src_nodes,
        dst_nodes=inp.dst_nodes,
        num_dst_nodes=inp.num_dst_nodes,
        src_time=inp.src_time,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        time_attr="time",
        temporal_strategy="uniform",
    )
