"""Build the eager full-feature graph with the shared TensorFrame schema."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
import torch_frame
from relbench.base import Database
from relbench.modeling.utils import to_unix_time
from torch_frame import stype
from torch_geometric.data import HeteroData
from torch_geometric.utils import sort_edge_index

from baseline.vanilla_baseline.features import (
    TensorFrameEncoder,
    TensorFrameFeatureSchema,
)

from .text_embedder import GloveTextEmbedder
from .types import ColStats


def build_graph(
    database: Database,
    *,
    feature_schema: TensorFrameFeatureSchema,
    cache_dir: Path | None = None,
    text_embedder: GloveTextEmbedder | None = None,
    text_batch_size: int = 256,
    remove_columns: list[tuple[str, str]] | None = None,
) -> tuple[HeteroData, ColStats, dict[str, dict[str, stype]]]:
    """Materialize all features while sharing online's schema and statistics."""
    del remove_columns
    encoder = TensorFrameEncoder(
        feature_schema,
        text_embedder,
        text_batch_size=text_batch_size,
    )
    data = HeteroData()
    col_stats = feature_schema.col_stats_dict
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for table_name, table in database.table_dict.items():
        frame = table.df
        if (
            table.pkey_col is not None
            and not (frame[table.pkey_col].to_numpy() == np.arange(len(frame))).all()
        ):
            raise ValueError(f"{table_name}: primary keys must be dense and zero-based")

        cache_path = None if cache_dir is None else cache_dir / f"{table_name}.pt"
        if cache_path is not None and cache_path.exists():
            tensor_frame, _ = torch_frame.load(str(cache_path))
        else:
            tensor_frame = encoder.encode(table_name, frame)
            if cache_path is not None:
                torch_frame.save(
                    tensor_frame,
                    col_stats[table_name],
                    str(cache_path),
                )
        data[table_name].tf = tensor_frame
        data[table_name].num_nodes = len(frame)
        if table.time_col is not None:
            timestamps = cast(pd.Series, frame[table.time_col]).dt.as_unit("ns")
            data[table_name].time = torch.from_numpy(to_unix_time(timestamps))

        for foreign_key, target_table in table.fkey_col_to_pkey_table.items():
            target = cast(pd.Series, frame[foreign_key])
            valid = cast(pd.Series, ~target.isna())
            source_ids = torch.arange(len(target))[torch.from_numpy(valid.to_numpy())]
            target_ids = torch.from_numpy(
                cast(pd.Series, target[valid]).astype(int).to_numpy()
            )
            if not (target_ids < len(database.table_dict[target_table])).all():
                raise ValueError(f"Dangling foreign key {table_name}.{foreign_key}")
            forward = (table_name, f"f2p_{foreign_key}", target_table)
            reverse = (target_table, f"rev_f2p_{foreign_key}", table_name)
            data[forward].edge_index = sort_edge_index(
                torch.stack((source_ids, target_ids))
            )
            data[reverse].edge_index = sort_edge_index(
                torch.stack((target_ids, source_ids))
            )

    data.validate()
    stypes = {
        table: {
            column: semantic_type for column, semantic_type in spec.col_to_stype.items()
        }
        for table, spec in feature_schema.tables.items()
    }
    return data, col_stats, stypes
