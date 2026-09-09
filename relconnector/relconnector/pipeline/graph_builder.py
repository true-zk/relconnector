"""Wrap ``make_pkey_fkey_graph`` with our chosen text embedder."""

from __future__ import annotations

from pathlib import Path

from relbench.base import Database
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.data import HeteroData

from .text_embedder import GloveTextEmbedder
from .types import ColStats


def infer_col_to_stype(database: Database) -> dict[str, dict[str, stype]]:
    return get_stype_proposal(database)


def build_graph(
    database: Database,
    *,
    cache_dir: Path | None = None,
    text_embedder: GloveTextEmbedder | None = None,
    text_batch_size: int = 256,
    remove_columns: list[tuple[str, str]] | None = None,
) -> tuple[HeteroData, ColStats, dict[str, dict[str, stype]]]:
    col_to_stype = infer_col_to_stype(database)
    text_cfg = None
    if text_embedder is not None:
        text_cfg = TextEmbedderConfig(
            text_embedder=text_embedder,
            batch_size=text_batch_size,
        )
    else:
        for table_stypes in col_to_stype.values():
            for column, column_stype in list(table_stypes.items()):
                if column_stype == stype.text_embedded:
                    table_stypes[column] = stype.categorical
    data, col_stats = make_pkey_fkey_graph(
        database,
        col_to_stype_dict=col_to_stype,
        text_embedder_cfg=text_cfg,
        cache_dir=None if cache_dir is None else str(cache_dir),
        remove_columns=remove_columns,
    )
    return data, col_stats, col_to_stype
