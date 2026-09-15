"""Streaming construction of an in-memory, feature-free CSC graph index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import torch
from relbench.modeling.utils import to_unix_time
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType

from relconnector.connector import BaseDatabaseReader, ForeignKey, TableSchema
from relconnector.connector.base import quote_identifier

from .index import GraphIndex


@dataclass(frozen=True)
class _TableLayout:
    schema: TableSchema
    node_count: int
    where_clause: str
    node_id_expression: str


@dataclass
class _RelationState:
    foreign_key: ForeignKey
    target_node_count: int
    forward_degree: torch.Tensor
    reverse_degree: torch.Tensor
    forward_row: torch.Tensor | None = None
    reverse_row: torch.Tensor | None = None
    forward_offset: torch.Tensor | None = None
    reverse_offset: torch.Tensor | None = None


class InMemoryGraphIndexBuilder:
    """Build PK/FK topology and temporal arrays without reading feature columns."""

    def __init__(
        self,
        reader: BaseDatabaseReader,
        *,
        scan_batch_size: int = 1_000_000,
    ) -> None:
        if scan_batch_size <= 0:
            raise ValueError("scan_batch_size must be greater than zero")
        self.reader = reader
        self.scan_batch_size = scan_batch_size

    def build(self, *, cutoff: pd.Timestamp | None = None) -> GraphIndex:
        data = HeteroData()
        layouts: dict[str, _TableLayout] = {}
        edge_types: list[EdgeType] = []
        colptr_dict: dict[str, torch.Tensor] = {}
        row_dict: dict[str, torch.Tensor] = {}

        for schema in self.reader.schemas().values():
            if schema.kind != "data":
                continue
            where_clause = _time_filter(schema, cutoff)
            node_id_expression = (
                quote_identifier(schema.primary_key)
                if schema.primary_key is not None
                else "rowid - 1"
            )
            node_count = self._count_rows(schema.name, where_clause, node_id_expression)
            layouts[schema.name] = _TableLayout(
                schema=schema,
                node_count=node_count,
                where_clause=where_clause,
                node_id_expression=node_id_expression,
            )
            data[schema.name].num_nodes = node_count
            if schema.time_column is not None:
                data[schema.name].time = self._read_node_time(
                    schema,
                    where_clause,
                    node_id_expression,
                    node_count,
                )

        edge_count = 0
        for source in layouts.values():
            if not source.schema.foreign_keys:
                continue
            relation_indices = self._read_table_relations(source, layouts)
            for foreign_key in source.schema.foreign_keys:
                forward, reverse = relation_indices[foreign_key.column]
                forward_type: EdgeType = (
                    source.schema.name,
                    f"f2p_{foreign_key.column}",
                    foreign_key.reference_table,
                )
                reverse_type: EdgeType = (
                    foreign_key.reference_table,
                    f"rev_f2p_{foreign_key.column}",
                    source.schema.name,
                )
                for edge_type, (colptr, row) in (
                    (forward_type, forward),
                    (reverse_type, reverse),
                ):
                    relation = _relation_name(edge_type)
                    edge_types.append(edge_type)
                    colptr_dict[relation] = colptr
                    row_dict[relation] = row
                    data[edge_type].num_edges = len(row)
                    edge_count += len(row)

        allocated_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in [*colptr_dict.values(), *row_dict.values()]
        )
        allocated_bytes += sum(
            store.time.numel() * store.time.element_size()
            for store in data.node_stores
            if "time" in store
        )
        return GraphIndex(
            data=data,
            edge_types=tuple(edge_types),
            colptr_dict=colptr_dict,
            row_dict=row_dict,
            node_count=sum(layout.node_count for layout in layouts.values()),
            edge_count=edge_count,
            allocated_bytes=allocated_bytes,
        )

    def _count_rows(self, table_name: str, where_clause: str, node_id: str) -> int:
        frame = self.reader.read_query(
            f"SELECT COUNT(*) AS row_count, MIN({node_id}) AS min_id, "
            f"MAX({node_id}) AS max_id FROM {quote_identifier(table_name)}"
            f"{where_clause}"
        )
        count = int(frame.iloc[0, 0])
        if count and (frame.iloc[0, 1] != 0 or frame.iloc[0, 2] != count - 1):
            raise ValueError(
                f"{table_name}: retained node IDs must be a dense zero-based "
                "prefix after time cutoff; arbitrary ID remapping is not supported"
            )
        return count

    def _read_node_time(
        self,
        schema: TableSchema,
        where_clause: str,
        node_id_expression: str,
        node_count: int,
    ) -> torch.Tensor:
        assert schema.time_column is not None
        output = torch.empty(node_count, dtype=torch.long)
        offset = 0
        query = (
            f"SELECT {quote_identifier(schema.time_column)} AS node_time "
            f"FROM {quote_identifier(schema.name)}{where_clause} "
            f"ORDER BY {node_id_expression}"
        )
        for frame in self.reader.iter_query(query, batch_size=self.scan_batch_size):
            values = torch.from_numpy(
                to_unix_time(
                    pd.to_datetime(
                        cast(pd.Series, frame["node_time"]),
                        format="mixed",
                    ).dt.as_unit("ns")
                )
            )
            output[offset : offset + len(values)] = values
            offset += len(values)
        if offset != node_count:
            raise RuntimeError(
                f"Expected {node_count} timestamps for {schema.name}, got {offset}"
            )
        return output

    def _read_table_relations(
        self,
        source: _TableLayout,
        layouts: dict[str, _TableLayout],
    ) -> dict[
        str,
        tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ]:
        states = {
            foreign_key.column: _RelationState(
                foreign_key=foreign_key,
                target_node_count=layouts[foreign_key.reference_table].node_count,
                forward_degree=torch.zeros(
                    layouts[foreign_key.reference_table].node_count,
                    dtype=torch.long,
                ),
                reverse_degree=torch.zeros(source.node_count, dtype=torch.long),
            )
            for foreign_key in source.schema.foreign_keys
        }
        query = self._relation_query(source)

        for frame in self.reader.iter_query(query, batch_size=self.scan_batch_size):
            source_ids = torch.from_numpy(frame["source_id"].to_numpy(dtype=np.int64))
            for state in states.values():
                valid_source, target_ids = _valid_edges(
                    frame,
                    source_ids,
                    state.foreign_key.column,
                    state.target_node_count,
                )
                state.forward_degree += torch.bincount(
                    target_ids,
                    minlength=state.target_node_count,
                )
                state.reverse_degree += torch.bincount(
                    valid_source,
                    minlength=source.node_count,
                )

        for state in states.values():
            forward_colptr = _degree_to_pointer(state.forward_degree)
            reverse_colptr = _degree_to_pointer(state.reverse_degree)
            state.forward_row = torch.empty(int(forward_colptr[-1]), dtype=torch.long)
            state.reverse_row = torch.empty(int(reverse_colptr[-1]), dtype=torch.long)
            state.forward_offset = forward_colptr[:-1].clone()
            state.reverse_offset = reverse_colptr[:-1].clone()

        for frame in self.reader.iter_query(query, batch_size=self.scan_batch_size):
            source_ids = torch.from_numpy(frame["source_id"].to_numpy(dtype=np.int64))
            for state in states.values():
                valid_source, target_ids = _valid_edges(
                    frame,
                    source_ids,
                    state.foreign_key.column,
                    state.target_node_count,
                )
                assert state.forward_row is not None
                assert state.reverse_row is not None
                assert state.forward_offset is not None
                assert state.reverse_offset is not None
                _fill_csc_rows(
                    state.forward_row,
                    state.forward_offset,
                    destination=target_ids,
                    source=valid_source,
                )
                reverse_positions = state.reverse_offset[valid_source]
                state.reverse_row[reverse_positions] = target_ids
                state.reverse_offset[valid_source] += 1

        output = {}
        for column, state in states.items():
            assert state.forward_row is not None
            assert state.reverse_row is not None
            assert state.forward_offset is not None
            assert state.reverse_offset is not None
            forward_colptr = _degree_to_pointer(state.forward_degree)
            reverse_colptr = _degree_to_pointer(state.reverse_degree)
            if not torch.equal(state.forward_offset, forward_colptr[1:]):
                raise RuntimeError(
                    f"Forward CSC fill mismatch for {source.schema.name}.{column}"
                )
            if not torch.equal(state.reverse_offset, reverse_colptr[1:]):
                raise RuntimeError(
                    f"Reverse CSC fill mismatch for {source.schema.name}.{column}"
                )
            output[column] = (
                (forward_colptr, state.forward_row),
                (reverse_colptr, state.reverse_row),
            )
        return output

    def _relation_query(self, source: _TableLayout) -> str:
        projections = [f"{source.node_id_expression} AS source_id"]
        projections.extend(
            quote_identifier(foreign_key.column)
            for foreign_key in source.schema.foreign_keys
        )
        # Temporal CSC requires neighbors ordered by source time within each
        # destination. Stable placement preserves this across SQL chunks.
        order = source.node_id_expression
        if source.schema.time_column is not None:
            order = f"julianday({quote_identifier(source.schema.time_column)}), {order}"
        return (
            f"SELECT {', '.join(projections)} "
            f"FROM {quote_identifier(source.schema.name)}{source.where_clause} "
            f"ORDER BY {order}"
        )


def _valid_edges(
    frame: pd.DataFrame,
    source_ids: torch.Tensor,
    column: str,
    target_node_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = cast(
        pd.Series,
        pd.to_numeric(cast(pd.Series, frame[column]), errors="coerce"),
    )
    valid = values.notna() & (values >= 0) & (values < target_node_count)
    mask = torch.from_numpy(valid.to_numpy(dtype=np.bool_))
    target = torch.from_numpy(cast(pd.Series, values[valid]).to_numpy(dtype=np.int64))
    return source_ids[mask], target


def _fill_csc_rows(
    row: torch.Tensor,
    write_offset: torch.Tensor,
    *,
    destination: torch.Tensor,
    source: torch.Tensor,
) -> None:
    if len(destination) == 0:
        return
    order = torch.argsort(destination, stable=True)
    sorted_destination = destination[order]
    sorted_source = source[order]
    unique_destination, counts = torch.unique_consecutive(
        sorted_destination,
        return_counts=True,
    )
    starts = torch.cumsum(counts, dim=0) - counts
    local_rank = torch.arange(len(destination)) - torch.repeat_interleave(
        starts,
        counts,
    )
    positions = write_offset[sorted_destination] + local_rank
    row[positions] = sorted_source
    write_offset[unique_destination] += counts


def _degree_to_pointer(degree: torch.Tensor) -> torch.Tensor:
    pointer = torch.empty(len(degree) + 1, dtype=torch.long)
    pointer[0] = 0
    torch.cumsum(degree, dim=0, out=pointer[1:])
    return pointer


def _time_filter(schema: TableSchema, cutoff: pd.Timestamp | None) -> str:
    if cutoff is None or schema.time_column is None:
        return ""
    timestamp = cutoff.isoformat().replace("'", "''")
    return (
        f" WHERE julianday({quote_identifier(schema.time_column)}) "
        f"<= julianday('{timestamp}')"
    )


def _relation_name(edge_type: EdgeType) -> str:
    return "__".join(edge_type)
