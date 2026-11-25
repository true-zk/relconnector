"""Build a feature-free relational topology from Arrow column batches."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from sqlalchemy import column, select, table
from sqlalchemy.sql.selectable import Select

from relx.conn import BaseDatabaseReader
from relx.core import DatabaseSchema, TableRef
from relx.task import ValidatedTask

from .graph_scope import select_graph_tables

if TYPE_CHECKING:
    from relx._relx_native import GraphIndex
    from relx.config import GraphConfig


def build_graph(
    reader: BaseDatabaseReader,
    schema: DatabaseSchema,
    task: ValidatedTask,
    config: GraphConfig,
    *,
    batch_size: int = 100_000,
) -> GraphIndex:
    """Build a native graph with per-node and per-edge availability times."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    from .graph_time import normalize_time_batch, resolve_graph_time

    try:
        from relx import _relx_native
    except ImportError as exc:
        raise RuntimeError(
            "RelX graph construction requires the native extension; "
            "run scripts/build_relx_native.py"
        ) from exc

    refs = select_graph_tables(schema, task, config)
    time_plan = resolve_graph_time(schema, task, config, refs)
    builder = _relx_native.GraphBuilder(time_plan.inclusive)
    identities: dict[TableRef, tuple[str, ...]] = {}
    for ref in refs:
        identity = schema.tables[ref].identity_key
        if not identity:
            raise ValueError(f"Graph table {ref} needs an identity key")
        identities[ref] = identity
        builder.register_table(str(ref), time_plan.columns[ref] is not None)

    for ref in refs:
        identity = identities[ref]
        time_column = time_plan.columns[ref]
        projection = (*identity, time_column) if time_column else identity
        query = _select_columns(ref, projection, order_by=identity)
        for batch in reader.iter_arrow_query(query, batch_size=batch_size):
            if time_column is not None:
                batch = normalize_time_batch(batch, time_column, time_plan.timezone)
            builder.add_nodes(str(ref), list(identity), batch)

    included = set(refs)
    for ref in refs:
        source = schema.tables[ref]
        relevant = [
            (index, fk)
            for index, fk in enumerate(source.foreign_keys)
            if fk.target_table in included
        ]
        if not relevant:
            continue
        projection = list(identities[ref])
        for _, fk in relevant:
            if fk.target_columns != identities[fk.target_table]:
                raise ValueError(
                    f"{ref}: foreign key target {fk.target_table} must use "
                    "its identity key"
                )
            projection.extend(fk.source_columns)
        query = _select_columns(ref, projection, order_by=identities[ref])
        names = {
            index: f"{ref}:{index}:{','.join(fk.source_columns)}->{fk.target_table}"
            for index, fk in relevant
        }
        for index, fk in relevant:
            builder.register_relation(names[index], str(ref), str(fk.target_table))
        for batch in reader.iter_arrow_query(query, batch_size=batch_size):
            for index, fk in relevant:
                builder.add_edges(
                    names[index],
                    str(ref),
                    str(fk.target_table),
                    list(identities[ref]),
                    list(fk.source_columns),
                    batch,
                )
    return builder.finish()


def _select_columns(
    ref: TableRef,
    columns: Iterable[str],
    *,
    order_by: tuple[str, ...],
) -> Select:
    names = tuple(dict.fromkeys(columns))
    source = table(
        ref.table_name,
        *(column(name) for name in names),
        schema=ref.schema_name,
    )
    return select(*(source.c[name] for name in names)).order_by(
        *(source.c[name] for name in order_by)
    )
