"""Select business tables for graph construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from relx.core import DatabaseSchema, TableRef
from relx.task import ValidatedTask

if TYPE_CHECKING:
    from relx.config import GraphConfig


def select_graph_tables(
    schema: DatabaseSchema, task: ValidatedTask, config: GraphConfig
) -> tuple[TableRef, ...]:
    """Apply explicit exclusions and always omit the current task table."""
    available = set(schema.tables)
    unknown = set(config.exclude_tables) - available
    if unknown:
        raise ValueError(
            "graph.exclude_tables contains unknown tables: "
            + ", ".join(sorted(map(str, unknown)))
        )

    excluded = {*config.exclude_tables, task.spec.table}
    required = {task.entity_table}
    if task.destination_table is not None:
        required.add(task.destination_table)
    missing = required & excluded
    if missing:
        raise ValueError(
            "graph.exclude_tables cannot remove task entity tables: "
            + ", ".join(sorted(map(str, missing)))
        )

    return tuple(ref for ref in schema.tables if ref not in excluded)
