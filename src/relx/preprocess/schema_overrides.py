"""Validate user supplied identity keys and missing relationships."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from sqlalchemy.engine.reflection import Inspector

from relx.core import ForeignKey, TableRef, TableSchema


@dataclass(frozen=True)
class SchemaOverrides:
    identity_keys: Mapping[TableRef, tuple[str, ...]] = field(default_factory=dict)
    foreign_keys: Mapping[TableRef, tuple[ForeignKey, ...]] = field(
        default_factory=dict
    )


def apply_schema_overrides(
    tables: dict[TableRef, TableSchema],
    inspector: Inspector,
    overrides: SchemaOverrides,
) -> dict[TableRef, TableSchema]:
    """Add configured keys and relationships after checking table and column names."""
    unknown = (set(overrides.identity_keys) | set(overrides.foreign_keys)) - set(tables)
    if unknown:
        raise ValueError(
            "Overrides refer to unselected tables: "
            + ", ".join(sorted(map(str, unknown)))
        )

    result = dict(tables)
    for ref, key in overrides.identity_keys.items():
        _check_columns(ref, key, {column.name for column in result[ref].columns})
        result[ref] = replace(result[ref], identity_key=key)

    for ref, extra_keys in overrides.foreign_keys.items():
        existing = result[ref].foreign_keys
        source_names = {column.name for column in result[ref].columns}
        for key in extra_keys:
            _check_columns(ref, key.source_columns, source_names)
            target = key.target_table
            if target in result:
                target_names = {column.name for column in result[target].columns}
            elif inspector.has_table(target.table_name, schema=target.schema_name):
                target_names = {
                    str(column["name"])
                    for column in inspector.get_columns(
                        target.table_name, schema=target.schema_name
                    )
                }
            else:
                raise ValueError(f"Unknown referenced table: {target}")
            _check_columns(target, key.target_columns, target_names)
            if len(key.source_columns) != len(key.target_columns):
                raise ValueError(f"Foreign key on {ref} has unequal column counts")
        result[ref] = replace(
            result[ref], foreign_keys=tuple(dict.fromkeys((*existing, *extra_keys)))
        )
    return result


def _check_columns(ref: TableRef, columns: tuple[str, ...], names: set[str]) -> None:
    if not columns or len(columns) != len(set(columns)):
        raise ValueError(f"Key on {ref} must contain distinct columns")
    missing = set(columns) - names
    if missing:
        raise ValueError(f"Unknown columns on {ref}: {', '.join(sorted(missing))}")
