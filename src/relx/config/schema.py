"""Parse schema corrections and feature exclusions outside the task section."""

from __future__ import annotations

from relx.core import ForeignKey, TableRef
from relx.preprocess import SchemaOverrides

from .common import fields, mapping, table_ref, text


def parse_schema(value: object, default_schema: str | None) -> SchemaOverrides:
    data = mapping(value, "schema")
    fields(data, {"identity_keys", "foreign_keys"}, "schema")
    raw_keys = mapping(data.get("identity_keys", {}), "schema.identity_keys")
    keys = {
        table_ref(name, "schema.identity_keys table", default_schema): _names(
            columns, f"schema.identity_keys.{name}"
        )
        for name, columns in raw_keys.items()
    }
    raw_foreign_keys = data.get("foreign_keys", [])
    if not isinstance(raw_foreign_keys, list):
        raise TypeError("schema.foreign_keys must be a list")
    foreign_keys: dict[TableRef, list[ForeignKey]] = {}
    for index, raw_item in enumerate(raw_foreign_keys):
        path = f"schema.foreign_keys[{index}]"
        item = mapping(raw_item, path)
        fields(
            item,
            {"source_table", "source_columns", "target_table", "target_columns"},
            path,
        )
        source = table_ref(
            item.get("source_table"), f"{path}.source_table", default_schema
        )
        target = table_ref(
            item.get("target_table"), f"{path}.target_table", default_schema
        )
        foreign_keys.setdefault(source, []).append(
            ForeignKey(
                _names(item.get("source_columns"), f"{path}.source_columns"),
                target,
                _names(item.get("target_columns"), f"{path}.target_columns"),
            )
        )
    return SchemaOverrides(
        identity_keys=keys,
        foreign_keys={ref: tuple(items) for ref, items in foreign_keys.items()},
    )


def parse_hidden_columns(
    value: object, default_schema: str | None
) -> tuple[tuple[TableRef, str], ...]:
    if not isinstance(value, list):
        raise TypeError("hidden_columns must be a list of [table, column] pairs")
    pairs: list[tuple[TableRef, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"hidden_columns[{index}] must be [table, column]")
        pairs.append(
            (
                table_ref(item[0], f"hidden_columns[{index}][0]", default_schema),
                text(item[1], f"hidden_columns[{index}][1]"),
            )
        )
    return tuple(dict.fromkeys(pairs))


def _names(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a nonempty list of column names")
    names = tuple(text(item, f"{path}[]") for item in value)
    if len(names) != len(set(names)):
        raise ValueError(f"{path} contains duplicate columns")
    return names
