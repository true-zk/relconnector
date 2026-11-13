"""Small strict YAML parsing primitives shared across sections."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from relx.core import TableRef


def mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{path} must be a mapping with string keys")
    return value


def fields(values: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown fields in {path}: {', '.join(sorted(unknown))}")


def text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a nonempty string")
    return value


def optional_text(value: object, path: str) -> str | None:
    return None if value is None else text(value, path)


def table_ref(value: object, path: str, default_schema: str | None = None) -> TableRef:
    if isinstance(value, str):
        parts = text(value, path).split(".")
        if len(parts) == 1:
            return TableRef(parts[0], default_schema)
        if len(parts) == 2 and all(parts):
            return TableRef(parts[1], parts[0])
    elif isinstance(value, dict):
        data = mapping(value, path)
        fields(data, {"name", "schema"}, path)
        return TableRef(
            text(data.get("name"), f"{path}.name"),
            optional_text(data.get("schema"), f"{path}.schema") or default_schema,
        )
    raise ValueError(f"{path} must be a table name or {{name, schema}} mapping")


def positive_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def fraction(value: object, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 < value < 1
    ):
        raise ValueError(f"{path} must be between 0 and 1")
    return float(value)


def nonnegative_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a nonnegative integer")
    return value
