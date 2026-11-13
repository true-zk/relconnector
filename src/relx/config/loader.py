"""Load one YAML file and dispatch sections to the owning configuration types."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .common import fields, mapping
from .runtime import (
    parse_database,
    parse_features,
    parse_model,
    parse_report,
    parse_training,
)
from .schema import parse_hidden_columns, parse_schema
from .task import parse_task
from .types import RunConfig


def load_config(path: str | Path) -> RunConfig:
    """Parse a complete run config without opening the database or writing files."""
    source = Path(path).expanduser().resolve()
    values: Any = yaml.safe_load(source.read_text(encoding="utf-8"))
    data = mapping(values, str(source))
    fields(
        data,
        {
            "database",
            "task",
            "schema",
            "hidden_columns",
            "features",
            "model",
            "training",
            "report",
        },
        str(source),
    )
    database = parse_database(data.get("database"), source.parent)
    task = parse_task(data.get("task"), database.schema_name)
    return RunConfig(
        database=database,
        task=task,
        schema=parse_schema(data.get("schema", {}), database.schema_name),
        hidden_columns=parse_hidden_columns(
            data.get("hidden_columns", []), database.schema_name
        ),
        features=parse_features(data.get("features", {}), database.schema_name),
        model=parse_model(data.get("model", {})),
        training=parse_training(data.get("training", {})),
        report=parse_report(data.get("report", {}), source.parent, task.name),
    )
