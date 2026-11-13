"""Parse settings used by database, feature, model, and reporting modules."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import cast

from relx.conn.factory import ReaderKind

from .common import (
    fields,
    mapping,
    nonnegative_int,
    optional_text,
    positive_int,
    table_ref,
    text,
)
from .types import (
    DatabaseConfig,
    FeatureColumnConfig,
    FeaturesConfig,
    ModelConfig,
    ReportConfig,
    TrainingConfig,
)


def parse_database(value: object, config_dir: Path) -> DatabaseConfig:
    data = mapping(value, "database")
    fields(data, {"url", "url_env", "reader", "schema"}, "database")
    if ("url" in data) == ("url_env" in data):
        raise ValueError("database requires exactly one of url or url_env")
    if "url_env" in data:
        name = text(data["url_env"], "database.url_env")
        location = os.environ.get(name)
        if not location:
            raise ValueError(f"Environment variable {name} is not set")
    else:
        location = text(data["url"], "database.url")
    if location == ":memory:":
        location = "sqlite+pysqlite:///:memory:"
    elif "://" not in location:
        location = str((config_dir / location).resolve())
    reader = text(data.get("reader", "sqlalchemy"), "database.reader")
    if reader not in {"sqlalchemy", "pandas", "connectorx"}:
        raise ValueError(f"Unknown database.reader: {reader}")
    return DatabaseConfig(
        location=location,
        reader=cast(ReaderKind, reader),
        schema_name=optional_text(data.get("schema"), "database.schema"),
    )


def parse_features(value: object, default_schema: str | None) -> FeaturesConfig:
    data = mapping(value, "features")
    fields(data, {"columns"}, "features")
    raw = data.get("columns", [])
    if not isinstance(raw, list):
        raise TypeError("features.columns must be a list")
    parsed: list[FeatureColumnConfig] = []
    seen: set[tuple[object, str]] = set()
    for index, raw_item in enumerate(raw):
        path = f"features.columns[{index}]"
        item = mapping(raw_item, path)
        fields(item, {"table", "column", "semantic_type", "processor", "params"}, path)
        ref = table_ref(item.get("table"), f"{path}.table", default_schema)
        column = text(item.get("column"), f"{path}.column")
        if (ref, column) in seen:
            raise ValueError(f"Duplicate feature configuration for {ref}.{column}")
        seen.add((ref, column))
        parsed.append(
            FeatureColumnConfig(
                table=ref,
                column=column,
                semantic_type=optional_text(
                    item.get("semantic_type"), f"{path}.semantic_type"
                ),
                processor=optional_text(item.get("processor"), f"{path}.processor"),
                params=mapping(item.get("params", {}), f"{path}.params"),
            )
        )
    return FeaturesConfig(tuple(parsed))


def parse_model(value: object) -> ModelConfig:
    data = mapping(value, "model")
    fields(data, {"name", "params"}, "model")
    return ModelConfig(
        name=text(data.get("name", "default"), "model.name"),
        params=mapping(data.get("params", {}), "model.params"),
    )


def parse_training(value: object) -> TrainingConfig:
    data = mapping(value, "training")
    fields(data, {"batch_size", "epochs", "lr", "seed"}, "training")
    lr = data.get("lr", 0.001)
    if (
        isinstance(lr, bool)
        or not isinstance(lr, (int, float))
        or not math.isfinite(lr)
        or lr <= 0
    ):
        raise ValueError("training.lr must be positive")
    return TrainingConfig(
        batch_size=positive_int(data.get("batch_size", 1024), "training.batch_size"),
        epochs=positive_int(data.get("epochs", 20), "training.epochs"),
        lr=float(lr),
        seed=nonnegative_int(data.get("seed", 42), "training.seed"),
    )


def parse_report(value: object, config_dir: Path, task_name: str) -> ReportConfig:
    data = mapping(value, "report")
    fields(data, {"directory"}, "report")
    directory = text(data.get("directory", f"runs/{task_name}"), "report.directory")
    path = Path(directory).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return ReportConfig(path.resolve())
