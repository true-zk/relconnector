"""Reproducibility metadata shared by baseline and online experiments."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from relbench_compat import CORRECTNESS_VERSION


class ExperimentMetadata(TypedDict):
    schema_version: int
    correctness_version: str
    implementation: str
    reader: str
    device: str
    executor: str
    feature_encoder: str
    feature_schema: str
    model: str
    sampling_policy: str
    database_path: str
    database_bytes: int
    database_mtime_ns: int
    config: dict[str, object]


def experiment_metadata(
    *,
    implementation: str,
    reader: str,
    device: str,
    executor: str,
    feature_encoder: str,
    feature_schema: str,
    model: str,
    sampling_policy: str,
    database_path: Path,
    config: dict[str, object],
) -> ExperimentMetadata:
    stat = database_path.stat()
    return {
        "schema_version": 2,
        "correctness_version": CORRECTNESS_VERSION,
        "implementation": implementation,
        "reader": reader,
        "device": device,
        "executor": executor,
        "feature_encoder": feature_encoder,
        "feature_schema": feature_schema,
        "model": model,
        "sampling_policy": sampling_policy,
        "database_path": str(database_path.resolve()),
        "database_bytes": stat.st_size,
        "database_mtime_ns": stat.st_mtime_ns,
        "config": config,
    }
