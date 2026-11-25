"""Compose the validated task, native graph, and semantic feature plan."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from relx.config import RunConfig, load_config
from relx.conn import connect
from relx.core import DatabaseSchema, TableRef
from relx.preprocess import build_graph, discover_schema, select_graph_tables
from relx.preprocess.features import FeaturePlan, FeatureProcessor, build_feature_plan
from relx.task import TaskDataReport, ValidatedTask, validate_task

if TYPE_CHECKING:
    from relx._relx_native import GraphIndex
    from relx.training.runner import TrainedRun


@dataclass(frozen=True)
class PreparationReport:
    table_count: int
    node_counts: Mapping[str, int]
    relation_count: int
    feature_count: int
    feature_types: Mapping[str, int]
    task_data: TaskDataReport | None
    graph_seconds: float
    feature_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class PreparedDataset:
    config: RunConfig
    schema: DatabaseSchema
    task: ValidatedTask
    graph: GraphIndex
    features: FeaturePlan
    feature_processors: Mapping[str, FeatureProcessor]
    report: PreparationReport


def prepare(
    config: RunConfig | str | Path,
    *,
    feature_processors: Mapping[
        str | tuple[TableRef | str, str], FeatureProcessor
    ] | None = None,
) -> PreparedDataset:
    """Validate and prepare a run without sampling, fitting, or training."""
    settings = load_config(config) if isinstance(config, (str, Path)) else config
    named, columns = _processor_overrides(feature_processors or {})
    started = perf_counter()
    with connect(settings.database.location, reader=settings.database.reader) as reader:
        schema = discover_schema(
            reader,
            schema_name=settings.database.schema_name,
            overrides=settings.schema,
        )
        task = validate_task(
            settings.task,
            schema,
            hidden_columns=settings.hidden_columns,
            reader=reader,
        )
        selected = select_graph_tables(schema, task, settings.graph)
        graph_started = perf_counter()
        graph = build_graph(reader, schema, task, settings.graph)
        graph_seconds = perf_counter() - graph_started
        feature_started = perf_counter()
        features = build_feature_plan(
            reader,
            schema,
            task,
            settings.features,
            settings.graph,
            selected,
            processors=named,
            column_processors=columns,
            sample_limit=settings.features.sample_limit,
            sample_seed=(
                settings.features.sample_seed
                if settings.features.sample_seed is not None
                else settings.training.seed
            ),
        )
        feature_seconds = perf_counter() - feature_started
    counts: dict[str, int] = {}
    for feature in features.features:
        counts[feature.semantic_type] = counts.get(feature.semantic_type, 0) + 1
    report = PreparationReport(
        table_count=len(selected),
        node_counts=graph.node_counts,
        relation_count=len(graph.relation_names()),
        feature_count=len(features.features),
        feature_types=counts,
        task_data=task.data_report,
        graph_seconds=graph_seconds,
        feature_seconds=feature_seconds,
        total_seconds=perf_counter() - started,
    )
    runtime_processors = {
        **named,
        **{
            f"custom:{ref}.{column}": processor
            for (ref, column), processor in columns.items()
        },
    }
    return PreparedDataset(
        settings, schema, task, graph, features, runtime_processors, report
    )


def run(
    config: RunConfig | str | Path,
    *,
    feature_processors: Mapping[
        str | tuple[TableRef | str, str], FeatureProcessor
    ] | None = None,
) -> TrainedRun:
    """Prepare, train with per-batch database reads, and write a report."""
    from relx.training.runner import train

    return train(prepare(config, feature_processors=feature_processors))


def _processor_overrides(
    overrides: Mapping[str | tuple[TableRef | str, str], FeatureProcessor],
) -> tuple[
    dict[str, FeatureProcessor],
    dict[tuple[TableRef, str], FeatureProcessor],
]:
    named: dict[str, FeatureProcessor] = {}
    columns: dict[tuple[TableRef, str], FeatureProcessor] = {}
    for key, processor in overrides.items():
        if not isinstance(processor, FeatureProcessor):
            raise TypeError(f"Feature processor {key!r} must inherit FeatureProcessor")
        if isinstance(key, str):
            named[key] = processor
        elif isinstance(key, tuple) and len(key) == 2:
            ref, column = key
            columns[(ref if isinstance(ref, TableRef) else TableRef(ref), column)] = (
                processor
            )
        else:
            raise TypeError("Feature processor keys must be names or (table, column)")
    return named, columns
