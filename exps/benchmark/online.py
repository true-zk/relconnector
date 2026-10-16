"""Measured online graph-index, sampling, and feature-fetch execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import pandas as pd

from relconnector.connector import create_reader
from relconnector.features import SqlFeatureFetcher
from relconnector.graph import InMemoryGraphIndexBuilder
from relconnector.sampling import PygLibNeighborSampler
from relconnector.task import EntitySeedBatch, SeedBatch, SqlSeedReader, TaskSpec

from .telemetry import TelemetryRecorder, TelemetryReport


class OnlineDataRunData(TypedDict):
    dataset: str
    task: str
    reader: str
    completed_batches: int
    completed_examples: int
    graph_nodes: int
    graph_edges: int
    graph_bytes: int
    queried_feature_rows: int
    telemetry: TelemetryReport


@dataclass(frozen=True)
class OnlineDataConfig:
    batch_size: int = 512
    num_neighbors: tuple[int, ...] = (128, 128)
    max_batches: int | None = None
    graph_scan_batch_size: int = 1_000_000
    seed_shuffle_block_size: int = 65_536
    feature_block_size: int = 4096
    feature_cache_bytes: int = 4 * 1024 * 1024 * 1024
    telemetry_interval_s: float = 0.05
    seed: int = 42


@dataclass(frozen=True)
class OnlineDataRunResult:
    dataset: str
    task: str
    reader: str
    completed_batches: int
    completed_examples: int
    graph_nodes: int
    graph_edges: int
    graph_bytes: int
    queried_feature_rows: int
    telemetry: TelemetryReport

    def to_dict(self) -> OnlineDataRunData:
        return {
            "dataset": self.dataset,
            "task": self.task,
            "reader": self.reader,
            "completed_batches": self.completed_batches,
            "completed_examples": self.completed_examples,
            "graph_nodes": self.graph_nodes,
            "graph_edges": self.graph_edges,
            "graph_bytes": self.graph_bytes,
            "queried_feature_rows": self.queried_feature_rows,
            "telemetry": self.telemetry,
        }


class OnlineDataExperiment:
    """Exercise the online path through feature retrieval, before model assembly."""

    def __init__(
        self,
        *,
        dataset: str,
        task: str,
        reader: str = "pandas",
        sqlite_dir: str | Path = Path("data/relbench"),
        config: OnlineDataConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.task_name = task
        self.reader_kind = reader
        self.sqlite_dir = Path(sqlite_dir)
        self.config = config or OnlineDataConfig()

    def run(self) -> OnlineDataRunResult:
        recorder = TelemetryRecorder(sample_interval_s=self.config.telemetry_interval_s)
        completed_batches = 0
        completed_examples = 0
        queried_feature_rows = 0

        with recorder.activate():
            with recorder.phase("database_open"):
                path = self.sqlite_dir / f"{self.dataset}.sqlite"
                reader = create_reader(self.reader_kind, path)
                reader.validate_catalog()
                metadata = reader.metadata()
            with recorder.phase("task_read"):
                task = TaskSpec.from_metadata(reader.task_metadata()[self.task_name])
            with recorder.phase("graph_index_build"):
                cutoff_value = metadata.get("test_timestamp")
                parsed_cutoff = (
                    None if cutoff_value in (None, "") else pd.Timestamp(cutoff_value)
                )
                cutoff = (
                    parsed_cutoff if isinstance(parsed_cutoff, pd.Timestamp) else None
                )
                graph = InMemoryGraphIndexBuilder(
                    reader,
                    scan_batch_size=self.config.graph_scan_batch_size,
                ).build(cutoff=cutoff)
            with recorder.phase("sampler_build"):
                sampler = PygLibNeighborSampler(
                    graph,
                    num_neighbors=list(self.config.num_neighbors),
                    base_seed=self.config.seed,
                )
                seeds = SqlSeedReader(
                    reader,
                    task,
                    batch_size=self.config.batch_size,
                    shuffle_block_size=self.config.seed_shuffle_block_size,
                    seed=self.config.seed,
                )
                fetcher = SqlFeatureFetcher(
                    reader,
                    hidden_columns=task.hidden_columns,
                    block_size=self.config.feature_block_size,
                    cache_bytes=self.config.feature_cache_bytes,
                    database_version=_database_version(path),
                )

            for seed_batch in seeds.iter_epoch(0):
                with recorder.operation("sampling"):
                    plan = sampler.sample(seed_batch)
                with recorder.operation("feature_fetch"):
                    fetcher.fetch(plan)
                completed_batches += 1
                completed_examples += _seed_count(seed_batch)
                queried_feature_rows += fetcher.last_stats.queried_rows
                if (
                    self.config.max_batches is not None
                    and completed_batches >= self.config.max_batches
                ):
                    break

        return OnlineDataRunResult(
            dataset=self.dataset,
            task=self.task_name,
            reader=self.reader_kind,
            completed_batches=completed_batches,
            completed_examples=completed_examples,
            graph_nodes=graph.node_count,
            graph_edges=graph.edge_count,
            graph_bytes=graph.allocated_bytes,
            queried_feature_rows=queried_feature_rows,
            telemetry=recorder.report(),
        )


def _seed_count(seed_batch: SeedBatch) -> int:
    if isinstance(seed_batch, EntitySeedBatch):
        return len(seed_batch.node_ids)
    return len(seed_batch.src_node_ids)


def _database_version(path: Path) -> str:
    stat = path.stat()
    return f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
