"""Composition root for online relational graph training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from relbench.base import TaskType

from .config import OnlineTrainingConfig
from .connector import BaseDatabaseReader, create_reader
from .connector.base import quote_identifier
from .features import (
    SqlFeatureFetcher,
    SqlTensorFrameSchemaBuilder,
    TensorFrameBatchAssembler,
    TensorFrameEncoder,
    TensorFrameFeatureSchema,
)
from .features.text import GloveTextEmbedder
from .graph import GraphIndex, InMemoryGraphIndexBuilder
from .runtime import (
    AsyncPipelineExecutor,
    AsyncRuntimeConfig,
    RuntimeComponents,
    RuntimeResult,
    SyncExecutor,
)
from .sampling import PygLibNeighborSampler
from .task import SqlSeedReader, TaskSpec
from .training import OnlineTrainer


@dataclass(frozen=True)
class OnlineTrainingSession:
    graph: GraphIndex
    task: TaskSpec
    feature_schema: TensorFrameFeatureSchema
    components: RuntimeComponents


class OnlineRelBenchModel:
    """Train without loading database features into memory globally."""

    def __init__(
        self,
        *,
        dataset: str,
        task: str,
        reader: str = "pandas",
        sqlite_dir: str | Path | None = None,
        config: OnlineTrainingConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.task_name = task
        self.reader_kind = reader
        self.sqlite_dir = (
            Path(sqlite_dir)
            if sqlite_dir is not None
            else Path(__file__).resolve().parents[1] / "data" / "relbench"
        )
        self.config = config or OnlineTrainingConfig()
        self.session: OnlineTrainingSession | None = None

    def prepare(self) -> OnlineTrainingSession:
        torch.set_num_threads(self.config.torch_num_threads)
        torch.manual_seed(self.config.seed)
        path = self.sqlite_dir / f"{self.dataset}.sqlite"
        reader = create_reader(self.reader_kind, path)
        reader.validate_catalog()
        task = TaskSpec.from_metadata(reader.task_metadata()[self.task_name])
        feature_schema = SqlTensorFrameSchemaBuilder(
            reader,
            hidden_columns=task.hidden_columns,
            scan_batch_size=self.config.graph_scan_batch_size,
            text_embedding_dim=GloveTextEmbedder.embedding_dim,
            cutoff=_cutoff(reader),
        ).build()
        graph = InMemoryGraphIndexBuilder(
            reader,
            scan_batch_size=self.config.graph_scan_batch_size,
        ).build(cutoff=_cutoff(reader))
        seeds = SqlSeedReader(
            reader,
            task,
            batch_size=self.config.batch_size,
            shuffle_block_size=self.config.seed_shuffle_block_size,
            seed=self.config.seed,
            max_batches=self.config.max_batches,
        )
        sampler = PygLibNeighborSampler(
            graph,
            num_neighbors=list(self.config.num_neighbors),
            base_seed=self.config.seed,
        )
        fetcher = SqlFeatureFetcher(
            reader,
            hidden_columns=task.hidden_columns,
            feature_columns=feature_schema.feature_columns,
            block_size=self.config.feature_block_size,
            cache_bytes=self.config.feature_cache_bytes,
            database_version=_database_version(path),
        )
        encoder = TensorFrameEncoder(
            feature_schema,
            GloveTextEmbedder(model_path=self.config.text_model_path),
            text_batch_size=self.config.text_batch_size,
        )
        assembler = TensorFrameBatchAssembler(encoder)
        torch.manual_seed(self.config.seed)
        if (
            self.config.device is not None
            and torch.device(self.config.device).type == "cuda"
        ):
            torch.cuda.manual_seed_all(self.config.seed)
        trainer = OnlineTrainer(
            graph,
            task,
            feature_schema=feature_schema,
            out_channels=_out_channels(reader, task, self.config.out_channels),
            channels=self.config.channels,
            num_layers=self.config.num_layers,
            aggr=self.config.aggr,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            device=self.config.device,
            seed=self.config.seed,
        )
        self.session = OnlineTrainingSession(
            graph=graph,
            task=task,
            feature_schema=feature_schema,
            components=RuntimeComponents(
                seeds=seeds,
                sampler=sampler,
                fetcher=fetcher,
                assembler=assembler,
                trainer=trainer,
            ),
        )
        return self.session

    def train(self) -> RuntimeResult:
        session = self.session or self.prepare()
        if self.config.executor == "sync":
            executor = SyncExecutor()
        else:
            executor = AsyncPipelineExecutor(
                AsyncRuntimeConfig(
                    seed_queue_bytes=self.config.seed_queue_bytes,
                    plan_queue_bytes=self.config.plan_queue_bytes,
                    ready_queue_bytes=self.config.ready_queue_bytes,
                )
            )
        return executor.run(session.components, epochs=self.config.epochs)


def _cutoff(reader: BaseDatabaseReader) -> pd.Timestamp | None:
    value = reader.metadata().get("test_timestamp")
    if value in (None, ""):
        return None
    timestamp = pd.Timestamp(value)
    return timestamp if isinstance(timestamp, pd.Timestamp) else None


def _out_channels(
    reader: BaseDatabaseReader,
    task: TaskSpec,
    configured: int | None,
) -> int:
    if configured is not None:
        return configured
    if task.task_type != TaskType.MULTICLASS_CLASSIFICATION:
        if task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            raise ValueError("multilabel tasks require config.out_channels")
        return 1
    if task.target_column is None:
        raise ValueError(f"Task {task.name!r} has no target column")
    schema = reader.schemas()[task.table_name]
    if schema.split_column is None:
        raise ValueError(f"Task table {task.table_name!r} has no split column")
    target = quote_identifier(task.target_column)
    split = quote_identifier(schema.split_column)
    table = quote_identifier(task.table_name)
    frame = reader.read_query(
        f"SELECT MAX({target}) AS maximum FROM {table} WHERE {split} = 'train'"
    )
    maximum = cast(int, frame.iloc[0, 0])
    return int(maximum) + 1


def _database_version(path: Path) -> str:
    stat = path.stat()
    return f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
