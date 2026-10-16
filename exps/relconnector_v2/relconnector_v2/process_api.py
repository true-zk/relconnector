"""Composition root for the independent multiprocess training runtime."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from relbench.base import TaskType

from . import replay
from .artifacts import CachedGraphIndexBuilder
from .config import OnlineTrainingConfig
from .connector import BaseDatabaseReader, create_reader
from .connector.base import quote_identifier
from .features import (
    SqlAssemblyWorkerFactory,
    SqlTensorFrameSchemaBuilder,
    TensorFrameFeatureSchema,
)
from .features.disk_text import (
    text_embedding_cache_location,
    text_model_identity,
)
from .features.text import GloveTextEmbedder
from .graph import GraphIndex, InMemoryGraphIndexBuilder
from .observability import OperationMetrics
from .runtime import (
    ProcessPipelineExecutor,
    ProcessRuntimeComponents,
    ProcessRuntimeConfig,
    RuntimeResult,
)
from .sampling import PygLibNeighborSampler
from .task import SqlSeedReader, TaskSpec
from .training import OnlineTrainer


@dataclass(frozen=True)
class MultiprocessTrainingSession:
    graph: GraphIndex
    task: TaskSpec
    feature_schema: TensorFrameFeatureSchema
    components: ProcessRuntimeComponents
    metrics: OperationMetrics
    initialization_cache: dict[str, bool]


class MultiprocessRelBenchModel:
    """Build only coordinator/trainer state in the parent process."""

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
            else Path(__file__).resolve().parents[2] / "data" / "relbench"
        )
        self.config = config or OnlineTrainingConfig()
        if self.config.cache_strategy in {"cost", "roi", "off"}:
            self.config = replace(
                self.config,
                feature_cache_bytes=0,
                encoded_feature_cache_bytes=0,
                text_cache_scope="local",
                text_embedding_cache_bytes=(
                    self.config.text_embedding_cache_bytes
                    if self.config.cache_strategy in {"cost", "roi"}
                    else 0
                ),
                disk_text_cache_mode=(
                    self.config.disk_text_cache_mode
                    if self.config.cache_strategy in {"cost", "roi"}
                    else "off"
                ),
            )
        self.session: MultiprocessTrainingSession | None = None
        self.replay_info: dict[str, object] | None = None

    def replay_identity(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "task": self.task_name,
            "database_version": _database_version(
                self.sqlite_dir / f"{self.dataset}.sqlite"
            ),
            "batch_size": self.config.batch_size,
            "seed": self.config.seed,
            "max_batches": self.config.max_batches,
            "num_neighbors": list(self.config.num_neighbors),
            "torch_version": str(torch.__version__),
            "sampling_contract": 1,
        }

    def prepare(self) -> MultiprocessTrainingSession:
        torch.set_num_threads(self.config.torch_num_threads)
        torch.manual_seed(self.config.seed)
        path = self.sqlite_dir / f"{self.dataset}.sqlite"
        database_version = _database_version(path)
        reader = create_reader(self.reader_kind, path)
        reader.validate_catalog()
        task = TaskSpec.from_metadata(reader.task_metadata()[self.task_name])
        cutoff = _cutoff(reader)
        cache_dir = (
            Path(self.config.cache_dir)
            if self.config.cache_dir is not None
            else self.sqlite_dir / ".relconnector-v2-cache"
        )
        effective_cache_dir = cache_dir if self.config.initialization_cache else None
        disk_text_cache_path, disk_text_cache_namespace = text_embedding_cache_location(
            cache_dir,
            database_path=path,
            database_version=database_version,
            model_identity=text_model_identity(
                self.config.text_model_path,
                default=GloveTextEmbedder.model_name,
            ),
            execution=self.config.text_execution,
            embedding_dim=GloveTextEmbedder.embedding_dim,
        )
        metrics = OperationMetrics(self.config.operation_telemetry)
        schema_builder = SqlTensorFrameSchemaBuilder(
            reader,
            hidden_columns=task.hidden_columns,
            scan_batch_size=self.config.graph_scan_batch_size,
            text_embedding_dim=GloveTextEmbedder.embedding_dim,
            cutoff=cutoff,
            require_stypes=True,
            cache_dir=effective_cache_dir,
        )
        feature_schema = schema_builder.build()
        graph_builder = (
            InMemoryGraphIndexBuilder(
                reader,
                scan_batch_size=self.config.graph_scan_batch_size,
            )
            if effective_cache_dir is None
            else CachedGraphIndexBuilder(
                reader,
                scan_batch_size=self.config.graph_scan_batch_size,
                cache_dir=effective_cache_dir,
            )
        )
        graph = graph_builder.build(cutoff=cutoff)
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
        workers = self.config.assembly_processes
        factory = SqlAssemblyWorkerFactory(
            database_path=str(path),
            database_version=database_version,
            reader_kind=self.reader_kind,
            hidden_columns=task.hidden_columns,
            feature_schema=feature_schema,
            feature_block_size=self.config.feature_block_size,
            feature_cache_bytes=self.config.feature_cache_bytes // workers,
            text_model_path=self.config.text_model_path,
            text_batch_size=self.config.text_batch_size,
            text_cache_bytes=self.config.text_embedding_cache_bytes // workers,
            text_cache_admission=self.config.text_cache_admission,
            text_execution=self.config.text_execution,
            shared_text_cache_bytes=(
                self.config.text_embedding_cache_bytes
                if self.config.text_cache_scope in {"shared", "adaptive"}
                else 0
            ),
            disk_text_cache_path=(
                None
                if self.config.disk_text_cache_mode == "off"
                else str(disk_text_cache_path)
            ),
            disk_text_cache_namespace=disk_text_cache_namespace,
            disk_text_cache_mode=self.config.disk_text_cache_mode,
            database_bytes=path.stat().st_size,
            disk_text_cache_min_database_bytes=(
                self.config.disk_text_cache_min_database_bytes
            ),
            disk_text_cache_min_model_inputs=(
                self.config.disk_text_cache_min_model_inputs
            ),
            disk_text_cache_max_memory_hit_rate=(
                self.config.disk_text_cache_max_memory_hit_rate
            ),
            disk_text_cache_min_model_time_s=(
                self.config.disk_text_cache_min_model_time_s
            ),
            encoded_cache_bytes=self.config.encoded_feature_cache_bytes // workers,
            shared_encoded_cache_bytes=(
                self.config.encoded_feature_cache_bytes
                if self.config.encoded_cache_scope == "shared"
                else 0
            ),
            encoded_cache_admission=self.config.encoded_cache_admission,
            operation_telemetry=self.config.operation_telemetry,
            adaptive_text_index=self.config.text_cache_scope == "adaptive",
            cost_strategy=self.config.cache_strategy in {"cost", "roi"},
            cost_disk_policy=(
                "roi" if self.config.cache_strategy == "roi" else "budget"
            ),
            cost_disk_bytes=self.config.cost_disk_cache_bytes // workers,
            training_epochs=self.config.epochs,
        )
        self.session = MultiprocessTrainingSession(
            graph=graph,
            task=task,
            feature_schema=feature_schema,
            components=ProcessRuntimeComponents(
                seeds=seeds,
                sampler=sampler,
                assembly_factory=factory,
                trainer=trainer,
            ),
            metrics=metrics,
            initialization_cache={
                "schema": schema_builder.cache_hit,
                "topology": bool(getattr(graph_builder, "cache_hit", False)),
            },
        )
        if self.config.replay_dir is not None:
            directory = Path(self.config.replay_dir)
            manifest = replay.validate(directory, self.replay_identity())
            self.replay_info = {
                "fingerprint": manifest["fingerprint"],
                "batches_per_epoch": len(manifest["batches"]),
            }
            self.session = replace(
                self.session,
                components=replace(
                    self.session.components,
                    seeds=replay.ReplaySeeds(directory, manifest["batches"]),
                    sampler=replay.ReplaySampler(directory),
                ),
            )
        return self.session

    def train(self) -> RuntimeResult:
        session = self.session or self.prepare()
        executor = ProcessPipelineExecutor(
            ProcessRuntimeConfig(
                assembly_processes=self.config.assembly_processes,
                seed_queue_bytes=self.config.seed_queue_bytes,
                plan_queue_bytes=self.config.plan_queue_bytes,
                ready_queue_bytes=self.config.ready_queue_bytes,
                window_batches=self.config.feature_window_batches,
                window_bytes=self.config.feature_window_bytes,
                max_inflight_windows=self.config.process_max_inflight_windows,
                initial_active_processes=self.config.process_initial_active_workers,
                adaptive_workers=self.config.process_adaptive_workers,
                worker_scale_interval_s=(self.config.process_worker_scale_interval_s),
                cross_epoch_prefetch=self.config.cross_epoch_prefetch,
                training_order=self.config.process_training_order,
                start_method=self.config.process_start_method,
                progress_interval_s=self.config.process_progress_interval_s,
            )
        )
        result = executor.run(session.components, epochs=self.config.epochs)
        if self.replay_info is not None:
            result.pipeline["replay"] = self.replay_info
        return result


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
    return int(cast(int, frame.iloc[0, 0])) + 1


def _database_version(path: Path) -> str:
    stat = path.stat()
    return f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
