"""Measured orchestration for the eager full-memory baseline."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from baseline.batch_baseline.config import TrainingConfig
from baseline.batch_baseline.dataset import default_sqlite_path, open_dataset
from baseline.batch_baseline.feature_store import (
    EagerFeatureStore,
    InMemoryTensorFrameFetcher,
)
from baseline.batch_baseline.objectives import resolve_output
from baseline.batch_baseline.text_embedder import GloveTextEmbedder
from baseline.vanilla_baseline.features import (
    SqlTensorFrameSchemaBuilder,
    TensorFrameBatchAssembler,
    TensorFrameEncoder,
)
from baseline.vanilla_baseline.graph import InMemoryGraphIndexBuilder
from baseline.vanilla_baseline.runtime import (
    AsyncPipelineExecutor,
    AsyncRuntimeConfig,
    RuntimeComponents,
    SyncExecutor,
)
from baseline.vanilla_baseline.sampling import PygLibNeighborSampler
from baseline.vanilla_baseline.task import SqlSeedReader, TaskSpec
from baseline.vanilla_baseline.training import OnlineTrainer

from .metadata import experiment_metadata
from .result import TrainingRunResult, add_training_config
from .telemetry import TelemetryRecorder
from .vanilla_wrappers import (
    MeasuredBatchAssembler,
    MeasuredFeatureFetcher,
    MeasuredSampler,
    MeasuredSeedSource,
    MeasuredTrainer,
)


class BaselineExperiment:
    """Run identical training semantics with all node features resident in RAM."""

    def __init__(
        self,
        dataset: str | None = None,
        task: str | None = None,
        *,
        model: str = "graphsage",
        reader: str = "pandas",
        sqlite_dir: str | Path | None = None,
        text_embedder: str | None = "glove",
        config: TrainingConfig | None = None,
    ) -> None:
        self.default_dataset = dataset
        self.default_task = task
        self.model_name = model
        self.reader_kind = reader
        self.sqlite_dir = Path(sqlite_dir) if sqlite_dir is not None else None
        self.text_embedder_name = text_embedder
        self.config = config or TrainingConfig()
        self.model: torch.nn.Module | None = None
        self.task = None
        self.last_result: TrainingRunResult | None = None

    def train(
        self,
        *,
        dataset: str | None = None,
        task: str | None = None,
        epochs: int | None = None,
    ) -> TrainingRunResult:
        dataset_name = dataset or self.default_dataset
        task_name = task or self.default_task
        if dataset_name is None or task_name is None:
            raise ValueError("Provide dataset and task in the constructor or train().")
        if self.model_name != "graphsage":
            raise ValueError("Strict comparison currently supports only graphsage")

        total_epochs = epochs if epochs is not None else self.config.epochs
        torch.set_num_threads(self.config.torch_num_threads)
        device = _resolve_device(self.config.device)
        recorder = TelemetryRecorder(sample_interval_s=self.config.telemetry_interval_s)

        with recorder.activate():
            with recorder.phase("database_open"):
                sqlite_path = self._sqlite_path(dataset_name)
                local = open_dataset(
                    dataset_name,
                    reader_kind=self.reader_kind,
                    sqlite_path=sqlite_path,
                )
            with recorder.phase("task_read"):
                task_object = local.load_task(task_name)
                task_spec = TaskSpec.from_metadata(
                    local.reader.task_metadata()[task_name]
                )
            with recorder.phase("feature_schema_prepare"):
                feature_schema = SqlTensorFrameSchemaBuilder(
                    local.reader,
                    hidden_columns=task_object.hidden_columns(),
                    encode_text=self.text_embedder_name == "glove",
                    cutoff=local.test_timestamp,
                    require_stypes=True,
                ).build()
            with recorder.phase("database_read"):
                database = local.get_db()
            with recorder.phase("text_encoder_init"):
                text_encoder = self._make_text_embedder()
                encoder = TensorFrameEncoder(
                    feature_schema,
                    text_encoder,
                    text_batch_size=self.config.text_batch_size,
                )
            with recorder.phase("feature_materialize"):
                cache_dir = (
                    local.path.with_suffix(".materialized")
                    / (self.text_embedder_name or "categorical_text")
                    / task_name
                    / feature_schema.fingerprint
                    if self.config.cache_materialization
                    else None
                )
                feature_store = EagerFeatureStore.materialize(
                    database,
                    encoder,
                    cache_dir=cache_dir,
                )
            with recorder.phase("graph_build"):
                graph = InMemoryGraphIndexBuilder(
                    local.reader,
                    scan_batch_size=self.config.graph_scan_batch_size,
                ).build(cutoff=local.test_timestamp)
            with recorder.phase("component_build"):
                seeds = SqlSeedReader(
                    local.reader,
                    task_spec,
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
                assembler = TensorFrameBatchAssembler(encoder)
                _, out_channels, _ = resolve_output(task_object)
                torch.manual_seed(self.config.seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(self.config.seed)
                trainer = OnlineTrainer(
                    graph,
                    task_spec,
                    feature_schema=feature_schema,
                    out_channels=out_channels,
                    channels=self.config.channels,
                    num_layers=self.config.num_layers,
                    aggr=self.config.aggr,
                    lr=self.config.lr,
                    weight_decay=self.config.weight_decay,
                    device=device,
                    seed=self.config.seed,
                )
                components = RuntimeComponents(
                    seeds=MeasuredSeedSource(seeds, recorder),
                    sampler=MeasuredSampler(sampler, recorder),
                    fetcher=MeasuredFeatureFetcher(
                        InMemoryTensorFrameFetcher(feature_store), recorder
                    ),
                    assembler=MeasuredBatchAssembler(assembler, recorder),
                    trainer=MeasuredTrainer(trainer, recorder),
                )
            with recorder.phase("train"):
                runtime_result = self._executor().run(
                    components,
                    epochs=total_epochs,
                )

        result = TrainingRunResult(
            dataset=dataset_name,
            task=task_name,
            task_type=task_object.task_type.value,
            model=self.model_name,
            reader=self.reader_kind,
            device=str(device),
            epochs=total_epochs,
            completed_batches=runtime_result.steps,
            completed_examples=runtime_result.examples,
            losses=[runtime_result.mean_loss],
            experiment=experiment_metadata(
                implementation="baseline",
                reader=self.reader_kind,
                device=str(device),
                executor=self.config.executor,
                feature_encoder="tensorframe-shared-v2",
                feature_schema=feature_schema.fingerprint,
                model="heteroencoder-graphsage",
                sampling_policy="shared-seed-pyg-lib-uniform-v4",
                database_path=sqlite_path,
                config={**asdict(self.config), "epochs": total_epochs},
            ),
            telemetry=add_training_config(recorder.report(), asdict(self.config)),
        )
        self.model = trainer.model
        self.task = task_object
        self.last_result = result
        print(
            f"[{dataset_name}/{task_name}] loss={runtime_result.mean_loss:.4f}, "
            f"batches={runtime_result.steps}, examples={runtime_result.examples}"
        )
        return result

    def _executor(self) -> SyncExecutor | AsyncPipelineExecutor:
        if self.config.executor == "sync":
            return SyncExecutor()
        return AsyncPipelineExecutor(
            AsyncRuntimeConfig(
                seed_queue_bytes=self.config.seed_queue_bytes,
                plan_queue_bytes=self.config.plan_queue_bytes,
                ready_queue_bytes=self.config.ready_queue_bytes,
            )
        )

    def _sqlite_path(self, dataset_name: str) -> Path:
        if self.sqlite_dir is not None:
            return self.sqlite_dir / f"{dataset_name}.sqlite"
        return default_sqlite_path(dataset_name)

    def _make_text_embedder(self) -> GloveTextEmbedder | None:
        if self.text_embedder_name is None:
            return None
        if self.text_embedder_name != "glove":
            raise ValueError(f"Unknown text embedder {self.text_embedder_name!r}")
        return GloveTextEmbedder()


def _resolve_device(name: str | None) -> torch.device:
    if name is not None:
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
