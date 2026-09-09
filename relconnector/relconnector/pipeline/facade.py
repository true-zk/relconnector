"""User-facing entry point: ``RelBenchModel(...).train()``."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from .config import TrainingConfig
from .dataset import (
    LocalRecommendationTask,
    default_sqlite_path,
    open_dataset,
)
from .graph_builder import build_graph
from .models import build_model
from .objectives import resolve_output
from .result import TrainingRunResult, add_training_config
from .sampler import make_train_loader
from .telemetry import TelemetryRecorder
from .text_embedder import GloveTextEmbedder
from .trainer import train_one_epoch


class RelBenchModel:
    """Compose SQL reading, PyG sampling, and model training."""

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
            raise ValueError(
                "Provide both `dataset` and `task` in the constructor or train()."
            )

        total_epochs = epochs if epochs is not None else self.config.epochs
        device = _resolve_device(self.config.device)
        torch.manual_seed(self.config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)

        recorder = TelemetryRecorder(sample_interval_s=self.config.telemetry_interval_s)
        losses: list[float] = []
        completed_batches = 0
        completed_examples = 0

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
            with recorder.phase("database_read"):
                database = local.get_db()
            with recorder.phase("text_encoder_init"):
                text_encoder = self._make_text_embedder()
            with recorder.phase("graph_build"):
                cache_dir = (
                    local.path.with_suffix(".materialized")
                    / (self.text_embedder_name or "categorical_text")
                    / task_name
                    if self.config.cache_materialization
                    else None
                )
                data, col_stats, _ = build_graph(
                    database,
                    cache_dir=cache_dir,
                    text_embedder=text_encoder,
                    text_batch_size=self.config.text_batch_size,
                    remove_columns=task_object.hidden_columns(),
                )
            with recorder.phase("loader_build"):
                loader = make_train_loader(data, task_object, self.config)
            with recorder.phase("model_build"):
                loss_fn, out_channels, target_dtype = resolve_output(task_object)
                model = build_model(
                    self.model_name,
                    data=data,
                    col_stats_dict=col_stats,
                    out_channels=out_channels,
                    channels=self.config.channels,
                    num_layers=self.config.num_layers,
                    aggr=self.config.aggr,
                    recommendation=isinstance(task_object, LocalRecommendationTask),
                ).to(device)
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=self.config.lr,
                    weight_decay=self.config.weight_decay,
                )

            for epoch in range(1, total_epochs + 1):
                with recorder.phase(f"train_epoch_{epoch}"):
                    stats = train_one_epoch(
                        model=model,
                        loader=loader,
                        task=task_object,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        device=device,
                        target_dtype=target_dtype,
                        recorder=recorder,
                        max_batches=self.config.max_batches,
                    )
                losses.append(stats.loss)
                completed_batches += stats.batches
                completed_examples += stats.examples
                print(
                    f"[{dataset_name}/{task_name}] epoch {epoch:03d}: "
                    f"loss={stats.loss:.4f}, batches={stats.batches}, "
                    f"examples={stats.examples}"
                )

        result = TrainingRunResult(
            dataset=dataset_name,
            task=task_name,
            task_type=task_object.task_type.value,
            model=self.model_name,
            reader=self.reader_kind,
            device=str(device),
            epochs=total_epochs,
            completed_batches=completed_batches,
            completed_examples=completed_examples,
            losses=losses,
            telemetry=add_training_config(
                recorder.report(),
                asdict(self.config),
            ),
        )
        self.model = model
        self.task = task_object
        self.last_result = result
        return result

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
