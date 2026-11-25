"""Train and evaluate a small entity prediction model from one YAML run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from relx.batch import (
    BatchIterator,
    OnlineBatchIterator,
    OnlineFeatureStore,
    PreparedBatch,
)
from relx.conn import connect
from relx.sampling import EntitySamplePlan, PygLibMultiHopSampler
from relx.task import (
    SqlEntitySeedSource,
    SqlRecommendationSeedSource,
)

from .loop import LabelCodec, run_epoch, run_recommendation_epoch
from .model import OneHopEntityModel
from .multi_hop_model import MultiHopEntityModel
from .relational_model import RelationalEntityModel, RelationalLinkModel

if TYPE_CHECKING:
    from relx.api import PreparedDataset


@dataclass(frozen=True)
class EpochReport:
    epoch: int
    train_loss: float
    validation_metric: float


@dataclass(frozen=True)
class TrainingReport:
    task_name: str
    model_name: str
    metric: str
    epochs: tuple[EpochReport, ...]
    test_metric: float | None
    train_rows: int
    validation_rows: int
    test_rows: int
    report_file: Path


@dataclass(frozen=True)
class Prediction:
    sample_key: tuple[object, ...]
    value: object


@dataclass
class TrainedRun:
    prepared: PreparedDataset
    model: OneHopEntityModel | MultiHopEntityModel | RelationalEntityModel | RelationalLinkModel
    report: TrainingReport
    test_pipeline: BatchIterator
    codec: LabelCodec | None

    def predict_test(self) -> tuple[Prediction, ...]:
        if self.codec is None:
            raise NotImplementedError("Recommendation ranking predictions are not available")
        if isinstance(self.model, RelationalLinkModel):
            raise TypeError("Recommendation model cannot predict entity labels")
        self.model.eval()
        results: list[Prediction] = []
        self.test_pipeline.start_epoch(0)
        with torch.no_grad():
            while (batch := self.test_pipeline.next_batch()) is not None:
                if not isinstance(batch, PreparedBatch):
                    raise TypeError("Entity prediction received recommendation samples")
                scores = self.model(batch).cpu().numpy()
                if not isinstance(batch.sample, EntitySamplePlan):
                    raise TypeError("Prediction requires entity samples")
                keys = batch.sample.sample_keys
                results.extend(
                    Prediction(key, self.codec.decode(score))
                    for key, score in zip(keys, scores, strict=True)
                )
        return tuple(results)


def train(prepared: PreparedDataset) -> TrainedRun:
    """Train using sampled, per-batch SQL feature reads."""
    config = prepared.config
    if config.model.name not in {
        "default", "one_hop_baseline", "multi_hop_baseline", "rdl", "relgnn"
    }:
        raise ValueError(f"Unsupported model.name {config.model.name!r}")
    params = config.model.params
    unknown = set(params) - {
        "channels", "fanout", "num_neighbors", "workers", "text_mode",
        "text_buckets", "glove_model_path", "glove_export_dir",
        "glove_cache_bytes", "glove_disk_cache",
    }
    if unknown:
        raise ValueError(f"Unknown model parameters: {sorted(unknown)}")
    channels = _positive_int(params.get("channels", 32), "model.params.channels")
    workers = _positive_int(params.get("workers", 1), "model.params.workers")
    text_buckets = _positive_int(
        params.get("text_buckets", 4_096), "model.params.text_buckets"
    )
    fanout = params.get("fanout", 10)
    if isinstance(fanout, bool) or not isinstance(fanout, int) or fanout < -1:
        raise ValueError("model.params.fanout must be -1 or nonnegative")
    raw_neighbors = params.get("num_neighbors")
    if config.model.name in {"multi_hop_baseline", "rdl", "relgnn"} and (
        raw_neighbors is None
    ):
        raise ValueError(f"{config.model.name} requires model.params.num_neighbors")
    if raw_neighbors is not None and "fanout" in params:
        raise ValueError("Use either fanout or num_neighbors, not both")
    if raw_neighbors is not None and (
        not isinstance(raw_neighbors, (list, tuple))
        or not raw_neighbors
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < -1
            for value in raw_neighbors
        )
    ):
        raise ValueError("model.params.num_neighbors must be a nonempty list of fanouts >= -1")
    if params.get("text_mode", "hash") not in {"hash", "glove"}:
        raise ValueError("model.params.text_mode must be hash or glove")
    if (
        any(feature.semantic_type == "text" for feature in prepared.features.features)
        and params.get("text_mode") is None
    ):
        raise ValueError(
            "Text features require model.params.text_mode: hash or glove"
        )
    torch.manual_seed(config.training.seed)
    torch.set_num_threads(workers)
    np.random.seed(config.training.seed)
    if prepared.task.spec.is_recommendation and config.model.name not in {
        "rdl", "relgnn"
    }:
        raise ValueError("Recommendation training requires model.name rdl or relgnn")
    source_class = (
        SqlRecommendationSeedSource if prepared.task.spec.is_recommendation
        else SqlEntitySeedSource
    )
    with connect(config.database.location, reader=config.database.reader) as reader:
        sources = {
            split: source_class(
                reader,
                prepared.task,
                prepared.graph,
                split=split,
                timezone=config.graph.timezone,
                batch_size=config.training.batch_size,
                shuffle_seed=config.training.seed,
            )
            for split in ("train", "validation", "test")
        }
        store = OnlineFeatureStore(
            prepared,
            reader,
            train_cutoff_us=sources["train"].max_time_us,
            native_workers=config.training.feature_native_workers,
        )
    codec = None
    if not prepared.task.spec.is_recommendation:
        labels = (
            list({
                value for source in sources.values()
                for value in source.labels() if value is not None
            })
            if prepared.task.spec.task_type.value != "regression" else []
        )
        codec = LabelCodec(prepared.task.spec.task_type, labels)
    sampler = PygLibMultiHopSampler(
        prepared.graph, prepared.schema,
        fanouts=(fanout,) if raw_neighbors is None else tuple(raw_neighbors),
        random_seed=config.training.seed,
    )
    pipelines = {
        split: OnlineBatchIterator(
            source, sampler, store,
            feature_workers=config.training.feature_workers,
            max_inflight=config.training.prefetch_batches,
            max_batches=config.training.max_batches,
        )
        for split, source in sources.items()
    }
    model: OneHopEntityModel | MultiHopEntityModel | RelationalEntityModel | RelationalLinkModel
    if prepared.task.spec.is_recommendation:
        assert raw_neighbors is not None
        model = RelationalLinkModel(
            store, prepared.task.entity_table, sampler.relations,
            kind=config.model.name, hops=len(raw_neighbors), channels=channels,
            text_buckets=text_buckets,
        )
    elif config.model.name in {"rdl", "relgnn"}:
        assert raw_neighbors is not None
        assert codec is not None
        model = RelationalEntityModel(
            store, prepared.task.entity_table, sampler.relations,
            kind=config.model.name, hops=len(raw_neighbors),
            channels=channels, out_channels=codec.out_channels,
            text_buckets=text_buckets,
        )
    elif raw_neighbors is None:
        assert codec is not None
        model = OneHopEntityModel(
            store, prepared.task.entity_table, sampler.relations,
            channels=channels, out_channels=codec.out_channels,
            text_buckets=text_buckets,
        )
    else:
        assert codec is not None
        model = MultiHopEntityModel(
            store, prepared.task.entity_table, sampler.relations,
            hops=len(raw_neighbors), channels=channels,
            out_channels=codec.out_channels, text_buckets=text_buckets,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    history: list[EpochReport] = []
    for epoch in range(config.training.epochs):
        if isinstance(model, RelationalLinkModel):
            loss, _ = run_recommendation_epoch(
                model, pipelines["train"], epoch, optimizer
            )
            _, metric = run_recommendation_epoch(
                model, pipelines["validation"], epoch
            )
        else:
            assert codec is not None
            loss, _ = run_epoch(model, pipelines["train"], codec, epoch, optimizer)
            _, metric = run_epoch(model, pipelines["validation"], codec, epoch)
        history.append(EpochReport(epoch, loss, metric))
    test_metric = None
    if prepared.task.data_report and prepared.task.data_report.test_labels_available:
        if isinstance(model, RelationalLinkModel):
            _, test_metric = run_recommendation_epoch(model, pipelines["test"], 0)
        else:
            assert codec is not None
            _, test_metric = run_epoch(model, pipelines["test"], codec, 0)
    counts = prepared.task.data_report
    assert counts is not None
    report_file = config.report.directory / "report.json"
    report = TrainingReport(
        prepared.task.spec.name,
        ("multi_hop_baseline" if raw_neighbors is not None
         and config.model.name not in {"rdl", "relgnn"} else config.model.name),
        "pairwise_accuracy" if codec is None else codec.metric_name,
        tuple(history),
        test_metric,
        counts.train_rows,
        counts.validation_rows,
        counts.test_rows,
        report_file,
    )
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        json.dumps(asdict(report), default=str, indent=2), encoding="utf-8"
    )
    return TrainedRun(prepared, model, report, pipelines["test"], codec)


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value
