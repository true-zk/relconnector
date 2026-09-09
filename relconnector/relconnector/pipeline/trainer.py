"""Measured training loops for entity and recommendation tasks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import torch
from relbench.base import TaskType
from torch_geometric.data import HeteroData

from .dataset import LocalEntityTask, LocalRecommendationTask, LocalTask
from .models import BaseLinkRelBenchModel, BaseRelBenchModel
from .telemetry import TelemetryRecorder

LinkBatch = tuple[HeteroData, HeteroData, HeteroData]


@dataclass(frozen=True)
class EpochStats:
    loss: float
    batches: int
    examples: int


def train_one_epoch(
    model: BaseRelBenchModel | BaseLinkRelBenchModel,
    loader: Iterable[HeteroData | LinkBatch],
    task: LocalTask,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_dtype: torch.dtype,
    recorder: TelemetryRecorder,
    *,
    max_batches: int | None = None,
) -> EpochStats:
    model.train()
    total_loss = 0.0
    total_examples = 0
    batches = 0
    iterator = iter(loader)

    while max_batches is None or batches < max_batches:
        try:
            with recorder.operation("sampling"):
                batch = next(iterator)
        except StopIteration:
            break

        if isinstance(task, LocalRecommendationTask):
            if not isinstance(model, BaseLinkRelBenchModel):
                raise TypeError("Recommendation task requires a link model")
            loss, batch_size = _link_step(
                model,
                cast(LinkBatch, batch),
                task,
                loss_fn,
                optimizer,
                device,
                recorder,
            )
        else:
            if not isinstance(model, BaseRelBenchModel):
                raise TypeError("Entity task requires an entity model")
            loss, batch_size = _entity_step(
                model,
                cast(HeteroData, batch),
                task,
                loss_fn,
                optimizer,
                device,
                target_dtype,
                recorder,
            )
        batches += 1
        total_examples += batch_size
        total_loss += loss * batch_size

    return EpochStats(
        loss=total_loss / max(total_examples, 1),
        batches=batches,
        examples=total_examples,
    )


def _entity_step(
    model: BaseRelBenchModel,
    batch: HeteroData,
    task: LocalEntityTask,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_dtype: torch.dtype,
    recorder: TelemetryRecorder,
) -> tuple[float, int]:
    with recorder.operation("h2d", synchronize_cuda=True):
        batch = batch.to(device)
    optimizer.zero_grad(set_to_none=True)
    with recorder.operation("forward", synchronize_cuda=True):
        prediction = model(batch, task.entity_table)
        target = batch[task.entity_table].y.to(target_dtype)
        if task.task_type not in {
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        }:
            prediction = prediction.view(-1)
        loss = loss_fn(prediction, target)
    with recorder.operation("backward", synchronize_cuda=True):
        loss.backward()
    with recorder.operation("optimizer_step", synchronize_cuda=True):
        optimizer.step()
    batch_size = int(batch[task.entity_table].batch_size)
    return float(loss.detach().cpu()), batch_size


def _link_step(
    model: BaseLinkRelBenchModel,
    batch: LinkBatch,
    task: LocalRecommendationTask,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    recorder: TelemetryRecorder,
) -> tuple[float, int]:
    with recorder.operation("h2d", synchronize_cuda=True):
        src_batch, pos_dst_batch, neg_dst_batch = (
            sampled.to(device) for sampled in batch
        )
    optimizer.zero_grad(set_to_none=True)
    with recorder.operation("forward", synchronize_cuda=True):
        positive, negative = model(
            src_batch,
            pos_dst_batch,
            neg_dst_batch,
            task.src_entity_table,
            task.dst_entity_table,
        )
        loss = loss_fn(positive, negative)
    with recorder.operation("backward", synchronize_cuda=True):
        loss.backward()
    with recorder.operation("optimizer_step", synchronize_cuda=True):
        optimizer.step()
    batch_size = int(src_batch[task.src_entity_table].batch_size)
    return float(loss.detach().cpu()), batch_size
