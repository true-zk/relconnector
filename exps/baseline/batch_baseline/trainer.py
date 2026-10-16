"""Eager training loops with optional, implementation-neutral instrumentation."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from relbench.base import TaskType
from torch_geometric.data import HeteroData

from .dataset import LocalEntityTask, LocalRecommendationTask, LocalTask
from .models import BaseLinkRelBenchModel, BaseRelBenchModel

LinkBatch = tuple[HeteroData, HeteroData, HeteroData]


class OperationObserver(Protocol):
    def operation(
        self, name: str, *, synchronize_cuda: bool = False
    ) -> AbstractContextManager[None]: ...


class _NoOpObserver:
    def operation(
        self, name: str, *, synchronize_cuda: bool = False
    ) -> AbstractContextManager[None]:
        return nullcontext()


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
    *,
    observer: OperationObserver | None = None,
    max_batches: int | None = None,
) -> EpochStats:
    observer = observer or _NoOpObserver()
    model.train()
    total_loss = 0.0
    total_examples = 0
    batches = 0
    iterator = iter(loader)

    while max_batches is None or batches < max_batches:
        try:
            with observer.operation("sample_and_prepare"):
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
                observer,
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
                observer,
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
    observer: OperationObserver,
) -> tuple[float, int]:
    with observer.operation("h2d", synchronize_cuda=True):
        batch = batch.to(device)
    optimizer.zero_grad(set_to_none=True)
    with observer.operation("forward", synchronize_cuda=True):
        prediction = model(batch, task.entity_table)
        target = batch[task.entity_table].y.to(target_dtype)
        if task.task_type not in {
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        }:
            prediction = prediction.view(-1)
        loss = loss_fn(prediction, target)
    with observer.operation("backward", synchronize_cuda=True):
        loss.backward()
    with observer.operation("optimizer_step", synchronize_cuda=True):
        optimizer.step()
    return float(loss.detach().cpu()), int(batch[task.entity_table].batch_size)


def _link_step(
    model: BaseLinkRelBenchModel,
    batch: LinkBatch,
    task: LocalRecommendationTask,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    observer: OperationObserver,
) -> tuple[float, int]:
    with observer.operation("h2d", synchronize_cuda=True):
        src_batch, pos_dst_batch, neg_dst_batch = (
            sampled.to(device) for sampled in batch
        )
    optimizer.zero_grad(set_to_none=True)
    with observer.operation("forward", synchronize_cuda=True):
        positive, negative = model(
            src_batch,
            pos_dst_batch,
            neg_dst_batch,
            task.src_entity_table,
            task.dst_entity_table,
        )
        loss = loss_fn(positive, negative)
    with observer.operation("backward", synchronize_cuda=True):
        loss.backward()
    with observer.operation("optimizer_step", synchronize_cuda=True):
        optimizer.step()
    return float(loss.detach().cpu()), int(src_batch[task.src_entity_table].batch_size)
