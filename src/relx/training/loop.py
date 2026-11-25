"""Label encoding, losses, and epoch metrics for entity tasks."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from relx.batch import BatchIterator, PreparedBatch, PreparedRecommendationBatch
from relx.sampling import EntitySamplePlan
from relx.task import TaskType

from .model import OneHopEntityModel
from .multi_hop_model import MultiHopEntityModel
from .relational_model import RelationalEntityModel, RelationalLinkModel


class LabelCodec:
    def __init__(self, kind: TaskType, labels: list[object]) -> None:
        self.kind = kind
        if kind in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}:
            distinct = set(labels)
            try:
                self.classes = tuple(sorted(distinct))
            except TypeError:
                self.classes = tuple(sorted(distinct, key=repr))
            self.indices = {value: index for index, value in enumerate(self.classes)}
            if len(self.classes) < 2:
                raise ValueError("Training requires at least two classes")
        elif kind is TaskType.REGRESSION:
            self.classes = ()
            self.indices = {}
        else:
            raise ValueError(f"One-hop baseline does not support task type {kind.value}")

    @property
    def out_channels(self) -> int:
        return len(self.classes) if self.kind is TaskType.MULTICLASS_CLASSIFICATION else 1

    @property
    def metric_name(self) -> str:
        return "mae" if self.kind is TaskType.REGRESSION else "accuracy"

    def targets(self, values: np.ndarray) -> torch.Tensor:
        if self.kind is TaskType.REGRESSION:
            return torch.as_tensor(values.astype(np.float32), dtype=torch.float32)
        return torch.as_tensor(
            [self.indices[value] for value in values], dtype=torch.long
        )

    def decode(self, score: np.ndarray) -> object:
        if self.kind is TaskType.REGRESSION:
            return float(score[0])
        if self.kind is TaskType.BINARY_CLASSIFICATION:
            return self.classes[int(score[0] >= 0)]
        return self.classes[int(score.argmax())]


def run_epoch(
    model: OneHopEntityModel | MultiHopEntityModel | RelationalEntityModel,
    pipeline: BatchIterator,
    codec: LabelCodec,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    model.train(optimizer is not None)
    pipeline.start_epoch(epoch)
    total_loss = 0.0
    total_metric = 0.0
    count = 0
    while (batch := pipeline.next_batch()) is not None:
        if not isinstance(batch, PreparedBatch):
            raise TypeError("Entity training received a recommendation batch")
        plan = batch.sample
        if not isinstance(plan, EntitySamplePlan):
            raise TypeError("Training requires entity labels")
        targets = codec.targets(plan.target)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(batch)
            if codec.kind is TaskType.REGRESSION:
                difference = (logits.squeeze(-1) - targets).abs()
                loss = difference.mean()
                contribution = float(difference.sum())
            elif codec.kind is TaskType.BINARY_CLASSIFICATION:
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), targets.float()
                )
                contribution = float(
                    ((logits.squeeze(-1) >= 0) == targets.bool()).sum()
                )
            else:
                loss = nn.functional.cross_entropy(logits, targets)
                contribution = float((logits.argmax(dim=1) == targets).sum())
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        size = len(targets)
        total_loss += float(loss.detach()) * size
        total_metric += contribution
        count += size
    if count == 0:
        return math.nan, math.nan
    metric = total_metric / count
    return total_loss / count, metric


def run_recommendation_epoch(
    model: RelationalLinkModel,
    pipeline: BatchIterator,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Train pairwise scores; report sampled positive-over-negative accuracy."""
    model.train(optimizer is not None)
    pipeline.start_epoch(epoch)
    total_loss = 0.0
    correct = 0
    count = 0
    while (batch := pipeline.next_batch()) is not None:
        if not isinstance(batch, PreparedRecommendationBatch):
            raise TypeError("Recommendation training requires three sampled graphs")
        with torch.set_grad_enabled(optimizer is not None):
            positive, negative = model(
                batch.source, batch.positive, batch.negative
            )
            loss = nn.functional.softplus(negative - positive).mean()
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        size = len(positive)
        total_loss += float(loss.detach()) * size
        correct += int((positive > negative).sum())
        count += size
    if count == 0:
        return math.nan, math.nan
    return total_loss / count, correct / count
