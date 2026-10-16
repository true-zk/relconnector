"""Task-type to loss, output shape, and target dtype resolution."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import torch
from relbench.base import TaskType

from .dataset import LocalEntityTask, LocalRecommendationTask, LocalTask


class PairwiseRankingLoss(torch.nn.Module):
    def forward(
        self, positive_score: torch.Tensor, negative_score: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.softplus(negative_score - positive_score).mean()


def resolve_output(
    task: LocalTask,
) -> tuple[torch.nn.Module, int, torch.dtype]:
    if isinstance(task, LocalRecommendationTask):
        return PairwiseRankingLoss(), 1, torch.float32
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        return torch.nn.BCEWithLogitsLoss(), 1, torch.float32
    if task.task_type == TaskType.REGRESSION:
        return torch.nn.L1Loss(), 1, torch.float32
    if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        train_df = task.get_table("train").df
        target = cast(pd.Series, train_df[task.target_col])
        maximum = cast(int | np.integer, target.max())
        num_classes = int(maximum) + 1
        return torch.nn.CrossEntropyLoss(), num_classes, torch.long
    if task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
        return (
            torch.nn.BCEWithLogitsLoss(),
            _multilabel_classes(task),
            torch.float32,
        )
    raise NotImplementedError(f"Unsupported task type: {task.task_type}")


def _multilabel_classes(task: LocalEntityTask) -> int:
    values = task.get_table("train").df[task.target_col]
    maximum = max(
        (int(np.max(value)) for value in values if len(value)),
        default=-1,
    )
    return maximum + 1
