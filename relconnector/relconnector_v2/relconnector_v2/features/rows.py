"""Row gathers for CPU feature production, preserving TensorFrame semantics."""

from __future__ import annotations

from typing import Any

import torch
from torch_frame import TensorFrame
from torch_frame.data import MultiEmbeddingTensor


def gather_rows(frame: TensorFrame, index: torch.Tensor) -> TensorFrame:
    """Copy selected rows without generic advanced indexing for dense features.

    The production path uses nonnegative, one-dimensional CPU long indices.
    Other indices retain upstream behavior, including negative-index handling.
    Ragged features also keep their native row selection. No gathered feature
    values alias the input, even for identity indices: batches can be mutated
    independently of cached rows. Embedding column offsets remain read-only
    metadata shared in the same way as upstream MultiEmbeddingTensor indexing.
    """
    if (
        index.device.type != "cpu"
        or index.dtype != torch.long
        or index.ndim != 1
        or frame.device != torch.device("cpu")
        or bool((index < 0).any())
    ):
        return frame[index]

    def select(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return torch.index_select(value, 0, index)
        if isinstance(value, MultiEmbeddingTensor):
            return MultiEmbeddingTensor(
                num_rows=index.numel(),
                num_cols=value.num_cols,
                values=torch.index_select(value.values, 0, index),
                offset=value.offset,
            )
        if isinstance(value, dict):
            return {key: select(item) for key, item in value.items()}
        return value[index]

    return TensorFrame(
        feat_dict={stype: select(value) for stype, value in frame.feat_dict.items()},
        col_names_dict=frame.col_names_dict,
        y=None if frame.y is None else select(frame.y),
        num_rows=index.numel(),
    )
