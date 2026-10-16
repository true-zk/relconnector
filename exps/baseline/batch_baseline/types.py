"""Shared type aliases for graph materialization and model construction."""

from torch_frame.data.stats import StatType

ColStats = dict[str, dict[str, dict[StatType, object]]]
