"""Minimal training adapter for prepared relational batches."""

from .model import OneHopEntityModel
from .multi_hop_model import MultiHopEntityModel

__all__ = ["MultiHopEntityModel", "OneHopEntityModel"]
