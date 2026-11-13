"""Dataset inspection and preparation before training."""

from .schema import discover_sqlite_schema

__all__ = ["discover_sqlite_schema"]
