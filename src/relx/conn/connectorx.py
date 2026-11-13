"""ConnectorX reader for materialized SQLite databases."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pandas as pd

from .base import BaseDatabaseReader


class ConnectorXDatabaseReader(BaseDatabaseReader):
    """Read complete results or stream bounded Arrow record batches."""

    def __init__(self, location: str | Path) -> None:
        super().__init__(location)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    def read_query(self, query: str) -> pd.DataFrame:
        try:
            import connectorx as cx
        except ImportError as exc:
            raise RuntimeError("ConnectorX reader requires connectorx") from exc
        return cast(pd.DataFrame, cx.read_sql(self.url, query, return_type="pandas"))

    def iter_query(self, query: str, *, batch_size: int) -> Iterator[pd.DataFrame]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        try:
            import connectorx as cx
            import pyarrow as pa
        except ImportError as exc:
            raise RuntimeError(
                "ConnectorX streaming requires connectorx and pyarrow"
            ) from exc

        stream = cast(
            pa.RecordBatchReader,
            cx.read_sql(
                self.url,
                query,
                return_type="arrow_stream",
                batch_size=batch_size,
            ),
        )
        try:
            for batch in stream:
                if batch.num_rows:
                    yield batch.to_pandas(types_mapper=pd.ArrowDtype)
        finally:
            stream.close()
