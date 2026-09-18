"""Connector-X implementation of the database reader contract."""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import pandas as pd

from .base import BaseDatabaseReader


class ConnectorXDatabaseReader(BaseDatabaseReader):
    """Read SQL through Connector-X as pandas frames or Arrow stream batches."""

    supports_batched_reads = True

    def read_query(self, query: str) -> pd.DataFrame:
        try:
            import connectorx as cx
        except ImportError as exc:
            raise RuntimeError(
                "Connector-X reader requires `pip install connectorx`"
            ) from exc
        return cast(
            pd.DataFrame,
            cx.read_sql(self.url, query, return_type="pandas"),
        )

    def iter_query(
        self,
        query: str,
        *,
        batch_size: int | None = None,
    ) -> Iterator[pd.DataFrame]:
        if batch_size is None:
            yield self.read_query(query)
            return
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        try:
            import connectorx as cx
            import pyarrow as pa
        except ImportError as exc:
            raise RuntimeError(
                "Connector-X streaming requires connectorx and pyarrow"
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
                yield batch.to_pandas()
        finally:
            stream.close()
