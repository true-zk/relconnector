"""Connector-X implementation of the database reader contract."""

from __future__ import annotations

import pandas as pd

from .base import BaseDatabaseReader


class ConnectorXDatabaseReader(BaseDatabaseReader):
    """Read SQL through Connector-X and return pandas dataframes."""

    def read_query(self, query: str) -> pd.DataFrame:
        try:
            import connectorx as cx
        except ImportError as exc:
            raise RuntimeError(
                "Connector-X reader requires `pip install connectorx`"
            ) from exc
        return cx.read_sql(self.url, query, return_type="pandas")
