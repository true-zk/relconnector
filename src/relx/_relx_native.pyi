"""Typing surface of the locally built C++ graph extension."""

from typing import Any

import numpy as np

class GraphIndex:
    @property
    def node_counts(self) -> dict[str, int]: ...
    def relation_names(self) -> list[str]: ...
    def relation_endpoints(self, name: str) -> tuple[str, str]: ...
    def relation(self, name: str) -> dict[str, Any]: ...
    def node_times(self, table: str) -> np.ndarray: ...
    def visible_neighbors(
        self, relation: str, destination: int, cutoff_us: int
    ) -> list[int]: ...
    def sample_neighbors(
        self,
        relation: str,
        destinations: list[int],
        cutoffs_us: list[int],
        fanout: int,
        random_seed: int,
        workers: int = 1,
    ) -> list[list[int]]: ...
    def sample_one_hop(
        self,
        seed_table: str,
        seed_ids: list[int],
        cutoffs_us: list[int],
        fanout: int,
        random_seed: int,
        workers: int = 1,
    ) -> dict[str, dict[str, np.ndarray]]: ...
    def sample_multi_hop(
        self,
        seed_table: str,
        seed_ids: list[int],
        cutoffs_us: list[int],
        fanouts: list[int],
        random_seed: int,
        workers: int = 1,
    ) -> dict[str, dict[str, np.ndarray]]: ...
    def lookup_nodes(
        self, table: str, key_columns: list[str], batch: object
    ) -> np.ndarray: ...


class GraphBuilder:
    def __init__(self, inclusive: bool = True) -> None: ...
    def register_table(self, table: str, timed: bool = False) -> None: ...
    def register_relation(self, relation: str, source: str, target: str) -> None: ...
    def add_nodes(self, table: str, key_columns: list[str], batch: object) -> None: ...
    def add_edges(
        self,
        relation: str,
        source: str,
        target: str,
        source_key_columns: list[str],
        foreign_key_columns: list[str],
        batch: object,
    ) -> None: ...
    def finish(self) -> GraphIndex: ...


class NativeFeatureStore:
    def add_numeric(
        self, table: str, column: str, values: np.ndarray,
        mean: float, std: float,
    ) -> None: ...
    def add_categorical(
        self, table: str, column: str, values: np.ndarray,
    ) -> None: ...
    def add_timestamp(
        self, table: str, column: str, values: np.ndarray,
        mean: float, std: float,
    ) -> None: ...
    def add_multicategorical(
        self, table: str, column: str, values: np.ndarray,
        offsets: np.ndarray,
    ) -> None: ...
    def add_embedding(
        self, table: str, column: str, values: np.ndarray,
    ) -> None: ...
    def add_text(
        self, table: str, column: str, values: list[str],
    ) -> None: ...
    def gather(
        self, node_ids: dict[str, np.ndarray], workers: int = 1
    ) -> dict[str, dict[str, dict[str, np.ndarray | None]]]: ...


class NativeBatchPipeline:
    def __init__(
        self,
        graph: GraphIndex,
        features: NativeFeatureStore,
        fanouts: list[int],
        multi_hop: bool,
        sampling_workers: int,
        feature_workers: int,
        feature_native_workers: int,
        max_inflight: int,
    ) -> None: ...
    def submit(
        self,
        sequence: int,
        seed_table: str,
        seed_ids: list[int],
        cutoffs_us: list[int],
        random_seed: int,
    ) -> None: ...
    def finish(self) -> None: ...
    def next(self) -> dict[str, Any] | None: ...
    def close(self) -> None: ...


class NativeFeatureReader: ...


class SqliteFeatureReader(NativeFeatureReader):
    def __init__(self, database_path: str) -> None: ...
    def add_table(
        self, name: str, schema: str, identity: list[str],
        rows: tuple[tuple[object, ...], ...],
    ) -> None: ...
    def add_feature(
        self, table: str, column: str, kind: str, mean: float = 0.0,
        std: float = 1.0, width: int = 1,
        vocab: dict[str, int] | None = None,
    ) -> None: ...
    def set_glove_model(
        self, directory: str, cache_bytes: int = 0,
        disk_cache_path: str = "",
    ) -> None: ...
    def set_row_cache_bytes(self, bytes: int) -> None: ...
    def fetch(
        self, node_ids: dict[str, np.ndarray],
    ) -> dict[str, dict[str, dict[str, np.ndarray | None]]]: ...


class MySqlFeatureReader(NativeFeatureReader):
    def __init__(
        self, host: str, port: int, user: str, password: str,
        database: str, unix_socket: str = "",
    ) -> None: ...
    def add_table(
        self, name: str, schema: str, identity: list[str],
        rows: tuple[tuple[object, ...], ...],
    ) -> None: ...
    def add_feature(
        self, table: str, column: str, kind: str, mean: float = 0.0,
        std: float = 1.0, width: int = 1,
        vocab: dict[str, int] | None = None,
    ) -> None: ...
    def set_glove_model(
        self, directory: str, cache_bytes: int = 0,
        disk_cache_path: str = "",
    ) -> None: ...
    def set_row_cache_bytes(self, bytes: int) -> None: ...
    def fetch(
        self, node_ids: dict[str, np.ndarray],
    ) -> dict[str, dict[str, dict[str, np.ndarray | None]]]: ...


class NativeOnlinePipeline:
    def __init__(
        self, reader: NativeFeatureReader, workers: int, max_inflight: int,
        max_ready_bytes: int = 268_435_456,
    ) -> None: ...
    def submit(self, sequence: int, node_ids: dict[str, np.ndarray]) -> None: ...
    def finish(self) -> None: ...
    def abort(self) -> None: ...
    def next(self) -> dict[str, Any] | None: ...
    def close(self) -> None: ...
