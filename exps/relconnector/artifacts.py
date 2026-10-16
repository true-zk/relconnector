"""Validated, atomic caches for small statistics and topology artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType

from .connector import BaseDatabaseReader
from .connector.base import sqlite_path

if TYPE_CHECKING:
    from .graph.index import GraphIndex


def artifact_key(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path, expected_key: str) -> object | None:
    if not path.exists():
        return None
    try:
        envelope = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if envelope.get("key") != expected_key:
        return None
    return decode_json_value(envelope.get("payload"))


def save_json(path: Path, key: str, payload: object) -> None:
    envelope = {"key": key, "payload": encode_json_value(payload)}
    _atomic_write(
        path,
        json.dumps(envelope, ensure_ascii=True, separators=(",", ":")).encode(),
    )


def load_graph(path: Path, expected_key: str) -> GraphIndex | None:
    from .graph.index import GraphIndex

    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("key") != expected_key:
            return None
        edge_types = tuple(tuple(item) for item in payload["edge_types"])
        data = HeteroData()
        for node_type, count in payload["node_counts"].items():
            data[node_type].num_nodes = int(count)
        for node_type, values in payload["node_times"].items():
            data[node_type].time = values
        for edge_type, count in zip(
            edge_types, payload["edge_counts"], strict=True
        ):
            data[cast(EdgeType, edge_type)].num_edges = int(count)
        return GraphIndex(
            data=data,
            edge_types=cast(tuple[EdgeType, ...], edge_types),
            colptr_dict=payload["colptr_dict"],
            row_dict=payload["row_dict"],
            node_count=int(payload["node_count"]),
            edge_count=int(payload["edge_count"]),
            allocated_bytes=int(payload["allocated_bytes"]),
        )
    except (
        AttributeError,
        EOFError,
        IndexError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return None


def save_graph(path: Path, key: str, graph: GraphIndex) -> None:
    payload = {
        "key": key,
        "edge_types": [list(edge_type) for edge_type in graph.edge_types],
        "edge_counts": [
            int(graph.data[edge_type].num_edges) for edge_type in graph.edge_types
        ],
        "node_counts": {
            node_type: int(graph.data[node_type].num_nodes)
            for node_type in graph.node_types
        },
        "node_times": {
            node_type: graph.data[node_type].time
            for node_type in graph.node_types
            if "time" in graph.data[node_type]
        },
        "colptr_dict": graph.colptr_dict,
        "row_dict": graph.row_dict,
        "node_count": graph.node_count,
        "edge_count": graph.edge_count,
        "allocated_bytes": graph.allocated_bytes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def encode_json_value(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return {
            "__type__": "tensor",
            "dtype": str(value.dtype).removeprefix("torch."),
            "value": value.tolist(),
        }
    if isinstance(value, np.generic):
        return encode_json_value(value.item())
    if isinstance(value, pd.Timestamp):
        return {"__type__": "timestamp", "value": value.isoformat()}
    if isinstance(value, pd.Timedelta):
        return {"__type__": "timedelta", "value": value.isoformat()}
    if isinstance(value, bytes):
        return {"__type__": "bytes", "value": value.hex()}
    if isinstance(value, tuple):
        return {"__type__": "tuple", "value": [encode_json_value(v) for v in value]}
    if isinstance(value, list):
        return [encode_json_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): encode_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, float) and not np.isfinite(value):
        return {"__type__": "float", "value": repr(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported artifact value: {type(value).__name__}")


def decode_json_value(value: object) -> object:
    if isinstance(value, list):
        return [decode_json_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("__type__")
    if kind == "tensor":
        dtype = getattr(torch, str(value["dtype"]))
        return torch.tensor(value["value"], dtype=dtype)
    if kind == "timestamp":
        return pd.Timestamp(value["value"])
    if kind == "timedelta":
        return pd.Timedelta(value["value"])
    if kind == "bytes":
        return bytes.fromhex(str(value["value"]))
    if kind == "tuple":
        return tuple(decode_json_value(item) for item in value["value"])
    if kind == "float":
        return float(str(value["value"]))
    return {key: decode_json_value(item) for key, item in value.items()}


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _temporary(path: Path) -> Path:
    return path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )


class CachedGraphIndexBuilder:
    """Cache topology while keeping the existing builder implementation isolated."""

    def __init__(
        self,
        reader: BaseDatabaseReader,
        *,
        scan_batch_size: int,
        cache_dir: str | Path,
    ) -> None:
        from .graph.builder import InMemoryGraphIndexBuilder

        self.reader = reader
        self.inner = InMemoryGraphIndexBuilder(
            reader, scan_batch_size=scan_batch_size
        )
        self.cache_dir = Path(cache_dir)
        self.cache_hit = False

    def build(self, *, cutoff: pd.Timestamp | None = None) -> GraphIndex:
        key = self._key(cutoff)
        path = self.cache_dir / "topology" / f"{key}.pt"
        cached = load_graph(path, key)
        if cached is not None:
            self.cache_hit = True
            return cached
        graph = self.inner.build(cutoff=cutoff)
        save_graph(path, key, graph)
        return graph

    def _key(self, cutoff: pd.Timestamp | None) -> str:
        database = Path(sqlite_path(self.reader.url))
        stat = database.stat()
        topology = []
        for schema in self.reader.schemas().values():
            if schema.kind != "data":
                continue
            topology.append(
                {
                    "table": schema.name,
                    "primary_key": schema.primary_key,
                    "time_column": schema.time_column,
                    "foreign_keys": [
                        [
                            foreign_key.column,
                            foreign_key.reference_table,
                            foreign_key.reference_column,
                        ]
                        for foreign_key in schema.foreign_keys
                    ],
                }
            )
        return artifact_key(
            {
                "algorithm": "csc-topology-v1",
                "database": {
                    "path": str(database.resolve()),
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                },
                "cutoff": None if cutoff is None else str(cutoff),
                "topology": topology,
            }
        )
