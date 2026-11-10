"""Pickle-safe SQL assembly workers for the process runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import resource
import time
from typing import Any, Literal

from relconnector_v2.connector import create_reader
from relconnector_v2.observability import OperationMetrics
from relconnector_v2.sampling import SamplePlan

from .assembler import TensorFrameBatchAssembler
from .contracts import PreparedBatch
from .node_block_text import NodeBlockTextCache
from .schema import TensorFrameEncoder, TensorFrameFeatureSchema
from .shared_encoded import SharedEncodedFeatureCache, create_shared_encoded_cache
from .shared_text import SharedTextEmbeddingCache
from .sql import SqlFeatureFetcher
from .text import GloveTextEmbedder


@dataclass(frozen=True)
class SqlAssemblyWorkerFactory:
    """Serializable configuration for one CPU assembly worker."""

    database_path: str
    database_version: str
    reader_kind: str
    hidden_columns: tuple[tuple[str, str], ...]
    feature_schema: TensorFrameFeatureSchema
    feature_block_size: int
    feature_cache_bytes: int
    text_model_path: str | None
    text_batch_size: int
    text_cache_bytes: int
    text_cache_admission: Literal["always", "second", "adaptive"]
    text_execution: Literal["official", "direct"]
    shared_text_cache_bytes: int
    disk_text_cache_path: str | None
    disk_text_cache_namespace: str
    disk_text_cache_mode: Literal["off", "on", "adaptive"]
    database_bytes: int
    disk_text_cache_min_database_bytes: int
    disk_text_cache_min_model_inputs: int
    disk_text_cache_max_memory_hit_rate: float
    disk_text_cache_min_model_time_s: float
    encoded_cache_bytes: int
    shared_encoded_cache_bytes: int
    encoded_cache_admission: Literal["always", "second", "adaptive"]
    operation_telemetry: bool
    feature_block_density_threshold: float = 0.25
    adaptive_text_index: bool = False
    shared_text_cache: SharedTextEmbeddingCache | None = None
    shared_encoded_cache: SharedEncodedFeatureCache | None = None
    cost_strategy: bool = False
    cost_disk_policy: Literal["budget", "roi"] = "budget"
    cost_disk_bytes: int = 0
    training_epochs: int = 1
    node_block_cache_path: str | None = None
    node_block_cache_namespace: str = ""
    node_block_cache_rows: int = 256
    node_block_cache_max_bytes: int = 8 * 1024**3
    node_block_warm_index: bool = False
    node_block_max_mapped_shards: int = 64

    def __call__(self, worker_index: int) -> SqlAssemblyWorker:
        return SqlAssemblyWorker(self, worker_index)

    def bind_process_context(
        self, context: Any
    ) -> tuple[SqlAssemblyWorkerFactory, tuple[object, ...]]:
        resources: list[object] = []
        text_cache = (
            None
            if self.shared_text_cache_bytes == 0
            else SharedTextEmbeddingCache(
                context,
                max_bytes=self.shared_text_cache_bytes,
                embedding_dim=GloveTextEmbedder.embedding_dim,
                adaptive_local_index=self.adaptive_text_index,
            )
        )
        encoded_cache = None
        if self.shared_encoded_cache_bytes:
            encoded_cache, resource = create_shared_encoded_cache(
                context,
                max_bytes=self.shared_encoded_cache_bytes,
                admission=self.encoded_cache_admission,
            )
            resources.append(resource)
        bound = replace(
            self,
            text_cache_bytes=0 if text_cache is not None else self.text_cache_bytes,
            encoded_cache_bytes=(
                0 if encoded_cache is not None else self.encoded_cache_bytes
            ),
            shared_text_cache=text_cache,
            shared_encoded_cache=encoded_cache,
        )
        return bound, tuple(resources)


class SqlAssemblyWorker:
    """Own SQL, caches, and encoding state for one long-lived process."""

    def __init__(self, config: SqlAssemblyWorkerFactory, worker_index: int) -> None:
        self.config = config
        self.worker_index = worker_index
        self.metrics = OperationMetrics(config.operation_telemetry)
        self.epoch_stage: dict[int, dict[str, float]] = {}
        self.epoch_operations: dict[int, dict[str, float]] = {}
        self.epoch_block_reads: dict[int, dict[str, int]] = {}
        reader = create_reader(config.reader_kind, config.database_path)
        self.text_embedder = GloveTextEmbedder(
            model_path=config.text_model_path,
            cache_bytes=config.text_cache_bytes,
            cache_admission=config.text_cache_admission,
            execution=config.text_execution,
            shared_cache=config.shared_text_cache,
            disk_cache_path=config.disk_text_cache_path,
            disk_cache_namespace=config.disk_text_cache_namespace,
            disk_cache_mode=config.disk_text_cache_mode,
            database_bytes=config.database_bytes,
            disk_cache_min_database_bytes=(config.disk_text_cache_min_database_bytes),
            disk_cache_min_model_inputs=config.disk_text_cache_min_model_inputs,
            disk_cache_max_memory_hit_rate=(config.disk_text_cache_max_memory_hit_rate),
            disk_cache_min_model_time_s=config.disk_text_cache_min_model_time_s,
            metrics=self.metrics,
            cost_strategy=config.cost_strategy,
            cost_disk_policy=config.cost_disk_policy,
            cost_disk_bytes=config.cost_disk_bytes,
            training_epochs=config.training_epochs,
        )
        self.node_block_cache = (
            None
            if config.node_block_cache_path is None
            else NodeBlockTextCache(
                config.node_block_cache_path,
                namespace=config.node_block_cache_namespace,
                embedding_dim=GloveTextEmbedder.embedding_dim,
                rows_per_block=config.node_block_cache_rows,
                max_bytes=config.node_block_cache_max_bytes,
                max_mapped_shards=config.node_block_max_mapped_shards,
            )
        )
        self.fetcher = SqlFeatureFetcher(
            reader,
            hidden_columns=config.hidden_columns,
            feature_columns=config.feature_schema.feature_columns,
            block_size=config.feature_block_size,
            block_density_threshold=config.feature_block_density_threshold,
            cache_bytes=config.feature_cache_bytes,
            database_version=config.database_version,
            metrics=self.metrics,
            node_block_text_cache=self.node_block_cache,
            text_columns={
                table: tuple(
                    column for column, kind in spec.col_to_stype.items()
                    if kind.value == "text_embedded"
                )
                for table, spec in config.feature_schema.tables.items()
            },
            text_embedder=self.text_embedder,
        )
        encoder = TensorFrameEncoder(
            config.feature_schema,
            self.text_embedder,
            text_batch_size=config.text_batch_size,
            metrics=self.metrics,
            preembedded_text=self.node_block_cache is not None,
        )
        self.assembler = TensorFrameBatchAssembler(
            encoder,
            encoded_cache_bytes=config.encoded_cache_bytes,
            encoded_cache_admission=config.encoded_cache_admission,
            encoded_cache=config.shared_encoded_cache,
        )

    def assemble_many(self, plans: list[SamplePlan]) -> list[PreparedBatch]:
        prepared: list[PreparedBatch] = []
        start = 0
        while start < len(plans):
            epoch = plans[start].key.epoch
            stop = start + 1
            while stop < len(plans) and plans[stop].key.epoch == epoch:
                stop += 1
            before = self.metrics.report()
            block_before = self.node_block_cache.snapshot() if self.node_block_cache is not None else None
            begun = time.perf_counter()
            if epoch > 0 and self.config.node_block_warm_index and self.node_block_cache is not None:
                self.node_block_cache.prepare_read_index()
            fetched = self.fetcher.fetch_many(plans[start:stop])
            read_s = time.perf_counter() - begun
            begun = time.perf_counter()
            prepared.extend(self.assembler.assemble_many(fetched))
            process_s = time.perf_counter() - begun
            operations = self.epoch_operations.setdefault(epoch, {})
            after = self.metrics.report()
            processing_in_fetch_s = 0.0
            for kind, name_key, value_key in (("timers", "operation", "total_s"), ("counters", "counter", "value")):
                old = {(item[name_key], item["table"], item["column"]): float(item[value_key]) for item in before[kind]}
                for item in after[kind]:
                    key = (item[name_key], item["table"], item["column"])
                    label = f"{kind}:{key[0]}"
                    delta = float(item[value_key]) - old.get(key, 0.0)
                    operations[label] = operations.get(label, 0.0) + delta
                    if kind == "timers" and key[0] in {"block_text_encode", "block_text_cache_write"}:
                        processing_in_fetch_s += delta
            stages = self.epoch_stage.setdefault(epoch, {"feature_read_s": 0.0, "feature_process_s": 0.0})
            stages["feature_read_s"] += read_s - processing_in_fetch_s
            stages["feature_process_s"] += process_s + processing_in_fetch_s
            if block_before is not None:
                block_after = self.node_block_cache.snapshot()
                reads = self.epoch_block_reads.setdefault(epoch, {"block_reads": 0, "block_read_bytes": 0})
                for key in reads:
                    reads[key] += int(block_after[key]) - int(block_before[key])
            start = stop
        return prepared

    def diagnostics(self) -> dict[str, object]:
        raw = self.fetcher.cache
        encoded = self.assembler.encoded_cache
        encoded_snapshot = (
            encoded.snapshot()
            if isinstance(encoded, SharedEncodedFeatureCache)
            else {
                "hit_rows": encoded.hit_rows,
                "miss_rows": encoded.miss_rows,
                "hit_rate": encoded.hit_rate,
                "inserted_rows": encoded.inserted_rows,
                "evictions": encoded.evictions,
                "admission_rejections": encoded.admission_rejections,
                "entries": encoded.entry_count,
                "current_bytes": encoded.current_bytes,
                "peak_bytes": encoded.peak_bytes,
                "epoch_stats": encoded.epoch_stats,
                "table_stats": encoded.table_stats,
            }
        )
        return {
            "worker": self.worker_index,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "feature_block_cache": {
                "hits": raw.hits,
                "misses": raw.misses,
                "hit_rate": raw.hit_rate,
                "insertions": raw.insertions,
                "evictions": raw.evictions,
                "current_bytes": raw.current_bytes,
                "peak_bytes": raw.peak_bytes,
            },
            "encoded_feature_cache": encoded_snapshot,
            "text_embedding": {
                **asdict(self.text_embedder.stats),
                "epoch_stats": self.text_embedder.epoch_stats,
                "shared": (
                    None
                    if self.config.shared_text_cache is None
                    else self.config.shared_text_cache.snapshot()
                ),
                "disk": self.text_embedder.disk_cache_stats,
                "node_block": (
                    None if self.node_block_cache is None
                    else self.node_block_cache.snapshot()
                ),
            },
            "operations": self.metrics.report(),
            "epoch_stage": self.epoch_stage,
            "epoch_operations": self.epoch_operations,
            "epoch_block_reads": self.epoch_block_reads,
            "feature_epoch_stats": {epoch: asdict(stats) for epoch, stats in self.fetcher.epoch_stats.items()},
        }

    def close(self) -> None:
        self.text_embedder.close()
        if self.node_block_cache is not None:
            self.node_block_cache.close()
