from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import nullcontext
from multiprocessing import get_context
from pathlib import Path
from unittest.mock import patch

import torch
from torch_frame import stype
from torch_frame.data.stats import StatType

from relconnector_v2.connector import ColumnSchema, TableSchema
from relconnector_v2.connector.pandas import PandasDatabaseReader
from relconnector_v2.features.assembler import TensorFrameBatchAssembler
from relconnector_v2.features.node_block_text import NodeBlockTextCache
from relconnector_v2.features.schema import (
    TableFeatureSchema,
    TensorFrameEncoder,
    TensorFrameFeatureSchema,
)
from relconnector_v2.features.sql import SqlFeatureFetcher
from relconnector_v2.sampling import EntitySamplePlan, SampledSubgraph
from relconnector_v2.task import BatchKey


class _TextModel:
    embedding_dim = 300

    def __init__(self) -> None:
        self.inputs = 0

    def epoch_context(self, _epoch: int):
        return nullcontext()

    def for_column(self, _table: str, _column: str):
        return self

    def __call__(self, values: list[str]) -> torch.Tensor:
        self.inputs += len(values)
        return torch.stack([torch.full((300,), float(len(value))) for value in values])


def _plan(epoch: int, batch: int, ids: list[int]) -> EntitySamplePlan:
    sample = SampledSubgraph(
        node_ids={"nodes": torch.tensor(ids)},
        edge_index={},
        batch=None,
        node_time={},
        num_sampled_nodes={},
        num_sampled_edges={},
        seed_node_type="nodes",
        seed_count=len(ids),
        seed_time=None,
    )
    return EntitySamplePlan(BatchKey(epoch, batch), sample, torch.zeros(len(ids)))


def _overlapping_cache_writer(path: str, worker: int, barrier, reports) -> None:
    cache = NodeBlockTextCache(
        path, namespace="shared", embedding_dim=300,
        rows_per_block=8, max_bytes=16 * 1024 * 1024,
    )
    try:
        shared = [cache.key("nodes", "body", index) for index in range(1_000)]
        unique = [cache.key("nodes", "body", 1_000 + 250 * worker + index)
                  for index in range(250)]
        barrier.wait(timeout=30)
        cache.put_many(shared + unique, torch.full((1_250, 300), float(worker)))
        reports.put(worker)
    finally:
        cache.close()


class NodeBlockTextTest(unittest.TestCase):
    def test_concurrent_overlapping_writers_avoid_orphan_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "blocks.sqlite")
            context = get_context("spawn")
            barrier, reports = context.Barrier(4), context.Queue()
            workers = [context.Process(
                target=_overlapping_cache_writer,
                args=(path, worker, barrier, reports),
            ) for worker in range(4)]
            try:
                for worker in workers:
                    worker.start()
                self.assertEqual(set(reports.get(timeout=45) for _ in workers), set(range(4)))
                for worker in workers:
                    worker.join(timeout=10)
                    self.assertEqual(worker.exitcode, 0)
                cache = NodeBlockTextCache(
                    path, namespace="shared", embedding_dim=300,
                    rows_per_block=8, max_bytes=16 * 1024 * 1024,
                )
                try:
                    self.assertEqual(cache.snapshot()["entries"], 2_000)
                    self.assertLess(len(list(Path(directory).glob("*.vectors/*.f32"))), 300)
                finally:
                    cache.close()
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                    worker.join(timeout=10)
                reports.close()

    def test_budget_rejects_a_new_shard_without_partial_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = NodeBlockTextCache(
                Path(directory) / "blocks.sqlite",
                namespace="small",
                embedding_dim=300,
                rows_per_block=4,
                max_bytes=1000,
            )
            try:
                cache.put_many([cache.key("nodes", "body", 0)], torch.ones(1, 300))
                self.assertEqual(cache.lookup([cache.key("nodes", "body", 0)]), {})
                self.assertEqual(cache.admission_rejections, 1)
            finally:
                cache.close()

    def test_warm_mixed_table_reads_only_non_text_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "mixed.sqlite"
            with sqlite3.connect(path) as db:
                db.execute(
                    "CREATE TABLE nodes (id INTEGER PRIMARY KEY, body TEXT, score REAL)"
                )
                db.executemany(
                    "INSERT INTO nodes VALUES (?, ?, ?)",
                    [(0, "long body", 1.0), (1, "another body", 2.0)],
                )
            body = ColumnSchema("body", 1, "object")
            score = ColumnSchema("score", 2, "float64")
            table = TableSchema("nodes", "id", columns=(body, score))
            reader = PandasDatabaseReader(str(path))
            model = _TextModel()
            cache = NodeBlockTextCache(
                root / "blocks.sqlite",
                namespace="mixed",
                embedding_dim=300,
                rows_per_block=4,
                max_bytes=1024 * 1024,
            )
            fetcher = SqlFeatureFetcher(
                reader,
                feature_columns={"nodes": (body, score)},
                block_size=8,
                read_policy="sparse",
                cache_bytes=0,
                node_block_text_cache=cache,
                text_columns={"nodes": ("body",)},
                text_embedder=model,  # type: ignore[arg-type]
            )
            queries: list[str] = []
            original_read = reader.read_query

            def counted_read(query: str):
                queries.append(query)
                return original_read(query)

            try:
                with (
                    patch.object(reader, "schemas", return_value={"nodes": table}),
                    patch.object(reader, "read_query", side_effect=counted_read),
                ):
                    ids = torch.tensor([0, 1])
                    first, _ = fetcher._fetch_with_node_block_text(
                        "nodes", ids, epoch=0
                    )
                    self.assertEqual(model.inputs, 2)
                    self.assertIn('"body"', queries[0])
                    queries.clear()
                    later, _ = fetcher._fetch_with_node_block_text(
                        "nodes", ids, epoch=1
                    )
                    self.assertEqual(model.inputs, 2)
                    self.assertEqual(len(queries), 1)
                    self.assertNotIn('"body"', queries[0])
                    self.assertIn('"score"', queries[0])
                    self.assertEqual(first["score"].tolist(), later["score"].tolist())
            finally:
                cache.close()

    def test_window_reuses_vectors_and_skips_sql_text_on_later_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "data.sqlite"
            with sqlite3.connect(path) as db:
                db.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY, body TEXT)")
                db.executemany(
                    "INSERT INTO nodes VALUES (?, ?)",
                    [(0, "first"), (1, "shared"), (2, "last"), (3, "new")],
                )
            column = ColumnSchema("body", 1, "object")
            table = TableSchema("nodes", "id", columns=(column,))
            reader = PandasDatabaseReader(str(path))
            model = _TextModel()
            cache = NodeBlockTextCache(
                root / "blocks.sqlite",
                namespace="test",
                embedding_dim=300,
                rows_per_block=4,
                max_bytes=1024 * 1024,
            )
            fetcher = SqlFeatureFetcher(
                reader,
                feature_columns={"nodes": (column,)},
                block_size=8,
                read_policy="sparse",
                cache_bytes=0,
                node_block_text_cache=cache,
                text_columns={"nodes": ("body",)},
                text_embedder=model,  # type: ignore[arg-type]
            )
            original_read = reader.read_query
            queries: list[str] = []

            def counted_read(query: str):
                queries.append(query)
                return original_read(query)

            schema = TensorFrameFeatureSchema(
                {
                    "nodes": TableFeatureSchema(
                        (column,),
                        {"body": stype.text_embedded},
                        {"body": {StatType.EMB_DIM: 300}},
                    )
                },
                "test",
            )
            encoder = TensorFrameEncoder(
                schema, model, text_batch_size=8, preembedded_text=True
            )
            try:
                with (
                    patch.object(reader, "schemas", return_value={"nodes": table}),
                    patch.object(reader, "read_query", side_effect=counted_read),
                ):
                    first = fetcher.fetch_many(
                        [
                            _plan(0, 0, [0, 1]),
                            _plan(0, 1, [1, 2]),
                        ]
                    )
                    self.assertEqual(model.inputs, 3)
                    self.assertEqual(len(queries), 1)
                    first_frame = first[0].subgraph.frames["nodes"]
                    first_tensor = encoder.encode("nodes", first_frame.frame)
                    prepared = TensorFrameBatchAssembler(encoder).assemble_many(first)
                    left = (
                        prepared[0].data["nodes"].tf.feat_dict[stype.embedding].values
                    )
                    right = (
                        prepared[1].data["nodes"].tf.feat_dict[stype.embedding].values
                    )
                    self.assertTrue(torch.equal(left[1], right[0]))
                    queries.clear()
                    later = fetcher.fetch_many(
                        [
                            _plan(1, 0, [2, 1]),
                            _plan(1, 1, [0, 1]),
                        ]
                    )
                    self.assertEqual(queries, [])
                    self.assertEqual(model.inputs, 3)
                    later_frame = later[0].subgraph.frames["nodes"]
                    later_tensor = encoder.encode("nodes", later_frame.frame)
                    self.assertTrue(
                        torch.equal(
                            first_tensor.feat_dict[stype.embedding].values,
                            later_tensor.feat_dict[stype.embedding].values,
                        )
                    )
                    baseline = TensorFrameEncoder(
                        schema, model, text_batch_size=8
                    ).encode(
                        "nodes",
                        original_read(
                            "SELECT body FROM nodes WHERE id < 3 ORDER BY id"
                        ),
                    )
                    self.assertTrue(
                        torch.equal(
                            later_tensor.feat_dict[stype.embedding].values,
                            baseline.feat_dict[stype.embedding].values,
                        )
                    )
                    self.assertEqual(cache.snapshot()["entries"], 3)
                    self.assertGreaterEqual(cache.snapshot()["block_reads"], 1)
                    queries.clear()
                    fetcher.fetch_many([_plan(2, 0, [2, 3])])
                    self.assertEqual(model.inputs, 7)  # 3 cold, 3 baseline, 1 new
                    self.assertEqual(len(queries), 1)
                    queries.clear()
                    fetcher.fetch_many([_plan(3, 0, [3])])
                    self.assertEqual(queries, [])
            finally:
                cache.close()


if __name__ == "__main__":
    unittest.main()
