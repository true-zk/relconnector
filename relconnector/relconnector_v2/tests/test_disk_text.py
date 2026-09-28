from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from unittest.mock import patch

import torch

from relconnector_v2.features.disk_text import (
    AdaptiveDiskTextEmbeddingCache,
    SqliteTextEmbeddingCache,
    _wal_checkpoint_target,
    _WalReadPolicy,
    text_embedding_cache_location,
)
from relconnector_v2.features.shared_text import SharedTextEmbeddingCache
from relconnector_v2.features.text import GloveTextEmbedder


class _FakeSentenceTransformer:
    encode_calls = 0

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def eval(self) -> None:
        pass

    def encode(self, sentences: list[str], **_kwargs: object) -> torch.Tensor:
        type(self).encode_calls += len(sentences)
        return torch.stack(
            [torch.full((300,), float(len(sentence))) for sentence in sentences]
        )


def _concurrent_disk_writer(path: str, worker: int, barrier, reports) -> None:
    cache = SqliteTextEmbeddingCache(
        path, namespace="concurrent", embedding_dim=2,
        wal_checkpoint_bytes=32 * 1024, write_shard_entries=32,
    )
    cache.lookup([])
    barrier.wait(timeout=30)
    for step in range(20):
        values = [f"shared-{step}-{i}" for i in range(8)]
        values += [f"worker-{worker}-{step}-{i}" for i in range(8)]
        found = cache.lookup_or_reserve(values)
        cache.publish(found.reservations, {
            position: torch.tensor([float(len(values[position])), 1.0])
            for position in found.reservations
        })
        found.hits.update(cache.wait(found.pending))
    reports.put(cache.snapshot())
    cache.close()


class SqliteTextEmbeddingCacheTest(unittest.TestCase):
    def test_checkpoint_budget_grows_with_index_and_respects_free_space(self) -> None:
        mib = 1024 * 1024
        self.assertEqual(_wal_checkpoint_target(1 * mib, 100_000 * mib), 64 * mib)
        self.assertEqual(_wal_checkpoint_target(200 * mib, 100_000 * mib), 800 * mib)
        self.assertEqual(_wal_checkpoint_target(2000 * mib, 100_000 * mib), 4096 * mib)
        self.assertEqual(_wal_checkpoint_target(2000 * mib, 1600 * mib), 100 * mib)

    def test_wal_read_policy_requires_sustained_reuse_and_exits_on_new_work(self) -> None:
        mib = 1024**2
        policy = _WalReadPolicy()
        policy.observe(0, 100 * mib, 0, 0)
        policy.observe(30, 110 * mib, 8192, 8192)
        self.assertFalse(policy.read_heavy)
        policy.observe(60, 120 * mib, 16384, 16384)
        self.assertTrue(policy.read_heavy)
        self.assertEqual(_wal_checkpoint_target(2000 * mib, 100_000 * mib,
                                               read_heavy=True), 512 * mib)
        # New unseen text must restore the larger write budget.
        policy.observe(90, 130 * mib, 24576, 17000)
        self.assertFalse(policy.read_heavy)
        policy.observe(120, 140 * mib, 32768, 25192)
        policy.observe(150, 150 * mib, 40960, 33384)
        self.assertTrue(policy.read_heavy)
        # Other workers can be writing even when this worker gets all hits.
        policy.observe(180, 450 * mib, 49152, 41576)
        self.assertFalse(policy.read_heavy)
        self.assertEqual(policy.transitions, 4)

    def test_wal_read_policy_ignores_tiny_samples_and_accounts_for_recycling(self) -> None:
        mib = 1024**2
        policy = _WalReadPolicy()
        policy.observe(0, 4000 * mib, 0, 0)
        policy.observe(30, 10 * mib, 10, 10)
        policy.observe(60, 20 * mib, 20, 20)
        self.assertFalse(policy.read_heavy)
        policy.observe(90, 30 * mib, 8212, 8212)
        policy.observe(120, 40 * mib, 16404, 16404)
        self.assertTrue(policy.read_heavy)
        self.assertEqual(_wal_checkpoint_target(2000 * mib, 1600 * mib,
                                               read_heavy=True), 100 * mib)

    def test_all_hit_lookup_can_checkpoint_without_another_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = SqliteTextEmbeddingCache(
                Path(directory) / "text.sqlite", namespace="warm-reader", embedding_dim=2,
            )
            try:
                expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
                cache.put_many(["one", "two"], expected)
                self.assertGreater(cache._wal_size(), 0)
                cache._wal_policy_checked = 0.0
                with patch("relconnector_v2.features.disk_text._wal_checkpoint_target",
                           return_value=1):
                    found = cache.lookup(["two", "one"])
                self.assertEqual(cache._wal_size(), 0)
                self.assertTrue(torch.equal(found.hits[0], expected[1]))
                self.assertTrue(torch.equal(found.hits[1], expected[0]))
                self.assertEqual(cache._writes, 2)
            finally:
                cache.close()

    def test_concurrent_writers_checkpoint_and_keep_shared_publications(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "text.sqlite")
            context = get_context("spawn")
            barrier, reports = context.Barrier(2), context.Queue()
            workers = [context.Process(target=_concurrent_disk_writer,
                args=(path, worker, barrier, reports)) for worker in range(2)]
            try:
                for worker in workers:
                    worker.start()
                stats = [reports.get(timeout=45) for _ in workers]
                for worker in workers:
                    worker.join(timeout=10)
                    self.assertEqual(worker.exitcode, 0)
                self.assertGreater(sum(s["checkpoint_attempts"] for s in stats), 0)
                self.assertEqual(sum(s["writes"] for s in stats), 480)
                reader = SqliteTextEmbeddingCache(path, namespace="concurrent", embedding_dim=2)
                self.assertEqual(reader.snapshot()["entries"], 480)
                values = [f"shared-{step}-{i}" for step in range(20) for i in range(8)]
                result = reader.lookup(values)
                self.assertEqual(len(result.hits), len(values))
                for position, value in enumerate(values):
                    self.assertTrue(torch.equal(result.hits[position], torch.tensor([float(len(value)), 1.0])))
                reader.close()
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                    worker.join(timeout=10)
                reports.close()

    def test_shard_mapping_lru_reopens_without_changing_returned_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = SqliteTextEmbeddingCache(
                Path(directory) / "text.sqlite", namespace="lru", embedding_dim=2,
                max_mapped_shards=2, write_shard_entries=1,
            )
            for index in range(5):
                cache.put_many([str(index)], torch.tensor([[index, index + 1.0]]))
            first = cache.lookup(["0"]).hits[0]
            first_map = getattr(next(iter(cache._mapped_shards.values())), "_mmap", None)
            assert first_map is not None
            cache.lookup(["1"])
            cache.lookup(["0"])  # Touch: shard 1 should be evicted before shard 0.
            cache.lookup(["2"])
            self.assertFalse(first_map.closed)
            result = cache.lookup(["4", "0", "3", "2", "1"])
            self.assertEqual(len(cache._mapped_shards), 2)
            self.assertTrue(first_map.closed)
            for position, index in enumerate([4, 0, 3, 2, 1]):
                self.assertTrue(torch.equal(result.hits[position], torch.tensor([index, index + 1.0])))
            self.assertTrue(torch.equal(first, torch.tensor([0.0, 1.0])))
            cache.close()
            self.assertTrue(torch.equal(result.hits[0], torch.tensor([4.0, 5.0])))
            self.assertEqual(cache._mapped_shards, {})

    def test_append_shards_are_visible_to_existing_reader_mappings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text.sqlite"
            writer = SqliteTextEmbeddingCache(
                path, namespace="append", embedding_dim=2, write_shard_entries=3,
            )
            reader = SqliteTextEmbeddingCache(path, namespace="append", embedding_dim=2)
            writer.put_many(["0"], torch.tensor([[0.0, 1.0]]))
            original = reader.lookup(["0"]).hits[0]
            writer.put_many(["1", "2", "3", "4"], torch.tensor(
                [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
            ))
            result = reader.lookup([str(i) for i in range(5)])
            self.assertEqual(len(list(path.with_suffix(".vectors").glob("*.f32"))), 2)
            for i in range(5):
                self.assertTrue(torch.equal(result.hits[i], torch.tensor([i, i + 1.0])))
            writer.close()
            reader.close()
            self.assertTrue(torch.equal(original, torch.tensor([0.0, 1.0])))

    def test_wal_checkpoint_reclaims_log_without_losing_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text.sqlite"
            cache = SqliteTextEmbeddingCache(
                path, namespace="checkpoint", embedding_dim=2, wal_checkpoint_bytes=1,
            )
            other = SqliteTextEmbeddingCache(path, namespace="checkpoint", embedding_dim=2)
            for i in range(6):
                lookup = cache.lookup_or_reserve([str(i)])
                cache.publish(lookup.reservations, {0: torch.tensor([i, i + 1.0])})
                self.assertEqual(cache._wal_size(), 0)
                self.assertTrue(torch.equal(other.lookup([str(i)]).hits[0], torch.tensor([i, i + 1.0])))
            self.assertGreater(cache._checkpoint_attempts, 1)
            self.assertEqual(cache._checkpoint_busy, 0)
            cache.close()
            other.close()

    def test_database_and_model_changes_use_different_cache_files(self) -> None:
        settings = {
            "database_path": "/data/rel-arxiv.sqlite",
            "execution": "direct",
            "embedding_dim": 300,
        }
        initial = text_embedding_cache_location(
            "/cache", database_version="v1", model_identity="glove-v1", **settings
        )
        changed_database = text_embedding_cache_location(
            "/cache", database_version="v2", model_identity="glove-v1", **settings
        )
        changed_model = text_embedding_cache_location(
            "/cache", database_version="v1", model_identity="glove-v2", **settings
        )
        self.assertEqual(len({initial, changed_database, changed_model}), 3)

    def test_embeddings_persist_across_cache_instances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text.sqlite"
            first = SqliteTextEmbeddingCache(path, namespace="test", embedding_dim=4)
            lookup = first.lookup_or_reserve(["alpha", "beta"])
            self.assertEqual(set(lookup.reservations), {0, 1})
            first.publish(
                lookup.reservations,
                {
                    0: torch.tensor([1.0, 2.0, 3.0, 4.0]),
                    1: torch.tensor([5.0, 6.0, 7.0, 8.0]),
                },
            )
            first.close()

            second = SqliteTextEmbeddingCache(path, namespace="test", embedding_dim=4)
            cached = second.lookup_or_reserve(["beta", "alpha"])
            self.assertEqual(cached.reservations, {})
            self.assertEqual(cached.pending, {})
            self.assertTrue(
                torch.equal(cached.hits[0], torch.tensor([5.0, 6.0, 7.0, 8.0]))
            )
            self.assertTrue(
                torch.equal(cached.hits[1], torch.tensor([1.0, 2.0, 3.0, 4.0]))
            )
            second.close()

    def test_waiter_reads_owner_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text.sqlite"
            owner = SqliteTextEmbeddingCache(path, namespace="test", embedding_dim=2)
            waiter = SqliteTextEmbeddingCache(path, namespace="test", embedding_dim=2)
            owned = owner.lookup_or_reserve(["shared"])
            pending = waiter.lookup_or_reserve(["shared"])
            self.assertEqual(set(owned.reservations), {0})
            self.assertEqual(set(pending.pending), {0})

            owner.publish(owned.reservations, {0: torch.tensor([3.0, 4.0])})
            hits = waiter.wait(pending.pending)
            self.assertTrue(torch.equal(hits[0], torch.tensor([3.0, 4.0])))
            owner.close()
            waiter.close()

    def test_connection_can_be_used_and_closed_across_threads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = SqliteTextEmbeddingCache(
                Path(directory) / "text.sqlite", namespace="threads", embedding_dim=2
            )

            def write() -> None:
                lookup = cache.lookup_or_reserve(["shared"])
                cache.publish(lookup.reservations, {0: torch.tensor([1.0, 2.0])})

            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(write).result()
            self.assertTrue(
                torch.equal(
                    cache.lookup_or_reserve(["shared"]).hits[0],
                    torch.tensor([1.0, 2.0]),
                )
            )
            cache.close()


class AdaptiveDiskTextEmbeddingCacheTest(unittest.TestCase):
    def test_small_database_never_creates_persistent_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text.sqlite"
            cache = AdaptiveDiskTextEmbeddingCache(
                path=path,
                namespace="small",
                embedding_dim=2,
                mode="adaptive",
                database_bytes=10,
                min_database_bytes=100,
                min_model_inputs=2,
                max_memory_hit_rate=0.1,
                min_model_time_s=0.1,
            )
            cache.observe_encoded(
                ["alpha", "beta"],
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                model_time_s=1.0,
                memory_hit_rate=0.0,
            )

            self.assertFalse(cache.active)
            self.assertFalse(path.exists())

    def test_small_database_ignores_existing_forced_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text.sqlite"
            forced = SqliteTextEmbeddingCache(path, namespace="small", embedding_dim=2)
            forced.put_many(["alpha"], torch.tensor([[1.0, 2.0]]))
            forced.close()
            cache = AdaptiveDiskTextEmbeddingCache(
                path=path,
                namespace="small",
                embedding_dim=2,
                mode="adaptive",
                database_bytes=10,
                min_database_bytes=100,
                min_model_inputs=2,
                max_memory_hit_rate=0.1,
                min_model_time_s=0.1,
            )
            self.assertIsNone(cache.lookup_or_reserve(["alpha"]))
            self.assertFalse(cache.active)
            cache.close()

    def test_high_memory_hit_rate_keeps_warmup_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text.sqlite"
            cache = AdaptiveDiskTextEmbeddingCache(
                path=path,
                namespace="reused",
                embedding_dim=2,
                mode="adaptive",
                database_bytes=1000,
                min_database_bytes=100,
                min_model_inputs=2,
                max_memory_hit_rate=0.1,
                min_model_time_s=0.1,
                write_batch_entries=3,
            )
            cache.observe_encoded(
                [str(index) for index in range(10)],
                torch.zeros((10, 2)),
                model_time_s=1.0,
                memory_hit_rate=0.9,
            )
            self.assertEqual(cache.snapshot()["warmup_entries"], 3)
            self.assertFalse(path.exists())
            cache.close()

    def test_large_low_reuse_workload_activates_and_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text.sqlite"
            cache = AdaptiveDiskTextEmbeddingCache(
                path=path,
                namespace="large",
                embedding_dim=2,
                mode="adaptive",
                database_bytes=1000,
                min_database_bytes=100,
                min_model_inputs=2,
                max_memory_hit_rate=0.1,
                min_model_time_s=0.1,
            )
            cache.observe_encoded(
                ["alpha", "beta"],
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                model_time_s=1.0,
                memory_hit_rate=0.0,
            )
            self.assertTrue(cache.active)
            self.assertEqual(
                cache.snapshot()["activation_reason"],
                "large-low-reuse-encoding-bound",
            )
            cache.close()

            reused = AdaptiveDiskTextEmbeddingCache(
                path=path,
                namespace="large",
                embedding_dim=2,
                mode="adaptive",
                database_bytes=1000,
                min_database_bytes=100,
                min_model_inputs=2,
                max_memory_hit_rate=0.1,
                min_model_time_s=0.1,
            )
            self.assertTrue(reused.active)
            lookup = reused.lookup_or_reserve(["beta"])
            self.assertIsNotNone(lookup)
            assert lookup is not None
            self.assertTrue(torch.equal(lookup.hits[0], torch.tensor([3.0, 4.0])))
            reused.close()

    def test_worker_attaches_after_another_worker_activates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text.sqlite"
            settings = {
                "path": path,
                "namespace": "shared",
                "embedding_dim": 2,
                "mode": "adaptive",
                "database_bytes": 1000,
                "min_database_bytes": 100,
                "min_model_inputs": 2,
                "max_memory_hit_rate": 0.1,
                "min_model_time_s": 0.1,
            }
            waiting = AdaptiveDiskTextEmbeddingCache(**settings)
            owner = AdaptiveDiskTextEmbeddingCache(**settings)
            self.assertIsNone(waiting.lookup_or_reserve(["alpha"]))
            owner.observe_encoded(
                ["alpha", "beta"],
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                model_time_s=1.0,
                memory_hit_rate=0.0,
            )
            owner.flush()

            lookup = waiting.lookup_or_reserve(["alpha", "beta"])
            self.assertIsNotNone(lookup)
            assert lookup is not None
            self.assertEqual(set(lookup.hits), {0, 1})
            self.assertEqual(waiting.snapshot()["activation_reason"], "existing-cache")
            waiting.close()
            owner.close()


class GloveDiskCacheIntegrationTest(unittest.TestCase):
    def test_adaptive_index_preserves_embeddings_without_reencoding(self) -> None:
        shared = SharedTextEmbeddingCache(
            get_context("spawn"),
            max_bytes=4 * 300 * 4,
            embedding_dim=300,
            adaptive_local_index=True,
        )
        index = shared._worker_index()
        index.window_calls = index.min_requests = 1
        index.min_lookup_s = 0.0
        _FakeSentenceTransformer.encode_calls = 0
        with patch(
            "sentence_transformers.SentenceTransformer", _FakeSentenceTransformer
        ):
            embedder = GloveTextEmbedder(execution="official", shared_cache=shared)
            with embedder.epoch_context(0):
                expected = embedder(["alpha", "beta", "alpha"])
                embedder(["alpha", "beta", "alpha"])
                embedder(["alpha", "beta", "alpha"])
            self.assertTrue(index.active)
            with embedder.epoch_context(1):
                actual = embedder(["alpha", "beta", "alpha"])
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(_FakeSentenceTransformer.encode_calls, 2)
            self.assertEqual(embedder.stats.model_inputs, 2)
            self.assertEqual(embedder.epoch_stats[1].get("model_inputs", 0), 0)
            self.assertGreater(shared.snapshot()["local_index"]["fast_hits"], 0)
            embedder.close()

    def test_later_epoch_reads_disk_when_shared_memory_is_full(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "text.sqlite")
            shared = SharedTextEmbeddingCache(
                get_context("spawn"), max_bytes=0, embedding_dim=300
            )
            _FakeSentenceTransformer.encode_calls = 0
            with patch(
                "sentence_transformers.SentenceTransformer",
                _FakeSentenceTransformer,
            ):
                first = GloveTextEmbedder(
                    execution="official",
                    shared_cache=shared,
                    disk_cache_path=path,
                    disk_cache_namespace="integration",
                    disk_cache_mode="on",
                )
                with first.epoch_context(0):
                    expected = first(["alpha", "beta", "alpha"])
                self.assertEqual(_FakeSentenceTransformer.encode_calls, 2)
                first.close()

                second = GloveTextEmbedder(
                    execution="official",
                    shared_cache=shared,
                    disk_cache_path=path,
                    disk_cache_namespace="integration",
                    disk_cache_mode="on",
                )
                with second.epoch_context(1):
                    actual = second(["alpha", "beta", "alpha"])
                self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(second.stats.model_inputs, 0)
                self.assertEqual(second.disk_cache_stats["hits"], 2)
                self.assertEqual(_FakeSentenceTransformer.encode_calls, 2)
                second.close()


if __name__ == "__main__":
    unittest.main()
