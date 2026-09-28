from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

from relconnector_v2.features import SharedTextEmbeddingCache


def _publish(cache: SharedTextEmbeddingCache) -> None:
    lookup = cache.lookup_or_reserve(["alpha", "beta"])
    cache.publish(
        lookup.reservations,
        {
            position: torch.full((4,), float(position + 1))
            for position in lookup.reservations
        },
    )


def _check_adaptive_child(cache: SharedTextEmbeddingCache) -> None:
    assert cache._worker_index().activations == 0
    lookup = cache.lookup_or_reserve(["alpha"])
    assert not lookup.reservations
    assert torch.equal(lookup.hits[0], torch.ones(4))


def _concurrent_roundtrip(cache: SharedTextEmbeddingCache, start) -> None:
    torch.set_num_threads(1)
    start.wait(timeout=20)
    for offset in range(30):
        keys = [str(value) for value in range(offset, offset + 16)]
        lookup = cache.lookup_or_reserve(keys)
        assert not lookup.uncached
        cache.publish(
            lookup.reservations,
            {
                position: torch.full((4,), float(keys[position]))
                for position in lookup.reservations
            },
        )
        hits = {**lookup.hits, **cache.wait(dict(lookup.pending))}
        for position, value in hits.items():
            assert torch.equal(value, torch.full((4,), float(keys[position])))


class SharedTextEmbeddingCacheTest(unittest.TestCase):
    def test_busy_process_lock_has_a_deadline(self) -> None:
        cache = SharedTextEmbeddingCache(
            mp.get_context("spawn"),
            max_bytes=16 * 16,
            embedding_dim=4,
            wait_timeout_s=0.05,
        )
        acquired = threading.Event()
        release = threading.Event()

        def hold_lock():
            with cache._condition:
                acquired.set()
                release.wait(timeout=5)

        thread = threading.Thread(target=hold_lock)
        thread.start()
        try:
            self.assertTrue(acquired.wait(timeout=5))
            with self.assertRaisesRegex(
                TimeoutError, "acquiring shared text cache lock"
            ):
                cache.lookup_or_reserve(["alpha"])
        finally:
            release.set()
            thread.join(timeout=5)

    def test_concurrent_publication_and_wait_are_consistent(self) -> None:
        context = mp.get_context("spawn")
        cache = SharedTextEmbeddingCache(
            context,
            max_bytes=16 * 256,
            embedding_dim=4,
            wait_timeout_s=20,
        )
        # Force parent-side views to exist before pickling for spawn.
        cache._metadata()
        start = context.Barrier(3)
        processes = [
            context.Process(target=_concurrent_roundtrip, args=(cache, start))
            for _ in range(3)
        ]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=40)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual(cache.snapshot()["published"], 45)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)

    def test_shared_hits_gather_outside_lock_and_do_not_alias_slab(self) -> None:
        cache = SharedTextEmbeddingCache(
            mp.get_context("spawn"),
            max_bytes=16 * 16,
            embedding_dim=4,
        )
        lookup = cache.lookup_or_reserve(["alpha"])
        cache.publish(lookup.reservations, {0: torch.ones(4)})
        gather = cache._gather

        def checked_gather(ready):
            self.assertFalse(cache._lock._semlock._is_mine())
            return gather(ready)

        with patch.object(cache, "_gather", side_effect=checked_gather):
            hits = cache.lookup_or_reserve(["alpha", "alpha"]).hits
            hits[0].zero_()
            self.assertTrue(torch.equal(hits[1], torch.ones(4)))
            self.assertTrue(
                torch.equal(cache.lookup_or_reserve(["alpha"]).hits[0], torch.ones(4))
            )

    def adaptive_cache(self) -> SharedTextEmbeddingCache:
        cache = SharedTextEmbeddingCache(
            mp.get_context("spawn"),
            max_bytes=4 * 4 * 16,
            embedding_dim=4,
            adaptive_local_index=True,
        )
        index = cache._worker_index()
        index.window_calls = index.min_requests = 1
        index.min_lookup_s = 0.0
        lookup = cache.lookup_or_reserve(["alpha"])
        cache.publish(lookup.reservations, {0: torch.ones(4)})
        cache.lookup_or_reserve(["alpha"])
        self.assertTrue(index.active)
        cache.lookup_or_reserve(["alpha"])
        return cache

    def test_adaptive_hits_bypass_shared_lock_and_return_safe_copies(self) -> None:
        cache = self.adaptive_cache()
        with patch.object(cache, "_condition") as condition:
            condition.acquire.side_effect = AssertionError("shared lock used")
            with patch.object(cache, "_find_slot", side_effect=AssertionError):
                hits = cache.lookup_or_reserve(["alpha", "alpha"]).hits
                cache.wait({})
                hits[0].zero_()
                self.assertTrue(torch.equal(hits[1], torch.ones(4)))
                self.assertTrue(
                    torch.equal(
                        cache.lookup_or_reserve(["alpha"]).hits[0], torch.ones(4)
                    )
                )
        stats = cache.snapshot()
        self.assertEqual(stats["hits"], 5)
        self.assertEqual(stats["local_index"]["fast_hits"], 3)

    def test_adaptive_misses_still_coordinate_shared_reservations(self) -> None:
        cache = self.adaptive_cache()
        owner = cache.lookup_or_reserve(["alpha", "beta"])
        self.assertEqual(set(owner.hits), {0})
        self.assertEqual(set(owner.reservations), {1})
        waiter = cache.lookup_or_reserve(["beta"])
        self.assertEqual(waiter.reservations, {})
        self.assertEqual(set(waiter.pending), {0})
        self.assertNotIn("beta", cache._worker_index().entries)
        cache.publish(owner.reservations, {1: torch.full((4,), 2.0)})
        self.assertTrue(
            torch.equal(cache.wait(dict(waiter.pending))[0], torch.full((4,), 2.0))
        )

    def test_adaptive_index_is_process_local_under_spawn(self) -> None:
        cache = self.adaptive_cache()
        process = mp.get_context("spawn").Process(
            target=_check_adaptive_child, args=(cache,)
        )
        process.start()
        process.join(timeout=30)
        self.assertEqual(process.exitcode, 0)
        self.assertEqual(cache.snapshot()["entries"], 1)

    def test_adaptive_full_cache_preserves_uncached_fallback(self) -> None:
        cache = SharedTextEmbeddingCache(
            mp.get_context("spawn"),
            max_bytes=16,
            embedding_dim=4,
            adaptive_local_index=True,
        )
        first = cache.lookup_or_reserve(["alpha"])
        cache.publish(first.reservations, {0: torch.ones(4)})
        second = cache.lookup_or_reserve(["alpha", "beta"])
        self.assertEqual(second.uncached, [1])
        self.assertFalse(second.reservations)
        self.assertTrue(torch.equal(second.hits[0], torch.ones(4)))

    def test_pending_lookup_waits_after_owner_publishes(self) -> None:
        context = mp.get_context("spawn")
        cache = SharedTextEmbeddingCache(
            context,
            max_bytes=4 * 4 * 4,
            embedding_dim=4,
        )
        owner = cache.lookup_or_reserve(["shared"])
        waiter = cache.lookup_or_reserve(["shared"])
        self.assertEqual(set(owner.reservations), {0})
        self.assertEqual(set(waiter.pending), {0})

        cache.publish(owner.reservations, {0: torch.full((4,), 3.0)})
        hits = cache.wait(dict(waiter.pending))
        self.assertTrue(torch.equal(hits[0], torch.full((4,), 3.0)))

    def test_reservations_are_visible_across_spawned_processes(self) -> None:
        context = mp.get_context("spawn")
        cache = SharedTextEmbeddingCache(
            context,
            max_bytes=4 * 4 * 8,
            embedding_dim=4,
        )
        process = context.Process(target=_publish, args=(cache,))
        process.start()
        process.join(timeout=30)

        self.assertEqual(process.exitcode, 0)
        lookup = cache.lookup_or_reserve(["beta", "alpha"])
        self.assertEqual(lookup.reservations, {})
        self.assertEqual(lookup.uncached, [])
        self.assertTrue(torch.equal(lookup.hits[0], torch.full((4,), 2.0)))
        self.assertTrue(torch.equal(lookup.hits[1], torch.full((4,), 1.0)))
        stats = cache.snapshot()
        self.assertEqual(stats["entries"], 2)
        self.assertEqual(stats["hits"], 2)

    def test_full_cache_rejects_new_values_with_bounded_probe(self) -> None:
        context = mp.get_context("spawn")
        cache = SharedTextEmbeddingCache(
            context,
            max_bytes=4 * 4 * 2,
            embedding_dim=4,
            max_probe=2,
        )
        first = cache.lookup_or_reserve(["alpha", "beta"])
        cache.publish(
            first.reservations,
            {
                position: torch.full((4,), float(position))
                for position in first.reservations
            },
        )
        second = cache.lookup_or_reserve(["gamma"])

        self.assertEqual(second.uncached, [0])
        self.assertEqual(cache.snapshot()["max_probe"], 2)


if __name__ == "__main__":
    unittest.main()
