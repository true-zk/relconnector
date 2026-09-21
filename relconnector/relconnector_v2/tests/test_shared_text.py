from __future__ import annotations

import unittest

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


class SharedTextEmbeddingCacheTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
