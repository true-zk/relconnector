from __future__ import annotations

import unittest

from relconnector_v2.features.adaptive_text_index import AdaptiveTextIndex


class AdaptiveTextIndexTest(unittest.TestCase):
    def make_index(self) -> AdaptiveTextIndex:
        return AdaptiveTextIndex(
            1024, window_calls=2, min_requests=20, min_lookup_s=0.01
        )

    def window(
        self,
        index: AdaptiveTextIndex,
        *,
        shared: int,
        local: int = 0,
        duration: float = 0.01,
    ) -> None:
        for _ in range(2):
            index.observe(
                requests=10,
                shared_hits=shared,
                local_hits=local,
                lookup_s=duration,
            )

    def test_requires_both_reuse_and_measured_lookup_cost(self) -> None:
        index = self.make_index()
        self.window(index, shared=1)
        self.assertFalse(index.active)
        self.window(index, shared=10, duration=0.00001)
        self.assertFalse(index.active)
        self.window(index, shared=9)
        self.assertTrue(index.active)

    def test_releases_index_when_reuse_disappears_and_can_recover(self) -> None:
        index = self.make_index()
        self.window(index, shared=10)
        index.remember("reused", 3)
        for _ in range(4):  # Two fill windows, then two low-reuse windows.
            self.window(index, shared=10)
        self.assertFalse(index.active)
        self.assertEqual(index.current_bytes, 0)
        self.assertEqual(len(index.entries), 0)
        for _ in range(4):
            self.window(index, shared=10)
            self.assertFalse(index.active)
        self.window(index, shared=10)
        self.assertTrue(index.active)
        self.assertEqual(index.activations, 2)

    def test_metadata_budget_evicts_and_rejects_oversized_text(self) -> None:
        index = self.make_index()
        self.window(index, shared=10)
        for n in range(20):
            index.remember(str(n), n)
        self.assertLessEqual(index.current_bytes, index.max_bytes)
        self.assertGreater(index.evictions, 0)
        self.assertIsNone(index.get("0"))
        self.assertEqual(index.get("19"), 19)
        index.remember("x" * 2048, 100)
        self.assertNotIn("x" * 2048, index.entries)


if __name__ == "__main__":
    unittest.main()
