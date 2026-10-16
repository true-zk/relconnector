from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import torch

from relconnector_v2.features.cost_cache import CostAwareTextCache, ReuseGate
from relconnector_v2.features.text import GloveTextEmbedder


class FakeTextModel:
    def __init__(self, *args, **kwargs):
        pass

    def eval(self):
        pass

    def encode(self, sentences, **kwargs):
        return torch.stack([torch.full((300,), float(len(s))) for s in sentences])


def encoder(values: list[str]) -> tuple[torch.Tensor, float]:
    return torch.tensor([[float(len(v)), float(sum(v.encode()))] for v in values]), 1.0


def cache(
    directory: Path | None = None,
    ram: int = 10000,
    *,
    disk_policy: Literal["budget", "roi"] = "budget",
) -> CostAwareTextCache:
    return CostAwareTextCache(
        max_bytes=ram,
        dimension=2,
        disk_dir=directory,
        disk_max_bytes=1000000,
        remaining_epochs=5,
        disk_policy=disk_policy,
    )


class CostCacheTest(unittest.TestCase):
    def test_roi_policy_writes_only_when_future_reads_repay_it(self):
        with tempfile.TemporaryDirectory() as temp:
            slow = cache(Path(temp) / "slow", ram=0, disk_policy="roi")
            slow.resolve(["long text"] * 100, column="x", epoch=0, encode=encoder)
            self.assertEqual(slow.stats["disk_write_blocks"], 1)
            self.assertGreater(slow.stats["disk_predicted_savings_s"], 0)

            fast = cache(Path(temp) / "fast", ram=0, disk_policy="roi")
            fast.resolve(
                ["short"] * 100,
                column="x",
                epoch=0,
                encode=lambda values: (encoder(values)[0], 0.00001),
            )
            self.assertEqual(fast.stats["disk_write_blocks"], 0)
            self.assertEqual(fast.stats["disk_roi_rejections"], 1)

    def test_roi_policy_skips_cold_epoch_disk_miss_probes(self):
        with tempfile.TemporaryDirectory() as temp:
            c = cache(Path(temp) / "blocks", ram=0, disk_policy="roi")
            c.resolve(["first"], column="x", epoch=0, encode=encoder)
            c.resolve(["different"], column="x", epoch=0, encode=encoder)
            self.assertEqual(c.stats["disk_miss_blocks"], 0)

    def test_roi_throughput_uses_all_successful_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            c = cache(Path(temp) / "blocks", ram=0, disk_policy="roi")
            for i in range(8):
                c.resolve([str(i)] * 100, column="x", epoch=0, encode=encoder)
            self.assertGreater(c.disk_written_bytes, 0)
            self.assertGreater(c.disk_write_s, 0)
            self.assertAlmostEqual(
                c.write_bytes_per_s,
                c._write_rate_bytes / c._write_rate_s,
            )

    def test_integration_preserves_normalization_expansion_and_epoch_accounting(self):
        with patch("sentence_transformers.SentenceTransformer", FakeTextModel):
            baseline = GloveTextEmbedder(execution="official")
            cached = GloveTextEmbedder(
                execution="official",
                cost_strategy=True,
                cache_bytes=10000,
                training_epochs=3,
            )
            values = ["alpha", "b", "alpha", None]
            expected = baseline(values)
            with cached.epoch_context(0):
                first = cached.for_column("t", "c")(values)
            with cached.epoch_context(1):
                second = cached.for_column("t", "c")(values)
            self.assertTrue(torch.equal(expected, first))
            self.assertTrue(torch.equal(expected, second))
            self.assertEqual(cached.epoch_stats[1].get("model_inputs", 0), 0)
            self.assertEqual(cached.epoch_stats[1]["cache_hits"], 3)
            baseline.close()
            cached.close()

    def test_disk_misses_back_off_and_new_epoch_reprobes(self):
        with tempfile.TemporaryDirectory() as temp:
            c = cache(Path(temp), ram=0)
            c.remaining_epochs = 1
            for i in range(17):
                c.resolve([str(i)], column="x", epoch=0, encode=encoder)
            self.assertEqual(c.stats["disk_miss_blocks"], 16)
            for i in range(20):
                c.resolve([str(i + 17)], column="x", epoch=0, encode=encoder)
            self.assertEqual(c.stats["disk_miss_blocks"], 16)
            c.resolve(["next"], column="x", epoch=1, encode=encoder)
            self.assertEqual(c.stats["disk_miss_blocks"], 17)

    def test_gate_bypasses_losing_work_then_reprobes(self):
        gate = ReuseGate()
        for _ in range(16):
            gate.observe(0, 0.001)
        self.assertFalse(gate.probe())
        for _ in range(63):
            self.assertFalse(gate.probe())
        self.assertTrue(gate.probe())
        for _ in range(16):
            gate.observe(0.1, 0.001)
        self.assertTrue(gate.probe())

    def test_ram_hits_preserve_order_and_own_storage(self):
        c = cache()
        first, _ = c.resolve(["cat", "b"], column="x", epoch=0, encode=encoder)
        first.fill_(1234)
        second, hits = c.resolve(["b", "cat"], column="x", epoch=0, encode=encoder)
        self.assertEqual(hits, 2)
        self.assertTrue(torch.equal(second, encoder(["b", "cat"])[0]))
        self.assertLessEqual(c.bytes, c.max_bytes)

    def test_mixed_hits_and_eviction(self):
        c = cache(ram=650)
        c.resolve(["cat", "b"], column="x", epoch=0, encode=encoder)
        result, _ = c.resolve(
            ["cat", "longer", "z"], column="x", epoch=0, encode=encoder
        )
        self.assertTrue(torch.equal(result, encoder(["cat", "longer", "z"])[0]))
        self.assertLessEqual(c.bytes, 650)
        self.assertGreater(c.stats["ram_evictions"], 0)

    def test_disk_survives_new_worker_and_epoch_without_encoding(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "blocks"
            writer = cache(directory, ram=0)
            writer.resolve(["cat", "b"], column="x", epoch=0, encode=encoder)
            self.assertEqual(writer.stats["disk_write_blocks"], 1)
            reader = cache(directory, ram=0)

            def fail(_):
                raise AssertionError("disk hit must avoid encoding")

            result, hits = reader.resolve(
                ["cat", "b"], column="x", epoch=2, encode=fail
            )
            self.assertEqual(hits, 2)
            self.assertTrue(torch.equal(result, encoder(["cat", "b"])[0]))

    def test_cheap_or_last_epoch_does_not_write(self):
        with tempfile.TemporaryDirectory() as temp:
            c = cache(Path(temp), ram=0)
            c.resolve(
                ["a"], column="x", epoch=0, encode=lambda v: (encoder(v)[0], 0.00001)
            )
            c.resolve(["b"], column="x", epoch=4, encode=encoder)
            self.assertEqual(list(Path(temp).rglob("*.bin")), [])

    def test_corrupt_disk_falls_back_to_correct_encoding(self):
        with tempfile.TemporaryDirectory() as temp:
            c = cache(Path(temp), ram=0)
            c.resolve(["cat"], column="x", epoch=0, encode=encoder)
            next(Path(temp).rglob("*.bin")).write_bytes(b"truncated")
            result, hits = cache(Path(temp), ram=0).resolve(
                ["cat"], column="x", epoch=1, encode=encoder
            )
            self.assertEqual(hits, 0)
            self.assertTrue(torch.equal(result, encoder(["cat"])[0]))

    def test_concurrent_publication_is_atomic(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)

            def publish(_):
                c = cache(directory, ram=0)
                return c.resolve(
                    ["hello", "world"], column="x", epoch=0, encode=encoder
                )[0]

            with ThreadPoolExecutor(4) as pool:
                results = list(pool.map(publish, range(8)))
            for result in results:
                self.assertTrue(torch.equal(result, encoder(["hello", "world"])[0]))
            self.assertEqual(len(list(directory.rglob("*.bin"))), 1)
            self.assertEqual(list(directory.rglob("*.tmp")), [])

    def test_key_has_length_delimiters_and_order(self):
        self.assertNotEqual(cache()._key(["ab", "c"]), cache()._key(["a", "bc"]))
        self.assertNotEqual(cache()._key(["a", "b"]), cache()._key(["b", "a"]))


if __name__ == "__main__":
    unittest.main()
