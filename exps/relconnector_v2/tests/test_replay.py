from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from relconnector_v2.replay import ReplaySampler, ReplaySeeds, materialize, validate
from relconnector_v2.sampling import (
    EntitySamplePlan,
    RecommendationSamplePlan,
    SampledSubgraph,
)
from relconnector_v2.task import BatchKey, EntitySeedBatch, RecommendationSeedBatch


class Seeds:
    def iter_epoch(self, epoch):
        yield EntitySeedBatch(
            BatchKey(epoch, 0),
            "a",
            torch.tensor([2]),
            torch.tensor([10]),
            torch.tensor([1.0]),
        )
        yield RecommendationSeedBatch(
            BatchKey(epoch, 1),
            "a",
            "b",
            torch.tensor([2]),
            torch.tensor([3]),
            torch.tensor([10]),
        )


class Sampler:
    def sample(self, seeds):
        graph = SampledSubgraph(
            {"a": torch.tensor([2, 4])},
            {("a", "r", "a"): torch.tensor([[0], [1]])},
            None,
            {},
            {},
            {},
            "a",
            1,
            torch.tensor([10]),
        )
        if isinstance(seeds, EntitySeedBatch):
            return EntitySamplePlan(seeds.key, graph, seeds.target)
        negative = SampledSubgraph(
            {"b": torch.tensor([5])}, {}, None, {}, {}, {}, "b", 1, None
        )
        return RecommendationSamplePlan(seeds.key, graph, graph, negative)


class ReplayTest(unittest.TestCase):
    def test_same_inputs_edges_targets_and_negative_samples_across_epochs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "plans"
            manifest = materialize(root, Seeds(), Sampler(), {"identity": "test"})
            validate(root, {"identity": "test"})
            source = ReplaySeeds(root, manifest["batches"])
            sampler = ReplaySampler(root)
            first = [sampler.sample(seed) for seed in source.iter_epoch(0)]
            last = [sampler.sample(seed) for seed in source.iter_epoch(4)]
            self.assertEqual(last[0].key, BatchKey(4, 0))
            self.assertTrue(torch.equal(first[0].target, last[0].target))
            self.assertTrue(
                torch.equal(
                    first[0].subgraph.edge_index[("a", "r", "a")],
                    last[0].subgraph.edge_index[("a", "r", "a")],
                )
            )
            self.assertTrue(
                torch.equal(
                    first[1].negative.node_ids["b"], last[1].negative.node_ids["b"]
                )
            )
            self.assertEqual(len(first), 2)

    def test_identity_and_content_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "plans"
            materialize(root, Seeds(), Sampler(), {"identity": "test"})
            with self.assertRaises(ValueError):
                validate(root, {"identity": "different database"})
            (root / "batch-000000.pt").write_bytes(b"bad")
            with self.assertRaises(ValueError):
                validate(root, {"identity": "test"})


if __name__ == "__main__":
    unittest.main()
