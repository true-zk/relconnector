from __future__ import annotations

import unittest

import torch
from torch_frame import TensorFrame, numerical

from relconnector_v2.features.cache import EncodedFeatureCache


class EncodedFeatureCacheTest(unittest.TestCase):
    def test_eviction_preserves_index_for_reinserted_table(self) -> None:
        cache = EncodedFeatureCache(1024, admission="always")
        for start in (0, 8, 16):
            ids = torch.arange(start, start + 8)
            frame = TensorFrame(
                feat_dict={numerical: ids.float().unsqueeze(1)},
                col_names_dict={numerical: ["value"]},
            )
            self.assertIsNotNone(cache.put("nodes", ids, frame))
            groups, misses = cache.lookup_many("nodes", ids)
            self.assertEqual(misses, [])
            self.assertEqual(sum(len(locations) for _, locations in groups), 8)
        self.assertGreater(cache.evictions, 0)


if __name__ == "__main__":
    unittest.main()
