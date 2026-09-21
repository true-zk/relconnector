from __future__ import annotations

import unittest
from typing import cast

import torch
import torch.multiprocessing as mp
from torch_frame import TensorFrame, numerical

from relconnector_v2.features.shared_encoded import (
    SharedEncodedFeatureCache,
    create_shared_encoded_cache,
)


def _publish(cache: SharedEncodedFeatureCache) -> None:
    node_ids = torch.tensor([3, 7], dtype=torch.long)
    frame = TensorFrame(
        feat_dict={numerical: torch.tensor([[3.0], [7.0]])},
        col_names_dict={numerical: ["value"]},
    )
    cache.put("nodes", node_ids, frame, epoch=0)


class SharedEncodedFeatureCacheTest(unittest.TestCase):
    def test_tensorframe_rows_are_shared_across_processes(self) -> None:
        context = mp.get_context("spawn")
        cache, resource = create_shared_encoded_cache(
            context,
            max_bytes=1024 * 1024,
            admission="always",
        )
        try:
            process = context.Process(target=_publish, args=(cache,))
            process.start()
            process.join(timeout=30)
            self.assertEqual(process.exitcode, 0)

            groups, misses = cache.lookup_many(
                "nodes",
                torch.tensor([7, 9, 3]),
                epoch=1,
            )
            self.assertEqual(misses, [1])
            values: dict[int, float] = {}
            for entry, positions in groups:
                feature = cast(torch.Tensor, entry.frame.feat_dict[numerical])
                for position, row in positions:
                    values[position] = float(feature[row, 0])
            self.assertEqual(values, {0: 7.0, 2: 3.0})
            snapshot = cache.snapshot()
            self.assertEqual(snapshot["hit_rows"], 2)
            self.assertEqual(snapshot["miss_rows"], 1)
        finally:
            resource.close()


if __name__ == "__main__":
    unittest.main()
