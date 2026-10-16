from __future__ import annotations

import pickle
import unittest
from multiprocessing.reduction import ForkingPickler
from typing import cast

import torch
import torch.multiprocessing as mp
from torch_frame import TensorFrame, numerical, text_embedded, timestamp
from torch_frame.data import MultiEmbeddingTensor

from relconnector_v2.features.shared_encoded import (
    SharedEncodedFeatureCache,
    _CacheOwner,
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
    def test_rpc_serialization_cannot_relocate_published_storage(self) -> None:
        owner = _CacheOwner(1024 * 1024, "always")
        ids = torch.tensor([3, 7])
        times = torch.arange(14).reshape(2, 1, 7)
        embedding = MultiEmbeddingTensor.from_tensor_list([torch.randn(2, 4)])
        frame = TensorFrame(
            {timestamp: times, text_embedded: embedding},
            {timestamp: ["created_at"], text_embedded: ["text"]},
            y=torch.tensor([1.0, 2.0]),
        )
        entry = owner.put("nodes", ids, frame, None)
        assert entry is not None
        cached_embedding = cast(
            MultiEmbeddingTensor, entry.frame.feat_dict[text_embedded]
        )
        tensors = [
            entry.node_ids,
            cast(torch.Tensor, entry.frame.feat_dict[timestamp]),
            cached_embedding.values,
            cached_embedding.offset,
            cast(torch.Tensor, entry.frame.y),
        ]
        # A reader may already hold any of these native data pointers when
        # another manager thread serializes the put response.
        pointers = [tensor.data_ptr() for tensor in tensors]
        self.assertTrue(all(tensor.is_shared() for tensor in tensors))
        received = pickle.loads(ForkingPickler.dumps(entry))
        self.assertEqual(pointers, [tensor.data_ptr() for tensor in tensors])
        torch.testing.assert_close(received.frame.feat_dict[timestamp], times)
        groups, misses = owner.lookup_many("nodes", ids.flip(0), None)
        self.assertEqual(misses, [])
        torch.testing.assert_close(
            groups[0][0].frame.feat_dict[timestamp], times.flip(0)
        )

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
            self.assertEqual(len(groups), 1)
            self.assertEqual(len(groups[0][0].frame), 2)
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

    def test_lookup_projects_rows_from_multiple_chunks(self) -> None:
        context = mp.get_context("spawn")
        cache, resource = create_shared_encoded_cache(
            context,
            max_bytes=1024 * 1024,
            admission="always",
        )
        try:
            for values in ([1, 2, 3], [4, 5, 6]):
                ids = torch.tensor(values, dtype=torch.long)
                frame = TensorFrame(
                    feat_dict={numerical: ids.float().unsqueeze(1)},
                    col_names_dict={numerical: ["value"]},
                )
                cache.put("nodes", ids, frame, epoch=0)

            groups, misses = cache.lookup_many(
                "nodes", torch.tensor([5, 9, 2]), epoch=1
            )
            self.assertEqual(misses, [1])
            self.assertEqual(len(groups), 1)
            entry, locations = groups[0]
            self.assertEqual(len(entry.frame), 2)
            feature = cast(torch.Tensor, entry.frame.feat_dict[numerical])
            actual = {position: float(feature[row, 0]) for position, row in locations}
            self.assertEqual(actual, {0: 5.0, 2: 2.0})
        finally:
            resource.close()


if __name__ == "__main__":
    unittest.main()
