from __future__ import annotations

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pandas as pd
import torch
from torch_frame import (
    TensorFrame,
    categorical,
    multicategorical,
    numerical,
    text_embedded,
    timestamp,
)
from torch_frame.data import MultiEmbeddingTensor, MultiNestedTensor
from torch_frame.data.multi_tensor import _MultiTensor

from relconnector_v2.features.assembler import TensorFrameBatchAssembler
from relconnector_v2.features.cache import EncodedFeatureCache
from relconnector_v2.features.rows import gather_rows


def assert_value_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, _MultiTensor):
        assert type(actual) is type(expected)
        torch.testing.assert_close(actual.values, expected.values, rtol=0, atol=0)
        torch.testing.assert_close(actual.offset, expected.offset, rtol=0, atol=0)
        assert actual.shape == expected.shape
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_value_equal(actual[key], expected[key])
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


class RowGatherTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        numbers = torch.randn(8, 4)[:, ::2]  # Deliberately non-contiguous.
        numbers[1, 0] = float("nan")
        self.frame = TensorFrame(
            {
                numerical: numbers,
                categorical: torch.tensor([[i, -1] for i in range(8)]),
                timestamp: torch.arange(56).reshape(8, 1, 7),
                text_embedded: MultiEmbeddingTensor.from_tensor_list(
                    [torch.randn(8, 3), torch.randn(8, 5)]
                ),
                multicategorical: MultiNestedTensor.from_tensor_mat(
                    [[torch.arange(i % 3)] for i in range(8)]
                ),
            },
            {
                numerical: ["n1", "n2"],
                categorical: ["c1", "c2"],
                timestamp: ["date"],
                text_embedded: ["title", "body"],
                multicategorical: ["tags"],
            },
            y=torch.arange(8),
        )

    def test_matches_upstream_for_repeated_shuffled_empty_and_fallback_indices(self):
        for index in (
            torch.tensor([7, 1, 7, 0, 2]),
            torch.tensor([], dtype=torch.long),
            torch.arange(8),
            torch.tensor([-1, 1, -2]),
            torch.tensor([True, False, True, False, True, False, True, False]),
        ):
            with self.subTest(index=index):
                expected = self.frame[index]
                actual = gather_rows(self.frame, index)
                self.assertEqual(actual.num_rows, expected.num_rows)
                self.assertEqual(actual.col_names_dict, expected.col_names_dict)
                assert_value_equal(actual.feat_dict, expected.feat_dict)
                assert_value_equal(actual.y, expected.y)

    def test_gathered_values_do_not_alias_source(self):
        result = gather_rows(self.frame, torch.arange(8))
        result.feat_dict[numerical][0, 0] = 999
        result.feat_dict[text_embedded].values[0, 0] = 999
        result.feat_dict[multicategorical].values.fill_(999)
        result.y[0] = 999
        self.assertNotEqual(float(self.frame.feat_dict[numerical][0, 0]), 999)
        self.assertNotEqual(
            float(self.frame.feat_dict[text_embedded].values[0, 0]), 999
        )
        self.assertFalse(
            bool((self.frame.feat_dict[multicategorical].values == 999).any())
        )
        self.assertEqual(int(self.frame.y[0]), 0)

    def test_featureless_frame_and_invalid_indices(self):
        frame = TensorFrame({}, {}, num_rows=8)
        self.assertEqual(gather_rows(frame, torch.tensor([3, 1, 3])).num_rows, 3)
        # Upstream allows any indices into a featureless (rows, 0) frame.
        self.assertEqual(
            gather_rows(frame, torch.tensor([8])).num_rows,
            frame[torch.tensor([8])].num_rows,
        )
        with self.assertRaises((IndexError, RuntimeError)):
            gather_rows(self.frame, torch.tensor([8]))

    def test_dictionary_feature_storage_matches_upstream(self):
        frame = TensorFrame(
            {numerical: {"values": torch.randn(8, 2), "mask": torch.ones(8, 2)}},
            {numerical: ["n1", "n2"]},
        )
        index = torch.tensor([2, 0, 2])
        assert_value_equal(gather_rows(frame, index).feat_dict, frame[index].feat_dict)

    def test_prefix_cache_hit_returns_requested_rows_only(self):
        cache = EncodedFeatureCache(1024 * 1024, admission="always")
        cache.put("nodes", torch.arange(8), self.frame)
        encoder = SimpleNamespace(
            metrics=SimpleNamespace(
                timer=lambda *a, **kw: nullcontext(), add=lambda *a, **kw: None
            )
        )
        assembler = TensorFrameBatchAssembler(encoder, encoded_cache=cache)
        result = assembler._encode_cached(
            "nodes", torch.tensor([0, 1, 2]), pd.DataFrame(index=range(3)), 0
        )
        self.assertEqual(len(result), 3)
        assert_value_equal(result.feat_dict, self.frame[torch.arange(3)].feat_dict)


if __name__ == "__main__":
    unittest.main()
