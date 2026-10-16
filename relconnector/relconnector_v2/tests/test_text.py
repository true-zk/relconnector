from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from relconnector_v2.features.text import GloveTextEmbedder
from relconnector_v2.observability import OperationMetrics


class _WordEmbeddings(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb_layer = torch.nn.Embedding(2, 300)
        self.calls: list[list[str]] = []

    def tokenize(self, sentences: list[str]) -> dict[str, object]:
        self.calls.append(list(sentences))
        return {"lengths": torch.tensor([len(value) for value in sentences])}

    def forward(self, features: dict[str, object]) -> dict[str, object]:
        return features


class _MeanPooling(torch.nn.Module):
    def get_pooling_mode_str(self) -> str:
        return "mean"

    def forward(self, features: dict[str, object]) -> dict[str, object]:
        lengths = features["lengths"]
        assert isinstance(lengths, torch.Tensor)
        features["sentence_embedding"] = lengths[:, None].expand(-1, 300).float()
        return features


class _CurrentSentenceTransformer(torch.nn.Module):
    word = _WordEmbeddings()
    pool = _MeanPooling()

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        super().__init__()

    def __iter__(self):
        return iter((self.word, self.pool))

    def encode(self, *_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("direct execution unexpectedly fell back to encode()")


class GloveTextEmbedderTest(unittest.TestCase):
    def test_direct_supports_current_tokenize_api_and_restores_row_order(self) -> None:
        metrics = OperationMetrics(True)
        _CurrentSentenceTransformer.word.calls.clear()
        with patch(
            "sentence_transformers.SentenceTransformer",
            _CurrentSentenceTransformer,
        ):
            embedder = GloveTextEmbedder(execution="direct", metrics=metrics)
            values = ["a", "the longest sentence", "medium"]
            encoded = embedder(values)
            embedder.close()

        self.assertEqual(encoded.shape, (3, 300))
        self.assertTrue(
            torch.equal(encoded[:, 0], torch.tensor([1.0, 20.0, 6.0]))
        )
        self.assertEqual(
            _CurrentSentenceTransformer.word.calls,
            [["the longest sentence", "medium", "a"]],
        )
        operations = {
            row["operation"] for row in metrics.report()["timers"]
        }
        self.assertIn("text_tokenize", operations)
        self.assertIn("text_lookup", operations)
        self.assertIn("text_pool", operations)
        self.assertNotIn("text_model", operations)


if __name__ == "__main__":
    unittest.main()
