from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, cast

from relconnector_v2.features.process_worker import SqlAssemblyWorker


class _Fetcher:
    def __init__(self) -> None:
        self.epochs: list[list[int]] = []

    def fetch_many(self, plans: list[Any]) -> list[Any]:
        epochs = [plan.key.epoch for plan in plans]
        if len(set(epochs)) != 1:
            raise ValueError("mixed epoch window")
        self.epochs.append(epochs)
        return plans


class _Assembler:
    def assemble_many(self, features: list[Any]) -> list[Any]:
        return features


class ProcessWorkerTest(unittest.TestCase):
    def test_cross_epoch_window_is_split_before_fetch(self) -> None:
        worker = cast(SqlAssemblyWorker, object.__new__(SqlAssemblyWorker))
        fetcher = _Fetcher()
        worker.fetcher = cast(Any, fetcher)
        worker.assembler = cast(Any, _Assembler())
        plans = [
            SimpleNamespace(key=SimpleNamespace(epoch=epoch, batch=batch))
            for epoch, batch in ((0, 0), (0, 1), (1, 0), (1, 1))
        ]

        prepared = worker.assemble_many(cast(Any, plans))

        self.assertEqual(prepared, plans)
        self.assertEqual(fetcher.epochs, [[0, 0], [1, 1]])


if __name__ == "__main__":
    unittest.main()
