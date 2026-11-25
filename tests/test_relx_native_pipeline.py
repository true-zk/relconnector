"""Native feature collection and bounded sampling-to-feature pipeline."""

from __future__ import annotations

import importlib.util
import unittest
from threading import Event, Thread

import numpy as np


@unittest.skipUnless(
    importlib.util.find_spec("relx._relx_native"), "native extension needed"
)
class NativeFeatureWorkersTest(unittest.TestCase):
    def test_six_default_feature_types_match_across_workers(self) -> None:
        from relx._relx_native import NativeFeatureStore

        store = NativeFeatureStore()
        store.add_numeric(
            "people", "amount", np.array([10.0, np.nan, 30.0]), 20.0, 10.0
        )
        store.add_categorical("people", "group", np.array([1, 0, 3]))
        store.add_timestamp(
            "people", "seen_at",
            np.array([100, np.iinfo(np.int64).min, 300], dtype=np.int64),
            200.0, 100.0,
        )
        store.add_multicategorical(
            "people", "tags", np.array([2, 5, 7], dtype=np.int64),
            np.array([0, 2, 2, 3], dtype=np.int64),
        )
        store.add_embedding(
            "people", "vector",
            np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        )
        store.add_text("people", "note", ["hello", "", "world"])
        indices = {"people": np.array([2, 0, 1, 2], dtype=np.int64)}
        serial = store.gather(indices, 1)
        threaded = store.gather(indices, 4)
        for name in ("amount", "group", "seen_at", "tags", "vector", "note"):
            np.testing.assert_array_equal(
                serial["people"][name]["values"],
                threaded["people"][name]["values"],
            )
        amounts = serial["people"]["amount"]["values"]
        assert isinstance(amounts, np.ndarray)
        np.testing.assert_array_equal(amounts.reshape(-1), [1, -1, 0, 1])
        np.testing.assert_array_equal(
            serial["people"]["amount"]["missing"], [False, False, True, False]
        )
        np.testing.assert_array_equal(
            serial["people"]["group"]["values"], [3, 1, 0, 3]
        )
        times = serial["people"]["seen_at"]["values"]
        assert isinstance(times, np.ndarray)
        np.testing.assert_array_equal(times.reshape(-1), [1, -1, 0, 1])
        np.testing.assert_array_equal(
            serial["people"]["tags"]["values"], [7, 2, 5, 7]
        )
        np.testing.assert_array_equal(
            serial["people"]["tags"]["offsets"], [0, 1, 3, 3, 4]
        )
        np.testing.assert_array_equal(
            serial["people"]["vector"]["values"],
            [[5, 6], [1, 2], [3, 4], [5, 6]],
        )
        np.testing.assert_array_equal(
            serial["people"]["note"]["values"],
            ["world", "hello", "", "world"],
        )
        empty = store.gather({"people": np.empty(0, dtype=np.int64)}, 4)
        empty_vectors = empty["people"]["vector"]["values"]
        assert isinstance(empty_vectors, np.ndarray)
        self.assertEqual(empty_vectors.shape, (0, 2))
        np.testing.assert_array_equal(empty["people"]["tags"]["offsets"], [0])
        with self.assertRaisesRegex(IndexError, "out of range"):
            store.gather({"people": np.array([3], dtype=np.int64)}, 2)


@unittest.skipUnless(
    importlib.util.find_spec("relx._relx_native")
    and importlib.util.find_spec("pyarrow"),
    "native extension and pyarrow needed",
)
class NativePipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        import pyarrow as pa

        from relx._relx_native import GraphBuilder, NativeFeatureStore

        builder = GraphBuilder()
        builder.register_table("people")
        builder.add_nodes("people", ["id"], pa.record_batch({"id": [0, 1, 2]}))
        self.graph = builder.finish()
        self.features = NativeFeatureStore()
        self.features.add_numeric(
            "people", "amount", np.array([10.0, 20.0, 30.0]), 20.0, 10.0
        )

    def test_bounded_queue_returns_completed_batches_in_order(self) -> None:
        from relx._relx_native import NativeBatchPipeline

        pipeline = NativeBatchPipeline(
            self.graph, self.features, [-1], False, 1, 2, 2, 1
        )
        submitted = Event()

        def submit_second() -> None:
            pipeline.submit(1, "people", [2], [100], 42)
            submitted.set()

        worker = Thread(target=submit_second)
        try:
            pipeline.submit(0, "people", [0, 1], [100, 100], 41)
            worker.start()
            self.assertFalse(submitted.wait(0.1))
            first = pipeline.next()
            assert first is not None
            self.assertEqual(first["sequence"], 0)
            np.testing.assert_array_equal(
                first["features"]["people"]["amount"]["values"],
                [[-1], [0]],
            )
            self.assertTrue(submitted.wait(2))
            pipeline.finish()
            second = pipeline.next()
            assert second is not None
            self.assertEqual(second["sequence"], 1)
            np.testing.assert_array_equal(
                second["features"]["people"]["amount"]["values"], [[1]]
            )
            self.assertIsNone(pipeline.next())
        finally:
            pipeline.close()
            if worker.ident is not None:
                worker.join(timeout=2)

    def test_sampling_error_reaches_consumer(self) -> None:
        from relx._relx_native import NativeBatchPipeline

        pipeline = NativeBatchPipeline(
            self.graph, self.features, [-1, -1], True, 1, 2, 1, 1
        )
        try:
            pipeline.submit(0, "people", [99], [100], 42)
            with self.assertRaisesRegex(IndexError, "out of range"):
                pipeline.next()
        finally:
            pipeline.close()

    def test_feature_error_reaches_consumer(self) -> None:
        from relx._relx_native import NativeBatchPipeline, NativeFeatureStore

        short_features = NativeFeatureStore()
        short_features.add_numeric(
            "people", "amount", np.array([10.0]), 10.0, 1.0
        )
        pipeline = NativeBatchPipeline(
            self.graph, short_features, [-1], False, 1, 2, 2, 1
        )
        try:
            pipeline.submit(0, "people", [2], [100], 42)
            with self.assertRaisesRegex(IndexError, "out of range"):
                pipeline.next()
        finally:
            pipeline.close()


if __name__ == "__main__":
    unittest.main()
