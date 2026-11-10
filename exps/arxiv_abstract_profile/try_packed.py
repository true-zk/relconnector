"""Compare a packed-token mean with the production padded-token GloVe path."""

from __future__ import annotations

import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch_frame import stype
from torch_frame.data.stats import StatType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exps" / "relconnector_v2"))

from relconnector_v2.features.text import GloveTextEmbedder, normalize_text  # noqa: E402
from relconnector_v2.connector import PandasDatabaseReader  # noqa: E402
from relconnector_v2.features.schema import (  # noqa: E402
    TableFeatureSchema, TensorFrameEncoder, TensorFrameFeatureSchema,
)

HERE = Path(__file__).resolve().parent
DB = ROOT / "exps" / "data" / "relbench" / "rel-arxiv.sqlite"


def packed(sentences, word_embeddings, chunk_size):
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]), reverse=True)
    ordered = [sentences[i] for i in order]
    outputs = []
    tokenize_s = bag_s = 0.0
    for start in range(0, len(ordered), chunk_size):
        chunk = ordered[start:start + chunk_size]
        tick = time.perf_counter()
        ids = [word_embeddings.tokenizer.tokenize(text) for text in chunk]
        offsets = [0]
        flat = []
        for row in ids:
            flat.extend(row)
            offsets.append(len(flat))
        tokens = torch.tensor(flat, dtype=torch.long)
        offsets_tensor = torch.tensor(offsets, dtype=torch.long)
        tokenize_s += time.perf_counter() - tick
        tick = time.perf_counter()
        vectors = F.embedding_bag(
            tokens, word_embeddings.emb_layer.weight, offsets_tensor,
            mode="mean", include_last_offset=True,
        )
        bag_s += time.perf_counter() - tick
        outputs.append(vectors)
    encoded = torch.cat(outputs)
    restored = torch.empty_like(encoded)
    restored[torch.tensor(order)] = encoded
    return restored, tokenize_s, bag_s


class PackedGloveTextEmbedder(GloveTextEmbedder):
    def _encode_missing(self, sentences):
        assert self._direct_modules is not None
        return packed(sentences, self._direct_modules[0], len(sentences))[0]


def end_to_end(records, texts):
    column = next(c for c in PandasDatabaseReader(DB).schemas()["papers"].columns
                  if c.name == "Abstract")
    schema = TensorFrameFeatureSchema(
        tables={"papers": TableFeatureSchema(
            columns=(column,), col_to_stype={"Abstract": stype.text_embedded},
            col_stats={"Abstract": {StatType.EMB_DIM: 300}},
        )}, fingerprint="arxiv-abstract-packed-v1",
    )
    encoders = {}
    for name, cls in (("production", GloveTextEmbedder),
                      ("packed", PackedGloveTextEmbedder)):
        embedder = cls(cache_bytes=64 * 1024 * 1024, execution="direct",
                       disk_cache_mode="off")
        encoders[name] = TensorFrameEncoder(schema, embedder, text_batch_size=256)
        embedder(["warmup sentence outside the sampled abstracts"])
    out = []
    for record in records:
        batch = [texts[i] for i in record["paper_ids"]]
        data = pd.DataFrame({"Abstract": batch})
        samples = {}
        for name in (("production", "packed") if record["repeat"] % 2 == 0
                     else ("packed", "production")):
            start = time.perf_counter()
            tf = encoders[name].encode("papers", data)
            elapsed = time.perf_counter() - start
            samples[name] = (elapsed, tf.feat_dict[stype.embedding].values)
        maximum = float((samples["production"][1] - samples["packed"][1]).abs().max())
        torch.testing.assert_close(samples["production"][1], samples["packed"][1],
                                   rtol=1e-4, atol=1e-5)
        out.append({"bucket": record["bucket"], "repeat": record["repeat"],
                    "rows": len(batch), "production_s": samples["production"][0],
                    "packed_s": samples["packed"][0],
                    "max_absolute_difference": maximum})
        print("TensorFrame", record["bucket"], record["repeat"],
              f"{samples['production'][0]:.4f}s -> {samples['packed'][0]:.4f}s",
              flush=True)
    return out


def run():
    torch.set_num_threads(1)
    baseline = json.loads((HERE / "results.json").read_text())
    records = [r for r in baseline["records"] if r["state"] == "cold"]
    ids = sorted({i for record in records for i in record["paper_ids"]})
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as con:
        texts = {
            i: normalize_text(con.execute(
                "SELECT Abstract FROM papers WHERE Paper_ID = ?", (i,)
            ).fetchone()[0]) for i in ids
        }
    embedder = GloveTextEmbedder(execution="direct", disk_cache_mode="off")
    assert embedder._direct_modules is not None
    word = embedder._direct_modules[0]
    results = []
    with torch.inference_mode():
        for record in records:
            batch = [texts[i] for i in record["paper_ids"]]
            if not batch:
                continue
            reference = embedder._encode_missing(batch)
            for chunk_size in (8, len(batch)):
                proposal, _, _ = packed(batch, word, chunk_size)
                max_abs = float((proposal - reference).abs().max())
                torch.testing.assert_close(proposal, reference, rtol=1e-4, atol=1e-5)
                result = {
                    "bucket": record["bucket"], "repeat": record["repeat"],
                    "rows": len(batch), "chunk_size": chunk_size,
                    "max_absolute_difference": max_abs,
                }
                samples = []
                # Alternate order to reduce cache/clock drift bias.
                for k in range(9):
                    if k % 2 == 0:
                        t = time.perf_counter()
                        embedder._encode_missing(batch)
                        reference_s = time.perf_counter() - t
                        t = time.perf_counter()
                        _, tok_s, bag_s = packed(batch, word, chunk_size)
                        packed_s = time.perf_counter() - t
                    else:
                        t = time.perf_counter()
                        _, tok_s, bag_s = packed(batch, word, chunk_size)
                        packed_s = time.perf_counter() - t
                        t = time.perf_counter()
                        embedder._encode_missing(batch)
                        reference_s = time.perf_counter() - t
                    samples.append({"production_s": reference_s, "packed_s": packed_s,
                                    "packed_tokenize_s": tok_s, "packed_bag_s": bag_s})
                result["samples"] = samples
                result["median_production_s"] = statistics.median(s["production_s"] for s in samples)
                result["median_packed_s"] = statistics.median(s["packed_s"] for s in samples)
                results.append(result)
                print(record["bucket"], record["repeat"], chunk_size,
                      f"{result['median_production_s']:.4f}s -> {result['median_packed_s']:.4f}s",
                      flush=True)
    frame_results = end_to_end(records, texts)
    output = {"method": "same GloVe tokenizer, F.embedding_bag mean, no padded token matrix",
              "validation": "allclose rtol=1e-4 atol=1e-5 against production direct output",
              "torch_threads": torch.get_num_threads(), "records": results,
              "tensorframe_records": frame_results}
    (HERE / "packed_results.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    run()
