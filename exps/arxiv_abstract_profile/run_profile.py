"""Profile the production Arxiv Abstract path on deterministic real rows.

Run from any directory: /home/zk/relconnector/.venv/bin/python run_profile.py
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import random
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch_frame import stype
from torch_frame.data.stats import StatType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exps" / "relconnector_v2"))

from relconnector_v2.connector import PandasDatabaseReader  # noqa: E402
from relconnector_v2.features.schema import (  # noqa: E402
    TableFeatureSchema,
    TensorFrameEncoder,
    TensorFrameFeatureSchema,
)
from relconnector_v2.features.text import GloveTextEmbedder, normalize_text  # noqa: E402
from relconnector_v2.features.shared_text import SharedTextEmbeddingCache  # noqa: E402
from relconnector_v2.observability import OperationMetrics  # noqa: E402

DB = ROOT / "exps" / "data" / "relbench" / "rel-arxiv.sqlite"
EDGES = (0, 64, 128, 256, 512, 10**9)
OPS = (
    "frame_prepare", "frame_convert", "map_text_embedded", "text_normalize",
    "text_cache_lookup", "shared_text_cache_lookup", "text_tokenize", "text_lookup", "text_pool",
    "text_expand", "disk_text_cache_lookup", "disk_text_cache_observe",
)


def timers(metrics: OperationMetrics) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for row in metrics.report()["timers"]:
        out[str(row["operation"])] += float(row["total_s"])
    return dict(out)


def difference(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {op: max(0.0, after.get(op, 0.0) - before.get(op, 0.0)) for op in OPS}


def bucket(length: int) -> str:
    for low, high in zip(EDGES, EDGES[1:], strict=False):
        if low <= length < high:
            return f"{low}-{high - 1}" if high < 10**9 else f"{low}+"
    raise AssertionError(length)


def sample_rows(tokenizer: object, per_bucket: int, repeats: int, modulus: int):
    wanted = per_bucket * repeats
    rows: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    seen: set[str] = set()
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as con:
        cursor = con.execute(
            "SELECT Paper_ID, Abstract FROM papers "
            "WHERE Abstract IS NOT NULL AND Paper_ID % ? = 0 "
            "ORDER BY Paper_ID", (modulus,)
        )
        for paper_id, abstract in cursor:
            abstract = normalize_text(abstract)
            if abstract in seen:
                continue
            seen.add(abstract)
            length = len(tokenizer.tokenize(abstract))
            key = bucket(length)
            counts[key] += 1
            # Fixed seed, reservoir sampling. Memory is bounded by requested rows.
            if len(rows[key]) < wanted:
                rows[key].append((paper_id, abstract, length))
            else:
                index = random.randrange(counts[key])
                if index < wanted:
                    rows[key][index] = (paper_id, abstract, length)
    for values in rows.values():
        random.shuffle(values)
    return rows, counts


def frame_values(frame):
    return frame.feat_dict[stype.embedding].values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-bucket", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modulus", type=int, default=11)
    parser.add_argument("--cache-kind", choices=("local", "shared"), default="local")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        name = "results.json" if args.cache_kind == "local" else "results_shared.json"
        args.output = Path(__file__).with_name(name)
    random.seed(20261016)
    torch.set_num_threads(1)
    metrics = OperationMetrics(True)
    shared_cache = (
        SharedTextEmbeddingCache(
            multiprocessing.get_context("spawn"), max_bytes=64 * 1024 * 1024,
            embedding_dim=300,
        ) if args.cache_kind == "shared" else None
    )
    embedder = GloveTextEmbedder(
        cache_bytes=64 * 1024 * 1024 if shared_cache is None else 0,
        cache_admission="always", execution="direct", disk_cache_mode="off",
        shared_cache=shared_cache, metrics=metrics,
    )
    assert embedder._direct_modules is not None
    word_embeddings = embedder._direct_modules[0]
    selected, candidates = sample_rows(
        word_embeddings.tokenizer, args.per_bucket, args.repeats, args.modulus
    )
    reader = PandasDatabaseReader(DB)
    abstract_col = next(
        col for col in reader.schemas()["papers"].columns if col.name == "Abstract"
    )
    schema = TensorFrameFeatureSchema(
        tables={"papers": TableFeatureSchema(
            columns=(abstract_col,),
            col_to_stype={"Abstract": stype.text_embedded},
            col_stats={"Abstract": {StatType.EMB_DIM: 300}},
        )}, fingerprint="arxiv-abstract-profile-v1",
    )
    encoder = TensorFrameEncoder(schema, embedder, text_batch_size=256, metrics=metrics)
    # Exercise lazy initialization before measured calls, with a row outside the sample.
    embedder(["warmup sentence outside the sampled abstracts"])
    records = []
    for key in [bucket(edge) for edge in EDGES[:-1]]:
        values = selected.get(key, [])
        for repeat in range(min(args.repeats, (len(values) + args.per_bucket - 1) // args.per_bucket)):
            part = values[repeat * args.per_bucket:(repeat + 1) * args.per_bucket]
            data = pd.DataFrame({"Abstract": [row[1] for row in part]})
            outputs = {}
            for state in ("cold", "warm"):
                before = timers(metrics)
                stats_before = embedder.stats
                started = time.perf_counter()
                encoded = encoder.encode("papers", data)
                wall = time.perf_counter() - started
                stats_after = embedder.stats
                outputs[state] = frame_values(encoded).clone()
                record = {
                    "bucket": key, "repeat": repeat, "state": state,
                    "paper_ids": [row[0] for row in part],
                    "token_lengths": [row[2] for row in part],
                    "rows": len(part), "wall_s": wall,
                    "timers_s": difference(timers(metrics), before),
                    "cache_hits": stats_after.cache_hits - stats_before.cache_hits,
                    "cache_misses": stats_after.cache_misses - stats_before.cache_misses,
                    "model_inputs": stats_after.model_inputs - stats_before.model_inputs,
                }
                records.append(record)
                print(key, repeat, state, f"{wall:.4f}s", flush=True)
            torch.testing.assert_close(outputs["cold"], outputs["warm"], rtol=0, atol=0)
            assert records[-2]["model_inputs"] == len(part)
            assert records[-1]["model_inputs"] == 0
            assert records[-1]["cache_hits"] == len(part)
    output = {
        "experiment": "arxiv-papers-Abstract-production-path",
        "database": str(DB), "model": embedder.model_name,
        "torch_threads": torch.get_num_threads(),
        "cache_kind": args.cache_kind,
        "text_batch_size": 256, "direct_batch_size": embedder.direct_batch_size,
        "cache_bytes": 64 * 1024 * 1024,
        "sample": {"rule": f"Paper_ID % {args.modulus} = 0, unique normalized Abstract; reservoir seed 20261016",
                   "per_bucket": args.per_bucket, "repeats": args.repeats,
                   "candidate_counts": dict(candidates)},
        "records": records,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
