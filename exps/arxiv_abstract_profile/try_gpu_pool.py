"""Measure GPU pooling and GPU GloVe lookup on the same real Abstract batches."""

from __future__ import annotations

import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exps" / "relconnector_v2"))
from relconnector_v2.features.text import GloveTextEmbedder, normalize_text  # noqa: E402
from try_packed import packed  # noqa: E402

HERE = Path(__file__).resolve().parent
DB = ROOT / "exps" / "data" / "relbench" / "rel-arxiv.sqlite"


def prepared_chunks(sentences, word, size=8):
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]), reverse=True)
    ordered = [sentences[i] for i in order]
    return [word(word.tokenize(ordered[i:i + size]))
            for i in range(0, len(ordered), size)]


def pool_chunks(chunks, pool, device):
    outputs = []
    for item in chunks:
        features = {"token_embeddings": item["token_embeddings"],
                    "attention_mask": item["attention_mask"]}
        outputs.append(pool(features)["sentence_embedding"])
    if device == "cuda":
        torch.cuda.synchronize()
    return torch.cat(outputs)


def encode_gpu(sentences, word, pool, weight_cuda, *, chunk_size, cpu_lookup):
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]), reverse=True)
    ordered = [sentences[i] for i in order]
    outputs = []
    for i in range(0, len(ordered), chunk_size):
        features = word.tokenize(ordered[i:i + chunk_size])
        if cpu_lookup:
            features = word(features)
            embeddings = features["token_embeddings"].to("cuda")
        else:
            embeddings = F.embedding(features["input_ids"].to("cuda"), weight_cuda)
        result = pool({"token_embeddings": embeddings,
                       "attention_mask": features["attention_mask"].to("cuda")})
        outputs.append(result["sentence_embedding"].cpu())
    torch.cuda.synchronize()
    encoded = torch.cat(outputs)
    restored = torch.empty_like(encoded)
    restored[torch.tensor(order)] = encoded
    return restored


def timed(call, repeats=9):
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        durations.append(time.perf_counter() - start)
    return durations


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    torch.set_num_threads(1)
    baseline = json.loads((HERE / "results.json").read_text())
    selected = [r for r in baseline["records"] if r["state"] == "cold"
                and r["bucket"] in ("64-127", "128-255")]
    ids = {i for record in selected for i in record["paper_ids"]}
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as con:
        texts = {i: normalize_text(con.execute(
            "SELECT Abstract FROM papers WHERE Paper_ID = ?", (i,)
        ).fetchone()[0]) for i in ids}
    embedder = GloveTextEmbedder(execution="direct", disk_cache_mode="off")
    assert embedder._direct_modules is not None
    word, pool = embedder._direct_modules
    start = time.perf_counter()
    weight_cuda = word.emb_layer.weight.to("cuda")
    torch.cuda.synchronize()
    setup_s = time.perf_counter() - start
    records = []
    with torch.inference_mode():
        for record in selected:
            batch = [texts[i] for i in record["paper_ids"]]
            cpu_chunks = prepared_chunks(batch, word)
            gpu_chunks = [{"token_embeddings": c["token_embeddings"].to("cuda"),
                           "attention_mask": c["attention_mask"].to("cuda")}
                          for c in cpu_chunks]
            torch.cuda.synchronize()
            reference = embedder._encode_missing(batch)
            methods = {
                "cpu_direct": lambda: embedder._encode_missing(batch),
                "packed_cpu_128": lambda: packed(batch, word, 128)[0],
                "gpu_pool_after_cpu_lookup_8": lambda: encode_gpu(
                    batch, word, pool, weight_cuda, chunk_size=8, cpu_lookup=True),
                "gpu_lookup_pool_8": lambda: encode_gpu(
                    batch, word, pool, weight_cuda, chunk_size=8, cpu_lookup=False),
                "gpu_lookup_pool_128": lambda: encode_gpu(
                    batch, word, pool, weight_cuda, chunk_size=128, cpu_lookup=False),
            }
            for name, call in methods.items():
                output = call()
                torch.testing.assert_close(output, reference, rtol=1e-4, atol=1e-5)
            samples = {
                "cpu_pool_only_16_chunks": timed(lambda: pool_chunks(cpu_chunks, pool, "cpu")),
                "gpu_pool_only_16_chunks_preloaded": timed(
                    lambda: pool_chunks(gpu_chunks, pool, "cuda")),
            }
            # Rotate starting method across records to reduce order bias.
            names = list(methods)
            shift = record["repeat"] % len(names)
            names = names[shift:] + names[:shift]
            for name in names:
                samples[name] = timed(methods[name])
            records.append({"bucket": record["bucket"], "repeat": record["repeat"],
                            "rows": len(batch), "samples_s": samples})
            print(record["bucket"], record["repeat"],
                  {k: round(statistics.median(v) * 1000, 2)
                   for k, v in samples.items()}, flush=True)
    result = {"gpu": torch.cuda.get_device_name(0),
              "weight_gpu_bytes": weight_cuda.numel() * weight_cuda.element_size(),
              "weight_transfer_setup_s": setup_s,
              "torch_threads": torch.get_num_threads(),
              "method": "single process; 128 real abstracts per batch; no training GPU contention",
              "validation": "allclose rtol=1e-4 atol=1e-5 vs CPU direct",
              "records": records}
    (HERE / "gpu_results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
