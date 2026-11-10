# CPU-only multi-epoch feature-cache and batch-size experiment

Date: 2026-10-27. This is a simulation and feature-supply replay; it does not
run GPU training or change the production training configuration.

## Question and method

For each of the ten local SQLite datasets, the experiment takes up to 512 real
IDs from four positions in every feature-bearing table. It uses the production
TensorFrame converter, direct CPU GloVe embeddings, and the **frozen phase-one
SQL window and density threshold** for that dataset. The schema and feature
statistics are the same cached production statistics used in the phase-two
column profile. The sample covers 71 tables and 518 columns.

The simulated five-epoch alternatives are:

1. **Direct:** each epoch reads the requested rows from SQLite and converts all
   features.
2. **Selective:** epoch 1 does the same work and writes selected encoded columns
   to files. Later epochs query and convert only uncached columns, load selected
   columns from files, then assemble a TensorFrame.
3. **All columns:** epoch 1 writes every encoded column. Later epochs load all
   columns and assemble a TensorFrame without SQL.

The files are a simple experimental ID-block cache, one compact serialized
column per table/sample. The same IDs recur in each simulated epoch, giving
100% cache hits. Every stored column is checked for exact roundtrip equality,
including NaNs. A column is selected when its phase-two conversion time exceeds
the measured file read and four future uses repay its file write. The direct
epoch value sums median SQL and conversion times; first-epoch values add measured
write time. Warm values time the complete replay path, including TensorFrame
reassembly. The five-epoch totals use `first + 4 × warm`.

## Multi-epoch results

Times below are milliseconds for the sampled rows across all feature-bearing
tables of one dataset. They are **not** full training epochs. Speedup compares
five simulated epochs with direct SQL plus conversion in every epoch.

| Dataset | Cached columns | Direct / epoch | Selective first | Selective later | Selective 5-epoch speedup | Sample cache MB |
|---|---:|---:|---:|---:|---:|---:|
| Amazon | 9/11 | 274 | 283 | 18 | 3.86× | 3.8 |
| Arxiv | 10/11 | 226 | 237 | 10 | 4.09× | 3.8 |
| Avito | 6/25 | 90 | 94 | 53 | 1.47× | 1.4 |
| Event | 23/120 | 223 | 237 | 124 | 1.51× | 3.9 |
| F1 | 23/45 | 188 | 206 | 61 | 2.09× | 5.1 |
| HM | 23/33 | 133 | 163 | 37 | 2.16× | 7.5 |
| Ratebeer | 107/132 | 900 | 981 | 134 | 2.97× | 31.2 |
| Salt | 13/13 | 72 | 86 | 8 | 3.05× | 5.0 |
| Stack | 28/32 | 398 | 426 | 40 | 3.40× | 9.5 |
| Trial | 83/96 | 840 | 946 | 91 | 3.20× | 28.2 |

The selected columns are 167/168 text, 107/108 categorical, 50/50 timestamps,
1/1 multicategorical, and **0/191 numerical**. Every numerical column took
less time to convert than to load as an individual serialized file; its SQL read
still occurs when other uncached columns need the table. The two other rejected
columns are `places.is_geo_override` in Ratebeer and
`sponsors.agency_class` in Trial: their single measured writes were high enough
that five epochs did not repay them. This is a noisy boundary, so it should be
re-estimated on a larger replay before deploying a fixed per-column list.

Selective caching stores 99.4 MB here versus 100.7 MB for all-column caching.
The saved numerical columns are small, while text embeddings dominate bytes.
Loading all columns is faster than selective loading on this sample, because it
removes the remaining SQL query and converter calls. Its five-epoch speedups
range from 2.30× to 4.40×; see the raw results for exact values. Thus selective
caching meets the requested slow-column policy, but it is **not the fastest
policy** when the whole encoded
table fits on storage.

An [earlier whole-dataset memory estimate](../benchmarks/batch-memory-estimate-20260927/estimates.json)
(different task run, so approximate)
projects about 124 GB for the selected columns versus 130 GB for all encoded
columns across the ten datasets. This estimate is much larger than the 99 MB
sample; the workspace filesystem had about 150 GB free during this run. It
makes hit rate, SSD capacity, and a bounded cache budget essential
for a production implementation. The existing production text cache is keyed by
content and still reads raw text from SQL; it is not the ID-keyed, SQL-skipping
cache simulated here.

## Warm page-cache sensitivity

The main warm times use normal OS file caching. A separate pass calls `fsync`
and `POSIX_FADV_DONTNEED` on each sample file before reading and assembling all
columns. Its full-cache read times were about 4–22× the normal warm reads across
the ten datasets. The eviction call is only a kernel hint, and an immediately
repeated selective read gave inconsistent results even for an all-cached table.
These stress numbers show that the warm speedups should not be extrapolated to
a cache larger than RAM or to a contended SSD. They are not a reliable cold-SSD
throughput estimate.

## Batch size and GPU starvation

Two CPU-only checks distinguish feature conversion chunks from *training seed*
batches. Holding the same 512 sampled rows per table fixed, converting them in
32-row chunks took 2.3–10.6× as long as one 512-row chunk, depending on the
dataset. This exposes fixed converter-call overhead but omits graph sampling
and does not map directly from feature rows to seed batch size.

For a direct seed-batch check, the F1 `driver-circuit-compete` task used real,
versioned sampled plans with 2-hop fanout `128,128`, four feature workers,
three epochs, no feature caches, and an empty CPU trainer. All batch sizes
processed the same 2,649 seeds per epoch. The phase-one SQL profile remained
at window 32 and density 0.6. A second run in reverse size order checks run
order effects.

| Seed batch size | Batches / 3 epochs | First CPU supply s | First seeds/s | Second CPU supply s | Second seeds/s |
|---:|---:|---:|---:|---:|---:|
| 256 | 33 | 16.52 | 481 | 16.05 | 495 |
| 512 | 18 | 14.41 | 551 | 14.61 | 544 |
| 1024 | 9 | 13.74 | 578 | 13.62 | 584 |
| 2048 | 6 | 13.47 | 590 | 13.48 | 590 |

Both passes show diminishing returns: 512 to 1024 improves CPU seed supply by
about 6–7%, and 512 to 2048 by about 7–8%. A larger seed batch can reduce
Python and per-batch overhead,
but it also changes sampled-node counts, GPU memory use, and the GPU step time.
This CPU replay cannot establish a GPU idle percentage. [Older F1](../benchmarks/v2-f1-gpu-direct-shared-20260925.jsonl)
and [Arxiv](../benchmarks/v2-arxiv-paper-citation-gpu-adaptive-mmap-warm-2x50-20260925.jsonl)
GPU runs reported ready-queue waits of 86% and 90% at batch size 512,
but they used earlier feature policies and are only evidence that data supply
has been a bottleneck. The practical next GPU comparison is 512 versus 1024
with ready-queue wait, examples/s, peak VRAM, and model quality recorded.

## Limits and artifacts

- The cache experiment repeats identical sampled IDs and does not model misses,
  changed epoch plans, eviction, concurrent worker I/O, or GPU training.
- Its SQL reads group sampled table IDs into synthetic virtual batches of about
  16 rows. The frozen window and density settings are used, but the query
  shapes do not reproduce the original 512-seed training batches.
- Main cache times sum table-level measurements on one CPU thread. Real training
  windows and CPU/GPU overlap can change end-to-end speedup.
- The F1 batch-size replay is one task; other datasets can have different graph
  overlap, text volume, memory use, and optimal batch size.
- The per-column file format is intentionally simple. Grouping columns into
  larger blocks could reduce read overhead; a production ID-keyed cache also
  needs schema/version invalidation and bounded admission.

Raw results: [five-epoch cache replay](../benchmarks/epoch-feature-cache-20261027/all10-rows512-5epoch-final.json),
[cold-read stress](../benchmarks/epoch-feature-cache-20261027/all10-cold-selective.json),
and [F1 CPU supply summary](../benchmarks/epoch-feature-cache-20261027/f1-batch-size-summary.json).
Reproduction entry points:
[`bench_epoch_feature_cache.py`](../relconnector_v2/relconnector_v2/bench_epoch_feature_cache.py)
and [`bench_epoch_cache_cold_read.py`](../relconnector_v2/relconnector_v2/bench_epoch_cache_cold_read.py).

```bash
PYTHONPATH=exps/relconnector_v2 .venv/bin/python -m relconnector_v2.bench_epoch_feature_cache \
  --output exps/benchmarks/epoch-feature-cache-20261027/reproduced.json
PYTHONPATH=exps/relconnector_v2 .venv/bin/python -m relconnector_v2.bench_epoch_cache_cold_read \
  --selective --input exps/benchmarks/epoch-feature-cache-20261027/reproduced.json \
  --output exps/benchmarks/epoch-feature-cache-20261027/reproduced-cold.json
```
