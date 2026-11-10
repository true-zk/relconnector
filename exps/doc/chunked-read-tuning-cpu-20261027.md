# CPU tuning of feature-read windows and block density

Date: 2026-10-27. This follows the [initial read-path experiment](chunked-read-cpu-20261027.md).

## Scope and method

This experiment sweeps fixed windows of **1, 2, 4, 8, 16, 32 batches** and
block density thresholds of **25%, 40%, 60%, 80%, 100%**. The block size is
4096 rows and the SQL `IN` cap is 10,000 IDs. Each of the ten local SQLite
datasets uses the same 32 batches of 512 IDs per representative table for
every candidate. Eight batches each are sparse, mixed, dense, and
87.5%-dense. Requests are deterministic and read actual database rows and all
non-key feature columns. The largest feature-bearing table and the widest
feature-bearing table are selected per dataset; they coincide for Amazon and
Avito, yielding 18 tables. A dataset's score is the sum of its selected table
median read times.

The grid uses seed 42, one untimed warm-up, and two timed repetitions. Leading
configurations are checked on seed 43 with one warm-up and three timed
repetitions. The fastest checked configuration per dataset is then measured on
a third, previously unused seed (45), again with one warm-up and three timed
repetitions. Candidate selection depends on elapsed time, so changing the
reference configuration does not change which candidate is fastest.

**Comparison baseline:** window size 1 and threshold 100%. Each batch gets
one exact-ID `IN` query, with no merging or deduplication *across* batches and
no whole-block `BETWEEN` reads. Duplicate IDs *within* a batch are requested
once and restored to their original positions. This is a batch-level exact
read, not one SQL query per ID. The confirmation script checks that each
representative table executes exactly 32 baseline queries and checks that
candidate feature values and order match the exact-read output.

Timing covers SQL reads, pandas conversion and decoding, deduplication, and
row restoration. The run is single-process, CPU-only, and has the raw feature
cache disabled.

## Results versus no-window exact reads

Times are the sum of median milliseconds across each dataset's representative
tables for the **same 32 batches**. Speedup is no-window exact-read time
divided by selected-configuration time. These are seed-45 confirmation results.

| Dataset | Window (batches) | Density threshold | Exact-read ms | Selected ms | Speedup |
|---|---:|---:|---:|---:|---:|
| rel-amazon | 32 | 60% | 357 | 163 | 2.19× |
| rel-arxiv | 32 | 60% | 587 | 258 | 2.28× |
| rel-avito | 32 | 40% | 308 | 117 | 2.65× |
| rel-event | 16 | 100% | 2536 | 1008 | 2.52× |
| rel-f1 | 32 | 60% | 662 | 194 | 3.41× |
| rel-hm | 32 | 40% | 965 | 389 | 2.48× |
| rel-ratebeer | 32 | 100% | 1812 | 734 | 2.47× |
| rel-salt | 32 | 80% | 783 | 312 | 2.51× |
| rel-stack | 32 | 80% | 653 | 280 | 2.33× |
| rel-trial | 32 | 60% | 1173 | 492 | 2.39× |

The median speedup across datasets is **2.48×** (range 2.19–3.41×;
geometric mean 2.50×). Across all 18 representative tables, SQL query count
fell from **576 to 36** and returned rows from **279,520 to 241,179**
(13.7% fewer). The larger windows chiefly reduce SQL calls by combining
requests and avoiding repeated reads across batches. Cross-batch
deduplication also removes repeated requested rows. For Event and Ratebeer, the selected
100% threshold keeps the selected window's reads exact-ID even in dense
regions.

The exact numerical threshold is not always identifiable from this workload.
For several datasets, thresholds from 40% to 80% produce the same query count
and returned-row count; differences between those settings are timing noise
unless confirmed on more workloads.

## Practical limit

These are the fastest tested **CPU microbenchmark configurations**, not proven
training settings. The old replay directories contain manifests but no saved
sampled plans, so the benchmark uses controlled requests rather than actual
neighbor-sampler output. It excludes raw/encoded cache reuse, text embedding,
parallel workers, queueing, and GPU training. The production process pipeline
also caps a window by bytes; a 32-batch target may be shortened by that cap.
Larger windows can increase time to the first ready batch and peak memory even
when total SQL read time improves. Production settings should be checked with
real sampled-plan replay or an end-to-end training run before adoption.

## Reproduce

```bash
PYTHONPATH=exps/relconnector_v2 .venv/bin/python -m relconnector_v2.tune_chunked_read \
  --repeats 2 --validation-repeats 3 \
  --output exps/benchmarks/chunked-read-20261027/tuned-all10.json
PYTHONPATH=exps/relconnector_v2 .venv/bin/python -m relconnector_v2.confirm_chunked_read \
  --tuning exps/benchmarks/chunked-read-20261027/tuned-all10.json \
  --baseline no_window_exact --seed 45 \
  --output exps/benchmarks/chunked-read-20261027/compare-no-window-seed45.json
```

The [grid result](../benchmarks/chunked-read-20261027/tuned-all10.json) and
[no-window confirmation](../benchmarks/chunked-read-20261027/compare-no-window-seed45.json)
retain all timings, query counts, and row counts.
