# CPU microbenchmark: window-aware SQLite feature reads

Date: 2026-10-27. This experiment isolates the feature-read path in
`relconnector_v2`. It does not use a GPU or run model training.

## Question and design

The production reader merges duplicate node IDs across a window, then chooses
`BETWEEN` for a 4096-row block when at least 25% of that block is requested;
otherwise it batches exact IDs into `IN` queries. To separate the two ideas,
the benchmark compares:

1. **Per-batch IN:** deduplicate within each batch and issue an `IN` query per
   batch.
2. **Window IN:** merge four batches, deduplicate across them, and issue a
   single `IN` query per window.
3. **Window hybrid:** the current production policy, with the same window.
4. **Window range:** issue a `BETWEEN` query for every touched block; run only
   for the dense scenarios to avoid unbounded overfetch on random IDs.

The timed path uses `SqlFeatureFetcher._fetch_table`, the production pandas
SQLite reader, full non-key feature columns, decoding, row ordering, and output
reconstruction. Its raw feature cache is disabled. Each policy receives exactly
the same IDs; the warm-up round checks every returned DataFrame against the
per-batch baseline, including duplicate rows and original order.

The old replay directories contain manifests but no `.pt` plan files. The
requests here are therefore deterministic synthetic windows over actual rows
of the ten local SQLite databases. For each dataset, the benchmark selects the
largest feature-bearing table by ID range and the widest such table; these
coincide for two datasets, yielding 18 tables. The main run has two windows of
four 512-row batches per table, with sparse, dense, and 80% dense / 20% sparse
patterns. The dense IDs are sampled from half a 4096-row block; the resulting
block density is about 32%, just above the 25% decision threshold. A second
run requests 3584 distinct IDs from a 4096-row block (87.5% density) in eight
512-row batches. Each run uses one untimed warm-up and the median of three
timed repetitions; policy order rotates between repetitions.

## Results

All ratios below are `comparison time / candidate time`; values above 1×
mean the candidate is faster. The three time columns are medians in
milliseconds for the **mixed** workload. The last column is from the separate
87.5%-density run.

| Dataset | Table (feature columns) | Per-batch IN | Window IN | Window hybrid | Hybrid / window IN, mixed | Hybrid / window IN, 87.5% |
|---|---|---:|---:|---:|---:|---:|
| rel-amazon | review (5) | 86 | 48 | 87 | 0.56× | 1.11× |
| rel-arxiv | citations (1) | 49 | 26 | 42 | 0.62× | 1.26× |
| rel-arxiv | papers (5) | 89 | 49 | 105 | 0.46× | 1.04× |
| rel-avito | SearchStream (5) | 118 | 67 | 125 | 0.53× | 1.15× |
| rel-event | event_attendees (2) | 95 | 60 | 97 | 0.61× | 1.27× |
| rel-event | events (108) | 987 | 549 | 1395 | 0.39× | 0.91× |
| rel-f1 | standings (4) | 68 | 33 | 59 | 0.57× | 1.13× |
| rel-f1 | results (11) | 149 | 77 | 170 | 0.45× | 1.03× |
| rel-hm | transactions (3) | 68 | 37 | 61 | 0.61× | 1.25× |
| rel-hm | article (24) | 168 | 84 | 193 | 0.44× | 0.96× |
| rel-ratebeer | beer_ratings (10) | 106 | 56 | 112 | 0.50× | 1.06× |
| rel-ratebeer | places (55) | 337 | 161 | 385 | 0.42× | 0.92× |
| rel-salt | salesdocumentitem (7) | 90 | 43 | 90 | 0.48× | 1.09× |
| rel-salt | salesdocument (12) | 103 | 52 | 112 | 0.47× | 1.04× |
| rel-stack | votes (2) | 56 | 30 | 49 | 0.62× | 1.04× |
| rel-stack | postHistory (7) | 102 | 55 | 121 | 0.46× | 0.98× |
| rel-trial | facilities_studies (1) | 57 | 30 | 47 | 0.64× | 1.32× |
| rel-trial | studies (27) | 226 | 115 | 271 | 0.42× | 0.91× |

Across the 18 tables, in the mixed workload:

- Window IN is **1.88× faster** than per-batch IN at the median (18/18 wins).
  It reduces the measured SQL query count from 8 to 2 for the two windows.
- The current hybrid policy is **0.49× as fast** as window IN at the median
  (0/18 wins). It reads **2.90×** as many rows at the median and makes 4 rather
  than 2 SQL queries: the dense block query and sparse `IN` query are separate.
- Relative to per-batch IN, the complete hybrid policy is **0.94× as fast** at
  the median (6/18 wins). Window merging's gain is often consumed by block
  overfetch.

For the 87.5%-density workload, the hybrid policy is **1.05× faster** than
window IN at the median (13/18 wins), with a range from 0.91× to 1.32×. The
range query reads only 1.14× as many rows as the exact-ID query. The widest
tables still tend to lose: Event `events` (108 columns) is 0.91×, Ratebeer
`places` (55) is 0.92×, and Trial `studies` (27) is 0.91×.

## Interpretation and limits

Window deduplication and bounded `IN` queries provide a clear CPU read benefit
for these workloads. The fixed 25% block threshold is too low for the tested
32%-dense and mixed patterns: the extra rows and decoding cost exceed any
range-scan advantage. A much denser block can favor range reads, particularly
for narrow tables. These observations motivate a row-width-aware threshold or
per-table calibration, but are not enough to change the production default.

This is a read-path microbenchmark. It excludes topology sampling, text
embedding, feature caching, parallel workers, and GPU training. The synthetic
requests are controlled rather than replayed from real training, and OS page
cache and other host activity can affect short timings. Thus the ratios must
not be presented as end-to-end training speedups.

## Reproduce

Run from the repository root, using the ten local databases in
`exps/data/relbench`:

```bash
PYTHONPATH=exps/relconnector_v2 .venv/bin/python -m relconnector_v2.bench_chunked_read \
  --output exps/benchmarks/chunked-read-20261027/all10.json
PYTHONPATH=exps/relconnector_v2 .venv/bin/python -m relconnector_v2.bench_chunked_read \
  --scenarios saturated --windows 1 --repeats 3 \
  --output exps/benchmarks/chunked-read-20261027/saturated-all10.json
```

The result JSON files retain every measured repetition, database size and
modification time, query count, returned-row count, and policy comparison.
