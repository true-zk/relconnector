# Phase 2: CPU feature processing by column

Date: 2026-10-27. Phase-one SQL read settings are frozen in
[`feature_read_profiles.py`](../relconnector_v2/relconnector_v2/feature_read_profiles.py).
The standalone benchmark CLI now chooses the dataset's tuned window and block
density threshold by default; its effective configuration records both values.
The SQL fetcher receives the threshold in both the single-process and process
worker paths.

## Measurement

All ten local SQLite datasets were profiled. The schema is the production
TensorFrame schema recorded by earlier task runs, so primary keys, foreign
keys, and task-hidden target columns are excluded. This leaves **71
feature-bearing tables and 518 feature columns**: 191 numerical, 108
categorical, 1 multicategorical, 50 timestamp, and 168 text. A table's sample
has up to **512 real rows**, taken from four positions in its ID range. Some
small tables have fewer rows. The SQL reads use each dataset's frozen window
(16 or 32 batches), 4096-row block size, and density threshold. The measured
feature conversion begins only after the rows have been fetched.

Each table was converted three times on one CPU thread. The report sums the
median per-column mapping times across the sampled tables. For text, it also
includes the measured text normalization in frame preparation. Category
includes the multicategorical column. Shared DataFrame copying and output
assembly are not assigned to individual types, so the type columns below are
**attributed processing time**, not an end-to-end wall-clock breakdown. The
raw file also contains each table's full encode time and separate SQL read
time. The GloVe sentence embedder uses the production direct CPU path; model
initialization is excluded and text/encoded reuse caches are disabled.

## Type-level results

Times are milliseconds for one sample of up to 512 rows **per feature-bearing
table** in the dataset; they must not be read as one training batch latency.
The last column is text's share of the summed column-attributed times.

| Dataset | Columns / text | Numerical ms | Category ms | Timestamp ms | Text ms | Text share |
|---|---:|---:|---:|---:|---:|---:|
| rel-amazon | 11 / 6 | 0.08 | 8.1 | 3.1 | 234.2 | 95.4% |
| rel-arxiv | 11 / 6 | 0.03 | — | 11.7 | 179.6 | 93.9% |
| rel-avito | 25 / 2 | 0.51 | — | 10.4 | 24.0 | 68.8% |
| rel-event | 120 / 6 | 1.93 | 24.1 | 10.7 | 28.7 | 43.9% |
| rel-f1 | 45 / 13 | 0.54 | 3.6 | 23.0 | 84.8 | 75.7% |
| rel-hm | 33 / 12 | 0.24 | 20.2 | 2.9 | 70.5 | 75.1% |
| rel-ratebeer | 132 / 54 | 0.65 | 86.9 | 18.0 | 623.8 | 85.5% |
| rel-salt | 13 / 8 | — | 4.4 | 8.6 | 35.6 | 73.2% |
| rel-stack | 32 / 15 | 0.14 | 12.5 | 18.2 | 280.5 | 90.1% |
| rel-trial | 96 / 46 | 0.32 | 59.8 | 33.5 | 550.0 | 85.5% |

![Feature processing by dataset and text length](../benchmarks/feature-processing-20261027/feature-processing-profile.png)

Across the ten datasets, text contributes **2,112 ms (85.3%)** of the
column-attributed time. Its 168 columns account for 81,282 sampled values.
Only 36,285 are distinct within their column samples, and 23,624 are empty;
the text mapper deduplicates repeated strings inside each call. The 191
numerical columns together take 4.4 ms of measured mapper time, but they
also incur shared frame preparation that is outside their per-column timer.

## Where text time goes

The direct GloVe path's nested timers, summed over text columns, are:

| Stage | Time across datasets | Share of text-attributed time |
|---|---:|---:|
| Tokenization | 643 ms | 30.4% |
| Embedding lookup | 251 ms | 11.9% |
| Mean pooling | 691 ms | 32.7% |
| Frame-preparation normalization | 106 ms | 5.0% |
| Embedder normalization | 29 ms | 1.4% |
| Other mapper overhead | 393 ms | 18.6% |

Tokenization plus pooling consume about **63%** of text-attributed time. The
longest sampled text columns also have the highest single-column times:

| Dataset | Table.column | Mean characters per sampled row | Time for column |
|---|---|---:|---:|
| Amazon | `review.review_text` | 648 | 116 ms |
| Trial | `studies.detailed_descriptions` | 1,178 | 114 ms |
| Arxiv | `papers.Abstract` | 1,102 | 97 ms |
| Stack | `posts.Body` | 1,153 | 94 ms |
| Stack | `postHistory.Text` | 677 | 65 ms |
| Trial | `studies.brief_summaries` | 619 | 64 ms |

Across the 168 text columns, log mean character length and log processing
time per value have a Pearson correlation of **0.78** on these samples. This
is descriptive, not proof of causality; token count, empty values, repeated
strings, and model chunking also affect time. Ratebeer and Trial have the
largest *aggregate* text cost mainly because they each have many text columns.

## Interpretation and limits

Text processing is the first candidate for phase-two optimization, especially
on long and mostly unique descriptions and bodies. Categorical conversion is
a secondary cost on Ratebeer and Trial. This experiment does not establish a
training speedup: it has no neighbor-sampler request stream, parallel workers,
GPU training, or warm text cache. The previously tuned SQL settings were kept
fixed, but the sampled feature rows form small virtual batches, so the density
threshold's exact query choice here is not representative of a 512-ID
training batch. A future optimization should be measured against this fixed
read profile with actual sampled plans or an end-to-end run.

## Reproduce and inspect

```bash
PYTHONPATH=exps/relconnector_v2 .venv/bin/python -m relconnector_v2.profile_feature_processing \
  --rows-per-table 512 --repeats 3 \
  --output exps/benchmarks/feature-processing-20261027/all10-rows512.json
```

[Raw results](../benchmarks/feature-processing-20261027/all10-rows512.json)
contain every column's runs, text stages, sampled character counts, and table
read/encode times. The [column CSV](../benchmarks/feature-processing-20261027/all10-rows512.columns.csv)
is convenient for sorting and plotting. The schema source path and checksum are
recorded per dataset in the raw result.
