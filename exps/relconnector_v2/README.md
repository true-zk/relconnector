# relconnector_v2

## v1 与 v2 的区别

v1 是基于单进程的数据准备优化版本，通过节点去重、跨批次窗口、缓存复用和进程内线程流水线减少重复读取与编码。实际训练进一步暴露出 assemble（特征读取、编码与批次组装）的供给瓶颈：即使减少重复工作，数据准备仍可能跟不上 GPU 消费。v2 因此升级为多进程架构，将采样、assemble 和训练拆分为独立的执行阶段，分别由采样进程、组装进程池和训练主进程承担，通过有界队列连接并重叠执行。这里的单进程不等于单线程，多进程也不改变邻居采样与模型训练的算法语义。

Independent implementation of the three-stage relational training pipeline:

```text
sample process -> assembly process pool -> GPU trainer
```

The package vendors the required connector, feature, graph, task, sampling, and
training code. It does not import the original `relconnector` package.

Use `PYTHONPATH=exps/relconnector_v2` when running from the repository root.
Runtime artifacts default to `.relconnector-v2-cache` and do not share the
original implementation's cache namespace.

Text embeddings use `--disk-text-cache-mode adaptive` by default. A large
database activates persistent caching after the observed text-model work is
substantial and the shared-memory hit rate is low; small databases such as
F1 keep using memory only. `--disk-text-cache-mode on` forces persistence,
and `off` disables it. The cache lives under the selected `--cache-dir` and
is keyed by database version, model identity, encoding path, and text content.
Each worker keeps at most 1,024 disk shard mappings open in an LRU. Eviction
closes the mapping, preserves persisted embeddings, and leaves returned tensors
valid; later access reopens the file without model encoding. New vectors are
appended to writer-owned, fixed-size 8,192-row shards. Writes trigger a TRUNCATE
checkpoint at an adaptive WAL threshold based on index size and free disk space
(normally 64 MiB to 4 GiB). This is a recycling trigger, not a hard size limit;
checkpoint contention and actual WAL peaks are reported in telemetry. Sustained
read-heavy traffic caps the budget at 512 MiB: two 30-second windows with at
least 4,096 disk requests each, at least 99% hits, and at most 4 MiB/s observed
WAL growth. Below 95% hits or above 8 MiB/s growth restores the write budget.
Warm lookups also check for recycling, at most once per second.

The default `--text-cache-scope adaptive` keeps shared embeddings and their
cross-worker encoding reservations. Each worker measures lookup cost and reuse.
After a costly, high-hit observation window it builds a bounded local index of
published shared slots, letting repeated texts bypass the shared lock without
duplicating embeddings or model work. Sustained low local reuse releases this
index; a cooldown prevents rapid switching. The index has a 32 MiB estimated
metadata budget per worker, in addition to the shared vector budget. The same
policy runs for every dataset and remains independent of disk persistence.
`shared` disables the local index for comparisons; `local` retains the explicit
per-worker embedding-cache option used in diagnostic experiments.

Shared lookups hash outside the process lock, probe zero-copy metadata views,
and gather published vectors in batches outside the lock. Publications also
copy vectors in batches. Lock acquisition has a deadline so a stalled owner
cannot silently block other workers forever.

The benchmark prints progress to stderr every 30 seconds, including trained
batches per epoch and seconds since the last trained batch. Use
`--progress-interval-s 0` to disable it. Seed-reader failures are propagated to
the trainer and trigger worker cleanup. Error delivery uses a synchronous channel
so resource exhaustion does not require starting another feeder thread.

The encoded-feature cache supports `--encoded-cache-scope shared` (default) and
`local`. Local scope splits the `--encoded-cache-mb` budget across assembly
workers and avoids manager RPC for repeated feature rows. HM's hot `article`
rows and mostly streaming `transactions` rows perform better with `local`;
the ten-task validation runner selects it with a 4096 MiB total budget for HM.

Example:

```bash
PYTHONPATH=exps/relconnector_v2 .venv/bin/python -m relconnector_v2.benchmark \
  --dataset rel-f1 \
  --task driver-circuit-compete \
  --epochs 2 \
  --assembly-processes 4 \
  --training-order ready
```

See `PLAN.md` for the staged performance-validation matrix and `STATUS.md` for
the current implementation state.

### Inspect and compact an existing disk text cache

The optional offline tool screens for a fragmented immutable embedding layout:

```bash
PYTHONPATH=exps/relconnector_v2 .venv/bin/python -m relconnector_v2.features.compact_disk_text /path/to/cache.sqlite --plan
```

To build an independent compact cache, pass a **new directory** as the destination:

```bash
PYTHONPATH=exps/relconnector_v2 .venv/bin/python -m relconnector_v2.features.compact_disk_text /path/to/cache.sqlite /path/to/new-cache/text_embeddings
```

The source index and vectors are unchanged. The tool copies indexed vectors into
8,192-row shards, verifies every vector byte-for-byte, rebuilds the SQLite index,
and writes `compaction.json` when finished. Existing destinations are rejected;
an interrupted build retains an `INCOMPLETE` marker. Only use a completed output.
The new database retains the original filename and namespace; select its parent
cache root with `--cache-dir`. Initialization artifacts are separate and must
also be available there or be rebuilt normally. This is an offline layout
optimization, not an automatic migration or an approximation of embeddings.

The 2026-09-27 targeted experiments also tested Ratebeer with
`--encoded-cache-scope local --encoded-cache-mb 2048` (512 MiB per worker with
four assembly workers). The warm short-run comparison favored local placement;
the original feature-read path was retained because moving cache lookup before
SQL added less than 1% independently. This remains an explicit experiment
configuration; the full validation matrix's Ratebeer default is unchanged until
long-run cache behavior is validated. See
`../doc/v2-bottleneck-targeted-optimization-20260927.md` for protocols and results.
