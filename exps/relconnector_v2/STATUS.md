# relconnector_v2 status

## Implemented

- Independent package namespace and vendored compatibility helpers.
- Separate `.relconnector-v2-cache` initialization-cache namespace.
- Spawn-based three-stage runtime:
  - seed reader and sampler process;
  - configurable Assembly process pool;
  - Trainer in the parent/GPU process.
- Byte-bounded seed, plan, and ready queues.
- Bounded in-flight windows and cross-epoch sampling/assembly prefetch.
- Strict FIFO and ready-order training modes with an epoch gate.
- Pre-spawned worker permits that scale up on ready starvation and scale down
  when the ready queue remains above its high watermark.
- Assembly workers own SQLite readers, raw caches, GloVe, and TensorFrame
  conversion; Pandas frames and raw strings never cross process boundaries.
- Shared-memory text embedding cache with miss reservation and publication.
- Batched shared text publication/gathers and zero-copy scalar metadata views;
  hashing and immutable-row gathers run outside the process lock. Process-local
  views are rebuilt after spawn, and lock acquisition has a timeout.
- Benchmark progress logging (30 seconds by default) and seed-reader exception
  propagation, preventing a stopped feeder from leaving the trainer waiting.
- Workload-driven local indexes for immutable shared text slots: bounded
  metadata, measured lookup cost/reuse, hysteresis and cooldown. Repeated
  accesses bypass the shared lock without copying the cache or duplicating
  encoding. Every dataset uses the same default text policy.
- Adaptive persistent text embedding cache backed by SQLite and float32 vector
  shards. Large, low-reuse workloads can reuse embeddings across batches,
  epochs, workers, and benchmark runs; small databases skip disk caching.
  A 1,024-shard LRU bounds open mappings and descriptors; eviction preserves
  persistent data and returned tensor copies. Fixed-size append shards reduce
  file creation, and TRUNCATE checkpoints recycle the WAL at a threshold adapted
  to index size, free disk space, and sustained disk-hit/WAL-growth observations.
  Read-heavy periods cap the WAL budget at 512 MiB; renewed misses or high
  shared WAL growth restore the write budget. Contention and peaks are measured.
  Worker errors use synchronous
  transport so mapping exhaustion cannot hide an error behind a failed thread.
- Completed training waits for all worker diagnostics (bounded by a 300-second
  timeout); missing reports raise instead of silently discarding cache metrics.
- Process-shared encoded TensorFrame cache with a single index owner. All
  admitted tensor storage is shared before publication, preventing RPC
  serialization from relocating storage while another request reads it.
- Optional CPU categorical bounds checks and preserved failed CPU inputs;
  training errors include the exact epoch, batch key, and completed count.
- PyTorch multiprocessing transport for prepared tensors.
- Independent benchmark CLI: `python -m relconnector_v2.benchmark`.

## Verified

- All ten GPU benchmark tasks completed two epochs, capped at 10,000 batches
  per epoch (76,186 batches total). The selected successful runs total
  46,242.51 seconds, versus 78,751.52 seconds for historical v1 scaled to
  matching per-task batch counts: 1.703x overall. Event/Trial remain slower;
  cache warmth and code revisions differ, and Amazon worker counters are
  unavailable. See [the full report](../doc/v2-full-matrix-20260925-final.md).
- Ratebeer completed 20,000 real GPU batches after the shared-storage fix,
  with all four worker reports and CPU categorical checks. The coordinated
  race reproducer fails before the fix and passes after it; four cache tests
  and a separate 128-batch GPU replay also pass.
- Strict-order 2-epoch synthetic execution.
- Ready-order 2-epoch execution with delayed early batches.
- Small in-flight limit and late epoch-count delivery.
- Text-cache reservation and visibility across spawned processes.
- Adaptive text-index activation, low-reuse fallback/recovery, metadata bounds,
  safe result copies, and unchanged model-input counts across reuse.
- Persistent text-cache reuse across embedder instances and late worker
  activation after another worker writes the cache.
- Encoded TensorFrame row visibility across spawned processes.
- No imports from `relconnector` or `relbench_compat`.
- Ruff, Pyright, compileall, and diff checks.
- Real F1 CPU smoke: 2 processes, 2 epochs x 2 batches, ready-order, shared
  caches, and cross-epoch prefetch completed end to end in 42.65 seconds.
- Matching 1-process smoke completed in 41.23 seconds with exactly the same
  mean loss (`0.6318540200591087`). The 3.4% small-workload regression confirms
  that process scaling must remain adaptive and feature-bound only.
- The F1 smoke shared text cache recorded 875 hits, 1,430 misses, and 246
  reservation waits. Those waits represent duplicate cross-process encoding
  work that was suppressed.

## Remaining performance work

- Measure Assembly scaling at 1/2/4/8 processes on Arxiv, Salt, and Ratebeer.
- Replace the manager-backed encoded-cache index if RPC exceeds 10% of
  Assembly time.
- Add direct stype payload production to bypass generic DataFrame conversion.
- Optional future 10/50-epoch steady-state validation beyond the completed
  two-epoch, 10,000-batch-per-epoch experiment.
