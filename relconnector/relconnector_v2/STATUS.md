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
- Process-shared encoded TensorFrame cache with a single index owner.
- PyTorch multiprocessing transport for prepared tensors.
- Independent benchmark CLI: `python -m relconnector_v2.benchmark`.

## Verified

- Strict-order 2-epoch synthetic execution.
- Ready-order 2-epoch execution with delayed early batches.
- Small in-flight limit and late epoch-count delivery.
- Text-cache reservation and visibility across spawned processes.
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
- Run all ten representative datasets for at least two epochs, followed by
  10/50-epoch steady-state validation.
