# Strict vanilla serial benchmark (2026-10-27)

The implementation is `baseline.vanilla_baseline.strict_benchmark`. The ten
established dataset/task pairs and their order are in
`baseline.vanilla_baseline.run_strict_matrix`. The live output directory is
`exps/benchmarks/strict-vanilla-20261027/`.

## Protocol

1. In a fresh process, prepare feature types/statistics and build one global
   feature-free CSC graph from the SQLite database. Persistent schema and graph
   initialization caches are disabled. The official `.stypes.json` file is
   treated as dataset metadata, not a computed cache.
2. For each epoch, stream task seeds in shuffled blocks. `max_batches=512`
   limits **each epoch**; smaller training sets finish sooner.
3. For each yielded batch, finish sampling, then read the sampled feature IDs
   using a direct `SELECT ... WHERE id IN (...)` for each sampled table,
   then encode and assemble features, then finish one GPU training step. Only
   then request the next batch. There is no application-level SQL block,
   text embedding, or encoded feature cache; there is no replay, window,
   prefetch, worker, or pipeline.

The seed stream itself reads task table rows in shuffled ranges to generate
training examples. The exact `IN` requirement applies to sampled **feature**
reads. SQLite and the operating system may still use their own page caches;
those are not controlled by this benchmark.

## Command

From the repository root:

```bash
PYTHONPATH=exps .venv/bin/python -m baseline.vanilla_baseline.run_strict_matrix \
  --output-dir exps/benchmarks/strict-vanilla-20261027 \
  --database-dir exps/data/relbench --device cuda:0 \
  --epochs 5 --max-batches 512 --batch-size 512 \
  --num-neighbors 128,128 --channels 128 --seed 42 \
  --torch-num-threads 1 --text-batch-size 256
```

The matrix runner executes one child process at a time, writes
`manifest.json` with source hashes and configuration, writes `status.json`,
and keeps a separate JSON result and log for each dataset. A completed
dataset's JSON has `status: "ok"` and five epoch records. Failed runs have
`status: "failed"` and a traceback; the matrix status is `incomplete` unless
all ten complete. `environment.json` records package versions, GPU names,
and hashes of the ten official `.stypes.json` metadata files.

## Measurement

`time.perf_counter` measures sampling, feature read plus assembly, and
training per batch. `torch.cuda.synchronize` separates CUDA stage boundaries.
`training_loop_s` includes seed generation and loop overhead; its component
timings therefore need not sum exactly to it. `prepare_s` includes all cold
preparation, while `prepare_breakdown_s.graph_build_s` isolates graph building.
`end_to_end_s` spans preparation and all epochs. Host peak is the process
maximum resident set size. GPU peaks are PyTorch's maximum allocated and
reserved memory on the selected device, including model and batch tensors;
they do not include memory allocated by unrelated processes or non-PyTorch
libraries.

## 2026-10-28 interruption and resume

The first Arxiv attempt reached epoch 1, batch 240/512, then its worker and
supervisor exited without a Python traceback. The underlying exit signal is
unknown. The detached recovery script could not resume because the original
runner compared JSON-loaded `cases` (lists) against Python tuples in its
manifest check. This was an orchestration error, not evidence that the Arxiv
batch itself failed. The partial Arxiv JSON/log and an incident record are
preserved in `interrupted-20261028/`.

`resume_supervisor.py` verifies every baseline Python source hash against the
original manifest, copies the exact source to `frozen-source/`, skips only the
four datasets whose JSON results have `status: "ok"`, and restarts Arxiv from
the beginning. It runs remaining datasets one at a time from the frozen
source. No partial training state is reused. Its status is written to the
same `status.json`; orchestration messages go to `resume-supervisor.log`.

On 2026-10-28, the root filesystem was found at 100% use with about 1.2 GiB
available, while `/home` had about 145 GiB available. This does not prove
the original Arxiv exit was caused by disk exhaustion: no system error or
exit signal was captured. The resumed supervisor now sets `TMPDIR` and
`SQLITE_TMPDIR` to its `tmp/` directory on `/home` for every child run.
The interrupted second Arxiv attempt was archived separately before this
change. The host reported ample available RAM and no cgroup OOM events at
the time of inspection.

## 2026-10-28 user amendment

At the user's request, Arxiv was stopped during its third epoch. Its two
completed epochs remain measured in `rel-arxiv.json`; the third epoch is
marked incomplete and excluded from completed-epoch timing. The original
Arxiv JSON and log at the moment of stopping are preserved in
`user-stopped-arxiv-20261028/`.

The remaining five datasets (`rel-salt`, `rel-stack`, `rel-amazon`,
`rel-ratebeer`, `rel-event`) now run **one epoch each**, sequentially, using
the same batch size, fanout, model, seed, exact SQL feature reads, and no
application cache. `amended-plan.json` records this change, and
`amended-summary.json` is updated after each dataset. The latter reports
the measured one-epoch time separately from three estimates:

- `projected_training_loop_5epoch_s = 5 × measured training_loop_s`;
- `requested_end_to_end_x5_s = 5 × measured end_to_end_s`;
- `prepare_once_plus_5epoch_s = measured prepare_s + 5 × measured training_loop_s`.

The requested end-to-end multiplication counts cold preparation five times,
so it is higher than the estimate that prepares once. Neither projected
five-epoch value is a measured five-epoch run. The amended matrix is complete
only when `status.json` says `complete_amended_plan` and all five new
one-epoch results have `status: "ok"`.

The amended plan completed on 2026-10-28 with no failed dataset. The final
`requested-five-epoch-report.json` and `.csv` apply the user's later request
to Arxiv as well: Arxiv uses its first complete epoch times five, while the
last five datasets use their only complete epoch times five. The first four
datasets retain their measured five-epoch times. The report preserves both
training-loop and end-to-end projections, with a separate one-preparation
estimate to show the effect of multiplying cold preparation five times.
