# Recovery Validation

This records short correctness checks for the four-directory refactor, not a
full performance study. Existing SQLite files and historical benchmark outputs
were left intact. No full 64-task experiment was restarted.

## Checks

- Ruff check and format check: passed.
- Pyright: 0 errors, 0 warnings.
- compileall for data, baseline, relconnector, benchmark and tests: passed.
- tests/: 28 assertions-based test cases passed.
- data/tests/: 4 test cases passed, including pandas and Connector-X round trips.
- Wheel built and inspected: only baseline, benchmark, relconnector and dist-info;
  no SQLite, JSONL, offline data tools, or bytecode.
- Isolated wheel imports outside the source directory: passed. Connector and
  telemetry imports do not import torch or the training implementations.

## CPU Training Smoke Tests

Earlier recovery smoke tests used truncated epochs. Strict parity was then tested
with batch_size=8, fanout=(2,2), channels=16, sync execution and one PyTorch
CPU thread.

| Implementation | Reader / Executor | Task | Batches |
| --- | --- | --- | --- |
| baseline | pandas / eager | rel-f1/driver-dnf | 4 |
| baseline | pandas / eager | rel-f1/driver-circuit-compete | 4 |
| online | pandas / async | rel-f1/driver-dnf | 4 |
| online | pandas / async | rel-f1/driver-circuit-compete | 4 |
| online | connector-x / sync | rel-f1/driver-dnf | 4 |

All used GloVe, including the local cached model on the online path. Outputs are
under benchmarks/baseline-recovery-temporal.jsonl,
benchmarks/online-recovery-temporal.jsonl and
benchmarks/online-final-connectorx-*.jsonl. The initial
online-recovery-entity.jsonl predates the timestamp-unit correction; do not use
it as the final correctness or performance reference.

The current strict outputs are benchmarks/baseline-strict-parity-v4.jsonl and
benchmarks/online-strict-parity-v4.jsonl for entity training, plus the matching strict-link-v4b files for recommendation. Both pairs have
identical feature
schema fingerprints, model/sampling metadata, completed work and exact loss. The
comparison CLI reports strictly-comparable. Earlier recovery files used the
pre-alignment encoder and must not be mixed with v4 results. These smoke runs
are correctness checks, not stable performance measurements.

## Important Runtime Finding

pandas 3 can infer datetime64[us]. RelBench 3.0.1 to_unix_time interprets the
underlying integer as nanoseconds, yielding timestamps 1000 times too small.
Both implementations now normalize node and task times to nanoseconds before
conversion. A temporal fixture with out-of-order IDs verifies CSC ordering and
excludes future neighbors; seed tests check actual UNIX seconds.

## Environment Caveats

The sandbox reports a restricted /proc/<pid>/task/<tid>/comm access during some
PyTorch process teardowns. Tests print OK and the training workers publish
successful 4-batch results, but the outer terminal tool can return nonzero.
This is not represented as a clean overall process exit. CUDA is unavailable
in this sandbox, so GPU memory and throughput remain unverified.

Queue limits cover queued payloads, not total RSS. The current threaded RNG
guard serializes sampling with training; feature preparation still overlaps.
Full-dataset epochs, multi-process sampling and production GPU performance
experiments remain separate follow-up work.
