# Debug Session: vanilla-arxiv-slow
- **Status**: [OPEN]
- **Issue**: The formal vanilla rel-arxiv/paper-citation validation is much slower than the batch baseline under the same 2-epoch, 10,000-step cap.
- **Debug Server**: Pending
- **Log File**: `.dbg/trae-debug-log-vanilla-arxiv-slow.ndjson`

## Reproduction Steps
1. Run `benchmark.run_validation_matrix` with the formal validation configuration.
2. Wait for `vanilla rel-arxiv/paper-citation`.
3. Observe low GPU utilization and multi-hour runtime while the worker remains CPU-active.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Expected signal |
|----|------------|------------|--------|-----------------|
| A | Synchronous per-batch SQL feature reads and encoding starve the GPU | High | Low | High CPU time, low GPU duty cycle, feature operations dominate |
| B | Repeated GloVe text encoding dominates runtime | High | Medium | Text/model encode stacks or telemetry dominate feature assembly |
| C | Neighbor sampling dominates runtime | Medium | Low | Sampling operation time exceeds feature fetch and encode time |
| D | SQLite queries miss the node_id index and perform table scans | Medium | Low | EXPLAIN QUERY PLAN reports SCAN instead of primary-key/index search |
| E | The worker is stalled on a lock or dead loop | Low | Low | No CPU/I/O/progress changes across repeated samples |

## Log Evidence
Pending.

## Verification Conclusion
Pending.
