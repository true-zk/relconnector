# relconnector_v2

Independent implementation of the three-stage relational training pipeline:

```text
sample process -> assembly process pool -> GPU trainer
```

The package vendors the required connector, feature, graph, task, sampling, and
training code. It does not import the original `relconnector` package.

Use `PYTHONPATH=relconnector_v2` when running directly from the repository.
Runtime artifacts default to `.relconnector-v2-cache` and do not share the
original implementation's cache namespace.

Example:

```bash
PYTHONPATH=relconnector_v2 python -m relconnector_v2.benchmark \
  --dataset rel-f1 \
  --task driver-circuit-compete \
  --epochs 2 \
  --assembly-processes 4 \
  --training-order ready
```

See `PLAN.md` for the staged performance-validation matrix and `STATUS.md` for
the current implementation state.
