# cache_baseline

冻结自 2026-09-16 正确性修复前的 `relconnector/`，包含节点去重、推荐分支合并、
最多四 batch feature window、raw block LRU 和 GloVe 整句 LRU。

本轮按用户授权回补正确性修复，详见 [修订报告](../../doc/cache-baseline-correctness-report.md)。
包内导入完全独立于 latest；benchmark 使用 `cache_online_training.py` / `cache_wrappers.py`。
不包含动态窗口、自适应准入或新增的编码并行算法。

```bash
../.venv/bin/python -m benchmark.online_runner --implementation cache \
  --dataset rel-f1 --task driver-circuit-compete --device cuda:0 \
  --output benchmarks/cache-corrected.jsonl
```

使用数据库旁的 `.stypes.json`；缺失或过期时先运行 `python -m data.initialize_stypes`。
缓存预算计量 Tensor payload；进程 RSS 仍包括 Python 对象、模型、索引及在途批次。
