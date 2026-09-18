# Frozen baselines

- `batch_baseline/`: eager training that materializes all TensorFrames before the loop.
- `vanilla_baseline/`: the previous online SQL implementation, frozen before feature
  deduplication and the implicit-rowid lookup fix.

These packages must not import the current `relconnector` package. Shared benchmark
instrumentation uses version-specific adapters so later contract changes cannot alter a
frozen baseline. Before each new optimization round, snapshot the current
`relconnector/` into a newly named baseline directory, then continue work only in
`relconnector/`.

## 正确性修订与 cache_baseline

`cache_baseline/` 保存本轮之前的节点去重、四 batch 合并窗口及有界 raw/text cache
策略，可通过 `benchmark.online_runner --implementation cache` 运行。它不导入 latest，
有独立测量 adapter；四版本仅共享独立的 `relbench_compat` 数据正确性契约。

用户已授权向所有基线回补 bug。`relbench-correctness-v1` 包含目标隐藏、官方 stype
artifact、rowid 索引、逻辑 dtype 恢复、默认路径、缓存 storage 和 SQL 统计口径修复。
vanilla 仍逐批编码；batch 仍全量物化；cache 仍使用原有静态窗口和 LRU。

修复前的源码在 `archives/pre-correctness-20260916.tar.gz`，对应 SHA256 文件逐项记录
原始源码。catalog 修复前 metadata 也已单独备份。历史 JSONL 保留；新结果包含
`experiment.correctness_version`，比较工具会拒绝跨正确性版本的严格比较。
