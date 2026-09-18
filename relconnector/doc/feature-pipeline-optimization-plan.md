# Feature Pipeline Optimization Plan

## 目标

在保持训练语义、采样结果和内存上界不变的前提下，修复隐式 SQLite `rowid`
查询退化，并减少在线特征读取和 TensorFrame/GloVe 编码的重复工作。

本轮只改性能路径，不混入 target column masking 等训练语义变更，保证实验归因单一。
磁盘持久化 dense embedding 暂缓。

## 版本与目录

```text
baseline/
  batch_baseline/    全量物化训练基线
  vanilla_baseline/  上一版在线 SQL 实现，冻结
relconnector/        当前最新优化实现
```

后续每轮优化前，将当前 `relconnector` 复制成新的命名 baseline，再只修改
`relconnector`。benchmark 必须显式记录 implementation 名称。

## 不变量

1. 每个 epoch 的任务行不重不漏，`BatchKey(epoch, batch)` 顺序保持 FIFO。
2. SamplePlan 的拓扑、节点局部顺序、重复节点、label 和 seed time 不变。
3. 只去重和缓存静态特征，不缓存 trainable HeteroEncoder/GNN 输出。
4. 时间相关表示继续按每个 batch 的 seed time 计算。
5. queue、window、raw cache 和 text cache 均由显式 byte/item budget 限制。
6. `vanilla_baseline` 可独立导入和运行，不引用最新 `relconnector`。

## Phase 1: SQL 与 node ID 契约

### 短期修复

无显式主键表仍向采样器暴露零基 node ID，但 SQL lookup 改为：

```sql
SELECT rowid - 1 AS __node_id__, ...
FROM table
WHERE rowid IN (...)
ORDER BY rowid
```

禁止在 `WHERE` 和 `ORDER BY` 中使用 `rowid - 1`，确保 SQLite 使用
`INTEGER PRIMARY KEY` lookup 而不是全表扫描和临时排序。

### 长期契约

新数据库中的无自然主键 data table 自动增加：

```sql
__relconnector_node_id__ INTEGER PRIMARY KEY
```

值严格为 `0..N-1`。已有数据库由离线工具事务性重建；迁移前验证旧 rowid
是连续 `1..N`，否则拒绝迁移，避免改变既有图节点映射。

验收：

- catalog 的 `primary_key` 和 columns 同步更新；
- `PRAGMA foreign_key_check` 为空；
- `EXPLAIN QUERY PLAN` 包含 `SEARCH ... USING INTEGER PRIMARY KEY`；
- 原始数据库保留，用副本执行迁移，避免破坏 vanilla 对照。

## Phase 2: 可观测性

每次 latest run 输出：

- requested node occurrences、unique rows、duplicate factor；
- queried rows、dense/sparse query count、SQL amplification；
- raw block cache hit/miss、row coverage、eviction、current/peak bytes；
- text occurrences、per-call unique count、cache hit/miss、model inputs、eviction；
- feature window count、mean/max batches；
- seed/plan/ready queue put/get wait、peak items 和 peak bytes；
- timeout progress snapshot：最后完成的 `BatchKey`、loss 和累计 fetch 指标。

SQL amplification 定义为：

```text
queried_rows / (unique_rows - cache_served_rows)
```

这样 cache 命中不会把查询放大率错误地压到 1 以下。

## Phase 3: 特征编码优化

### 3.1 batch 内唯一节点

对每个 table：

```text
node_ids = [5, 2, 5, 7]
unique   = [2, 5, 7]
inverse  = [1, 0, 1, 2]
```

SQL 和 TensorFrame 只处理 `unique`，最终用 `encoded_unique[inverse]` 恢复原局部顺序。
推荐任务的 source、positive、negative 三个分支联合去重。

### 3.2 有界跨 batch 合并

feature worker 阻塞获取第一个 SamplePlan 后，从 plan queue 非阻塞 drain 最多 K 个计划，
同时受 `feature_window_batches` 和 `feature_window_bytes` 限制。窗口内按 table 合并 ID，
统一 fetch/encode，再按原 `BatchKey` 顺序放入 ready queue。

### 3.3 frozen text embedding cache

使用 byte-bounded LRU 缓存：

```text
normalized text -> frozen GloVe vector
```

每次调用先去重，只将 cache miss 送入 sentence transformer。缓存只保存冻结文本向量，
不保存 trainable feature encoder 或 GNN 的输出。

## 实验顺序

1. 单元测试验证 rowid、迁移、inverse 恢复、推荐三分支、window FIFO、cache budget。
2. Ruff、Pyright、compileall 和完整测试。
3. 在原始数据库上做 vanilla/latest paired smoke，要求 loss 完全一致。
4. 在迁移副本上验证显式 node ID lookup。
5. 选编码瓶颈任务做 GPU 诊断实验，再决定是否扩大到 10 个正式任务。

## 当前状态

- [x] 冻结 `batch_baseline` 和 `vanilla_baseline`，latest 从 vanilla 复制后独立修改。
- [x] 修复隐式 rowid lookup。
- [x] 新 writer 自动创建显式 node ID。
- [x] 实现离线迁移工具，并生成、VACUUM `data/relbench-explicit-node-id/` 副本；10 个库无 keyless data table，稳定占用 33GB。
- [x] 增加 fetch、cache、text、window、queue 和 timeout progress 指标。
- [x] 实现 batch 内、推荐三分支和跨 batch 窗口去重。
- [x] 实现 byte-bounded frozen GloVe LRU。
- [x] 增加 `--implementation vanilla|latest` 配对 runner。
- [x] 通过静态检查与单元测试。
- [x] rel-f1 六任务、每任务两 batch 的 CPU paired smoke loss 完全一致；累计 batch assemble 从 5.172s 降至 1.050s（4.92x）。
- [x] 在 Salt 和 Stack 上执行正式 GPU 诊断实验；端到端分别加速 4.99x 和
  1.22x，结果见
  [gpu-feature-optimization-experiment.md](gpu-feature-optimization-experiment.md)。
