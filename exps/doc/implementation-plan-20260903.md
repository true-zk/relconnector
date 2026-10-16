# 大规模关系数据库图训练迭代计划

本文档描述从当前 ReDeLEx 数据链路迭代到大规模流式图训练架构的实施步骤。
架构背景和方案讨论见
[architecture-discussion-20260903.md](architecture-discussion-20260903.md)。

## 1. 目标架构

```text
Database
  -> Connector-X Arrow stream
  -> offline graph/feature materialization

CSR/CSC graph index
  -> pyg-lib Sampler workers
  -> SamplePlan queue

mmap feature store
  -> Fetcher workers
  -> PreparedBatch queue

PreparedBatch
  -> GPU Trainer
  -> Metrics / Checkpoint
```

核心边界：

- `pyg-lib` 负责异构、多跳和时间邻居采样。
- Connector-X 负责数据库并行查询和 Arrow 流式读取。
- Python/PyTorch 负责调度、向量化编码、batch 组装和模型训练。
- 图索引和特征存储绑定到固定数据库 snapshot。
- Sampler、Fetcher 和 Trainer 通过显式数据契约解耦。

## 2. 迭代原则

- 不一次性重写当前链路。
- 每个阶段都提供独立、可运行、可回退的交付物。
- 先证明结果等价，再优化性能。
- 新旧路径通过配置开关切换。
- 每个阶段都有 correctness、memory 和 throughput 验收。
- 未经 profile 证明的热点不提前下沉到 C++。

建议长期保留三种运行模式：

```text
legacy:
  pandas + complete HeteroData + NeighborLoader

local_pipeline:
  mmap graph/features + local Sampler/Fetcher + Trainer

distributed_pipeline:
  partitioned graph/features + distributed Sampler/Fetcher + Trainer
```

## 3. 迭代 0：建立基线

### 目标

固定当前实现的行为、性能和实验结果，为后续重构提供 reference。

### 工作项

1. 选择最小 benchmark：
   - 一个静态节点预测任务。
   - 一个时间节点预测任务。
   - 一个包含多种 edge type 的数据集。
2. 固定：
   - 数据库 snapshot。
   - train/val/test split。
   - schema 和 task 定义。
   - 模型配置。
   - 随机种子。
3. 记录图统计：
   - 每种 node type 的节点数。
   - 每种 edge type 的边数。
   - dangling FK 数量。
   - 每种节点和边的时间范围。
4. 保存若干 reference batch：
   - seed nodes。
   - `n_id_dict`。
   - `edge_index_dict`。
   - target。
   - seed time。
5. 增加阶段耗时和资源指标：
   - 数据库读取时间。
   - 图构建时间。
   - 采样时间。
   - feature fetch 时间。
   - batch 组装时间。
   - H2D 时间。
   - GPU step 时间。
   - CPU/GPU 峰值内存。

### 交付物

```text
benchmark fixtures
reference sampled batches
reference metrics
pipeline profiling schema
```

### 验收条件

- legacy 模式在固定 seed 下可重复运行。
- reference batch 可以被测试代码自动比较。
- 指标能够区分数据读取、采样、fetch 和训练瓶颈。

## 4. 迭代 1：拆分接口但保持内存实现

### 目标

把 `NeighborLoader` 隐式完成的采样、特征过滤和 batch 组装拆成独立接口，同时继续使用
完整内存 `HeteroData`。

### 数据契约

```python
SamplePlan(
    snapshot_id,
    epoch_id,
    batch_id,
    seed_nodes,
    n_id_dict,
    edge_index_dict,
    num_sampled_nodes,
    num_sampled_edges,
    batch_size,
    seed_time=None,
)
```

```python
PreparedBatch(
    snapshot_id,
    epoch_id,
    batch_id,
    hetero_data,
)
```

### 工作项

1. 定义 `Sampler` protocol。
2. 定义 `FeatureFetcher` protocol。
3. 定义 `BatchAssembler` protocol。
4. 实现 `InMemorySampler`。
5. 实现 `InMemoryFeatureFetcher`。
6. 将 Trainer 改为只依赖 `PreparedBatch`。
7. 保持单进程同步执行。

### 运行路径

```text
complete HeteroData
  -> InMemorySampler
  -> SamplePlan
  -> InMemoryFeatureFetcher
  -> PreparedBatch
  -> existing model
```

### 验收条件

- 新旧路径使用相同 seed 时，seed nodes 完全一致。
- 节点集合、边集合、target 和时间约束一致。
- 现有 SAGE/DBFormer 不需要了解 Fetcher 实现。
- 固定 batch 上的模型输出和 loss 一致。

## 5. 迭代 2：接入 Connector-X Arrow Stream

### 目标

替换 `pandas.read_sql_query` 全表读取，建立受 batch size 控制的数据库流式入口。

### 工作项

1. 实现 Connector-X reader adapter。
2. 使用 `return_type="arrow_stream"`。
3. 支持显式 SQL 列投影：
   - PK。
   - FK。
   - feature columns。
   - time column。
4. 支持 query partition、手动 query list 和 batch size。
5. 实现 Arrow schema 到 ReDeLEx stype 的映射。
6. 覆盖以下类型：
   - nullable/non-nullable integer。
   - float。
   - boolean。
   - string/dictionary。
   - timestamp。
   - decimal。
   - list。
7. 明确 Arrow C Data buffer 的 ownership。
8. 增加空表、空 batch 和 schema drift 检查。

### 实施约束

- 不经过 pandas。
- 不将全部 RecordBatch 合并为一个 Arrow Table。
- 每个 RecordBatch 处理后及时释放。
- 数据库读取必须绑定固定 snapshot。

### 验收条件

- Connector-X 与 SQLAlchemy 路径的行数、列值和 null 语义一致。
- 峰值读取内存由 `batch_size` 控制。
- Arrow buffer 无 use-after-free 和内存泄漏。
- 读取吞吐不低于 legacy 路径。

## 6. 迭代 3：构建 Dense ID 和 CSR/CSC 图索引

### 目标

不加载完整 DataFrame，离线构建可由 pyg-lib 使用的图索引。

### 工作项

1. 为每种 node type 建立稳定 dense ID。
2. 持久化：

```text
raw PK -> dense node ID
```

3. 分块读取 FK 并生成 typed COO edge shards。
4. 对 COO shards 做外部排序。
5. 为每种 edge type 生成 CSR/CSC。
6. 生成反向关系索引。
7. 时间图索引按时间排序。
8. 设计 graph manifest：
   - snapshot ID。
   - schema version。
   - node counts。
   - edge counts。
   - dtypes。
   - file offsets。
   - checksums。
9. 实现 mmap graph index reader。

### 关键决策

- 单个 node type 节点数小于 `2^31` 时优先使用 `int32`。
- PK/FK 映射在数据库、DuckDB 或外部 merge join 中执行。
- 不使用全量 pandas merge。
- dangling FK 的语义与当前实现保持一致。

### 验收条件

- node/edge counts 与 legacy 图一致。
- PK/FK 随机抽查映射正确。
- 正向和反向索引相互一致。
- mmap 打开图索引时，RSS 不随完整边数等比例立即增长。
- 不读取节点特征也能执行邻居采样。

## 7. 迭代 4：构建特征存储

### 目标

将数据库列转换为可按 dense node ID 随机访问的 Torch 输入。

### Feature Manifest

```text
snapshot_id
encoding_version
node_type
column_name
stype
dtype
shape
shard row range
file path
normalization metadata
categorical vocabulary metadata
```

### 工作项

1. 只使用训练数据拟合：
   - numerical statistics。
   - missing value policy。
   - categorical vocabulary。
   - text encoder configuration。
2. 使用 Connector-X Arrow stream 分块读取。
3. Python/PyTorch 向量化编码：
   - Arrow numeric -> Torch tensor。
   - validity bitmap -> mask。
   - dictionary indices -> categorical IDs。
   - timestamp -> int64。
4. 文本提前转换为固定 embedding 或 token representation。
5. 将固定长度特征写入 mmap tensor shards。
6. 实现：

```python
MMapFeatureStore.get(node_type, node_ids)
```

7. 实现 `TensorFrame` batch 组装。
8. 保证 Arrow owner 在零拷贝 tensor 生命周期内存活。

### 验收条件

- 随机节点的逐列特征与 legacy 路径一致。
- 训练、验证和测试使用相同训练期编码规则。
- Fetcher RSS 随 batch 规模增长，不随全表规模增长。
- 重复 fetch 同一节点结果稳定。

## 8. 迭代 5：单进程端到端 MVP

### 目标

在没有 multiprocessing 的情况下，打通新的完整训练链路。

### 运行路径

```text
mmap CSR/CSC
  -> pyg-lib sampler
  -> SamplePlan
  -> mmap feature fetcher
  -> TensorFrame/HeteroData
  -> Trainer
```

### 工作项

1. 将 graph index adapter 接入 pyg-lib。
2. 支持：
   - homogeneous sampling。
   - heterogeneous sampling。
   - temporal sampling。
3. 从 task table 构造 epoch seed list。
4. Fetcher 附加 target、seed time 和 batch metadata。
5. 保证目标 node type 的 seed 位于 batch 前部。
6. 接通现有模型和 loss。
7. checkpoint 保存：
   - snapshot ID。
   - encoding version。
   - epoch。
   - seed cursor。
   - sampler RNG state。

### 验收条件

- 一个 epoch 内 seed 不重复、不遗漏。
- 时间任务不访问未来节点和边。
- 固定 PreparedBatch 上的新旧模型输出一致。
- 小数据集指标与 legacy 基线处于预期误差范围。
- 主机内存不再依赖完整节点特征规模。

完成本阶段后得到功能性 MVP。

## 9. 迭代 6：Epoch-Aware 异步流水线

### 目标

让采样、特征读取和 GPU 训练重叠。

### 消息协议

```text
BeginEpoch(epoch_id)
Batch(epoch_id, batch_id)
EndEpoch(epoch_id)
WorkerError(worker_id, epoch_id, error)
Shutdown
```

### 工作项

1. 启动单独 Sampler process。
2. 启动单独 Fetcher process。
3. 增加有界 `SamplePlan` queue。
4. 增加有界 `PreparedBatch` queue。
5. 实现 backpressure。
6. 实现 worker heartbeat、timeout 和优雅退出。
7. 使用稳定 RNG key：

```text
hash(base_seed, epoch_id, batch_id, hop, edge_type, node_id)
```

8. Trainer 在 `EndEpoch(e)` 前不得消费 epoch `e + 1`。
9. 记录 queue wait、queue occupancy 和 GPU idle。

### 验收条件

- 同步和异步模式的采样结果一致。
- queue 容量固定时内存存在确定上界。
- worker 异常不会导致永久阻塞。
- epoch、scheduler、validation 和 checkpoint 边界正确。
- GPU idle time 相比单进程模式下降。

这是第一版建议交付的完整 MVP 边界。

## 10. 迭代 7：多 Worker 与 Cache 优化

### 目标

在已有正确流水线的基础上，通过 profile 提升端到端 steps/s。

### Sampler 优化顺序

1. 调整单 worker 的 pyg-lib 内部线程数。
2. 增加 sampler workers。
3. 设置 CPU affinity。
4. 优化 CSR/CSC 数据布局。
5. 实施 NUMA-aware 图索引放置。

### Fetcher 优化顺序

1. 按 node type 合并请求。
2. node IDs 去重和排序。
3. 按 shard 批量 gather。
4. 使用 mmap 和 OS page cache。
5. 增加小型进程内 block LRU。
6. 预热高频节点对应的 block。
7. 增加 fetcher workers。
8. 引入 pinned memory buffer pool。
9. 跨 batch request coalescing。

### Cache 规则

允许缓存：

- numerical tensors。
- categorical IDs。
- timestamps。
- fixed text embeddings。
- feature blocks。

禁止缓存：

- GNN outputs。
- trainable categorical embeddings。
- row encoder outputs。

### 进程间通信

使用：

```text
shared/pinned tensor buffer pool
  + descriptor queue
```

descriptor 只包含：

```text
epoch_id
batch_id
buffer_slot
tensor shapes
node/edge type offsets
```

Trainer 消费完成后归还 buffer slot，避免 pickle 完整 `HeteroData`。

### 性能指标

每次只修改一个变量，并记录：

```text
sample plans / second
fetched nodes / second
fetch p50/p95/p99
cache hit rate
queue occupancy
CPU utilization
memory bandwidth
GPU utilization
training steps / second
```

### 验收条件

- worker 增加带来端到端 steps/s 提升。
- 没有明显线程池超卖。
- cache 命中率提升能够转化为 fetch 性能收益。
- 正确性和可复现性不受影响。

## 11. 迭代 8：按 Profile 下沉 C++ 热点

### 目标

只把已经确认的 Python/Torch 热点下沉。

候选操作：

- Arrow validity bitmap 解包。
- dictionary encoding。
- node ID sort/unique/restore。
- 多列 fused gather。
- mmap shard gather。
- pinned buffer 写入。

推荐暴露纯 tensor custom operator：

```python
features, masks = torch.ops.redelex.fetch_features(
    node_type,
    node_ids,
    feature_manifest,
)
```

以下控制逻辑保留在 Python：

- epoch/task 调度。
- worker 生命周期。
- queue/backpressure。
- 模型训练和评估。
- checkpoint 和指标上报。

验收条件：

- 下沉前有 profile 证明该路径占比显著。
- C++ 实现与 Python reference 有逐列一致性测试。
- ownership、dtype、ABI 和错误传播有明确契约。
- 优化反映在端到端 steps/s，而不只是 microbenchmark。

## 12. 迭代 9：评估训练时在线数据库 Fetch

### 默认选择

```text
Connector-X 离线或定期摄取
  -> mmap feature store
  -> 训练时本地 fetch
```

只有以下条件成立时才引入在线查询：

- 特征更新频率高，不能接受 snapshot 延迟。
- 本地持久化成本不可接受。
- 数据库可以稳定提供足够高的随机批量查询吞吐。

在线路径需要实现：

- 合并多个 SamplePlan 的 node IDs。
- 临时表 JOIN 或范围查询。
- 数据库连接数和并发 query 限制。
- snapshot isolation。
- 带 snapshot/version 的 cache key。
- 数据库限流时的 pipeline backpressure。

如果在线查询无法持续喂满 Trainer，应回退到 mmap feature store。

## 13. 迭代 10：多机分布式扩展

### 前置条件

只有单机流水线达到内存容量或吞吐上限后才进入本阶段。

### 工作项

- graph partition。
- node/edge owner mapping。
- remote feature fetch。
- 跨 shard 邻居采样。
- ghost node/cache。
- 全局 epoch seed partition。
- worker failure recovery。
- snapshot 一致性协议。

### 采样正确性

当一个节点的邻接表跨 shard 时，不能让每个 shard 独立采固定数量邻居。正确方案是：

- 一个节点的完整邻接表由单一 owner 采样；或
- 根据 shard degree 分配采样额度，再做无放回合并。

### 验收条件

- 分布式与单机 reference sampler 的采样分布一致。
- seed 在全局范围不重复、不遗漏。
- worker 数变化不改变确定性采样结果。
- 横向扩展带来端到端吞吐提升。

## 14. MVP 范围

第一版 MVP 停在迭代 6：

```text
Connector-X Arrow 流式摄取
CSR/CSC mmap 图索引
mmap tensor feature store
pyg-lib 单机采样
Python/Torch Fetcher
单 Sampler process
单 Fetcher process
单 GPU Trainer
epoch-aware bounded queues
```

MVP 不包含：

- 多机图分片。
- 训练时实时数据库随机查询。
- 复杂共享 LRU。
- Fetcher 全量 C++ 化。
- 动态图增量更新。

MVP 首先验证：

- 与 legacy 的结果一致性。
- 数据和采样的科学正确性。
- CPU/GPU 内存上界。
- 流水线能否稳定喂满 Trainer。

在 MVP 达标后，再依据 profile 选择多 worker、cache、C++ 下沉或分布式扩展。
