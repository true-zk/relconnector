# 大规模关系数据库图训练技术路线讨论记录

## 1. 背景与目标

当前 ReDeLEx 的主要数据路径是：

```text
SQL 数据库
  -> 全表读取为 pandas DataFrame
  -> RelBench Database/Table
  -> 完整 PyG HeteroData
  -> NeighborLoader 采样
  -> GNN 训练
```

该实现要求完整数据库、节点特征和图结构能够进入主机内存。本文讨论的目标场景是：

- 数据库规模过大，不能对每张表执行无条件的全列、全量读取。
- CPU 内存有限，但经过压缩的 PK/FK 图索引可以放入内存或 mmap。
- GPU 显存有限，必须采用子图 mini-batch 训练。
- 采样、特征读取和模型训练需要解耦并形成异步流水线。
- 在提高吞吐的同时，保证 epoch、随机采样、时间约束和数据切分的正确性。

## 2. 已确认的总体架构

系统分为三个核心阶段：

```text
Sampler
  -> SamplePlan
Fetcher
  -> PreparedBatch
Trainer
  -> Metrics / Checkpoint
```

技术职责划分如下：

### 2.1 pyg-lib

`pyg-lib` 负责图结构上的高性能采样：

- 基于 CSR/CSC 的邻居访问。
- 异构图按 edge type 的 fanout 采样。
- 多跳邻居采样。
- 时间图采样。
- 多 worker 并行采样。
- 输出节点 ID、局部边和每跳采样统计，不负责节点特征读取。

### 2.2 Connector-X

Connector-X 负责数据库读取：

- 数据库连接和类型解析。
- SQL query partition。
- 多线程并行读取。
- 流式 Arrow RecordBatch 输出。
- 离线构建节点特征存储。
- 必要时处理 Fetcher cache miss 的批量查询。

Connector-X 主体由 Rust 实现，`connectorx-cpp` 提供 C ABI。系统应使用
`arrow_stream`/RecordBatch iterator，不使用 pandas 作为中间格式。

### 2.3 Python + PyTorch

Python 作为控制面和胶水层：

- epoch 生命周期和 seed 调度。
- sampler/fetcher worker 管理。
- 有界队列、backpressure 和异常处理。
- Arrow 到 Torch 的向量化编码。
- 特征 cache 管理。
- `TensorFrame`/`HeteroData` 组装。
- CPU pinned memory 到 GPU 的异步传输。
- 模型 forward/backward、优化器、评估和 checkpoint。

Python 不应执行逐行、逐节点的数据处理。热点计算应由 PyTorch、PyArrow、
pyg-lib 或后续自定义 C++ operator 完成。

## 3. 离线数据准备

### 3.1 PK/FK 到图索引

每张表的一行对应一种 node type 下的一个节点。原始 PK 需要转换为连续的
`dense_node_id`，FK 则转换为目标表的 `dense_node_id`。

推荐使用两阶段处理：

```text
阶段 1：为每张表建立 raw PK -> dense_node_id 映射
阶段 2：分块读取 FK，通过映射生成每种 edge type 的边
```

映射和边构建不能依赖完整 pandas DataFrame merge。可选实现包括：

- 在数据库中通过临时映射表 JOIN。
- 使用 DuckDB/SQLite/分区 Parquet 保存映射。
- 对 PK/FK 排序后执行外部 merge join。

图索引应按 edge type 存储为 CSR/CSC。为支持双向消息传播，需要为关系建立
正向和反向索引。节点数允许时优先使用 `int32`，降低边索引内存占用。

### 3.2 节点特征

Connector-X 分块读取实际特征列，并以 Arrow RecordBatch 流形式输出。离线阶段完成：

- 数值缺失值处理和归一化。
- categorical 字符串到整数 ID 的映射。
- timestamp 到统一 int64 时间单位的转换。
- 文本 token 或固定文本 embedding 的预计算。
- 按 `(node_type, dense_node_id)` 写入特征 shard。

固定长度特征优先写入连续 mmap tensor。Parquet 适合离线扫描和交换，但不适合作为
高频随机行读取的唯一存储。

归一化统计、类别词表和缺失值规则只能由训练数据拟合，再应用到验证和测试数据。

## 4. 运行时流水线

### 4.1 Sampler 输出

Sampler 只处理图索引并输出轻量结构：

```python
SamplePlan(
    epoch_id,
    batch_id,
    seed_nodes,
    n_id_dict,          # 每种 node type 的全局 dense node IDs
    edge_index_dict,    # 当前子图内的局部边索引
    num_sampled_nodes,
    num_sampled_edges,
    batch_size,
    seed_time=None,
)
```

Sampler 不读取节点特征，也不创建最终 `TensorFrame`。

### 4.2 Fetcher 输出

Fetcher 接收 `SamplePlan`：

1. 按 node type 合并和去重 node IDs。
2. 按 shard 和 offset 排序，提高访问局部性。
3. 从 mmap/cache/数据库批量读取特征。
4. 恢复采样节点的局部顺序。
5. 应用必要的 dtype cast、null mask 和轻量在线转换。
6. 组装模型兼容的 `TensorFrame` 和 `HeteroData`。

```python
PreparedBatch(
    epoch_id,
    batch_id,
    hetero_data,
)
```

生成的 `HeteroData` 至少需要包含：

```text
batch[node_type].tf
batch[node_type].n_id
batch[node_type].batch_size
batch[node_type].y
batch[edge_type].edge_index
```

### 4.3 Trainer

Trainer 只消费 `PreparedBatch`：

```text
queue.get()
  -> non_blocking GPU copy
  -> forward
  -> loss
  -> backward
  -> optimizer.step
```

训练侧不感知数据库读取和图采样的具体实现。

## 5. Epoch 和随机性正确性

一个标准 epoch 应定义为：

- 对全部训练 seed nodes 做一次确定性 shuffle。
- 每个 seed 在该 epoch 中出现一次。
- 每个 epoch 重新随机采样 seed 的多跳邻居。

图索引只加载一次，不在 epoch 之间重建或清空。epoch 切换时只需要：

- 重置 seed iterator。
- 更新 epoch RNG。
- 等待当前 epoch 的 batch 全部消费完成。

队列协议需要显式包含：

```text
BeginEpoch(epoch_id)
Batch(epoch_id, batch_id)
...
EndEpoch(epoch_id)
```

Sampler 可以提前生成未来 epoch，但 Trainer 在收到当前 epoch 的 `EndEpoch` 之前，
不能消费下一 epoch 的 batch。

为消除 worker 调度对采样结果的影响，随机数应由稳定键派生：

```text
hash(base_seed, epoch_id, batch_id, hop, edge_type, node_id)
```

多 worker 输出可以按 `batch_id` 重排以保证完全可复现，也可以在 epoch 内乱序消费以
提高吞吐。后者保证数据分布正确，但不会产生完全相同的优化轨迹。

## 6. 多进程采样

单机多进程不需要复杂的分布式图算法，但必须保证：

- seed batch 不重复、不遗漏。
- 所有 worker 共享同一个只读图 snapshot。
- RNG 不依赖 worker 数和执行顺序。
- 时间采样只访问 `neighbor_time <= seed_time` 的邻居。
- 异构 fanout 按 edge type 正确执行。

真正多机分片时，需要额外解决全局均匀采样。不能简单地在每个 shard 各采 `k` 个邻居，
否则会改变采样分布。可选方案：

- 一个节点的完整邻接表由单一 owner 管理并负责采样。
- 根据各 shard 的邻居数量分配采样额度，再进行无放回合并。

采样通常是随机内存访问和内存带宽受限任务。增加进程数不保证线性提速，必须对
`1/2/4/8` workers 分别 benchmark。

## 7. Fetcher 并行与 Cache

### 7.1 Cache 内容

适合缓存：

- 编码后的数值 tensor。
- categorical IDs。
- timestamp。
- 固定文本 embedding。
- 已解析的特征 shard/page。

不适合缓存：

- GNN 输出 embedding。
- trainable categorical embedding。
- row encoder 输出。

后面三类结果依赖持续更新的模型参数，会快速过期。

### 7.2 Cache 组织

推荐以 block 为缓存单位：

```text
cache_key = (snapshot_id, node_type, feature_group, node_id // block_size)
```

第一版建议使用：

```text
mmap feature store
  + OS page cache
  + 小型进程内 LRU
  + 高频节点/特征块预热
```

不要一开始实现复杂的跨进程共享 LRU。共享 cache 会引入锁、引用计数和淘汰协调，
可能抵消收益。

Fetcher 可以合并多个 `SamplePlan` 的 node ID 请求，统一去重、排序和读取，再拆回各
batch，从而降低随机 IO。

## 8. Connector-X 与 Torch 编码边界

Connector-X 输出 Arrow RecordBatch，Python 使用 PyArrow 和 Torch 做向量化转换。

对于连续且无 null 的 `int32/int64/float32/float64`：

```text
Arrow buffer -> NumPy view -> torch.from_numpy
```

可以接近零拷贝。以下类型需要额外处理：

| Arrow 类型 | 处理 |
| --- | --- |
| nullable 数值 | values tensor + validity mask |
| boolean | Arrow bit-packed bitmap 解包 |
| string | dictionary encode 或 tokenize |
| dictionary | indices 转 categorical ID |
| timestamp | 转换为统一单位的 int64 |
| decimal | 转定点整数或 float |
| list | values + offsets tensors |

当 Torch tensor 直接引用 Arrow buffer 时，必须保持 Arrow RecordBatch 存活。跨进程时
不应 pickle Arrow Python 对象，而应将结果写入共享 tensor buffer。

Python/Torch 编码是第一版的推荐方案。只有 profiling 证明 nullable bitmap 解包、
dictionary encoding、多列 fused gather 或 TensorFrame 小对象创建成为主要瓶颈后，
再将对应算子下沉为 C++/PyTorch custom operator。

## 9. 进程通信和资源控制

不要通过普通 multiprocessing queue 传输完整 Python `HeteroData`。推荐：

```text
共享内存或 pinned tensor buffer pool
  + 只传 descriptor 的有界队列
```

descriptor 包含：

```text
epoch_id
batch_id
buffer_slot
tensor shapes
node/edge type offsets
```

Trainer 消费完成后归还 buffer slot。`SamplePlan` 和 `PreparedBatch` 队列只预取少量
batch，例如 2 到 4 个，通过 backpressure 限制内存。

CPU 资源需要统一规划：

```text
sampler workers * sampler threads
  + fetcher workers * fetcher threads
  <= 可用物理 CPU cores
```

需要避免 pyg-lib、Connector-X/Rayon、PyTorch/OpenMP 各自创建完整线程池导致超卖。
在多 NUMA 节点机器上，应为 worker 设置 CPU affinity，并尽量让图索引、feature shard
和 worker 位于同一 NUMA node。

## 10. 数据一致性和评估

- 整个训练运行必须绑定到固定数据库 snapshot/version。
- target 列不能进入节点特征。
- 静态任务需要明确 transductive 或 inductive 边界。
- 时间任务必须过滤未来节点和未来边。
- 训练、验证、测试使用一致的 ID 映射和特征编码规则。
- 验证和测试应使用固定采样种子、固定采样子图，或全邻居采样。
- checkpoint 需要记录 epoch、seed cursor、采样 RNG 和数据 snapshot ID。

## 11. 推荐实施顺序

1. Connector-X `arrow_stream` 替换 pandas 全表读取。
2. 实现 PK/FK dense ID 映射和 CSR/CSC 离线构建。
3. 实现 Arrow -> Torch 向量化编码及 mmap feature shard。
4. 使用 pyg-lib 实现单 worker `SamplePlan`。
5. 实现 Python Fetcher 和模型兼容的 `PreparedBatch`。
6. 接通单 sampler、单 fetcher、单 trainer 的端到端链路。
7. 增加有界预取、shared/pinned buffer pool。
8. 分别扩展 sampler/fetcher workers 并进行吞吐 benchmark。
9. 根据 profile 决定是否将 Fetcher 热点下沉到 C++。
10. 最后再评估多机图分片和分布式采样。

## 12. 核心结论

最终确认的技术路线是：

```text
pyg-lib 负责采样
Connector-X 负责数据库并行流式读取
Python/PyTorch 负责调度、向量化编码、batch 组装和模型训练
```

图索引常驻内存或 mmap，Sampler 和 Fetcher 独立并行，通过 epoch-aware 的有界流水线
向 Trainer 提供 batch。性能优化的重点依次是数据布局、批量访问、局部性、并行度和
零/少拷贝，而不是把所有逻辑一次性下沉到 C++。
