# 三阶段分离多进程流水线优化计划

## 1. 目标

将完整训练过程视为一个跨 epoch 的数据供应系统，拆分为三个独立执行节点：

1. Sample Node：持续产生采样子图。
2. Assembly Pool：多进程完成 SQL 读取、特征转换、文本编码和 batch 组装。
3. Train Node：持续消费已完成 batch，尽量保持 GPU 满载。

优化目标不是单独缩短某个函数，而是提高端到端稳态吞吐，减少 GPU 因等待
batch 而产生的空闲时间。缓存、预读和 assembly 结果允许跨 epoch 保留；epoch
只约束训练语义和指标归属，不再作为数据准备边界。

本计划只修改 latest，`batch_baseline`、`vanilla_baseline` 和
`cache_baseline` 继续冻结。

## 2. 当前证据

现有遥测已经证明 Assembly 是大型文本任务的关键路径：

| workload | steps | wall time | batch assembly | train step | ready queue 等待 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Arxiv latest 完整 1 epoch | 1044 | 2855s | 2746s | 290s | 2516s |
| Salt latest 2x50 | 100 | 141s | 261 CPU-s | 42s | 99s |
| Ratebeer latest 2x20 | 40 | 53s | 81 CPU-s | 14s | 39s |

异步阶段会重叠，因此 operation time 不能直接相加；但 ready queue 等待接近
wall time，同时 plan queue 长时间阻塞，已经构成明确的生产者快、Assembly 慢、
GPU 饥饿证据。

Salt 的细粒度数据进一步表明，优化后不只 GloVe 模型本身耗时：

- `frame_convert`: 132.3s；
- 其中 `map_text_embedded`: 121.9s；
- `text_cache_lookup`: 47.2s；
- `frame_gather`: 42.9s；
- Direct GloVe tokenize/lookup/pool 合计约 5.5s。

这些计时存在包含关系，不能求和，但说明应同时处理多进程扩展、TensorFrame
转换和数据搬运，而不是继续只优化 GloVe 调用。

## 3. 设计原则

1. 三阶段只通过有界队列和共享内存描述符通信。
2. GPU 进程是唯一创建 CUDA context 的进程。
3. Assembly 进程只使用 CPU，每进程默认 `torch_num_threads=1`。
4. 不跨进程传输 Pandas DataFrame 或原始长文本。
5. 大 tensor 放入共享 slab，队列只传 ID、shape、offset 和生命周期 token。
6. 缓存生命周期覆盖整个训练 run，不在 epoch 边界清空。
7. 训练默认不跨 epoch，但允许下一 epoch 提前采样和组装。
8. 提供 strict-order 和 throughput-order 两种模式，先隔离性能变化，再验证
   非严格顺序的训练效果。
9. 所有队列以 retained bytes 而不是 batch 数量作为硬预算。
10. 调度依据实时供需反馈，不以固定 worker 数和固定 look-ahead 作为最终策略。

## 4. 目标架构

```text
Seed source
    |
    v
Sample process
    |
    | SamplePlanDescriptor(epoch, batch, topology handles)
    v
Bounded plan queue
    |
    v
Coordinator / dispatcher
    |
    +----> Assembly process 0 ----+
    +----> Assembly process 1 ----+--> completion queue
    +----> Assembly process N ----+
                                      |
                                      v
                                Ready batch pool
                                      |
                                      v
                              GPU trainer process
```

Assembly process 内部执行：

```text
SamplePlan
  -> shared encoded-cache lookup/reservation
  -> worker-local SQLite read for misses
  -> direct feature mapping / Direct GloVe
  -> publish encoded rows to shared slabs
  -> gather batch feature payload
  -> publish PreparedBatchDescriptor
```

初版不保留中央 SQL fetch 阶段。让 Assembly 进程直接读取 SQLite，可以避免把
包含长字符串的 DataFrame pickle 到其他进程。SQLite 文件通过独立只读连接访问，
操作系统 page cache 仍由所有进程共享。

## 5. 共享缓存

### 5.1 Encoded node cache

主要 key：

```text
(database_version, schema_fingerprint, table, node_id)
```

epoch 不进入 key。只要数据库版本、schema、隐藏列和编码规则不变，同一个节点
在 10 或 50 个 epoch 中都可以复用。

缓存拆为三层：

1. Index：key 到 `(slab_id, row, generation)` 的映射。
2. Data slab：按 stype 保存共享 tensor。
3. State：`MISSING -> RESERVED -> READY`，用于避免多个进程重复编码同一 miss。

第一版由 coordinator 批量处理 index 请求。每个 Assembly worker 一次提交一组
node IDs，避免逐节点 IPC。worker 获得 hit slots 和 reserved misses，编码完成后
批量 publish。

后续如果 coordinator 成为瓶颈，再按 `(table, hash(node_id) % shards)` 分片；
不在第一版引入分布式索引复杂度。

### 5.2 Text cache

优先保证 encoded node cache 跨进程共享，因为它能同时跳过 SQL 解码、全部列
mapper 和 GloVe。文本内容缓存作为二级优化：

- 固定宽 300 维向量放共享 slab；
- normalized text 索引可先按 worker 分片；
- 只有在 encoded cache miss 较高且文本跨节点重复明显时，才升级为全局共享索引；
- 不恢复硬盘 embedding cache。

### 5.3 生命周期和淘汰

- 缓存预算对整个进程池全局生效，不按 worker 复制；
- slot 带 generation，防止淘汰后旧 descriptor 误读；
- ready batch 引用的 slot 必须 pin；
- Trainer 完成 H2D 后发送 release token；
- admission 统计跨 epoch 延续；
- epoch 边界只记录快照，可以做频率衰减，但不清空数据；
- 剩余 epoch 越多，允许更积极地准入首轮出现的节点。

## 6. 跨 epoch 预取

当前“窗口不跨 epoch”限制调整为：

1. Sample 和 Assembly 可以提前进入后续 epoch。
2. 同一个 assembly window 可以包含相邻 epoch 的计划，只要所有输入特征是
   不随训练状态变化的纯函数。
3. batch 始终保留原始 `(epoch, batch)` key。
4. Trainer 在默认模式下完成 epoch N 后才消费 epoch N+1。
5. next-epoch ready 数据受独立 byte budget 限制，不能挤占当前 epoch 的供应。
6. 学习率调度器、early stopping 和 epoch-end hook 仍在 Train Node 串行执行。

这样可以在 epoch N 尾部利用 CPU 空闲预热 epoch N+1，同时让 epoch 0 建立的
encoded cache、text cache 和 raw-page locality 服务全部后续 epoch。

长训练下重点报告：

- epoch 0 冷启动时间；
- epoch 1 稳态时间；
- epoch 2+ 稳态均值和方差；
- 各 epoch cache hit rate；
- 跨 epoch 预取命中率和被丢弃的预取字节。

## 7. TensorFrame 路径

TensorFrame 对象只是 `feat_dict + col_names_dict` 容器。当前真正昂贵的是
DataFrame 到 stype tensor 的 converter、文本 mapper、cache expand 和 gather。

分两步处理：

### Phase A：保持语义，只移出关键线程

- Assembly 进程继续使用现有 TensorFrame converter；
- TensorFrame 内部 tensor 写入 shared memory；
- completion queue 只传 tensor descriptor；
- Trainer 重建轻量 TensorFrame view；
- RelBench `HeteroEncoder` 完全不变。

这一步用于单独证明多进程和通信掩盖的收益。

### Phase B：直接构建 stype payload

新增内部 `EncodedFeaturePayload`：

```text
feat_dict
col_names_dict
num_rows
shared-memory ownership tokens
```

Assembly worker 直接写 numerical、categorical、timestamp、embedding 等 stype
tensor，不再经过通用 DataFrame converter。Trainer 在调用 HeteroEncoder 前做
O(stype count) 的 TensorFrame 包装。

只有在该包装本身被证明仍有显著开销时，才考虑让模型直接接受 `feat_dict`。
第一阶段不替换 RelBench HeteroEncoder，避免无必要的模型语义偏移。

## 8. 非严格训练顺序

单个 epoch 内的样本本来已经 shuffle，采样和特征生产不依赖模型参数，因此
不必因某个慢窗口阻塞所有已完成 batch。

提供两种模式：

1. `strict`：按原始 BatchKey FIFO，作为 correctness 和性能归因基准。
2. `ready`：从已完成池消费，不等待缺失的早期窗口。

`ready` 模式不能简单永久采用完成先后顺序，否则短文本或小子图会系统性提前。
计划采用有界 ready pool：

- 在当前 epoch 的已完成 batch 中选择；
- 使用独立、可记录的 scheduler RNG；
- 设置最大 reorder lag，避免单个 batch 无限延迟；
- 记录实际消费顺序用于复现；
- 每个 batch 内部随机数继续由 BatchKey 派生。

改变 SGD 更新顺序会改变精确 loss 轨迹。验收目标从逐 step 完全一致调整为：

- feature tensor 与 strict 模式逐值一致；
- 相同 batch 集合，无遗漏和重复；
- 3 个训练 seed 的最终指标均值和方差无显著退化；
- 仍保留 strict 模式供严格回归使用。

Trainer 不跨 epoch 乱序，避免破坏 epoch scheduler、early stopping 和指标语义。

## 9. 反馈式调度

预先创建固定上限的 Assembly 进程，通过 permit 控制 active worker 数，避免频繁
spawn。调度器每个控制周期读取：

- sampler、assembly、trainer 的 service rate；
- plan/ready queue 当前字节和高低水位；
- queue put/get wait time；
- GPU utilization 和 GPU idle time；
- 每个 worker 的任务时长、CPU 和 RSS；
- shared cache hit/miss、reservation wait；
- IPC bytes、共享 slab 分配和回收延迟。

基础控制规则：

| 状态 | 动作 |
| --- | --- |
| plan 高水位、ready 低水位、GPU 等待 | 增加 active Assembly worker |
| plan 低水位、Assembly 等待 | 提高 sampler 并行度或减小 Assembly worker |
| ready 高水位、GPU 满载 | 限制 Assembly，避免无效占用内存 |
| RSS 接近预算 | 缩小 look-ahead、next-epoch budget 和 cache admission |
| cache hit 高、miss 少 | 减少 worker，避免调度和 gather 竞争 |
| 单窗口尾延迟高 | 拆分窗口或启用 work stealing |

第一版只实现离散级别 `1/2/4/N` worker 和高低水位控制，不实现复杂 PID 控制器。

## 10. 通信与故障语义

- 使用 `multiprocessing` 的 `spawn` context，禁止 CUDA 初始化后的 `fork`；
- SamplePlan 中的 topology tensor 使用 shared memory；
- SQLite、Pandas 和原始字符串只存在于 Assembly worker；
- Prepared feature tensor 使用 shared slab；
- queue 只发送小型 descriptor；
- worker heartbeat 和 error queue 单独存在；
- 任一 worker 失败时，coordinator 取消未完成 reservation；
- Trainer 失败时广播 cancel，并回收所有 pinned slot；
- 所有 queue、worker、slab 都必须支持确定性关闭；
- 不允许 orphan process 和泄漏的 `/dev/shm` segment。

## 11. 实施阶段

### Stage 0：冻结基线和补齐遥测

- 等当前正式矩阵完成，或在独立 worktree 中开发；
- 记录 current latest 的 queue wait、GPU idle、assembly service rate；
- 为 IPC bytes、shared-memory allocation 和 cache reservation 增加 telemetry；
- 不改变训练行为。

### Stage 1：多进程 Assembly MVP

- 新增 process runtime backend；
- SamplePlan 直接发送给 Assembly process；
- 每个 worker 独立 SQLite 只读连接和 GloVe 实例；
- 保留 strict FIFO；
- 暂用 worker-local cache，短实验只用于测量 1/2/4/8 进程扩展曲线；
- 记录序列化、IPC、进程启动和 worker RSS。

Go/No-Go：Arxiv 4 worker 相比 1 worker assembly throughput 至少提升 1.8x；
否则先解决 IPC/TensorFrame payload，再继续共享缓存。

### Stage 2：共享 encoded cache

- 建立 shared stype slabs；
- coordinator 批量 lookup/reserve/publish；
- 实现 generation、pin/release 和 byte-bounded eviction；
- 跨 epoch 保持 cache；
- 对比 local-cache、shared-cache 和 no-cache。

### Stage 3：轻量 TensorFrame payload

- worker 直接构建 `EncodedFeaturePayload`；
- Trainer 只重建轻量 TensorFrame view；
- 对 Salt 的 frame_convert、map_text 和 frame_gather 做前后对比；
- 保持 HeteroEncoder 和最终 feature tensor 不变。

### Stage 4：ready-order 和跨 epoch pipeline

- 移除全局 head-of-line blocking；
- 实现 bounded ready pool 和可复现消费顺序；
- 允许后续 epoch 提前采样、组装和 cache 预热；
- Trainer 保留 epoch gate；
- 增加 10/50 epoch 稳态指标。

### Stage 5：反馈式协调

- worker permit 动态调整；
- adaptive look-ahead 与 worker 数联动；
- 根据 GPU starvation、queue watermarks 和 RSS 自动收敛；
- 输出每次策略变化的原因和前后服务率。

## 12. 实验矩阵

### 快速阶段

| workload | 目的 | 配置 |
| --- | --- | --- |
| F1 | 小任务开销和回归 | 完整 2 epoch |
| Arxiv | 长文本、多字段、多进程扩展 | 2x50、2x200 |
| Salt | 高 locality、共享缓存价值 | 2x50、2x200 |
| Ratebeer | 低 locality、长文本、内存带宽 | 2x20、2x100 |

每个 workload 测：

- current thread 1；
- thread 2/4；
- process 1/2/4/8；
- process 4 + shared cache；
- process 4 + shared cache + lightweight payload；
- strict order 与 ready order。

### 正式阶段

1. 十个代表数据集全部至少 2 epoch；
2. Arxiv、Salt、Ratebeer 各跑 10 epoch；
3. 选择一个可在合理时间完成的任务跑 50 epoch；
4. 对 long-run 单独报告冷启动和稳态，不只报告总平均。

## 13. 验收标准

### 正确性

- strict 模式与 current latest 的 batch feature tensor 逐值一致；
- strict 模式固定输入 loss 满足现有正确性阈值；
- ready 模式无 batch 丢失、重复或跨 epoch 提前训练；
- ready 模式 3 seeds 最终指标无显著退化；
- 所有数据集至少完成 2 epoch；
- worker 异常、取消和恢复测试通过。

### 性能

- Arxiv process 4 相比 current latest wall time 改善至少 1.8x；
- Salt 改善至少 1.3x；
- Ratebeer 改善至少 1.2x；
- feature-bound workload 稳态 GPU utilization 目标不低于 80%；
- ready queue starvation time 降至训练 wall time 的 10% 以下；
- F1 等非 feature-bound 小任务退化不超过 5%。

### 资源

- GPU 显存峰值相对 current latest 增长不超过 5%；
- 共享 cache 和 ready pool 均严格遵守 byte budget；
- 默认配置主存峰值不超过 current latest 的 1.5x；
- 无共享内存、进程、文件描述符泄漏。

## 14. 回退边界

- 多进程 backend 通过配置开关启用，thread backend 保留；
- strict order 始终可用；
- shared cache 可独立关闭；
- lightweight TensorFrame payload 可回退现有 converter；
- cross-epoch prefetch 可设为 0；
- 任一新路径失败时只回退 latest，不改三个冻结 baseline。

## 15. 决策顺序

1. 先证明多进程 Assembly 的扩展曲线。
2. 再用共享 cache 消除多进程重复工作。
3. 再移除 DataFrame/TensorFrame converter 的冗余路径。
4. 最后放宽训练顺序并启用跨 epoch 预取。
5. 以 GPU starvation 和端到端 wall time 决策，不以单个函数微基准决策。
