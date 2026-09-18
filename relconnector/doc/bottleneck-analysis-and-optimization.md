# RelConnector 瓶颈分析与优化方案

> 当前执行范围：已先冻结 cache_baseline 并回补正确性问题，见
> [正确性修订报告](cache-baseline-correctness-report.md)。以下耗时来自修复前历史实验，
> 不代表修复后性能。第一轮动态优化已经实施，结果见
> [动态特征生产优化实施报告](adaptive-feature-optimization-report.md)。
> 字段 stype 始终遵循 RelBench；已完成两轮 GPU 短回归，完整任务仍待正式验收。

## 1. 分析范围

本文基于以下三组正式 GPU 实验：

- 全量物化：`benchmarks/gpu-real-baseline.jsonl`
- 历史在线实现：`benchmarks/gpu-real-online.jsonl`
- 当前优化实现：`benchmarks/gpu-feature-optimized.jsonl`

统一配置为 1 epoch、batch size 512、fanout 128x128、channels 128、
CUDA A30、单 Torch CPU 线程、异步执行器和单任务 14400 秒 timeout。

异步阶段会重叠，因此 sampling、feature fetch、batch assemble 和 train step
耗时不能直接相加。判断关键路径时，优先使用 ready queue 等待时间，再结合各阶段
服务时间、缓存指标和每批处理的数据量。

## 2. 总体结论

当前系统已经解决了全量特征物化造成的主机内存容量问题，但训练吞吐的主要瓶颈
已经转移到 CPU 特征生产：

```text
SamplePlan
  -> SQL fetch/decode
  -> DataFrame prepare
  -> text embedding
  -> TensorFrame conversion/index restore
  -> ready queue
  -> GPU train step
```

Amazon、Arxiv 和 Stack 的训练线程分别有 97.3%、89.0% 和 86.5% 的时间等待
ready queue。GPU train step 并不是这些任务的关键路径，继续单独优化 GNN kernel
不会显著降低端到端时间。

瓶颈不是单一问题，而是四类问题：

1. TensorFrame 组装和文本特征处理占据主要 CPU 时间。
2. 单个 feature worker 串行执行 SQL、解码和编码，不能利用更多 CPU 核。
3. Ratebeer 每批节点规模大、复用率低，缓存和去重收益不足。
4. Event 等短训练任务被 schema statistics 和拓扑初始化时间主导。

## 3. 关键证据

### 3.1 训练关键路径

| 数据集 | Train | Ready queue 等待 | Batch assemble / Train | Feature fetch / Train |
| --- | ---: | ---: | ---: | ---: |
| Amazon | 13683.6s | 97.3% | 94.3% | 5.7% |
| Arxiv | 2827.8s | 89.0% | 97.1% | 2.9% |
| Stack | 6368.8s | 86.5% | 89.8% | 10.2% |
| HM | 1541.3s | 65.6% | 63.9% | 36.0% |
| Salt | 2788.2s | 55.3% | 90.6% | 9.4% |
| Trial | 193.7s | 60.4% | 86.1% | 13.5% |
| Avito | 933.7s | 50.6% | 8.4% | 90.9% |

这些比例是并行 worker 的累计服务时间与 train wall time 的比值，不能横向求和。
它们说明：

- Amazon、Arxiv、Stack 的首要瓶颈是 assemble。
- HM、Salt、Trial 仍明显受 assemble 限制，但 sampling/train step 已开始参与
  关键路径。
- Avito 与其他任务不同，首要瓶颈是 SQL fetch 和 DataFrame decode。

### 3.2 文本编码和缓存

| 数据集 | Text occurrences | Model inputs | Cache hit | Evictions |
| --- | ---: | ---: | ---: | ---: |
| Amazon | 151.49M | 111.52M | 23.44% | 111.07M |
| Arxiv | 97.33M | 36.93M | 62.05% | 36.48M |
| Stack | 147.36M | 44.35M | 34.62% | 43.90M |
| HM | 234.22M | 0.38M | 99.23% | 0 |
| Salt | 586.09M | 0.15M | 99.95% | 0 |
| Trial | 4.24M | 0.92M | 23.92% | 0.47M |

当前缓存键是完整 normalized text，值是 300 维 GloVe 向量。Amazon 和 Stack
的工作集远大于 512MiB，几乎每次插入都触发淘汰，形成 LRU 抖动。

另一方面，Salt 和 HM 即使文本缓存命中率超过 99%，batch assemble 仍然很重。
原因是命中只省掉 GloVe 模型调用，系统仍需为每次 occurrence 执行：

- Python 字符串归一化和字典查找；
- `torch.stack` 和 inverse 展开；
- DataFrame copy/map；
- TensorFrame converter；
- 节点 inverse indexing 和 TensorFrame 切片。

因此，下一层缓存必须位于完整文本编码之后，而不是继续扩大整句向量缓存。

### 3.3 Ratebeer 专项

Ratebeer 在 4 小时内完成 15686/20743 batch，即 75.6%。其进度指标为：

| 指标 | 数值 |
| --- | ---: |
| 平均 wall time | 约 0.903s/batch |
| Requested rows | 122151/batch |
| Unique rows | 110528/batch |
| SQL queried rows | 88253/batch |
| Duplicate factor | 1.105 |
| Row-cache coverage | 20.2% |
| SQL amplification | 1.0007 |

SQL amplification 接近 1，说明 rowid 修复已经消除了扫描放大；问题是必须处理的
真实节点数量本身很大。相比之下，batch baseline 的 sampling 和 train step
分别约为 0.144s/batch 和 0.128s/batch。当前约 0.903s/batch 的流水线速度说明
feature path 是新的限制因素。

Ratebeer 还包含大量被推断为 `text_embedded` 的高基数字符串。抽样 100000 行：

| 字段 | 唯一率 | 平均长度 |
| --- | ---: | ---: |
| `beer_ratings.comments` | 95.0% | 195.3 |
| `beers.name` | 100.0% | 25.0 |
| `beers.description` | 92.8% | 231.2 |
| `beers.tags` | 79.5% | 766.0 |
| `brewers.email` | 99.6% | 23.4 |
| `places.website` | 93.5% | 31.3 |

这些高基数字段解释了文本执行成本和整句 LRU 的低收益，但不能据此人工改变
stype。RelBench 3.0.1 官方 `get_stype_proposal()` 对每张表随机采样最多 1000
行并调用 `torch_frame.utils.infer_df_stype()`，仅将推断出的 `embedding`
改为 `multicategorical`；它不按 URL、邮箱、电话、UPC 或邮编等列名做特判。
本项目应保持该处理语义，优化其执行过程，而不是自行重定义输入特征。

要让 Ratebeer 进入 4 小时预算，吞吐至少还需提升约 32.2%。

### 3.4 SQL 与原始特征缓存

大部分任务的 SQL amplification 已接近 1，SQL 扫描不再是全局问题。raw block
cache 的收益高度依赖节点局部性：

- Salt row-cache coverage 98.7%，收益显著。
- Arxiv 88.6%，收益显著。
- Ratebeer 20.2%，只能部分缓解读取量。
- Amazon 0%，当前训练访问模式无法形成 block cache 命中。

Avito 是例外。其 model inputs 只有 23521，batch assemble 仅占 train 的 8.4%，
但 feature fetch 占 90.9%。该任务应单独优化 SQL 查询次数、返回行数和
DataFrame decode，而不是继续调文本缓存。

### 3.5 初始化阶段

Event 总 wall time 为 960.5 秒，其中 prepare 为 954.4 秒，训练只有 6.0 秒。
历史细分数据中，schema statistics 约 793.6 秒，graph build 约 139.2 秒。

统计量不参与训练循环、占用空间很小，应按数据库版本、cutoff 和 schema
fingerprint 持久化。拓扑索引也属于项目允许保存的图结构数据，可以独立缓存。

### 3.6 内存和 GPU

当前十项 latest 实验的峰值 RSS 均低于 10GiB，Ratebeer timeout 前为 8.74GiB。
因此当前阶段不应以降低内存为首要目标，也不应无条件增大所有缓存。

GPU 端的主要问题是供给不足，而不是模型计算过慢。Amazon 的 train step 累计
仅占 train wall time 的约 2.5%，但 ready queue 等待占 97.3%。应先提高特征
生产吞吐，再重新评估 GPU kernel 和 batch size。

### 3.7 相关代码位置

- 单 feature worker 串行 fetch/assemble：
  `relconnector/runtime/async_pipeline.py:106-137`
- window 内 TensorFrame 编码和 inverse 恢复：
  `relconnector/features/assembler.py:29-76`
- DataFrameToTensorFrameConverter：
  `relconnector/features/schema.py:343-426`
- CPU GloVe、整句 LRU 和 embedding 展开：
  `relconnector/features/text.py:34-123`
- raw DataFrame block cache：
  `relconnector/features/cache.py:19-61`
- indexed rowid/primary-key SQL：
  `relconnector/features/sql.py:302-365`

## 4. 优化方案

### 4.1 P0：补齐细粒度观测

当前 `batch_assemble` 同时包含文本、类别、时间、DataFrame converter 和 inverse
恢复，Ratebeer timeout 结果又只有累计 fetch counters。实施优化前应增加：

- 每 table、每 column 的输入行数、文本字符数和 token 数；
- `normalize`、cache lookup、GloVe model、embedding expand 的独立耗时；
- `_prepare_frame`、TensorFrame converter、inverse slice 的独立耗时；
- SQL execute、row decode、DataFrame concat/reindex 的独立耗时；
- 每个 queue 的周期性深度、字节数和 producer/consumer 空闲时间；
- timeout progress 中同步保存 operation timers 和 text-cache stats。

验收标准：

- 所有细分耗时可以汇总回现有 `feature_fetch` 和 `batch_assemble`；
- 遥测开销低于 wall time 的 2%；
- 能输出 Ratebeer 前 500 至 1000 batch 的按表、按列火焰表。

### 4.2 P1：增加 encoded TensorFrame block cache

当前跨窗口缓存只有 raw DataFrame block 和完整文本向量，没有缓存最终编码节点。
建议增加 byte-bounded `EncodedFeatureBlockCache`：

```text
(database_version, schema_fingerprint, table, block)
    -> immutable CPU TensorFrame block
```

读取命中时直接执行 TensorFrame 索引，不再重复 DataFrame prepare、文本展开和
converter。缓存必须：

- 按实际 Tensor bytes 计费；
- 采用有界 LRU/TinyLFU；
- 支持 oversized rejection；
- 记录 hit rows、evictions、bytes 和 avoided encode time；
- 不缓存 GNN hidden state 或依赖 seed time 的动态结果。

预期收益最高的是 Salt、Arxiv 和 HM。Ratebeer、Amazon、Stack 应使用准入策略，
避免低复用节点污染 encoded cache。

### 4.3 P1：保持 RelBench stype 的文本执行优化

#### 固化官方 stype proposal

RelBench 3.0.1 的官方流程是：

```text
每张表 df.sample(min(1000, len(df)))
  -> infer_df_stype
  -> embedding 改为 multicategorical
  -> remove_pkey_fkey
  -> Dataset.materialize
```

当前 `SqlTensorFrameSchemaBuilder` 先排除主外键和任务隐藏列，再从 cutoff
过滤后的数据中读取按 node ID 排序的前 1000 行；官方是在完整表上随机采样，
之后才由建图/materialize 流程移除主外键和隐藏列。虽然 batch baseline、
vanilla 和 latest 当前共享同一份本地 schema，彼此可比，但不保证与官方
RelBench proposal 完全一致。

已实现 `data.initialize_stypes`：按官方 pandas 抽样规则有界读取特征，调用同一
`infer_df_stype()` 并保留 RelBench 类型转换，以官方函数对照测试验收后固化小型 metadata artifact。batch baseline、vanilla 和 latest 必须读取同一份
artifact，不再各自推断。artifact 应记录 RelBench、PyTorch Frame 版本、数据库
版本和 stype fingerprint。

禁止为了性能按列名人工修改 stype、删除文本列或把文本改成 hash/categorical。
这些做法会改变 RelBench 原始输入语义，不纳入本项目的优化路径。

#### Token-level GloVe

当前模型本质是词向量平均。对于高基数长文本，应缓存 token vector，而不是完整
句向量：

```text
normalized text
  -> tokenize
  -> unique token IDs
  -> GloVe lookup/cache
  -> segmented mean
```

不同评论往往不重复整句，但会共享大量 token。该方案不需要持久化 dense row
embedding，符合“计算换存储”的约束。

#### 自适应缓存准入

- 统计窗口内 reuse distance 和二次访问率；
- 一次性文本不进入 LRU，避免 1 亿级 put/evict；
- 对高命中列保留完整文本 cache；
- 对低命中列切换 token cache 或 no-cache；
- cache 策略按列配置，而不是全库共用一种策略。

#### 减少命中后的展开成本

- 避免为每个 256-row text chunk 创建大量 Python Tensor 对象；
- 缓存使用连续 Tensor slab 和整数 offset；
- 合并同一 table/window 的 inverse gather；
- 对多列文本进行批量 token lookup 和 segmented reduction。

### 4.4 P1：拆分 feature pipeline 并增加并行度

当前单个 feature worker 串行执行 `fetch_many` 和 `assemble_many`。建议改为：

```text
sampler
  -> plan queue
  -> SQL fetch worker
  -> fetched-feature queue
  -> N encode workers
  -> ordered reorder buffer
  -> ready queue
  -> GPU trainer
```

约束：

- 每个 SQL worker 使用独立只读连接；
- 所有 queue 继续按 bytes 限制；
- 通过 `BatchKey` reorder buffer 保持训练顺序；
- RNG 不进入 feature worker，保持确定性；
- encoded cache 分片或集中管理，避免锁竞争；
- 首轮只测试 2 个 encode worker，再根据 CPU 利用率扩展。

该方案可以重叠 SQLite decode 和 TensorFrame 编码，并利用多核 CPU。Amazon、
Arxiv、Stack 和 Ratebeer 是主要受益对象。

### 4.5 P1：Avito 的 SQL/fetch 专项

Avito 应保持编码路径不变，单独测试：

- 根据每 table 历史密度自适应选择 sparse `IN`、range 或 block read；
- 将相邻 node ID 合并为多个 range；
- 增大高查询开销表的 feature window；
- 减少 DataFrame concat、reindex 和 dtype decode 次数；
- 比较 pandas 与 Connector-X 在真实 sparse query 上的差异；
- 记录每 table 的 query count、returned rows、bytes 和 decode time。

目标是降低 feature fetch wall time，而不是单纯追求更高 raw-cache hit rate。

### 4.6 P1：持久化 schema statistics 和 topology

建议新增小型 metadata artifact：

```text
cache/schema/<database-version>/<schema-fingerprint>.json
cache/topology/<database-version>/<topology-fingerprint>.pt
```

cache key 至少包含：

- 数据库绝对路径、大小和 mtime；
- task cutoff 和 hidden columns；
- stype policy、统计算法版本；
- topology 表/外键版本。

使用临时文件加原子 rename 写入，读取后校验 fingerprint。该缓存不包含完整节点
特征或预采样训练集，不违反项目存储约束。Event 的重复运行预计可直接减少约
15 分钟初始化时间。

### 4.7 P2：自适应去重和 feature window

固定四 batch window 对不同任务并不最优：

- Salt、Arxiv：高重复，应扩大 window，直到达到 byte 上限。
- Stack、Amazon、Ratebeer：低重复，应缩小 window或按表跳过 `torch.unique`。
- Avito：可根据 SQL query latency 适度扩大 window，减少查询次数。

决策应基于最近若干 window 的 table-level duplicate factor、cache coverage、
encode rows/s 和 queue starvation，而不是全任务单一阈值。

### 4.8 P2：采样优化

当 feature path 加速后，sampling 会成为下一瓶颈。Ratebeer batch baseline 的
sampling 已约为 2992 秒，Salt 当前 sampling 也占 train wall 的 45.5%。

可依次评估：

- 多 sampler worker，并按 BatchKey 独立派生 RNG；
- sampler 输出的更紧凑表示，降低 plan queue bytes；
- 合并重复 seed 时间和边类型元数据；
- 检查 pyg-lib temporal/disjoint sampling 的 CPU 并行度。

在 ready queue 等待显著下降前，不应优先投入这一层。

## 5. 推荐实施顺序

### 阶段 A：不改变训练语义

1. 将当前 `relconnector` 冻结为新的版本化 baseline。
2. 离线生成并固化官方 RelBench stype proposal，三套实现共同读取。
3. 增加按表、按列细粒度遥测。
4. 持久化 schema statistics 和 topology。
5. 实现 encoded TensorFrame block cache。
6. 拆分 fetch/encode stage，并测试 2 个 encode worker。
7. 实现 table-level adaptive dedup/window。

### 阶段 B：文本执行优化

1. 连续 Tensor slab cache，减少 Python 对象和 `torch.stack`。
2. Token-level GloVe lookup 与 segmented mean。
3. 自适应 cache admission。

这些改动应保持 GloVe 300 维输出语义，并用固定输入逐值验证。

### 阶段 C：独立正确性修复

1. 单独修复 target leakage。
2. 验证离线 stype artifact 与官方 `get_stype_proposal()` 输出逐表一致。
3. 验证 batch baseline、vanilla 和 latest 使用相同的 stype fingerprint。

该阶段不以性能为理由改变 RelBench 的字段语义。

## 6. 实验矩阵与验收

每轮只引入一种优化，并至少覆盖：

- Ratebeer：剩余 timeout 和每批超大特征量；
- Amazon：低复用、高文本 cache churn；
- Stack：低节点重复、高基数文本；
- Salt：高复用、命中后展开成本；
- Avito：SQL/fetch 主导；
- Event：初始化主导。

核心验收指标：

- Ratebeer 完整 epoch 小于 14400 秒；
- 所有任务峰值 RSS 保持低于 32GiB，并保留至少 20% 系统余量；
- SQL amplification 保持接近 1；
- feature-bound 任务的 ready queue 等待比例降至 50% 以下；
- encoded/text cache 报告有效节省的 encode time，而不只报告 hit rate；
- 三套实现的 stype artifact 与官方 RelBench proposal 一致且 fingerprint 相同；
- batch 顺序、采样节点、模型输入和 loss 满足既定一致性要求；
- 不持久化完整 dense GloVe row embedding 或预采样训练集。

## 7. 最终判断

当前优化已经证明在线训练可以突破全量物化的内存上限。下一阶段不应继续泛化为
“加大缓存”，而应针对不同任务采用不同策略：

- 高复用任务：缓存最终 encoded features，消除重复组装。
- 高基数文本任务：token-level GloVe、缓存准入和多核编码。
- SQL 主导任务：按表优化读取模式和 decode。
- 短训练大数据库：持久化 schema statistics 和 topology。

Ratebeer 是下一轮首要验收任务。它至少需要 32.2% 的吞吐提升，最有可能通过
“优化同语义文本执行 + 并行 feature stage + 降低 TensorFrame 构造成本”的组合
进入 4 小时预算。
