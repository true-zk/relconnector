# Baseline 与 Relconnector 在线训练 GPU 实验报告

## 1. 摘要

本轮对本地 10 个 RelBench SQLite 数据库各选一个任务，按“同一任务先 baseline、后 online，再进入下一个任务”的顺序执行，共 20 次独立进程实验。使用正常训练参数、完整 1 epoch，不设置 max_batches；每次 worker 上限为 14,400 秒。

- **Baseline：10/10 完成完整 epoch；online：5/10 完成，5/10 Timeout。** 没有记录到 OOM；超时不等于完整训练成功。
- 除小库 rel-f1 外，成功的 online 实验显著降低主机 RSS：Salt 从 17.72 GiB 降至 5.15 GiB，Stack 从 19.45 GiB 降至 3.72 GiB。
- Amazon baseline 峰值达到 103.77 GiB，在本机运行成功，但不适合目标中的 32 GiB 内存环境。online 截止前峰值为 4.23 GiB，不能视作完整 epoch 峰值。
- **瓶颈不能统称为“磁盘太慢”：已验证无主键表的 SQL 谓词导致全表扫描；成功任务还表明逐 batch CPU 特征编码可主导耗时。**
- Event、Trial 的 online 总时间更短，是省去全量物化带来的收益；训练循环本身仍更慢。
- 还发现四个任务的目标列未从输入特征中排除，存在标签泄漏；四对成功运行的 loss 未通过现有严格比较。因此这些是**当前实现的资源测量**，不是已验收的 RelBench 模型效果结果。

本报告只增加文档，不修改训练代码、数据库、比较阈值或原始结果，也不重跑完整训练。

## 2. 实验口径

### 2.1 数据来源

- [Baseline JSONL](file:///workspace/relconnector/benchmarks/gpu-real-baseline.jsonl)：10 条，全部 status=ok。
- [Online JSONL](file:///workspace/relconnector/benchmarks/gpu-real-online.jsonl)：5 条成功，5 条 TaskTimeout。
- 两文件任务顺序一致：F1、Salt、Arxiv、Event、Avito、Trial、HM、Stack、Ratebeer、Amazon。
- 核对时，10 个 SQLite 大小和修改时间均与 baseline 元数据一致；五对成功运行的 schema fingerprint、模型、采样策略、共有训练参数和完成工作量均通过比较器对应检查。timeout 没有同等完整元数据。

数据源 SHA-256：

```text
gpu-real-baseline.jsonl
378056d2222625b618019455c73d98132787e0234ab2f4ac1ae89e2203f5187d
gpu-real-online.jsonl
99d0b08d7573ea9766e19cdb5f7082979dba474973a7c9463601f9b33db6dd1b
```

### 2.2 统一配置

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA A30，24 GiB，cuda:0 |
| 软件（JSONL 记录） | Python 3.12.14，PyTorch 2.7.1+cu128，CUDA build 12.8 |
| 数据 / reader | 本地 SQLite / pandas，本轮未比较 Connector-X |
| epoch / batch size | 1 / 512，最后一批允许不足 512 |
| 采样 / GNN | 两层 fanout=(128,128)，channels=128，aggr=sum |
| 模型 | TensorFrame + HeteroEncoder + HeteroTemporalEncoder + HeteroGraphSAGE + task head |
| 优化器 | Adam，lr=0.001，weight_decay=0 |
| 随机性 | seed=42，按 epoch/batch 派生采样和训练 seed |
| CPU / 调度 | torch_num_threads=1 / async，两边相同 |
| 文本 | 本地 GloVe 300 维，冻结、CPU 编码；TensorFrame 文本分批配置 256 |
| 种子 / 图扫描 | shuffle block=65,536 行，scan batch=1,000,000 行 |
| 队列预算 | seed 64 MiB、plan 2 GiB、ready 8 GiB，各最多 64 items |
| Online 缓存 | 4 GiB DataFrame block LRU，block size=4096 |
| Baseline 物化 | 全量内存 TensorFrame，cache_materialization=False，不保存 .pt 快照 |
| 工作量 / 超时 | max_batches=None，每个 worker 整个生命周期最多 14,400 秒 |

单 PyTorch CPU 线程是本轮控制 pyg-lib 跨进程采样一致性的设置，不等于进程只有一个 OS 线程，也不是 CPU 调优配置。训练使用本地 task catalog、数据和 GloVe，不重新下载 RelBench。

### 2.3 指标解释

1. 成功任务总时间取 telemetry.overall.duration_s，包含准备和训练，但不包含全部进程启动/导入成本；完整进程时间另存 process.duration_s。timeout 只能使用后者。
2. 成功 RSS 取 telemetry.overall.peak.rss_mb（约 50 ms 采样）；timeout 取 process.peak_rss_mb（约 100 ms）。采样频率不同，都可能漏掉瞬时峰值。
3. *_mb 字段实际按 1024² 换算，为 MiB；报告再除以 1024 得 GiB。RSS 不是整机内存，不包含所有内核文件页缓存。
4. 显存取 overall 的 CUDA allocator 峰值，分别列 allocated、reserved，不是 nvidia-smi 总占用。
5. loss 为训练 batch 按 examples 加权的平均值，不是最终单批 loss，也不是 val/test 指标。
6. 异步 operation 会重叠，累计时间不能直接相加；train_step 包含 H2D、前向、反向、optimizer 和同步相关开销，不是纯 GPU kernel 时间。

依据：[TelemetryRecorder](file:///workspace/relconnector/benchmark/telemetry.py#L169-L245)、[显存峰值](file:///workspace/relconnector/benchmark/telemetry.py#L310-L326)、[进程测量](file:///workspace/relconnector/benchmark/process.py#L41-L104)、[测量包装器](file:///workspace/relconnector/benchmark/wrappers.py#L20-L73)、[加权 loss](file:///workspace/relconnector/relconnector/runtime/async_pipeline.py#L107-L141)。

## 3. 两种训练方法的执行流程

### 3.1 共同边界

data/ 是离线建库部分，本轮直接使用 data/relbench/*.sqlite。两边都在内存中保留 PK/FK 拓扑、正反向 CSC 邻接和节点时间，采样不读取业务特征。stype 和统计量在训练循环前准备，采用相同 test_timestamp 截断和 task hidden columns。

共享 schema 包含数值统计、类别词表、时间统计、文本维度与列顺序。两边各自构建并通过 fingerprint 核对，不是跨进程共享同一 Python 对象。统计量和 GloVe 不参与梯度更新，HeteroEncoder、GNN、任务 head 参与训练。

```text
SeedBatch(epoch, batch)
  -> SamplePlan(node_ids, edge_index, seed_time, target)
  -> FeatureBatch
  -> PreparedBatch(HeteroData with TensorFrame)
  -> OnlineTrainer.train_step()
```

每 epoch 一次是指每条训练任务行一次，不是每个全局实体 ID 一次。时间任务中同一实体可对应不同时间的任务行，邻居也可以被重复采样。

### 3.2 Baseline：全量读取和物化

**本轮入口是 benchmark.runner -> BaselineExperiment.train()，不是直接调用旧 baseline/trainer.py、baseline/sampler.py。** 为控制训练语义，实验入口复用了 online 的 sampler、assembler、trainer、executor。

```text
SQLite catalog / task manifest / task splits
  -> 共享 schema 与统计量准备
  -> 全量读表为 DataFrame，构造 RelBench Database
  -> test_timestamp 截断、主外键校验
  -> 全表 GloVe / 数值 / 类别 / 时间转换
  -> EagerFeatureStore：全量 TensorFrame 常驻内存
  -> 构建内存 CSC
  -> 每 batch 采样
  -> 按 node_ids 从内存 TensorFrame 切片
  -> 组装 HeteroData
  -> H2D -> 前向 -> loss -> 反向 -> Adam.step()
```

| 步骤 | 代码点位 |
| --- | --- |
| worker 与入口 | [runner.py:62](file:///workspace/relconnector/benchmark/runner.py#L62)、[BaselineExperiment.train](file:///workspace/relconnector/benchmark/baseline.py#L68-L180) |
| task、schema、全量读库 | [baseline.py:95-108](file:///workspace/relconnector/benchmark/baseline.py#L95-L108)、[LocalRelBenchDataset.get_db](file:///workspace/relconnector/baseline/dataset.py#L90-L107) |
| SQL 恢复为 Database | [read_relbench_database](file:///workspace/relconnector/relconnector/connector/base.py#L181-L203) |
| 全量物化 | [EagerFeatureStore.materialize](file:///workspace/relconnector/baseline/feature_store.py#L31-L60) |
| 图和训练组件装配 | [baseline.py:130-175](file:///workspace/relconnector/benchmark/baseline.py#L130-L175) |
| 内存特征切片 | [InMemoryTensorFrameFetcher](file:///workspace/relconnector/baseline/feature_store.py#L63-L93) |

训练期间除 TensorFrame，还持有 database DataFrame、task splits、拓扑、模型和队列；物化过程中另有临时副本。RSS 不等于所有 Tensor 大小之和。本轮没有写全量特征快照。

### 3.3 Online：拓扑常驻，特征按需读取、编码

```text
SQLite catalog / task manifest
  -> 共享 schema 与统计量准备
  -> 分块扫描 PK/FK/时间，构建内存 CSC
  -> 初始化 SQL fetcher、block cache、CPU GloVe
  -> 读取一批任务行并采样拓扑
  -> 按表去重查询 ID、查缓存
  -> 高密度读整块，低密度合并 IN 查询
  -> SQL decode，恢复采样行顺序和重复行
  -> 在线转 TensorFrame，组装 HeteroData
  -> H2D -> 前向 -> loss -> 反向 -> Adam.step()
  -> 释放 batch，仅保留缓存策略接受的 DataFrame block
```

| 步骤 | 代码点位 |
| --- | --- |
| worker 与测量入口 | [online_runner.py:43-70](file:///workspace/relconnector/benchmark/online_runner.py#L43-L70)、[OnlineTrainingBenchmark.run](file:///workspace/relconnector/benchmark/online_training.py#L38-L88) |
| 全部组件准备 | [OnlineRelBenchModel.prepare](file:///workspace/relconnector/relconnector/api.py#L68-L144) |
| schema 与 fingerprint | [SchemaBuilder](file:///workspace/relconnector/relconnector/features/schema.py#L65-L164)、[fingerprint](file:///workspace/relconnector/relconnector/features/schema.py#L477-L519) |
| 图索引 | [GraphIndexBuilder.build](file:///workspace/relconnector/relconnector/graph/builder.py#L55-L132)、[CSC 两遍构建](file:///workspace/relconnector/relconnector/graph/builder.py#L180-L271) |
| 种子调度 | [SqlSeedReader.iter_epoch](file:///workspace/relconnector/relconnector/task/seeds.py#L78-L129) |
| SQL 请求与缓存 | [SqlFeatureFetcher._fetch_table](file:///workspace/relconnector/relconnector/features/sql.py#L117-L173)、[FeatureBlockCache](file:///workspace/relconnector/relconnector/features/cache.py#L19-L59) |
| SQL 构造 / reader | [sql.py:189-242](file:///workspace/relconnector/relconnector/features/sql.py#L189-L242)、[PandasDatabaseReader](file:///workspace/relconnector/relconnector/connector/pandas.py#L18-L43) |
| 文本和 TensorFrame 编码 | [GloveTextEmbedder](file:///workspace/relconnector/relconnector/features/text.py#L20-L43)、[TensorFrameEncoder.encode](file:///workspace/relconnector/relconnector/features/schema.py#L396-L425) |

缓存和队列预算限制各自核算的 payload，不是整个 RSS 硬上限；拓扑、schema、GloVe、处理中 batch 和临时对象另占内存。本轮未通过 cgroup 强制 32 GiB 总内存限制。

### 3.4 共享异步与 GPU 训练

```text
seed-reader 线程 -> seed_queue
sampler 线程 -> plan_queue
feature-fetcher 线程：fetch + assemble 串行 -> ready_queue
主线程：GPU train_step
```

- [AsyncPipelineExecutor](file:///workspace/relconnector/relconnector/runtime/async_pipeline.py#L30-L141) 每阶段一个线程、FIFO 保序，按 epoch 顺序连续生产，可在队列预算内预取下一 epoch；本轮仅 1 epoch，未测跨 epoch 预取。
- [ByteBoundedQueue](file:///workspace/relconnector/relconnector/runtime/queue.py#L13-L69) 同时限制字节数与 item 数，超大单项报错；本轮没有该错误。
- [PygLibNeighborSampler](file:///workspace/relconnector/relconnector/sampling/pyg_lib.py#L43-L164) 使用共享 CSC 和逐 batch seed；时间任务采用 temporal/disjoint 采样，推荐任务采 source/positive/negative 三个子图。
- [Assembler](file:///workspace/relconnector/relconnector/features/assembler.py#L26-L78) 直接复用 baseline TensorFrame，对 online DataFrame 调用 encoder。
- [模型](file:///workspace/relconnector/relconnector/training/model.py#L16-L106) 使用可训练 HeteroEncoder、相对时间编码、GraphSAGE、任务 head。
- [Trainer](file:///workspace/relconnector/relconnector/training/trainer.py#L60-L144) 二分类 BCEWithLogits、多分类 CE、回归 L1、推荐 softplus(negative_score-positive_score)。

[全局 RNG 锁](file:///workspace/relconnector/relconnector/runtime/async_pipeline.py#L118-L119)覆盖训练调用，sampler 使用同一锁。因此采样关键区与训练并不完全并行，主要是特征准备与它们重叠；这是两边共同的实现约束。

## 4. 结果对比

### 4.1 总时间、RSS 与状态

B=baseline，O=online。时间单位为分钟，RSS 为 GiB。examples/batches 是完整 epoch 工作量，五个 online 成功任务均与 baseline 相同。timeout 完成批数未落盘，不是 0。

| 数据库 | 任务 | examples | batches | B 总时间 | B RSS | O 总时间 / 状态 | O RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rel-f1 | driver-circuit-compete | 2,649 | 6 | 0.17 | 2.29 | 0.22 / 完成 | 2.44 |
| rel-salt | item-incoterms | 1,622,789 | 3,170 | 18.27 | 17.72 | 235.45 / 完成 | 5.15 |
| rel-arxiv | paper-citation | 534,233 | 1,044 | 4.43 | 7.33 | 240.01 / Timeout | 4.29 |
| rel-event | user-attendance | 19,239 | 38 | 20.20 | 31.93 | 16.90 / 完成 | 3.91 |
| rel-avito | searchstream-click | 2,212,750 | 4,322 | 12.46 | 17.44 | 240.01 / Timeout | 4.26 |
| rel-trial | eligibilities-adult | 234,839 | 459 | 6.89 | 22.07 | 5.35 / 完成 | 2.94 |
| rel-hm | user-churn | 3,832,692 | 7,486 | 10.85 | 14.35 | 240.01 / Timeout | 3.89 |
| rel-stack | user-badge | 3,386,276 | 6,614 | 14.19 | 19.45 | 130.44 / 完成 | 3.72 |
| rel-ratebeer | beer_ratings-total_score | 10,620,177 | 20,743 | 58.17 | 20.29 | 240.01 / Timeout | 3.86 |
| rel-amazon | item-churn | 2,536,014 | 4,954 | 42.23 | 103.77 | 240.01 / Timeout | 4.23 |

成功配对 online/baseline 总时间比：F1 1.24x、Salt 12.89x、Event 0.84x、Trial 0.78x、Stack 9.20x；RSS 降幅为 -6.6%、71.0%、87.8%、86.7%、80.9%。这些是当前运行观测值，受第 6 节限制约束。timeout 的 240 分钟是截止时间，不是完成时间，不能据此说“完整 epoch 只慢了 N 倍”。

### 4.2 显存

单位 GiB，每格为 peak allocated / peak reserved；五个成功配对的 allocator 峰值在原始数据中一致。

| 数据库 | Baseline | Online |
| --- | ---: | ---: |
| rel-f1 | 14.06 / 18.57 | 14.06 / 18.57 |
| rel-salt | 8.32 / 22.81 | 8.32 / 22.81 |
| rel-arxiv | 3.98 / 7.66 | 未记录（Timeout） |
| rel-event | 2.50 / 3.18 | 2.50 / 3.18 |
| rel-avito | 1.61 / 6.17 | 未记录（Timeout） |
| rel-trial | 0.31 / 0.38 | 0.31 / 0.38 |
| rel-hm | 1.26 / 7.75 | 未记录（Timeout） |
| rel-stack | 0.85 / 1.10 | 0.85 / 1.10 |
| rel-ratebeer | 3.49 / 7.38 | 未记录（Timeout） |
| rel-amazon | 1.00 / 1.75 | 未记录（Timeout） |

主要节省的是主机全量特征内存，不是每 batch GPU 激活内存。峰值相同与同规格 batch/模型设计一致，但不是输入逐元素相同的证明。

### 4.3 Loss

| 数据库 | 类型 / loss | Baseline loss | Online loss | 当前比较器 |
| --- | --- | ---: | ---: | --- |
| rel-f1 | 推荐 / pairwise softplus | 1.2480664871 | 1.2480977695 | not-equivalent:loss |
| rel-salt | 多分类 / CE | 0.0378852230 | 0.0389236337 | not-equivalent:loss |
| rel-arxiv | 二分类 / BCE | 0.5389136476 | 未记录 | 缺少成功配对 |
| rel-event | 回归 / L1 | 0.4145591924 | 0.4145629266 | not-equivalent:loss |
| rel-avito | 二分类 / BCE | 0.0339962412 | 未记录 | 缺少成功配对 |
| rel-trial | 二分类 / BCE | 0.0111412018 | 0.0111412018 | strictly-comparable |
| rel-hm | 二分类 / BCE | 0.4425577025 | 未记录 | 缺少成功配对 |
| rel-stack | 二分类 / BCE | 0.1414530431 | 0.1414556502 | not-equivalent:loss |
| rel-ratebeer | 回归 / L1 | 0.3781105505 | 未记录 | 缺少成功配对 |
| rel-amazon | 二分类 / BCE | 0.5124713909 | 未记录 | 缺少成功配对 |

[比较器](file:///workspace/relconnector/benchmark/compare.py#L51-L89)用 math.isclose(rel_tol=1e-7, abs_tol=1e-7)。F1、Event、Stack 绝对差约 3.13e-5、3.73e-6、2.61e-6。CUDA 非确定性是候选解释，但没有逐 batch 输入/梯度指纹或确定性重跑，不能认定已证明根因。

Salt 绝对差约 0.00103841，相对 baseline 平均 loss 为 2.74%，不是预测精度下降 2.74 个百分点，也不应以“浮点误差”结案。Trial 通过比较器仅表示满足已有配对检查，不表示没有标签泄漏。

### 4.4 准备与训练循环

单位秒。B 准备=overall-train；O 准备取 prepare phase，phase 外少量开销使其与 overall 不完全相加。

| 数据库 | B 准备 | B 训练循环 | O 准备 | O 训练循环 |
| --- | ---: | ---: | ---: | ---: |
| rel-f1 | 7.287 | 3.138 | 6.666 | 6.118 |
| rel-salt | 147.341 | 948.610 | 38.710 | 14,088.186 |
| rel-arxiv | 69.911 | 195.802 | 未记录 | 未记录 |
| rel-event | 1,207.912 | 4.122 | 953.209 | 60.799 |
| rel-avito | 444.331 | 303.533 | 未记录 | 未记录 |
| rel-trial | 378.820 | 34.456 | 61.129 | 259.596 |
| rel-hm | 268.145 | 383.148 | 未记录 | 未记录 |
| rel-stack | 435.164 | 415.949 | 53.816 | 7,772.289 |
| rel-ratebeer | 497.930 | 2,992.161 | 未记录 | 未记录 |
| rel-amazon | 2,374.483 | 159.098 | 未记录 | 未记录 |

Baseline 准备阶段重要分项如下，online 当前仅有总 prepare，无法从 JSONL 拆出同样分项。

| 数据库 | schema 统计 | 全量读库 | 特征物化 | 图构建 |
| --- | ---: | ---: | ---: | ---: |
| rel-f1 | 1.086 | 0.297 | 0.155 | 0.404 |
| rel-salt | 17.664 | 14.779 | 89.790 | 15.114 |
| rel-arxiv | 5.314 | 4.759 | 42.729 | 10.311 |
| rel-event | 793.575 | 157.734 | 112.132 | 139.207 |
| rel-avito | 154.873 | 55.264 | 124.355 | 95.463 |
| rel-trial | 32.828 | 19.663 | 297.868 | 22.710 |
| rel-hm | 110.657 | 32.604 | 32.689 | 79.602 |
| rel-stack | 25.530 | 18.747 | 355.214 | 23.130 |
| rel-ratebeer | 130.962 | 89.330 | 160.554 | 81.158 |
| rel-amazon | 115.948 | 81.129 | 2,081.762 | 85.244 |

Event、Trial 的 online 训练循环约为 baseline 的 14.7、7.5 倍，单 epoch 节省的准备成本抵消了代价。多 epoch 只付一次准备成本，不能直接延用单 epoch 结论。

允许预计算统计量不等于计算成本为零。[分位数和时间统计](file:///workspace/relconnector/relconnector/features/schema.py#L166-L331)仍扫描、排序列，Event schema 阶段就约 794 秒。未来可复用小型元数据，本轮两边都现场准备并计入时间。

## 5. Timeout 分析

### 5.1 直接机制与观测缺口

五条 timeout 都是 process.timed_out=true、returncode=-9、进程时间约 14400.3-14400.4 秒。[父进程](file:///workspace/relconnector/benchmark/process.py#L49-L89)到上限后 SIGKILL 整个进程组，[runner](file:///workspace/relconnector/benchmark/online_runner.py#L92-L134)写 TaskTimeout。这里 -9 是主动超时终止，不是 OOM 证据；上限包含导入、准备、训练。

Worker 在[整个 run 返回后](file:///workspace/relconnector/benchmark/online_runner.py#L43-L70)才写 JSON，硬终止丢失内存 telemetry，所以不知道完成到第几批、阶段比例、最终 loss 和 CUDA 峰值。不能将这些缺失量写成 0。

此前抽样看到 CPU 活跃、逻辑读取增长、GPU 间歇执行，支持低吞吐解释，但不是完整追踪，不能严格排除所有停顿。以下区分已验证机制、定量证据与任务级推断。

### 5.2 已验证：rowid 表达式导致全表扫描

无主键时，[`_node_id_expression`](file:///workspace/relconnector/relconnector/features/sql.py#L237-L242)返回 rowid - 1，同时用于投影、WHERE、ORDER BY。对现有 HM 数据库只读 EXPLAIN：

```sql
SELECT rowid - 1 AS __node_id__, t_dat
FROM transactions WHERE rowid - 1 IN (0,1,2) ORDER BY rowid - 1;
```

```text
SCAN transactions
USE TEMP B-TREE FOR ORDER BY
```

保持零基 node ID 语义，将谓词和排序改为裸 rowid：

```sql
SELECT rowid - 1 AS __node_id__, t_dat
FROM transactions WHERE rowid IN (1,2,3) ORDER BY rowid;
```

```text
SEARCH transactions USING INTEGER PRIMARY KEY (rowid=?)
```

这是报告阶段的查询计划验证，**没有修改代码或数据库，没有把改写后的性能冒充正式实验结果**。范围请求也应把 node ID [start,stop] 映射到 rowid BETWEEN start+1 AND stop+1，返回时保留零基 ID。

| 数据库 | 无显式主键且有候选特征的表 | 只读 EXPLAIN |
| --- | --- | --- |
| rel-amazon | review | 表达式版本 SCAN；裸 rowid 版本 SEARCH |
| rel-arxiv | citations、paperAuthors、paperCategories | 同上 |
| rel-avito | SearchStream、VisitStream、PhoneRequestsStream | 同上 |
| rel-hm | transactions | 同上 |
| rel-ratebeer | beer_upcs | 同上 |

这些表并非都能走“无特征直接返回常量”分支：Arxiv 三关系表保留 Submission_Date，HM 保留 t_dat/price/sales_channel_id。本地 MAX(rowid) 显示 HM transactions 约 1,545 万、Avito SearchStream 约 925 万、Amazon review 约 2,086 万行号规模。这不是 cutoff 后采样行数，但反映单次扫描可能涉及很大表。

索引查询主要随所需 k 行和定位成本增长；退化后每次可能扫描 N 行。当前每次 IN 最多 10,000 个 ID，一个 batch 可拆多次查询，再重复数千批，扫描与排序成本被放大。Baseline 循环从内存切片，不执行这些特征 SQL。

EXPLAIN 没有记录本轮实际每批查哪些表、查几次，无法量化它占 timeout 的百分比。Salt、Stack 均可按显式整数主键检索却依然慢，说明还有别的瓶颈。

### 5.3 定量证据：重复编码

[fetcher](file:///workspace/relconnector/relconnector/features/sql.py#L132-L173)先对 SQL ID 去重，再 reindex(ids) 恢复采样重复行；[assembler](file:///workspace/relconnector/relconnector/features/assembler.py#L48-L57)随后编码全部恢复行。temporal/disjoint 下，同一全局节点可重复出现在多个 seed 子图，造成 batch 内重复编码；跨 batch 也没有编码结果缓存。

```text
baseline 编码成本 ≈ 所有保留节点编码一次
online 编码成本   ≈ 每批采样节点出现次数逐次编码
```

固定统计量不会消除分词、GloVe、时间解析、类别映射、DataFrame 拷贝与 TensorFrame 构建。[GloVe 在 CPU 执行](file:///workspace/relconnector/relconnector/features/text.py#L24-L43)，不是由空闲 GPU 自动承担。

成功 online 的累计 operation 时间（秒）如下。fetch 和 assemble 同线程串行，其余列可与它们重叠。

| 数据库 | sampling | feature_fetch | batch_assemble | train_step | train phase |
| --- | ---: | ---: | ---: | ---: | ---: |
| rel-f1 | 0.481 | 1.664 | 3.940 | 4.109 | 6.118 |
| rel-salt | 1,270.060 | 1,169.564 | 12,917.582 | 1,107.794 | 14,088.186 |
| rel-event | 0.873 | 47.806 | 12.837 | 3.555 | 60.799 |
| rel-trial | 80.964 | 40.598 | 218.615 | 93.370 | 259.596 |
| rel-stack | 932.609 | 694.864 | 7,075.623 | 893.282 | 7,772.289 |

- Salt batch_assemble 占 train phase 约 **91.7%**，每批平均 4.075 秒，fetch 0.369 秒；单独提高 SQL 吞吐无法消除主要开销。
- Stack batch_assemble 占约 **91.0%**，每批平均 1.070 秒，fetch 0.105 秒。
- Event 以 fetch 为主，约占 train phase **78.6%**，不能给不同任务统一归因。

batch_assemble 不是纯 GloVe 时间。TensorFrame converter 每次调用还创建 mapper：[dataset.py:288-304](file:///workspace/.venv/lib/python3.12/site-packages/torch_frame/data/dataset.py#L288-L304)，类别 mapper 建 Series 并 merge：[mapper.py:80-109](file:///workspace/.venv/lib/python3.12/site-packages/torch_frame/data/mapper.py#L80-L109)。尚无逐列 profiler 数据量化这些分项。

### 5.4 缓存与调度为什么没隐藏开销

[缓存策略](file:///workspace/relconnector/relconnector/features/sql.py#L140-L167)要求 4096 行 block 内同批密度至少 0.25，通常即至少 1024 个不同 ID，才整块读入并缓存。更稀疏的请求合并 IN，但返回行不缓存。缓存对象是 DataFrame，不是 TensorFrame/GloVe embedding，命中仍需编码；推荐任务三个子图也没有合成一次唯一 ID 请求。

所以 4 GiB 是容量上限，不等于已有效缓存热节点或编码结果。随机采样分散到很多 block 时，可能长期走不缓存的 sparse 分支。本轮没把 last_stats、命中率、唯一节点数、缓存使用字节写入 JSONL，不能给出实测命中率。

忽略启动、排空、竞争和锁，理想流水线稳态间隔至少为：

```text
batch 间隔 >= max(采样服务时间, fetch + assemble 服务时间, train 服务时间)
```

fetch 与 assemble 在[同一线程](file:///workspace/relconnector/relconnector/runtime/async_pipeline.py#L87-L105)串行，扩大 ready queue 只能缓冲波动，不能提高长期产能。Salt、Stack 的 fetch+assemble 累计时间几乎等于 train phase，直接反映特征生产限制流水线。

Python 处理、CPU GloVe、共享 RNG 锁、单 PyTorch 线程也可能影响吞吐。sampling 计时包含 sampler 内等待 RNG 锁的时间，不能把它全部当成 pyg-lib 计算慢；需分离锁等待、队列等待和服务时间。

### 5.5 各 Timeout 任务

每批预算=14400/完整 epoch batches，尚未扣准备时间，只是所需平均间隔的宽松上界，不是 online 实测每批耗时。

| 数据库 | batches | 每批预算（秒） | 当前判断 |
| --- | ---: | ---: | --- |
| rel-arxiv | 1,044 | 13.793 | 三关系表查询 SCAN，Title/Abstract 等走 CPU 文本编码；仅 1044 批也超时，不能只用批数多解释，SQL/编码占比待测。 |
| rel-avito | 4,322 | 3.332 | seed 自身 SearchStream 无主键，另有 VisitStream 等，重复扫描是强候选瓶颈，还有 Title/SearchQuery 编码。 |
| rel-hm | 7,486 | 1.924 | transactions 扫描，article 多描述列及 postal_code 被推断为文本，高批数放大查询和重复编码。 |
| rel-ratebeer | 20,743 | 0.694 | beer_ratings 本身可主键检索，不能说所有表都全扫；beer_upcs 有扫描问题，邻接表多文本。批数最多，单批预算最紧，主因占比未测。 |
| rel-amazon | 4,954 | 2.907 | review 扫描，加上商品描述/评论文本重复编码。B 一次物化 2081.762 秒后训练仅 159.098 秒；O 将静态转换重复放入循环。 |

这些机制说明存在可消除的工程成本，但不能推算精确完成时间，也不能断言修正一项后一定能在四小时内完成。

### 5.6 不等于物理磁盘带宽耗尽

此前抽样 rchar/syscr 增长明显，Avito、HM 尤甚，但相应 read_bytes 为 0。rchar 是系统调用返回的逻辑读取字节，也含非数据库读取，不等于物理磁盘 I/O。

这更支持页缓存重复读取、SQLite 扫描和 CPU 转换，而非已证明 SSD 带宽耗尽。没有冷缓存控制或完整磁盘统计；真正 32 GiB 环境中页缓存更小，性能可能改变。

## 6. 正确性与结论边界

### 6.1 四个任务目标列仍进入输入

对当前 catalog 和同规则 stype 推断做只读核对，将前 1000 条 train 任务行与实体行关联，发现：

| 任务 | 未排除输入列 | stype | 核对结果 |
| --- | --- | --- | --- |
| rel-salt/item-incoterms | salesdocumentitem.ITEMINCOTERMSCLASSIFICATION | numerical | 1000 条非空，全部等于 target |
| rel-avito/searchstream-click | SearchStream.IsClick | numerical | 1000 条非空，全部等于 target |
| rel-trial/eligibilities-adult | eligibilities.adult | categorical | 1000 条非空，统一二值表示后全部等于 target |
| rel-ratebeer/beer_ratings-total_score | beer_ratings.total_score | numerical | 1000 条非空，全部等于 target |

[候选列逻辑](file:///workspace/relconnector/relconnector/features/schema.py#L110-L117)只排除 PK/FK/manifest hidden columns，不自动排除实体 target；[converter](file:///workspace/relconnector/relconnector/features/schema.py#L368-L393)也未设置 target_col。采样包含 seed，assembler 未额外屏蔽其目标特征。

至少在上述记录中存在直接标签泄漏通路，两边共同受影响。输入一致不等于任务定义正确，低 training loss 不是有效模型证据。应按任务语义统一修正目标屏蔽，再建新版本和 fingerprint，不静默改写历史结果。

### 6.2 其他限制

- 单次、单 epoch、单 seed，无重复实验、置信区间、val/test 或收敛验证。
- 元数据对齐不等于完整 epoch 每个 Tensor、子图、梯度都逐一核验，四对 loss 根因待验证。
- CPU 小规模一致性不能代替完整 GPU 确定性验证，不应直接放宽容差让结果通过。
- 先 baseline 后 online，未控制 OS 页缓存，不是严格冷启动 I/O 对比。
- timeout RSS 仅覆盖已执行部分，不含完整工作量峰值、吞吐与最终 loss。
- 自动 stype 虽然两边一致，不保证语义最优，例如 Salt 某些代码字段被当成文本/时间；改 stype 应单独对照。
- 没有独立记录 SQL 次数/计划、逐列编码、queue wait、H2D、锁等待、GPU 利用率序列，限制更细归因。

## 7. 后续改进顺序

保留本轮结果，按单变量原则分别验证：

1. **正确性。** 统一修正目标屏蔽；对同 BatchKey 记录 seed、SamplePlan、PreparedBatch、初始参数指纹，定位 loss 首次分歧，以新版本区别旧结果。
2. **观测。** 周期性落盘完成 batch、prepare 子阶段、query 数、唯一/重复节点数、缓存、各类编码、锁/队列等待、H2D；timeout 留最后快照和完整配置。
3. **只改 rowid 查询。** WHERE/ORDER BY 用裸 rowid，保持零基语义，EXPLAIN 验证 SCAN 到 SEARCH，相同 ID 验证值和顺序，不新增全量副本。
4. **减少重复编码。** 唯一节点先编码，再恢复 disjoint 顺序，有界缓存冻结 GloVe 或静态 TensorFrame。不能缓存随梯度变化的 HeteroEncoder/GNN 隐向量，也不能错误复用依赖 seed_time 的相对时间编码。
5. **缓存与 reader。** 分别评估 sparse 行缓存、跨批合并读取、连接生命周期复用，再比较 pandas/Connector-X，避免同时改变多个维度。
6. **并行。** 结合确定性缩小 RNG 锁范围，采用独立 RNG 或多进程 sampler/encoder，配合反压，不能只靠扩大队列。
7. **复测。** 保持 batch 512、fanout 128x128、channels 128，施加真实内存约束，分别测单/多 epoch、冷/热缓存。小 workload 不替代正式结果。

结论：**拓扑常驻、特征在线获取的内存方向成立；当前实现尚未做到高效索引查询和充分复用静态编码。先消除额外工程成本，再评价在线方法的时间与内存取舍。**

## 8. 复现入口

在 /workspace/relconnector 下运行；以下用新输出名，不覆盖原结果。每任务两条命令顺序执行，不并发占 GPU。报告编写期间未执行这些重跑命令。

```bash
../.venv/bin/python -m benchmark.runner \
  --dataset rel-salt --task item-incoterms \
  --reader pandas --epochs 1 --batch-size 512 \
  --num-neighbors 128,128 --channels 128 \
  --device cuda:0 --torch-num-threads 1 --executor async \
  --task-timeout 14400 \
  --output benchmarks/gpu-report-rerun-baseline.jsonl --resume

../.venv/bin/python -m benchmark.online_runner \
  --dataset rel-salt --task item-incoterms \
  --reader pandas --epochs 1 --batch-size 512 \
  --num-neighbors 128,128 --channels 128 \
  --device cuda:0 --torch-num-threads 1 --executor async \
  --feature-cache-mb 4096 --task-timeout 14400 \
  --output benchmarks/gpu-report-rerun-online.jsonl --resume

../.venv/bin/python -m benchmark.compare \
  benchmarks/gpu-real-baseline.jsonl benchmarks/gpu-real-online.jsonl
```

SQL 根因的最小只读复核：

```bash
../.venv/bin/python - <<'PY'
import sqlite3
from contextlib import closing

with closing(sqlite3.connect(
    "file:data/relbench/rel-hm.sqlite?mode=ro", uri=True
)) as con:
    for predicate, order in (
        ("rowid - 1 IN (0,1,2)", "rowid - 1"),
        ("rowid IN (1,2,3)", "rowid"),
    ):
        query = (
            "EXPLAIN QUERY PLAN "
            "SELECT rowid - 1 AS __node_id__, t_dat FROM transactions "
            f"WHERE {predicate} ORDER BY {order}"
        )
        print(predicate, [row[3] for row in con.execute(query)])
PY
```

代码链接指当前 workspace 行号，依赖库链接指本机 .venv，升级后可能变化。原始 JSONL 是数值结果的最终依据。
