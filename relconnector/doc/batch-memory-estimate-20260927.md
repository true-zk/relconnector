# Batch 全量加载与编码内存估算（2026-09-27）

本次没有运行训练，没有执行全量特征编码，也没有同时将全库加载进内存。只读取元数据、扫描计数和抽样解码；另用独立进程加载一次 CPU GloVe，作为运行时开销参考。估算针对当前 SQLite / stypes / 任务隐藏列配置，不能视为历史 Batch 实验的重测。

## 汇总（GiB，1 GiB = 2^30 bytes）

| 数据集 | 原始表：未截断全库 | 原始表：Batch 保留 | 编码 TensorFrame | 保留原始表＋编码 | 图索引估算 | 基础常驻 RSS 估算 |
|---|---:|---:|---:|---:|---:|---:|
| rel-amazon | 13.760 | 9.342 | 32.844 | 42.186 | 0.677 | 44.254 |
| rel-arxiv | 0.450 | 0.399 | 1.291 | 1.691 | 0.107 | 3.152 |
| rel-avito | 1.981 | 1.843 | 9.805 | 11.648 | 1.222 | 14.305 |
| rel-event | 4.745 | 3.918 | 12.844 | 16.762 | 1.295 | 19.395 |
| rel-f1 | 0.007 | 0.006 | 0.012 | 0.018 | 0.004 | 1.361 |
| rel-hm | 1.151 | 1.251 | 5.252 | 6.503 | 0.803 | 8.718 |
| rel-ratebeer | 8.212 | 6.990 | 22.036 | 29.026 | 0.825 | 31.509 |
| rel-salt | 0.902 | 0.794 | 11.955 | 12.750 | 0.255 | 14.395 |
| rel-stack | 2.945 | 2.347 | 10.057 | 12.404 | 0.199 | 14.013 |
| rel-trial | 2.899 | 2.657 | 14.989 | 17.646 | 0.234 | 19.234 |

## 口径与估算公式

- 原始表未截断全库：SQLite 中所有 data 表全量读入 pandas 后的逻辑内存估算，不是 SQLite 文件大小，也不包含所有任务表。Batch 代码会先读这些表，再按 test timestamp 截断。
- Batch 保留：各数据集 test_timestamp 之前（含当天）的记录；无时间列的表保留全部。时间过滤产生的行索引按每行 8 字节计入；非时间表使用 RangeIndex。
- 编码 TensorFrame：与所选任务相同，排除主键、外键和任务隐藏列；无特征表采用常量列。保留的原始表不会被编码替换，当前 BaselineExperiment.train 中 database 变量仍持有它们，因此两者相加。
- 基础常驻 RSS 估算 = 保留原始表 + 编码张量 + 双向 CSC/时间索引 + 所选任务全部 split + 独立 CPU 运行时 RSS 参考。
- CPU 运行时参考实测为 1.338 GiB，其中 GloVe 参数张量为 0.447 GiB；包含 baseline 依赖导入及 GloVe 加载，不含数据、CUDA context 或训练。
- 最后一列是加法模型的基础常驻量估计，不是实测 RSS、严格下界、峰值或可保证的机器容量。未计入完整统计字典、CPU 模型状态、CUDA 主机开销、采样/预取队列、编码和 SQL 转换临时副本、分配器碎片/未归还页。各项逻辑字节也可能重复计算共享 Python 对象，因此没有给出无依据的峰值误差区间。

## 计算方法

- 对每张表执行只读聚合，获得全量行数、时间截断后行数和 ID 范围。行数不是抽样估计。
- 每表均匀随机抽取最多 8192 行，复用实际 PandasDatabaseReader 和 decode_frame；固定随机种子 20260927。使用 pandas memory_usage(deep=True)，并补计 JSON 列表/字典的嵌套对象。原始字符串和嵌套对象大小按行数外推。
- 对抽样前后交错分半，记录 raw_half_sample_spread_pct 作为敏感性诊断；这不是置信区间，也不覆盖长尾漏采、对象共享和分配器误差。
- numerical：float32，每行每列 4 bytes。categorical：int64，每行每列 8 bytes。timestamp：7 个 int64，每行每列 56 bytes。
- text_embedded：冻结 GloVe 300 维 float32，每行每列 1200 bytes；模型 channels=128 不决定持久特征维度。即使文本为空或重复，全量 TensorFrame 仍按每行保存向量。
- multicategorical：int64 去重后的类别值 + int64 offsets。每行去重类别个数由样本估计。除这一可变长类型外，编码体积由真实保留行数及当前 stypes 直接推算，并计入小型 embedding offsets。
- 图索引：每条有效 FK 边双向各一个 int64 row；前后向 CSC 指针为 8×(源节点数+目标节点数+2)，时间列为每节点 8 bytes。有效边比例由样本估计。
- 所选任务与最近十库汇总一致；按当前任务隐藏列逻辑计算，详见 summary.csv 的 task 列。

## 与历史 RSS 的关系

- 该表回答最终数据结构占用多少内存，而不是编码过程达到过多少峰值。读取全表、时间截断、字符串转换、文本向量拼接、图构建和分配器保留都会改变峰值。
- 特别是当前 Ratebeer 编码体积约 22.04 GiB，单此一项已超过历史报告的 Batch RSS 20.29 GiB。当前估算与旧报告不能视为同配置同口径；原始历史 Batch JSONL 缺失，无法确定具体是配置/类型映射变化还是旧记录问题。不能用旧值校准、覆盖本次估算，也不能把本次估算伪装成历史实测。
- HM 的过滤后原始表估算略大于过滤前，是因为大部分交易仍被保留，而过滤新增了显式行索引。

## 复现与文件

```bash
.venv/bin/python benchmarks/batch-memory-estimate-20260927/estimate.py
.venv/bin/python benchmarks/batch-memory-estimate-20260927/task_memory.py
.venv/bin/python benchmarks/batch-memory-estimate-20260927/runtime_reference.py
.venv/bin/python benchmarks/batch-memory-estimate-20260927/report.py
```

- `estimates.json`：每表行数、抽样数、每列 stype/编码字节、图估算、输入文件 mtime/大小。
- `summary.csv`：十库汇总；`tables.csv`：每表明细。
- `runtime-reference.json`：独立 CPU 运行时实测参考；`task-memory.json`：任务 split 估算。

代码依据：`benchmark/baseline.py`、`baseline/batch_baseline/feature_store.py`、`baseline/vanilla_baseline/features/schema.py`、`baseline/vanilla_baseline/graph/builder.py` 以及安装的 torch_frame/data/mapper.py。
