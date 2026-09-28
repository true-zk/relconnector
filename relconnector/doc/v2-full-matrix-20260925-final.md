# v2 十任务 GPU benchmark（实验 20260925，完成）

有效完成 10/10 项。成功运行耗时合计 46242.51 秒；历史 v1 按相同各任务批次数折算为 78751.52 秒，整体加速 1.703×，耗时减少 41.28%。

Event、Trial 仍慢于历史 v1；整体仍慢于 Batch 基线。下文保留各项结果、超时估算和缓存冷热差异，不能将全部差异归因于算法。

- 配置：2 epoch、每 epoch 最多 10,000 batch、batch 512、fanout 128×128、channels 128、CUDA、4 个组装进程、自适应磁盘文本缓存。
- 历史数据：`doc/gpu-feature-optimization-experiment.md` 的 1-epoch batch/vanilla/v1 结果。速度比按端到端秒/batch 换算，v2 的启动成本摊薄在两轮中；旧实验硬件和初始化缓存状态不保证相同，因此不是严格配对实验。
- Ratebeer 的 v2 每轮上限 10,000 batch，历史完整轮次为 20,743 batch；比较只使用实际完成 batch 的平均耗时。
- Arxiv 在本轮开始前已有磁盘 embedding；Stack 修复后续跑也复用诊断及前次运行留下的磁盘 embedding。Amazon 修复续跑保留旧缓存并合并短测缓存，起始已有 6,584,936 条 embedding，属于部分热缓存。其他任务需以各自结果中的 activation_reason 和编码数判断冷热状态。
- Amazon 完整两轮训练结束，但旧版 10 秒诊断收集期限导致 worker 统计缺失；其模型编码次数和磁盘命中数标记 unknown，不能解释为 0。收集期限与静默遗漏处理已修复，Ratebeer 从初始化阶段重新启动使用该修复和 WAL 读取阶段策略。
- Salt 的复测要求是 adaptive 文本查询策略；HM 的复测要求是 local 4096 MiB 编码特征缓存。表格使用各任务最后一次成功结果，汇总只纳入通过复测配置及两轮完整性检查的结果。HM 和 Salt 复测的初始化缓存状态与原运行可能不同。
- Ratebeer 初次在约 3,804 个已完成 batch 后出现 CUDA 索引断言；诊断重跑在 batch 7371 保存到错误的时间戳秒值。随后独立复现共享特征缓存发布后、RPC 序列化搬迁存储与并发读取的竞争（修复前段错误，预先共享存储后通过），已改为共享存储完成后再发布缓存。该缺陷可解释特征损坏；首次 CUDA 失败没有保存输入，不能确认两次是同一触发链。最终重跑保留 CPU 类别校验和失败输入保存，检查开销包含在总时间中。详见 v2-ratebeer-regression-20260926.md。
- 前九项沿用各自已经完成的有效运行，没有在 Ratebeer 的共享缓存发布修复后全部重测；本表是逐项修复续跑结果，不是最终同一代码快照的十任务重新测量。
- 汇总耗时只累加各任务选中的有效成功运行；失败尝试、排查短测和人工修复时间不计入 benchmark 耗时，因此不等于整个开发与实验过程的日历耗时。
- 通过当前复测要求的任务：10/10。

## 逐任务结果

| Task | v2 batches | v2 wall (2e) | v2 s/batch | vs batch | vs vanilla | vs v1 | disk | model e0/e1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| rel-amazon | 9908 | 14699.6 | 1.484 | 0.35x | >1.96x | 1.89x | unknown | unknown |
| rel-arxiv | 2088 | 981.6 | 0.470 | 0.55x | >29.34x | 5.82x | on | 184,724/1,316 |
| rel-avito | 8644 | 1604.8 | 0.186 | 0.94x | >17.95x | 1.49x | off | 23,521/0 |
| rel-event | 76 | 2194.0 | 28.868 | 1.11x | 0.93x | 0.88x | off | 1,977/0 |
| rel-f1 | 12 | 26.6 | 2.214 | 1.20x | 1.37x | 1.20x | off | 2,160/374 |
| rel-hm | 14972 | 1760.1 | 0.118 | 0.75x | >16.36x | 1.98x | off | 389,772/11,935 |
| rel-ratebeer | 20000 | 15444.9 | 0.772 | 0.22x | >0.90x | ~1.19x* | off | 20,245,311/20,503,129 |
| rel-salt | 6340 | 5208.2 | 0.821 | 0.42x | 5.43x | 1.09x | off | 146,635/0 |
| rel-stack | 13228 | 3665.7 | 0.277 | 0.47x | 4.27x | 3.51x | on | 0/6,942 |
| rel-trial | 918 | 656.9 | 0.716 | 1.28x | 0.99x | 0.79x | on | 919,545/0 |

Ratios above 1x favor v2. Historical times cover one epoch, v2 covers two; s/batch normalizes batch count but amortizes startup differently. *Ratebeer v1 timed out after 15,686 batches.

| Task | outside pipeline | pipeline | epoch 0 | epoch 1 | disk hits | disk writes | final disk entries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rel-amazon | 51.2 | 14648.5 | 6306.8 | 8277.5 | unknown | unknown | unknown |
| rel-arxiv | 10.5 | 971.1 | 491.9 | 283.3 | 939,742 | 186,040 | 719,105 |
| rel-avito | 12.2 | 1592.6 | 800.3 | 780.1 | 0 | 0 | 0 |
| rel-event | 2166.5 | 27.4 | 6.9 | 6.9 | 0 | 0 | 0 |
| rel-f1 | 9.8 | 16.7 | 3.1 | 1.9 | 0 | 0 | 0 |
| rel-hm | 3.8 | 1756.3 | 875.0 | 866.7 | 0 | 0 | 0 |
| rel-ratebeer | 12.4 | 15432.6 | 7774.4 | 7626.8 | 0 | 0 | 0 |
| rel-salt | 12.0 | 5196.2 | 2473.3 | 2676.7 | 0 | 0 | 0 |
| rel-stack | 11.2 | 3654.4 | 1795.6 | 1842.4 | 56,902,720 | 6,942 | 2,891,661 |
| rel-trial | 127.4 | 529.5 | 288.0 | 223.3 | 534,805 | 919,545 | 919,545 |

Missing worker diagnostics (unknown, not zero): rel-amazon. Completed training batches/times are retained; cache/model counters cannot be verified for these runs.

Aggregate comparison (validated full-run results only):
- Completed: 10/10; pending: none.
- v2 observed total: 46242.51 s.
- v1 scaled to the same per-task batch counts: 78751.52 s.
- Aggregate v1/v2 ratio: 1.703x; >1 favors v2.
- batch scaled total: 19043.67 s; batch/v2 ratio: 0.412x.
- vanilla scaled total: >175742.20 s; vanilla/v2 ratio: >3.800x.
- Vanilla timeout entries use 14,400 s as a lower bound on full-epoch time; these are lower-bound linear extrapolations, not measured completed runs.
- Tasks still slower than v1: rel-event, rel-trial.
- All ten tasks are included in this comparison.
- Initialization, cache warmth, and implementation revisions differ; this is not a strictly paired experiment.
- Ratebeer v1 uses observed timeout throughput (14,400 s / 15,686 batches), not a completed full epoch.

F1：磁盘缓存关闭；第一轮/第二轮模型编码 2,160/374 条。
Arxiv：磁盘缓存启用；第一轮/第二轮模型编码 184,724/1,316 条。
Ratebeer：磁盘缓存关闭；第一轮/第二轮模型编码 20,245,311/20,503,129 条。
Ratebeer 本轮累计内存文本命中率约 88.67%，未触发当前低命中率磁盘启用条件。第二轮仍有模型编码，不能声称所有大库都已实现免重复编码，也不能仅凭命中率认定当前策略最优；模型输入计数不是全局不同文本的数量。

## 未加速任务与解释范围

Event：首次 schema/topology 初始化占主要耗时；events 宽表有 96 个数值列，逐列扫描及排序求统计量成本高。
Trial：初始化之外，共享文本查询、磁盘查询和发布也有明显开销；第二轮模型编码为 0，说明持久化复用已经发生，但总耗时仍回退。
这两项保留较早完成的 v2 结果，没有用后续全部修复重测；历史 v1 原始 JSONL 当前缺失，依据历史报告比较，不能严格拆分相对 v1 的各阶段差异。详见 v2-full-matrix-20260925.md。

逐项完整性检查：`benchmarks/v2-full-matrix-20260925/audit.json`（实际两轮 batch 数、采样/训练数、样本数、有限损失和诊断完整性）。

原始数据和运行日志：`benchmarks/v2-full-matrix-20260925/v2.jsonl`、`benchmarks/v2-full-matrix-20260925/run.log`。
