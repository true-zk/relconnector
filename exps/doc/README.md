# Project Documents

## v1 与 v2 的区别

v1 是基于单进程的数据准备优化版本，通过节点去重、跨批次窗口、缓存复用和进程内线程流水线减少重复读取与编码。实际训练进一步暴露出 assemble（特征读取、编码与批次组装）的供给瓶颈：即使减少重复工作，数据准备仍可能跟不上 GPU 消费。v2 因此升级为多进程架构，将采样、assemble 和训练拆分为独立的执行阶段，分别由采样进程、组装进程池和训练主进程承担，通过有界队列连接并重叠执行。这里的单进程不等于单线程，多进程也不改变邻居采样与模型训练的算法语义。

过程性设计、实施计划、验证记录和实验报告统一保存在本目录。

当前汇报入口：[24页整合版PPT](slides/performance-explanation-20260927/RelConnector-优化复盘-按数据类型解释.pptx)及[PDF](slides/performance-explanation-20260927/RelConnector-优化复盘-按数据类型解释.pdf)；当前论文入口：[方法与定向验证整合稿](paper/relconnector-background-method-revised-20260927/RelConnector-背景动机与方法-修订版.pdf)。新缓存位置与分片布局优化已并入原方法章节，旧演进稿作为历史材料。

- [优化演进详细报告：vanilla → v1 → v2（2026-09-26）](optimization-evolution-report-20260926.md)：以有限内存训练为主线，梳理优化机制、退化修复、实验口径和资源边界；batch 仅作传统训练背景。
- [优化演进幻灯片（24 页）](slides/optimization-evolution-20260926/README.md)：可编辑 PPTX、PDF 与逐页讲解备注。
- [优化演进幻灯片：简洁图解版（18 页，推荐）](slides/optimization-evolution-simple/README.md)：白底简洁布局，增加流程图、时序图和每页信息量；Office 可编辑 PPTX、PDF 与讲解备注。

## 架构与计划

- [早期架构讨论](architecture-discussion-20260903.md)
- [早期实施计划](implementation-plan-20260903.md)
- [特征管线优化计划](feature-pipeline-optimization-plan.md)

## 验证与实验

- [Ratebeer 编码缓存位置消融](v2-ratebeer-cache-placement-20260927.md)：同预算本地缓存短程总时间降低18.71%，流水线降低11.26%；缓存前移独立收益不足，候选归档。

- [按历史瓶颈制定的针对性优化计划与进展](v2-bottleneck-targeted-optimization-20260927.md)：十类任务证据、缓存前移的负结果，以及布局整理实验。
- [Stack 持久化向量布局优化](v2-stack-compaction-20260927.md)：预热后交错实验总时间降低12.04%，包含离线整理成本和适用边界。

- [v2 后续优化计划](v2-next-optimization-plan-20260927.md)：按实际成本选择缓存与读取策略，包含分阶段验收条件。
- [v2 共享缓存投影优化尝试](v2-cache-projection-optimization-20260927.md)：候选正确性及四次 GPU 对照；局部查询改善但未确认整体提速，候选归档、默认代码保持上一轮版本。
- [v2 专用行选择优化](v2-row-gather-optimization-20260927.md)：上一轮 Salt 定向实验与已保留的优化。
- [v2 十任务完整 GPU benchmark（2026-09-27 完成）](v2-full-matrix-20260925-final.md)：两轮、每轮最多 10,000 batch；按实际每批耗时对比 Batch、Vanilla、v1，保留缓存冷热与缺失统计说明。
- [Ratebeer 两次训练故障与共享缓存修复](v2-ratebeer-regression-20260926.md)：故障时间线、并发复现与修复后完整验证。
- [四目录重构验证](recovery-validation.md)
- [首轮 GPU 训练报告](gpu-training-report.md)
- [特征优化 GPU 实验计划](gpu-feature-optimization-experiment.md)
- [瓶颈分析与优化方案](bottleneck-analysis-and-optimization.md)

模块使用方式仍记录在各目录的 `README.md`；原始 JSONL 实验结果保存在
`benchmarks/`。

- [cache_baseline 冻结与正确性修复计划](cache-baseline-correctness-plan.md)
- [cache_baseline 正确性修订报告](cache-baseline-correctness-report.md)

命名兼容：对外材料统一使用 v1；复现命令中的 `--implementation latest`、历史文件路径及原始数据的 `latest_*` 字段仍表示 v1，保留原样以兼容现有 runner 和结果文件。
