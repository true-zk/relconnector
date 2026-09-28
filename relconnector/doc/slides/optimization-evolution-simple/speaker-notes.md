# 逐页讲解备注

## 1. RelConnector 优化演进：在有限内存下完成关系图训练

本文使用2026-09-26详细报告快照。batch不做性能排名。vanilla已有混合SQL读取、raw cache、线程执行器，未优化是相对后续系统优化。标题页兼作研究范围说明。

来源：doc/optimization-evolution-report-20260926.md
本页基于原报告快照，未重新执行实验。

## 2. 1. 优化路线：减少重复工作 → 提高并行度 → 降低管理成本

历史v1表代表早期缓存版本，不等于当前relconnector目录完整实现。cache_baseline是早期v1冻结版本，不额外作为主对照。v2有独立命名空间，当前工作区并非固定发布tag。

来源：doc/optimization-evolution-report-20260926.md
本页基于原报告快照，未重新执行实验。

## 3. 2. 瓶颈定位：GPU 主要在等待 CPU 特征准备

97.3%、89.0%、86.5%来自历史版本；不代表当前v2。不能简单将多个worker服务时间相加解释训练wall。

来源：doc/bottleneck-analysis-and-optimization.md
本页基于原报告快照，未重新执行实验。

## 4. 3. SQL 优化：把扫描变成按 ID 查找，避免过量读取

稀疏/稠密混合读取是现有基础能力，不能全部归为v1新增。后续还支持显式连续node ID。示意格子不是实际测量。

来源：doc/feature-pipeline-optimization-plan.md；cache-baseline-correctness-report.md
本页基于原报告快照，未重新执行实验。

## 5. 4. 去重与窗口：只做一次特征生产，再恢复训练输入

图示数字为教学例子，不是性能实验。历史固定window4batch/512MiB，动态窗口后按复用率和缓存覆盖调整。去重不改变训练任务样本数。

来源：doc/optimization-evolution-report-20260926.md
本页基于原报告快照，未重新执行实验。

## 6. 5. 缓存逐层上移：越接近最终特征，省掉的重复工作越多

encoded以(table,node_id)索引，整块持有但仅请求miss编码。不能缓存GNN trainable hidden states。raw节点常量、text和encoded的开销层级不同。

来源：doc/adaptive-feature-optimization-report.md
本页基于原报告快照，未重新执行实验。

## 7. 6. v1 第二轮：减少通用框架开销，并让 SQL 与编码重叠

direct固定文本与official逐值一致，但CUDA短回归loss未全面达到严格阈值。Salt/Ratebeer组合修改无法拆成单项贡献。表中训练时间不是全流程wall。

来源：doc/adaptive-feature-optimization-report.md
本页基于原报告快照，未重新执行实验。

## 8. 7. 正确性：先对齐输入，再比较性能

autocomplete目标隐藏、Ratebeer external勘误、官方随机stype抽样与dtype还原均回补。四版本fixture通过不等于全任务CUDA与ready都逐位一致。

来源：doc/cache-baseline-correctness-report.md
本页基于原报告快照，未重新执行实验。

## 9. 8. v2 多进程架构：组装进程完成“读取到特征”的完整路径

图为概念架构，Assembly N指可配置池，默认4。图中的SQLite/拓扑来源合并示意，实际采样使用共享拓扑，worker自行读取SQLite。shared encoded索引由manager管理。

来源：doc/../relconnector_v2/relconnector_v2/runtime/process_pipeline.py
本页基于原报告快照，未重新执行实验。

## 10. 9. 调度：ready 减少队头等待，但训练顺序会改变

图为示意，假设单batch训练1时间单位，无真实测量意义。ready示例0到1空闲、3到4等待。当前epoch全部完成后才训练下一轮。动态控制是进程许可，不是按需spawn回收。

来源：doc/optimization-evolution-report-20260926.md
本页基于原报告快照，未重新执行实验。

## 11. 10. 文本缓存路径：先复用，再编码；高复用时绕过共享锁

这是逻辑查找顺序示意，actual local-index/shared-cache/disk reservations需要协调；向下命中到向量输出，右下GloVe仅表示所有层未命中，不能把命中向量再编码。缓存发布回共享/磁盘省略回流线。

来源：doc/v2-salt-regression-20260926.md；v2-disk-text-cache-experiment-20260925.md
本页基于原报告快照，未重新执行实验。

## 12. 11. HM / Salt：命中缓存以后，访问成本才是瓶颈

HM等预算短测：206.0→183.5→157.7s支持RPC归因；完整结果改变容量。HM wall2.87x还包含初始化暖。Salt修复包含位置索引和共享查询批量化，相对历史v1仅1.088x。

来源：doc/v2-hm-regression-20260925.md；v2-salt-regression-20260926.md
本页基于原报告快照，未重新执行实验。

## 13. 12. Stack / Amazon：并行训练还需要控制系统资源

Stack完整结果暖磁盘；旧早期停滞缺调用栈不能武断说都是死锁。后续mmap资源链有现场/隔离复现。Amazon续跑已经合并6584936条向量，不能当冷启动。

来源：doc/v2-stack-regression-20260926.md；v2-amazon-regression-20260926.md
本页基于原报告快照，未重新执行实验。

## 14. 13. 磁盘缓存：首轮构建成本与后续复用收益分开看

两个Assembly worker、ready顺序，cold是新的embedding缓存并含首次初始化。磁盘激活并不保证每任务更快，需要多轮摊销。

来源：doc/v2-disk-text-cache-experiment-20260925.md
本页基于原报告快照，未重新执行实验。

## 15. 14. 结果总览：8 项成功记录，仍有退化和未完成任务

数据见报告§8；历史A30，v2文档RTX4090，HM local4GiB，其他多shared2GiB；各任务并非最终同版本统一重跑。Event初始化主导；Trial仍有ready等待。Ratebeer矩阵上限每轮10000，而历史完整20743，不能当完整epoch。

来源：doc/optimization-evolution-report-20260926.md
本页基于原报告快照，未重新执行实验。

## 16. 15. 内存目标是否达成？必须统计整个进程树

各进程RSS相加可能重复计算共享页。拓扑仍常驻；构建临时数组和操作系统余量需要纳入。HM encoded例外local总4GiB。默认队列16GiB不能当作适合任意16/32GiB机器的安全配置。

来源：doc/optimization-evolution-report-20260926.md
本页基于原报告快照，未重新执行实验。

## 17. 16. 下一轮实验：统一约束，拆开验证每项优化

主对照排除batch。缓存容量可不同，但必须受统一总内存限额，消融则按等预算/容量。HM目前encoded配置按数据集特配，需要披露，未来可以转为运行时策略。

来源：doc/optimization-evolution-report-20260926.md
本页基于原报告快照，未重新执行实验。

## 18. 17. 总结：每一轮优化都在解决新的关键路径问题

主来源doc/optimization-evolution-report-20260926.md。其他文档：gpu-feature-optimization-experiment.md、adaptive-feature-optimization-report.md、cache-baseline-correctness-report.md、v2各任务回归报告。原始v2结果benchmarks/v2-full-matrix-20260925/v2.jsonl。此版18页，白底黑字，简单原生流程图、表格和可编辑Office图表，无卡片、阴影、装饰页。

来源：doc/optimization-evolution-report-20260926.md
本页基于原报告快照，未重新执行实验。