# 文献检索、证据映射与写作边界

检索与代码核对日期：2026-09-27。正文为 `manuscript.md`；本文件是写作与复核附件，不是完整相关工作章节。

## 1 检索路径

围绕三组问题检索：关系数据库为何需要关系学习；现有统一框架如何加载数据；大规模图训练如何处理采样、特征访问与流水线。使用关系深度学习、RelBench、ReDeLEx、memory、scalability、GNN data preparation、feature gathering、sampling、caching 等组合关键词。正式论述依据原论文、官方会议论文集及作者仓库，未以搜索摘要或二手综述替代机制依据。

| 文献 | 阅读重点 | 用于本稿的论证 | 不据此作出的推断 |
| --- | --- | --- | --- |
| [RDL position paper, ICML 2024](https://proceedings.mlr.press/v235/fey24a.html) | 关系表示、主外键图、跨表学习动机 | 背景：表关系为何包含预测信息 | 不宣称 RelConnector 发明关系图表示 |
| [RelBench, NeurIPS 2024](https://cs.stanford.edu/~jure/pubs/relbench-neurips24.pdf) | 基准、任务、时间划分、统一接口 | 背景：可重复建模基础 | 原始七库与本地十库不混为一谈 |
| [ReDeLEx, 2025](https://arxiv.org/html/2506.22199v1) | §3 工作流、§4.1 时序邻域采样 | 统一模型研究框架已存在 | 原文没有证明“完全不能扩展”；不能说其没有小批次训练 |
| [ReDeLEx pinned code](https://github.com/jakubpeleska/redelex/tree/3d1b392d90de6f6bf8acbac2d539d6fbaf3f32fa) | db_dataset.py 全表读取；data/graph.py 特征物化 | 将扩展性问题落到指定加载路径 | 不代表仓库所有路径或未来版本 |
| [SALIENT, MLSys 2022](https://proceedings.mlsys.org/paper_files/paper/2022/file/afacc5db3e0e85b446e6c7727cd7dca5-Paper.pdf) | §3 瓶颈；§4 采样、切片及传输重叠 | 用端到端供给视角理解训练 | 不把其采样内核优化当作本系统贡献 |
| [BGL, NSDI 2023](https://www.usenix.org/system/files/nsdi23-liu-tianfeng.pdf) | 数据准备、缓存、资源隔离；§5.5 | 并行阶段之间也会竞争资源 | 不借用其加速数字为本系统背书 |
| [Ginex, PVLDB 2022](https://arxiv.org/pdf/2208.09151) | inspector-executor；按采样计划管理特征访问 | 计划可指导后续数据准备 | 本系统没有实现其最优替换算法 |
| [MariusGNN, EuroSys 2023](https://arxiv.org/pdf/2202.02365) | 存储层级、COMET、分区与采样 | 指明完整外存图训练已有工作 | 本系统拓扑仍驻内存，不主张相同能力 |
| [PyTorch Frame, 2024](https://arxiv.org/abs/2404.00776) | 异构表格表示与模型模块 | 区分静态张量化与可训练编码器 | 静态缓存不能复用随参数变化的表示 |
| [RelBench v2, 2026](https://arxiv.org/abs/2602.12606) | 基准扩展、ReDeLEx 及外部集合整合 | 辅助确认当前基准演进；未作为核心贡献对照 | 其 v2 与 RelConnector v2 无关；官方归为 ICLR DATA-FM workshop |

## 2 六张图的逐项来源

| 图 | 来源 | 口径与限制 |
| --- | --- | --- |
| 图1 内存 | `doc/gpu-feature-optimization-experiment.md` 十库汇总 | 历史单进程峰值；超时项是终止前峰值；不画不可比的 v2 主进程峰值 |
| 图2 架构 | 当前 `relconnector_v2` 的 prepare、sampling、process_pipeline、features 实现 | 逻辑架构图，无模拟数据；实际仍常驻全图拓扑 |
| 图3a 初始化 | `doc/v2-initialization-optimization-20260927.md`；两个 `benchmarks/v2-event-init-*20260927/result.json` | 独立空应用缓存、未清操作系统缓存、单次前后；组件可堆叠 |
| 图3b 等待 | `doc/bottleneck-analysis-and-optimization.md` §3.1 | 正确性修订前的历史诊断；不是当前最终基准 |
| 图4 需求 | 上述历史诊断 Ratebeer 计数；`gpu-feature-optimization-experiment.md` Salt / Ratebeer 重复因子 | 表示需求重复与剩余行数，不是采样内核加速 |
| 图5 Salt v1 | `doc/gpu-feature-optimization-experiment.md` Salt 表 | 3170批次；操作时间重叠；历史数值等价性未完全验收 |
| 图6a/b Salt v2 | `doc/v2-salt-regression-20260926.md` 完整自适应复测 | 同为6340批次；共享查询批量化与索引联合修复；worker时间不可与wall相加 |
| 图6c HM | `doc/v2-hm-regression-20260925.md` 定向对照 | 同为1000批次、2GiB编码缓存总预算、复用初始化；单次对照无误差棒 |

## 3 实现与方法章节映射

相对于项目根目录 `/home/zk/relconnector/relconnector`：

| 机制 | 代码 |
| --- | --- |
| 数值预算分组、精确分位数与回退 | `relconnector_v2/relconnector_v2/features/schema.py`，`_cache_key` 与数值统计构建路径 |
| 既有时序采样与确定性种子派生 | `relconnector_v2/relconnector_v2/sampling/pyg_lib.py` |
| 单采样进程、多组装进程、队列预算与轮次边界 | `relconnector_v2/relconnector_v2/runtime/process_pipeline.py` |
| 合并唯一行、逆映射、稀疏/稠密数据库访问 | `relconnector_v2/relconnector_v2/features/sql.py` |
| 编码与最终批次恢复 | `relconnector_v2/relconnector_v2/features/assembler.py` |
| 共享文本与首次编码预留 | `relconnector_v2/relconnector_v2/features/shared_text.py` |
| 本地位置索引的启停与元数据预算 | `relconnector_v2/relconnector_v2/features/adaptive_text_index.py` |
| 冻结文本、默认磁盘启发式参数 | `relconnector_v2/relconnector_v2/features/text.py` |
| 持久化激活条件与缓存命名空间 | `relconnector_v2/relconnector_v2/features/disk_text.py` |
| 编码行缓存准入与本地/共享配置 | `relconnector_v2/relconnector_v2/features/cache.py`、`shared_encoded.py`、`process_worker.py` |

只读核对当前代码，没有在本次写作中更改训练实现，也没有重跑训练。原报告中的单元测试数量和一致性结果均明确归属于历史验证，不能理解为本次重新执行的测试。

## 4 应保留的实验边界

1. 时间比较需同时记录硬件、CPU线程、批次数、完整/截断轮次、缓存冷热和并发负载。早期诊断注明 A30，部分后续定向实验使用 RTX 4090，不能用跨系列结果直接归因代码改动。
2. 阶段服务时间可能并行或嵌套；只有不重叠的 prepare 组件可以画堆叠总时间。
3. v2 的 `RUSAGE_SELF` 仅统计主进程；作业内存需另行测量。简单相加各进程RSS也会重复计算共享页。
4. Event 冷初始化3.57倍属于同机单次前后对照；旧冷启动与新热启动的巨大差值不是初始化算法加速。
5. Arxiv 磁盘缓存短测只观察到204.7→190.5秒；首次写入396.6秒。不能把跨系列全部提升归给磁盘缓存。
6. Salt完整v2修复3.51倍是相对原v2；v1已获得大量复用收益。应分别解释两个阶段。
7. HM本地4GiB是显式配置；同预算因果对照采用2GiB三组。自动部署缓存位置属于后续方向。
8. Ratebeer的v2为两轮各截断10000批次，共20000批次，15444.93秒；历史v1四小时完成15686/20743批次。不能声称同任务同超时预算下只有v2成功。
9. 既有计划文档中的多采样进程、完整描述符零复制、跨轮混合窗口、词元级缓存等，不应据计划文字写成已实现。
10. 作业级内存、统一硬件重复实验、准确率与顺序影响仍需完整实验章节解决；本稿以已有证据建立动机和解释方法。

## 5 可复现文件

- `build_figures.py`：六张原始图的可编辑绘图脚本。
- `figure-data.csv`：图中实验数值、单位与来源。
- `build_document.py`：Markdown 转 Word 的排版脚本。
- `figures/`：每图 PNG 预览、PDF 与 SVG 矢量版本。
- `manuscript.md`：正文源文件；Word 与 PDF 为其排版导出。
