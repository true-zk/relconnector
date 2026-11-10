# 学位论文章节草案：面向关系表格学习的按需数据准备与训练流水线

整理日期：2026-10-23。本文档只规划学位论文中 **RelX 按需训练框架**这一研究部分。学位论文还包含此前的 MetaRTL 模型工作；两部分共同的背景、总题目与全篇衔接见[双研究部分背景初稿](thesis-background-metartl-and-framework.md)。历史实验与稿件中的 `RelConnector` 是 RelX 的旧称，文件路径仍保留原名作为证据索引。

## 一、建议的研究问题与章节主张

**研究问题：**在关系数据库的图学习训练中，如何避免原始表和编码特征全量常驻内存，同时让采样后的特征供给尽量跟上模型训练，并保持关系、时间与特征语义一致？

**本章主张：**将离线元数据准备与在线的采样、特征准备、模型训练分开；在线阶段传递轻量采样计划，按需读取和编码特征，使用有界队列、多进程组装与分层复用控制吞吐和内存。这里优化的是训练数据路径，不声称提出新的图模型或邻居采样算法。当前图拓扑仍驻留主机内存，队列和缓存预算也不等于整个作业的严格内存上限。

建议题名：**“面向关系表格学习的按需特征供给与流水线训练方法”**。若整篇论文偏系统实现，可用“设计与实现”；若偏实验研究，可用“性能分析与优化”。

## 二、章节大纲

### X.1 引言与问题定义

- 关系表格学习需要跨表主外键、异构列与预测时间；RelBench 提供任务和评价协议。[1–3]
- 全量特征物化的容量压力，与按需加载后的数据准备瓶颈。用 Amazon 历史峰值 RSS（Batch 103.82 GiB、v1 7.23 GiB）和阶段等待分析引出问题；注明这是历史不同版本的观测，不作为最终配对结果。
- 明确研究对象：单机、SQLite 数据源、图拓扑驻主机内存、训练按采样批次供给节点特征；给出模型语义与预算约束。
- 列出可由实验支持的贡献：①按需特征供给与语义一致性；②分阶段、有界的并行流水线；③根据复用与访问成本设计的缓存路径；④宽表冷初始化优化。将“全自动最优策略”留作未来工作。

### X.2 背景、相关工作与设计目标

- **关系学习与基准：**关系数据库到时序异构图、RelBench、ReDeLEx、PyTorch Frame。[1–4, 12]
- **图训练数据路径：**GraphSAGE 邻域采样；SALIENT、BGL 对采样、特征读取、传输和流水线的优化。[5–7]
- **外存与缓存：**Ginex、MariusGNN 的计划驱动特征访问与存储层级；Plumber 的输入流水线诊断。[8–10]
- 用一张对比表写清对象和边界：上述图系统主要处理既有图及特征存储；本项目还需处理 SQL 行读取、异构列转换、冻结文本编码与批次重建。该区别应表述为**侧重点不同**，不要声称已有系统没有小批次训练或缓存。
- 设计目标：语义正确、内存可控、端到端耗时低、冷/热运行可解释、实验可复现。

### X.3 系统架构与语义约束

- 给出四阶段总图：离线 schema/拓扑/统计准备 → 种子与时序邻域采样 → 特征读取、静态编码与 batch 组装 → GPU 模型训练。
- 在线三阶段的生产者—消费者关系：采样进程、组装进程池、训练进程；说明批内依赖与跨批重叠、采样计划内容、严格顺序和就绪顺序两种训练模式。
- 有界 seed、plan、ready 队列及在途窗口；解释缓存 payload 预算、进程内对象、共享页和 OS page cache 的不同口径。
- 不变性契约：相同任务 split、时间可见性、邻居 fanout、节点/边 ID、特征类型、静态编码规则和模型输入。区分冻结文本向量与随参数更新的可训练编码器。

### X.4 按需数据准备与复用方法

- **冷初始化：**仅准备必要的拓扑和统计量；宽表数值列按内存预算分组扫描，精确分位数选择及单列回退。报告 Event 冷初始化 2164.00→606.01 秒的单次前后测量，并注明未清 OS 文件缓存。
- **采样计划与需求压缩：**按节点唯一行合并请求、保留逆映射恢复原批次语义；短窗口跨批复用；SQL 行读取的稀疏/稠密选择。这里减少物理特征访问，不改邻居采样算法。
- **特征组装与并行：**组装进程持有 SQLite reader、原始行缓存、编码器，避免跨进程传递 DataFrame 和原始字符串；按需生成 TensorFrame 并交给训练进程。
- **缓存层次：**原始行、已编码行、共享文本向量、进程本地位置索引、磁盘文本向量。说明缓存键、版本失效、准入、总预算、命中成本与首次写入成本；显式配置与运行时自适应逻辑分开表述。
- 用工作负载表说明策略适用条件：长文本/跨轮复用，重复实体多，低复用大邻域，宽数值表，短任务。只把有受控对照的具体机制写成独立收益。

### X.5 实验设计与结果分析

- **实验对象与对照：**Batch（全量物化特征，但仍可小批次训练）、Vanilla、v1、v2；列出十个本地数据库、任务、模型、硬件与完整配置。这十库不能直接称为 RelBench 原论文的七库，也不能把后续 RelBench v2 与本项目 v2 混淆。[2, 11]
- **主指标：**冷/热端到端时间、实际完成 batch 的吞吐、训练端等待时间、进程树 PSS/或 cgroup 峰值、磁盘占用和任务质量。单独报告准备时间、在线训练时间、启动和排空；各 worker 服务时间不可相加为墙钟时间。
- **公平对照：**固定源码快照、机器、batch 数、epoch、采样种子、缓存预算与初始状态；至少多次独立重复，交错执行顺序；记录完整/截断轮次及失败。准确率、loss 和输入张量一致性共同验证语义。
- **消融：**组装 worker 数、普通特征缓存、文本缓存、跨批窗口、磁盘持久化与缓存位置。同预算配对比较；缓存层有收益遮蔽，不能累加单项加速比。参考现有消融协议。
- **已有证据放置：**历史十库矩阵和四版本复盘用作动机及工作负载分析；Event 冷初始化作为定向结果；2026-09-29 的十库 5×100 固定回放 ROI 缓存实验可作为受控短程结果（相对同回放全关缓存，总时间 3992.15→2739.55 秒，7/10 库改善），注明 F1/Event 实际批数不足 100、三个库回归，且不外推完整训练。
- **结果讨论：**按瓶颈类型分析加速与退化；给出内存—吞吐—磁盘空间权衡。ROI 实验已测进程树 PSS，Salt 42.64 GiB、Arxiv 22.90 GiB，说明仍不能笼统声称“32 GiB 内稳定运行所有任务”。

### X.6 本章小结与局限

- 总结已验证的按需供给、并行组装、复用机制和冷初始化结果。
- 局限：拓扑尚未外存化；历史十库不同冷热和轮数；短程缓存实验不能代表长跑；当前自动策略不是全局最优；整作业内存与最终代码的统一配对评测仍需补齐。
- 与学位论文后续章节衔接：如果另有模型算法或系统部署章节，在此只提供数据供给接口与实验基础。

## 三、优先补齐的图表与证据

| 优先级 | 图表/证据 | 当前材料与动作 |
| --- | --- | --- |
| 必需 | 系统四阶段图、在线三阶段时序图、语义不变性表 | 基于现有修订稿和代码重绘，正文使用 RelX，历史来源保留原文件名。 |
| 必需 | 同条件冷/热端到端对照与质量结果 | 当前历史十库比较条件不齐；冻结版本后重新配对。 |
| 必需 | 整作业峰值内存 | 采用进程树 PSS 或隔离 cgroup；不可用主进程 RSS 代表整作业。 |
| 必需 | worker/缓存/窗口消融 | 使用现有 `v2-ablation-design-20260928.md` 协议并记录实际命中与激活。 |
| 可先写 | Event 初始化、ROI 十库固定回放、Arxiv 摘要编码微基准 | 分别标为单次定向、短程和微基准，避免互相代替。 |

## 四、参考文献初选与引用位置

以下按本章首次可能出现的顺序编号，给出可核对的原论文或正式论文页；最终按学校要求转换为 GB/T 7714 或指定格式。网页源码是**实现证据**，不替代论文条目。

| 编号 | 文献 | 用于本章 |
| --- | --- | --- |
| [1] | Fey M, et al. [Position: Relational Deep Learning—Graph Representation Learning on Relational Databases](https://proceedings.mlr.press/v235/fey24a.html). ICML, 2024. | 关系数据库转时序异构图的研究动机。 |
| [2] | Robinson J, et al. [RelBench: A Benchmark for Deep Learning on Relational Databases](https://proceedings.neurips.cc/paper_files/paper/2024/hash/25cd345233c65fac1fec0ce61d0f7836-Abstract-Datasets_and_Benchmarks_Track.html). NeurIPS Datasets and Benchmarks, 2024. | 任务、时间约束、基准与评价协议。 |
| [3] | Peleška J, Šír G. [REDELEX: A Framework for Relational Deep Learning Exploration](https://arxiv.org/abs/2506.22199). arXiv:2506.22199, 2025. | 多数据库统一研究框架；引用其论文机制时不要把指定版本代码的全表读取直接归为论文结论。 |
| [4] | Hu W, et al. [PyTorch Frame: A Modular Framework for Multi-Modal Tabular Learning](https://arxiv.org/abs/2404.00776). arXiv:2404.00776, 2024. | 异构列与 TensorFrame；区分静态编码和可训练模块。 |
| [5] | Hamilton W L, Ying R, Leskovec J. [Inductive Representation Learning on Large Graphs](https://papers.nips.cc/paper_files/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html). NeurIPS, 2017. | 邻域采样与聚合的经典背景。 |
| [6] | Kaler T, et al. [Accelerating Training and Inference of Graph Neural Networks with Fast Sampling and Pipelining](https://proceedings.mlsys.org/paper_files/paper/2022/hash/afacc5db3e0e85b446e6c7727cd7dca5-Abstract.html). MLSys, 2022. | SALIENT：采样、特征切片和传输流水线。 |
| [7] | Liu T, et al. [BGL: GPU-Efficient GNN Training by Optimizing Graph Data I/O and Preprocessing](https://www.usenix.org/conference/nsdi23/presentation/liu-tianfeng). NSDI, 2023. | 数据预处理、缓存与阶段间资源竞争。 |
| [8] | Park Y, Min S, Lee J W. [Ginex: SSD-enabled Billion-scale Graph Neural Network Training on a Single Machine via Provably Optimal In-memory Caching](https://www.vldb.org/pvldb/vol15/p2626-park.pdf). PVLDB, 15(11), 2022. | 采样计划指导特征访问与内存缓存。 |
| [9] | Waleffe R, Mohoney J, Rekatsinas T, Venkataraman S. [MariusGNN: Resource-Efficient Out-of-Core Training of Graph Neural Networks](https://doi.org/10.1145/3552326.3567501). EuroSys, 2023. | 外存图训练及存储层级；界定本项目拓扑驻内存的范围。 |
| [10] | Kuchnik M, et al. [Plumber: Diagnosing and Removing Performance Bottlenecks in Machine Learning Data Pipelines](https://proceedings.mlsys.org/paper_files/paper/2022/hash/d0e90e9a9310570dfa643aa3b2da6e89-Abstract.html). MLSys, 2022. | 输入流水线瓶颈、并行/预取/缓存资源约束。 |
| [11] | Gu J, et al. [RelBench v2: A Large-Scale Benchmark and Repository for Relational Data](https://arxiv.org/abs/2602.12606). arXiv:2602.12606, 2026. | 后续基准扩展背景；与项目 v2 名称区分。 |
| [12] | Wang M, et al. [4DBInfer: A 4D Benchmarking Toolbox for Graph-Centric Predictive Modeling on RDBs](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2fd67447702c8eff5683dda507a1b0a2-Abstract-Datasets_and_Benchmarks_Track.html). NeurIPS Datasets and Benchmarks, 2024. | 关系预测任务与统一评测工具的相关工作。 |

**阅读顺序：**先读 [1][2][3] 确定对象和任务，再读 [6][7][8][9] 梳理系统差异；写实验方法时读 [10]。若正文篇幅有限，[11][12] 可放综述章节，[5] 只在介绍邻域采样时简引。

## 五、项目内写作证据入口

- 已有背景/方法源稿：[修订版 manuscript](paper/relconnector-background-method-revised-20260927/manuscript.md)；它已写到方法和定向验证，可重用措辞与图，但实验条件需按最终版本核对。
- 文献与实现映射：[research-and-evidence](paper/relconnector-background-method-revised-20260927/research-and-evidence.md)。
- 十库历史对照：[v2 性能复盘](v2-dataset-performance-review-20260927.md)；结果来自混合缓存状态及不同 epoch/批数。
- 消融协议：[v2 消融设计](v2-ablation-design-20260928.md)。
- 近期短程受控数据：[ROI 十库 5×100 实验](v2-roi-all10-100x5-20260929.md)。
- 新的局部瓶颈证据：[Arxiv Abstract 分阶段实验](../arxiv_abstract_profile/REPORT.md)；紧凑 token 仍是原型，不能写成生产系统已取得端到端收益。
