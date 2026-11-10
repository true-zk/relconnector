# 《面向关系表格学习的语义建模与高效训练方法研究》

# 第1章 绪论（初稿）

> 本稿依据用户提供的 MetaRTL 论文、RelX 项目实现和实验记录整理。正文采用现用名称 **RelX**；旧材料中的 `RelConnector` 仅作为历史文件名保留。参考文献编号为本章临时编号，整篇论文合稿后统一调整。

## 1.1 研究背景与意义

关系数据库是组织结构化数据的重要方式。现实场景中的预测对象通常与其他实体共同构成多表数据：用户与交易、商品与类别、论文与作者等信息分别存放在不同表中，并通过主键和外键关联。一个预测结果可能同时取决于目标记录自身的属性、关联记录的属性以及这些关联发生的时间。传统学习流程往往需要先选择连接路径、聚合规则和时间窗口，将多表数据转化为单张特征表，再进行模型训练。这一过程依赖任务经验，也会预先限定模型能够利用的关系信息。[1]

关系表格学习尝试直接利用数据库的表结构与跨表联系。关系深度学习将表中的记录映射为节点，将主外键联系映射为边，得到带有节点类型和边类型的异构图；对于带预测时间的任务，还需约束模型仅使用预测时刻允许观察的信息。在此表示上，表格编码器处理数值、类别、文本等异构字段，图模型则沿跨表关联传递信息。RelBench 为这一方向提供了公开数据库、预测任务和评价协议，使不同方法能够在统一任务上比较。[1–3]

要让关系表格学习服务于大规模数据库，需要同时处理**表示能力**和**训练效率**。从表示角度看，相同的两个实体类型可以通过不同路径发生联系，各路径所表达的语义及其对预测任务的重要性可能不同。采用多层图神经网络反复聚合邻域，虽然能够扩大可见范围，也增加了采样和消息传递成本；仅将各类邻居混合聚合，还可能难以突出任务相关的连接路径。[4–7] 因而，如何以适度的模型计算提取有用的跨表语义，是关系表格学习面临的一个问题。

从训练角度看，模型采用小批次更新，并不保证数据准备也只占用小批次的内存。若训练前加载所有原始表并物化编码特征，主机内存需求会随数据库规模增长；若改为按采样结果逐批读取，数据库查询、异构字段转换、文本编码和批次张量组装又会成为持续开销。数据准备速度不足时，训练设备需要等待输入，模型计算上的改进难以完整体现为端到端收益。已有图训练系统对采样、特征访问、传输和并行调度进行了优化，说明训练效率需要从整个数据供给过程考察。[9–13] 对关系数据库而言，按需 SQL 读取与表格特征编码又增加了专门的数据准备步骤。

基于上述背景，本文从两个层面研究关系表格学习：一方面，通过元路径语义建模提高跨表表示的针对性并控制模型计算；另一方面，通过按需数据准备和流水线执行，降低特征常驻内存需求并改善训练数据供给。两项研究针对同一类多表学习任务，但关注不同的计算环节，共同服务于关系表格学习在较大规模数据上的应用。

## 1.2 国内外研究现状

### 1.2.1 关系表格语义建模

图表示为多表关系的学习提供了统一入口。GraphSAGE 通过邻域采样与特征聚合支持小批次图学习；关系深度学习进一步利用主外键将数据库组织为异构图，并结合表格编码器与图模型完成预测。[1][4] PyTorch Frame 提供处理多模态表格字段的模块化表示，RelBench 则使不同数据库和任务上的建模效果能够被系统评价。[2][3]

在异构图研究中，元路径用节点类型与边类型的序列描述复合关系。HAN 利用注意力区分元路径语义，SeHGNN 则将部分邻域聚合移至预处理阶段，以减少训练时的重复计算。[5][6] 针对关系数据库，RelGNN 从关系图的结构特点出发设计复合消息传递，ReDeLEx 提供跨多个关系数据库研究模型表现的统一框架。[7][8] 这些研究分别推进了语义建模、模型结构和评价工具，但模型仍需在关系表格的异构字段、复杂连接路径及大规模训练成本之间作出取舍。本文第一项工作由此关注：在较少可训练消息传递计算下，怎样保留并选择与任务相关的跨表路径信息。

### 1.2.2 大规模图学习与训练数据流水线

大规模图训练的开销并非只来自模型前向和反向传播。SALIENT 对采样、特征切片及数据传输进行协同优化，BGL 进一步讨论特征读取、缓存与数据准备阶段间的资源竞争。[9][10] 当图数据或特征不能全部放入内存时，Ginex 利用采样计划组织后续特征访问和缓存，MariusGNN 则从存储布局、数据移动和采样等方面支持外存图训练。[11][12] 通用机器学习输入流水线研究也表明，并行度、预取和缓存需要结合资源约束与真实瓶颈调整。[13]

上述工作为本研究提供了采样计划、存储层级和流水线调度方面的启示。对于直接以关系表格为数据源的训练，还需根据采样节点到数据库中查询原始行，处理不同类型的列，并重建模型所需的批次表示。不同数据库的表规模、文本长度、访问重复率和任务时长差异明显，使固定的数据准备及缓存配置难以在所有负载上保持相同效果。本文第二项工作因而聚焦于关系表格训练中的特征供给路径：在不预先常驻全部编码特征的条件下，如何组织读取、编码、缓存和并行执行。这里是从已有图训练系统与本项目工作负载分析得到的研究定位，并不意味着现有系统缺少小批次训练或缓存机制。

## 1.3 研究问题与目标

本文围绕两个相互关联的问题展开。

**问题一：跨表关系如何得到有效表达。**关系数据库中的同一目标实体可以沿多种主外键路径连接到其他表，不同路径承载的语义并不相同。依赖深层可训练消息传递扩大关系范围会增加训练开销，而不区分路径的聚合可能削弱有用信号。本文希望结合表格字段表示和元路径语义，以较轻量的方式生成并选择跨表特征，提高关系预测任务的建模能力。

**问题二：大规模训练如何持续获得所需数据。**全量特征物化具有训练期间读取方便的优势，但主机内存需求较高；按需读取可以减小常驻工作集，却可能使特征准备落后于训练消费。本文希望在保持任务划分、时间可见性、采样和特征语义的前提下，减少重复数据准备，并通过受预算约束的流水线使采样、组装和训练在不同批次间重叠。

两个问题分别对应模型计算路径与训练数据路径。本文将两项工作作为独立的研究内容评价：MetaRTL 主要使用预测效果、模型开销与模块消融验证；RelX 主要使用正确性、端到端时间、整作业内存与系统消融验证。两者使用相近的关系学习背景，并不意味着当前实验已经完成 MetaRTL 在 RelX 上的联合部署。

## 1.4 本文的研究内容与主要工作

围绕第一个问题，本文研究基于元路径的关系表格语义建模方法 **MetaRTL**。该方法先使用表格编码器与浅层异构图网络进行有限轮次的预训练，得到节点初始表示；随后通过非参数传播构造多条元路径对应的关系特征，并使用 MetaAttn 融合路径语义及全局节点信息。其核心是将初始表示学习与后续语义聚合分开，使不同跨表路径能够参与任务相关的表示生成。本文将通过分类、回归及模块消融实验分析该方法的效果与计算开销。具体模型结构和实验设置见第3章。

围绕第二个问题，本文设计关系表格学习的按需训练框架 **RelX**。框架将可复用的结构与统计准备同在线训练过程区分，在线阶段依次执行种子与邻域采样、特征读取与批次组装、模型训练。不同批次的阶段通过有界队列衔接，并使用独立组装进程、请求去重及多层特征复用降低重复工作和训练等待。RelX 保留现有模型与采样的训练语义，重点优化原始关系表到模型输入之间的数据路径。本文将按冷/热状态分别测量初始化与训练过程，并通过端到端时间、整作业内存、输入一致性和消融实验分析收益及适用边界。框架设计与实验分别见第4章和第5章。

两项工作的联系体现在统一的关系表格学习对象：MetaRTL 研究跨表信息**怎样形成有效表示**，RelX 研究模型训练时**怎样及时获得所需数据**。现阶段的实验支持分别讨论两项成果；对两者的直接集成及联合效果，本文不作未经验证的结论。

## 1.5 论文组织结构

全文共六章。第1章介绍研究背景、相关研究、问题与主要工作。第2章给出关系数据库图表示、异构表格编码、元路径、时序邻域采样和训练数据流水线的基础概念，并详细比较与本文相关的方法和系统。第3章介绍 MetaRTL 的模型结构、训练过程及预测实验。第4章说明 RelX 的总体架构、按需特征准备、缓存和流水线执行机制。第5章给出 RelX 的正确性、时间、内存和消融实验，并分析不同数据库负载下的表现。第6章总结全文工作，讨论方法与系统的适用范围以及后续研究方向。

## 本章参考文献（待按学校模板统一格式）

[1] Fey M, Hu W, Huang K, et al. [Position: Relational Deep Learning—Graph Representation Learning on Relational Databases](https://proceedings.mlr.press/v235/fey24a.html). ICML, 2024.

[2] Robinson J, Ranjan R, Hu W, et al. [RelBench: A Benchmark for Deep Learning on Relational Databases](https://proceedings.neurips.cc/paper_files/paper/2024/hash/25cd345233c65fac1fec0ce61d0f7836-Abstract-Datasets_and_Benchmarks_Track.html). NeurIPS Datasets and Benchmarks, 2024.

[3] Hu W, Yuan Y, Zhang Z, et al. [PyTorch Frame: A Modular Framework for Multi-Modal Tabular Learning](https://arxiv.org/abs/2404.00776). arXiv:2404.00776, 2024.

[4] Hamilton W L, Ying R, Leskovec J. [Inductive Representation Learning on Large Graphs](https://papers.nips.cc/paper_files/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html). NeurIPS, 2017.

[5] Wang X, Ji H, Shi C, et al. [Heterogeneous Graph Attention Network](https://arxiv.org/abs/1903.07293). WWW, 2019.

[6] Yang X, Yan M, Pan S, et al. [Simple and Efficient Heterogeneous Graph Neural Network](https://ojs.aaai.org/index.php/AAAI/article/download/26283/26055). AAAI, 2023.

[7] Chen T, Kanatsoulis C, Leskovec J. [RelGNN: Composite Message Passing for Relational Deep Learning](https://proceedings.mlr.press/v267/chen25ad.html). ICML, 2025.

[8] Peleška J, Šír G. [REDELEX: A Framework for Relational Deep Learning Exploration](https://arxiv.org/abs/2506.22199). arXiv:2506.22199, 2025.

[9] Kaler T, Stathas N, Ouyang A, et al. [Accelerating Training and Inference of Graph Neural Networks with Fast Sampling and Pipelining](https://proceedings.mlsys.org/paper_files/paper/2022/hash/afacc5db3e0e85b446e6c7727cd7dca5-Abstract.html). MLSys, 2022.

[10] Liu T, Chen Y, Li D, et al. [BGL: GPU-Efficient GNN Training by Optimizing Graph Data I/O and Preprocessing](https://www.usenix.org/conference/nsdi23/presentation/liu-tianfeng). NSDI, 2023.

[11] Park Y, Min S, Lee J W. [Ginex: SSD-enabled Billion-scale Graph Neural Network Training on a Single Machine via Provably Optimal In-memory Caching](https://www.vldb.org/pvldb/vol15/p2626-park.pdf). PVLDB, 15(11), 2022.

[12] Waleffe R, Mohoney J, Rekatsinas T, Venkataraman S. [MariusGNN: Resource-Efficient Out-of-Core Training of Graph Neural Networks](https://doi.org/10.1145/3552326.3567501). EuroSys, 2023.

[13] Kuchnik M, Klimovic A, Simsa J, et al. [Plumber: Diagnosing and Removing Performance Bottlenecks in Machine Learning Data Pipelines](https://proceedings.mlsys.org/paper_files/paper/2022/hash/d0e90e9a9310570dfa643aa3b2da6e89-Abstract.html). MLSys, 2022.

## 写作依据与定稿前核对（不纳入论文正文）

- MetaRTL 的方法及实验口径来自用户提供的 [MetaRTL 原稿](/home/zk/.codex/attachments/4aa4a3da-74f8-4da9-98fc-133303b7cb0f/Meta_IJCNN.pdf)；本稿未据附件文件名判断发表状态。
- RelX 的机制来自项目内的 [架构与方法修订稿](paper/relconnector-background-method-revised-20260927/manuscript.md)及[实现状态](../relconnector_v2/STATUS.md)。正式论文应以冻结的最终源码和实验记录为准。
- 历史十库成绩具有不同轮数、缓存冷热状态和代码版本；[性能复盘](v2-dataset-performance-review-20260927.md)可用于解释瓶颈，正式性能结论还需统一条件的对照。v2 的主进程 RSS 不代表整作业内存；近期[短程 ROI 实验](v2-roi-all10-100x5-20260929.md)提供进程树 PSS，但不代表完整长程任务。
- 第1.2节是绪论中的研究现状概述。第2章应补充各工作的问题定义、方法差异和局限，避免简单重复本章。
