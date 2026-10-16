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

## 2 修订版的图表与公式

本版保留两张图：十库历史内存对比与四阶段架构。原图3至6全部改为表格，正文编号已经同步更新。

| 位置 | 内容及来源 |
| --- | --- |
| 图1 | 历史十库 RSS；`doc/gpu-feature-optimization-experiment.md` |
| 图2 | 当前四阶段架构；采样与组装执行代码 |
| 表1 | ReDeLEx 可借鉴设计；指定提交的接口、键映射、类型、缓存与加载器实现 |
| 表2 | Event 冷初始化，`doc/v2-initialization-optimization-20260927.md` |
| 表3 | 历史等待与服务时间，`doc/bottleneck-analysis-and-optimization.md`；正确性修订前诊断 |
| 表4 | Salt/Ratebeer 重复因子及 Ratebeer 行需求；上述两类历史报告 |
| 表5 | Salt Vanilla/v1，`doc/gpu-feature-optimization-experiment.md` |
| 表6 | Salt v2 共享查询修复，`doc/v2-salt-regression-20260926.md` |
| 表7 | HM 同预算缓存对照，`doc/v2-hm-regression-20260925.md` |
| 表10 | 工作负载到策略的映射 |
| 表11 | 公式与实现边界的映射 |

十四个编号公式都是对本系统语义或成本取舍的形式化表达，不是摘录为新贡献的他人定理。其中精确统计与唯一行恢复直接对应实现；流水线、全作业内存、读取代价、缓存净收益及持久化摊销是明确简化条件的分析模型，未宣称已经实现参数拟合、全局优化或最优性证明。

## 3 实现与方法章节映射

相对于项目根目录 `/home/zk/relconnector/exps`：

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

## 5 本次深入核对的 ReDeLEx 路径

提交固定为 `3d1b392d90de6f6bf8acbac2d539d6fbaf3f32fa`，工作树干净。原论文与当前仓库实现分开归因。

- `redelex/db/interface.py`：统一生命周期、模式、键、SQL、整表和 RelBench 转换接口。
- `redelex/db/remote_db.py`：SQLAlchemy 反射、类型规范化、全表 DataFrame 查询。
- `redelex/db/relbench_db.py`：已加载的 RelBench 表注册到每实例独立 DuckDB 连接。
- `redelex/db/utils.py`：SQL 到 Pandas 类型、复合键批量重索引；仅键列连接、空值保护。
- `redelex/datasets/db_dataset.py`：下载各表、人工主键、外键映射、时间配置、原键保留选项。
- `redelex/data/semantic_schema.py`：物理类型与学习语义分离；元数据、基数与列名启发式。
- `redelex/data/graph.py`：逐表物化缓存、文本批量配置256、特征挂载图对象。
- `experiments/original/dbgnn_hyperparams.py`：既有 NeighborLoader、时间约束、可配置 worker 与持久化 worker；并非默认所有实验均多进程。

从这些机制可提出列投影/谓词下推、分块返回和离线键映射持久化，但这些扩展不是 ReDeLEx 已提供的预算化特征流水线。本次没有基于其代码做独立性能实测，没有给出其单项机制的加速倍数。

## 6 可复现文件与格式

- `manuscript.md`：修订正文；公式以 LaTeX 显示块保存。
- `build_document.py`：公式通过 MathML 转为 Word 原生 OMML，可继续编辑；不将公式截图嵌入。
- `build_figures.py`：重建三张整合图；当前配对结果与PPT共用数据源，各提供 PNG、PDF、SVG。
- `evidence-data.csv`：保留原61条图中测量值，补入19条服务时间、比例与推导值，共80条；各项注明报告或计算规则。
- `RelConnector-背景动机与方法-修订版.docx` / `.pdf`：编辑与阅读版本。

正文依赖：python-docx、latex2mathml、mathml2omml、lxml；图形依赖 NumPy 与 Matplotlib；PDF 使用 LibreOffice 导出。本次依赖安装在临时目录，不修改训练环境。没有重跑训练，也没有更改训练代码。

## 整合验证：v2每batch优化，2026-09-27

主指标取每次运行的流水线总时间除以实际完成批数，再对两次重复取均值；它包含内部启动、等待和排空，不是逐批独立延迟或严格稳态。整次运行时间除批数作为单独补充，不与主指标混用。

| 正文位置 | 数据与来源 | 判定边界 |
| --- | --- | --- |
| §6.4、表8、式10 | Ratebeer原共享A4/A5、原本地C1/C2，`benchmarks/v2-cache-before-sql-20260927/ratebeer-controlled-summary.json` | 原特征读取代码；同2GiB总预算；缓存部署为显式配置，非新自动算法 |
| §6.5、表9、式12–13 | Stack原布局A2/A3、合并布局B2/B3，`benchmarks/v2-stack-compaction-20260927/primary-summary.json` | 同逻辑缓存；逐字节校验2,891,661条向量；离线整理144.43秒另计 |
| §7.1、图3、式14 | `targeted-batch-results.json`，由两份原始汇总自动计算 | 64批/200批短程预热对照；不外推完整训练、不更新RSS |
| §7.1负结果 | Avito共享整次仅减少0.17%；Ratebeer本地上缓存前移仅再减少0.87% | 候选已归档，生产原路径保留，不将局部SQL收益作系统成果 |

新增结果均来自本地实测，不修改既有文献论断。新机制已经并入原编码缓存和持久化缓存章节；均值结果表改为配对点图，现共11表、14公式、3图。
