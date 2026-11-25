# 在线特征读取约束与迁移方案

## 不可破坏的运行约束

RelX 的训练数据源是关系型数据库。训练中的每个 sampled batch 必须根据子图节点 ID，向数据库读取所需的原始特征；已命中的**部分特征缓存**可以免去对应的数据库查询。不得把所有业务表的原始特征预读到进程内存，再让训练只从这份全量副本取数。

图拓扑索引、节点 ID 到数据库业务键的映射、训练特征的统计量与词表是准备阶段的元数据，不等于原始特征快照。统计量与词表可以流式从训练可见数据拟合，内存用量应由状态大小而非表的全部原始特征决定。

当前 `src/relx/batch/feature_store.py` 在初始化时读取每张特征表，并把默认特征整体注册到 `NativeFeatureStore`。`NativeBatchPipeline` 随后只访问该内存副本。这是过渡原型，**不符合上面的约束，不能作为正式训练路径或端到端性能结论**。

## 当前已接通的在线路径

`relx.run()` 现在使用 `PygLibMultiHopSampler`、`OnlineBatchIterator` 和 `OnlineFeatureStore`。准备阶段只保留图索引、节点业务键和训练可见特征的拟合状态；拟合按数据库结果块流式进行。Python 采样后把各表节点 ID 交给 `NativeOnlinePipeline`。C++ worker 每个线程持有一条 SQLite 或 MySQL 连接，按表和组合业务键去重后，以参数化批量 SQL 查询未缓存节点；SQLite 每次最多查询 500 行，MySQL 每次最多查询 128 行，并按采样顺序恢复输出。C++ 内完成数值、类别、多类别、时间、向量及文本处理。结果按提交顺序返回，已完成结果同时按 batch 数和特征字节数限流。

每个 C++ worker 可缓存最近读取的节点特征，默认总额 64 MiB，可用 `training.feature_cache_bytes: 0` 关闭。缓存命中时不查询该节点，未命中仍回源数据库。这个缓存只在当前 worker 会话中有效；如果训练期间会修改业务数据，应关闭缓存或使用固定数据库快照。文本模式 `model.params.text_mode: glove` 在准备时从本地 SentenceTransformer GloVe 模型导出词表和权重，训练 batch 的分词与平均词向量在 C++ 完成；可设置 `glove_cache_bytes` 和 `glove_disk_cache`。后者默认为关闭，磁盘缓存位于导出目录，改变模型权重后需重新导出并清理旧文本缓存。

GloVe 首次导出要求 `sentence-transformers` 已安装，且 `model.params.glove_model_path` 指向本地模型或默认模型已存在于本地 Hugging Face 缓存；导出不会在训练时自动下载权重。训练模型可设为 `model.name: rdl` 或 `model.name: relgnn`，并提供 `model.params.num_neighbors`，例如 `[8, 4]`。

原生在线路径目前支持文件型 SQLite 与 MySQL，时间特征要求 UTC。准备阶段的特征状态拟合、pyg-lib 采样仍在 Python；用户自定义 Python `FeatureProcessor` 的原始列由 C++ 读库，批次返回后调用用户的 Python `transform()`。PostgreSQL 原生适配器、数据库快照协调及与 v2 的完整同语义性能对照仍待实现。已有全表快照原型的性能数字不能当成在线路径收益。

MySQL 使用本机 `mysqlclient` C API 编译；`scripts/build_relx_native.py` 在找到 `mysql_config` 时自动启用。YAML 的 `database.url` 可使用 `mysql+pymysql://...`，在线 batch 会自动选 `MySqlFeatureReader`。当前 URL 额外参数仅接受 `charset=utf8mb4` 与 `unix_socket`；其他参数会明确报错。MySQL C++ 会话设为 UTC，因此若数据库使用 `TIMESTAMP` 类型，准备阶段的 Python/ConnectorX 会话也应使用 UTC，避免预测时间与批次特征的时区解释不一致。

## 目标流水线

```text
Python seed reader → pyg-lib 多跳采样 → 有界 sampled queue
                                          ↓
                                  C++ assembler workers
                                          ↓
                 缓存命中 / 按节点业务键批量查询原数据库
                                          ↓
                     数值、类别、时间、文本处理与缓存写入
                                          ↓
                 有界 ready queue → Python 模型训练
```

Python 仅提交 pyg-lib 采样结果并消费完整批次；中间样本由 C++ worker 持有。批次队列按数量和估计字节数限流，worker 可乱序完成，训练侧按提交顺序获取。输出包括图局部索引、节点特征张量、缺失掩码与 ragged offsets。模型专属的轻量 PyTorch 对象包装仍在 Python。

当前实现按 `training.prefetch_batches` 限制在途 batch 数，按 `training.prefetch_bytes`（默认 256 MiB）限制 C++ 已完成结果的特征载荷；原生 worker 数为 `min(prefetch_batches, feature_workers × feature_native_workers)`。为避免乱序结果造成死锁，轮到消费的 batch 允许独自超过字节阈值。线程正在处理的结果、Python 采样计划及 PyTorch 张量不在这个字节计数内，因此它不是进程内存硬上限。

## 数据库与缓存边界

1. `GraphIndex` 必须保留 `table + internal_node_id → identity_key` 的反向映射；一个 identity key 可以是多列和字符串。assembler 先按表去重 sampled ID，再把缓存未命中的 ID 转为业务键，使用参数化批量 SQL 查询必要列，最后按 sampled ID 顺序还原输出。
2. 定义 `NativeFeatureReader` 接口；每个 worker 持有自己的只读连接。SQLite、MySQL、PostgreSQL 等数据库由独立适配器处理标识符、参数、类型与一致性快照。ConnectorX 的现有实验 C ABI 缺少安全的参数与错误返回，不能直接承担正式在线读取。
3. 当前节点缓存只存当前 worker 会话从数据库读取并处理过的特征，以表及内部节点 ID 索引；它依赖同一会话内数据库内容与特征规格不变。磁盘文本缓存以原文为键，必须与导出的 GloVe 权重配套使用；切换权重时清理旧缓存。更严格的数据库版本与处理器指纹仍需后续补齐。
4. 特征状态拟合允许准备阶段流式查询数据库，但不得保留全表原始列。模型训练若要求读到固定数据版本，应由只读快照或显式数据库版本约束保证，而不是假设不同 worker 的独立查询自动一致。

## 迁移顺序

1. **已完成：**训练默认路径从 `FeatureStore` 全表快照切走；状态拟合改为有界流式扫描。保留当前快照代码仅作对照基准，`relx.run()` 不使用它。
2. **已完成核心协议：**新增 `NativeOnlinePipeline` 接收 pyg-lib 的采样节点 ID，C++ worker 执行读库和变换；提交/返回按序、有界队列、异常和取消传播已接通。旧 `NativeBatchPipeline` 仅保留快照基准用途。
3. **SQLite 和 MySQL 已接通：**`NativeFeatureReader` / `NativeFeatureSession` 分离读取器配置和 worker 连接；两种数据库均已实现批量参数化 SQL、连接复用、节点特征缓存。MySQL 5.7 的隔离实例已通过组合键、跨块查询、长文本、BLOB、缺失行、多线程队列及 YAML 到训练报告测试。下一步补 PostgreSQL，并在真实 rel-f1 MySQL 数据库核对完整输出和吞吐。
4. **默认变换和 GloVe 已接通：**C++ 文本内存缓存及可选 SQLite 磁盘缓存已实现；从 rel-f1 各文本列取 250 条不同文本与本地 SentenceTransformer 300 维 GloVe 对照，最大绝对误差为 `5.96e-8`。下一步完善时区和文本缓存版本管理，并测量冷/热负载。
5. 在 `rel-f1` 上与 v2 对照同样的任务、采样、特征输出、冷/热缓存、批次数及模型工作；分别报告准备、在线 SQL、编码、队列等待和完整训练耗时与峰值内存。

`src/agentdocs/REL_F1_CPP_PIPELINE_BENCHMARK.md` 测的是内存特征副本的原型，不能用其中的 12.44× 推断本项目最终在线训练的加速比。

## 本轮运行验证

在本地 `rel-f1` SQLite 的 `driver-dnf` 任务上，两跳 `[8, 4]`、batch 128、GloVe 300 维、1 epoch，RDL GraphSAGE 与 RelGNN 均完成训练、验证和测试；各得到 0.7792 验证准确率、0.7051 测试准确率。它们共用 RelX 的表特征编码器，并非 v2 的完整 TensorFrame/相对时间编码组合；这些数字只证明端到端流程能运行，不能用于判断模型效果或 v2 加速倍数。

从 YAML 调用 `relx.run()`（包含准备、状态拟合、训练和评估）的本地单次耗时：RDL 12.32 秒，RelGNN 7.67 秒；RDL 单独复跑为 12.16 秒。运行配置和 JSON 报告位于被忽略的 `build/rel_f1_final/`。测试在同一机器顺序运行，尚未控制 CPU/文件缓存等条件，不能据此比较两种模型的稳态训练性能。

同一批 40 个预采样 batch、2 个 C++ worker、1 MiB ready 队列阈值下，缓存关闭的三次读取耗时约 0.60、0.70、0.68 秒；64 MiB 节点缓存的三次耗时约 0.51、0.58、0.62 秒。包含 Python 包装和 C++ SQL/GloVe 处理，排除了采样、状态拟合与模型计算。差距较小且有波动，暂不能宣称稳定倍数。

对 250 条 rel-f1 短文本单独测 C++ GloVe：不使用磁盘缓存的热运行约 1.5–1.6 毫秒，SQLite 磁盘缓存热读约 9 毫秒，首次冷写约 163 毫秒。故磁盘缓存保留为可选项，默认关闭；它更适合编码本身明显昂贵的长文本，不适合据此优化 rel-f1 的短文本。
