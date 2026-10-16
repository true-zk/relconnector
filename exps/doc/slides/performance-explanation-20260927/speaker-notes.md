# 整合版逐页说明

## 01 REL CONNECTOR / 优化复盘 / 从特征供给瓶颈，到自适应在线训练

INTEGRATED_V2_20260927
重组旧稿，更新至十任务完成及最新Event/Trial复测。所有历史对比均保留缓存与版本差异。性能图不含Batch；内存图按用户要求包含Batch。

证据与口径：
doc/v2-dataset-performance-review-20260927.md

## 02 01 / VANILLA 的问题 / 按需读取省下内存，却把特征生产放进训练循环

INTEGRATED_V2_20260927
历史Vanilla五项在4小时超时。不要把Vanilla描述为完全没有优化；旧SQL修复已回补到当前代码，历史成绩代表当时快照。

证据与口径：
doc/gpu-feature-optimization-experiment.md
doc/optimization-evolution-report-20260926.md §2–3

## 03 02 / v1 如何优化 / 把重复请求合并，只对缺失部分生产特征

INTEGRATED_V2_20260927
去重只减少特征生产，重复训练位置通过inverse恢复。文本快路径保留GloVe计算语义；不兼容时回退。初始化持久化缓存并非v2独有。

证据与口径：
doc/optimization-evolution-report-20260926.md §4–5

## 04 02 / v1 的收益边界 / 重复结构不同，同一缓存策略的收益就不同

INTEGRATED_V2_20260927
Salt约4.99×，Stack约1.22×。历史阶段/worker计时会重叠，不求和为wall。此处仅解释v1在不同数据结构上的收益差异。

证据与口径：
doc/gpu-feature-optimization-experiment.md

## 05 03 / 又发现了什么问题 / assemble 成为瓶颈：单进程优化后仍供给不足

INTEGRATED_V2_20260927
本页是诊断归纳，不宣称每个问题只有单一原因。Ratebeer另有共享存储发布竞争，属于正确性问题，在后面独立说明。

证据与口径：
doc/v2-full-matrix-20260925.md
doc/v2-hm-regression-20260925.md

## 06 04 / V2 架构 / 采样、特征组装、GPU 训练分进程执行

INTEGRATED_V2_20260927
epoch gate限制当前轮训练，下一轮可预取；特征窗口不能跨epoch混合。活跃worker数可根据供给状态调整，但预先spawn的进程暂停不等于归还内存。图拓扑仍常驻。

证据与口径：
relconnector_v2/relconnector_v2/runtime/process_pipeline.py
relconnector_v2/relconnector_v2/process_api.py

## 07 04 / 算法 D：冷初始化数值统计 / 根据行数与内存预算，自动合并数值列扫描

INTEGRATED_V2_20260927
空应用缓存对照，不清操作系统文件缓存。预算只针对数值数组，非进程RSS上限。schema缓存版本更新，旧schema需重算；图缓存可复用。新旧完整冷测分别2164.0016/606.0092秒。

证据与口径：
doc/v2-initialization-optimization-20260927.md
benchmarks/v2-event-init-optimized-20260927/result.json

## 08 04 / 阶段二：采样需求 / 采样输出计划，组装只处理唯一物理行

INTEGRATED_V2_20260927
重复因子来自历史诊断；不是采样内核或端到端加速比。单采样进程，不改变邻居采样算法。

证据与口径：
doc/targeted-batch-results-20260927.json
doc/v2-ratebeer-cache-placement-20260927.md
doc/v2-stack-compaction-20260927.md

## 09 04 / 算法 A：高复用文本 / 重复命中时，只查本地槽位索引

INTEGRATED_V2_20260927
每worker索引元数据预算32MiB，LRU淘汰；激活后两个窗口预热，反复失效延长冷却，最多32个窗口。Salt端到端3.51×包含初始化状态差异；查询计时是并行累计。

证据与口径：
doc/v2-salt-regression-20260926.md

## 10 04 / 编码缓存：统一考虑传输粒度与位置 / 缓存放在哪里，要看访问成本与额外复用

INTEGRATED_V2_20260927
HM为旧1000批同预算对照；Ratebeer为64批×每配置两次预热后对照，不能横向比较两库秒数。Ratebeer查询22.378→2.175、发布36.642→8.931工作进程秒；主指标是流水线时间/批数。完整矩阵默认尚未切换。

证据与口径：
doc/targeted-batch-results-20260927.json
doc/v2-ratebeer-cache-placement-20260927.md
doc/v2-stack-compaction-20260927.md

## 11 04 / 算法 B：文本工作集较大 / 把跨轮可复用的 embedding 留在磁盘

INTEGRATED_V2_20260927
Arxiv矩阵模型输入184724/1316。Trial371.80秒为热初始化/文本缓存，不能全部归因于代码优化。Ratebeer内存累计命中约88.67%，未触发低命中门槛，第二轮仍有大量编码。

证据与口径：
doc/v2-full-matrix-20260925-final.md
doc/v2-trial-rerun-20260927.md

## 12 04 / 持久化缓存：从命中走向廉价访问 / 向量已缓存，物理碎片仍会拖慢 assemble

INTEGRATED_V2_20260927
原布局A2/A3、合并布局B2/B3，每组2次；文件缓存预热后。磁盘查询工作进程时间118.304→54.904秒，下降53.59%。显式离线工具，按分片数/映射预算筛选，不按数据集名判断。不是运行时自动迁移，不宣称RSS下降。

证据与口径：
doc/targeted-batch-results-20260927.json
doc/v2-ratebeer-cache-placement-20260927.md
doc/v2-stack-compaction-20260927.md

## 13 04 / 算法 C：控制并发资源成本 / 缓存能命中，还要能长期稳定运行

INTEGRATED_V2_20260927
写阶段预算参考4倍索引、通常64MiB至4GiB并受空闲空间1/16约束。连续两窗口高命中、低WAL增长进入读阶段，上限512MiB；命中下降或增长回升退出。Amazon约60GB WAL已收回；不能把稳定性修复与性能贡献混为单一倍率。

证据与口径：
doc/v2-stack-regression-20260926.md
doc/v2-amazon-regression-20260926.md

## 14 04 / 特征复制与读取：以关键路径决定取舍 / 少做一个操作，不一定让一批更快

INTEGRATED_V2_20260927
Salt每版2次60批；投影试验受外部负载变化影响，无整体收益证据。对应各专项报告。所有操作计时允许重叠，不可相加成墙钟。

证据与口径：
doc/targeted-batch-results-20260927.json
doc/v2-ratebeer-cache-placement-20260927.md
doc/v2-stack-compaction-20260927.md

## 15 04 / 正确性是优化的前提 / Ratebeer：先修复发布竞争，再评价吞吐

INTEGRATED_V2_20260927
Ratebeer完整结果15444.93秒，包含CPU类别检查开销；20k批为截断工作量。最终模型输入约20245311/20503129，ready等待11389.19秒。

证据与口径：
doc/v2-ratebeer-regression-20260926.md

## 16 04 / 工作负载 → 成本 → 策略 / 把新旧优化放在同一套选择逻辑中

INTEGRATED_V2_20260927
全部机制在相应前文页解释。本地缓存位置尚非自动学习；离线整理显式调用。

证据与口径：
doc/targeted-batch-results-20260927.json
doc/v2-ratebeer-cache-placement-20260927.md
doc/v2-stack-compaction-20260927.md

## 17 05 / 当前 v2 定向结果 / 新的每 batch 结果：供给时间与等待一起下降

INTEGRATED_V2_20260927
数据源为targeted-batch-results-20260927.json。等待是流水线的一部分，整次运行又包含流水线；三行不可相加。Ratebeer墙钟每批下降18.71%，Stack12.04%；不得当完整矩阵或RSS更新。

证据与口径：
doc/targeted-batch-results-20260927.json
doc/v2-ratebeer-cache-placement-20260927.md
doc/v2-stack-compaction-20260927.md

## 18 05 / 十库历史参考 + 当前定向证据 / 跨版本概览，与本轮 v2 对照分层展示

INTEGRATED_V2_20260927
历史Vanilla/v1单轮、v2两轮且缓存冷热不同；Event49.83×包含热初始化，冷初始化仅3.57×。Ratebeer v1四小时超时，v2运行15444.93秒完成2×10000批，不是同预算完整遍历。TO柱为状态符号。图中v2 RSS只含主进程。最新64/200批对照不替换历史端到端柱，不外推十库成绩。

证据与口径：
doc/targeted-batch-results-20260927.json
doc/v2-ratebeer-cache-placement-20260927.md
doc/v2-stack-compaction-20260927.md

## 19 05 / 内存：保留实测口径 / 新优化测了吞吐，没有产生新的作业内存结论

INTEGRATED_V2_20260927
来自历史rss-data.csv；Amazon独立线性量程，其余共用另一个量程。此次无新作业级内存测量，不能用新时间结果补造RSS。

证据与口径：
doc/targeted-batch-results-20260927.json
doc/v2-ratebeer-cache-placement-20260927.md
doc/v2-stack-compaction-20260927.md

## 20 07 / 内存结论与下一步 / 减少常驻特征，还需要覆盖整个进程树

INTEGRATED_V2_20260927
不直接相加进程RSS，因为共享页会重复计算。v2默认ready队列16GiB为上限而非实际占用，也非全系统内存承诺。通用启发式已实现，但HM本地配置不是自动学习，所有十库尚未用新统计代码重测。

证据与口径：
doc/optimization-evolution-report-20260926.md §9
doc/v2-dataset-performance-review-20260927.md

## 21 (a) Batch｜全量物化训练架构 / 训练前生成全部特征；训练时从常驻 TensorFrame 切片。虚线框表示执行范围。

Source: benchmark/baseline.py; baseline/batch_baseline/feature_store.py. Full table reads and EagerFeatureStore.materialize precede training. Graph topology is resident. The async baseline reuses sampler/feature threads, but feature fetch is an in-memory TensorFrame slice. The node/edge/time/label information in SamplePlan is preserved during assembly. Feature enc. in the GPU block is the trainable model encoder, distinct from frozen GloVe/statistical preprocessing. Task-seed queue is omitted here for clarity.
中文：全量物化不等于全批量梯度下降；本架构仍采用邻居采样和mini-batch训练。

## 22 (b) Vanilla｜按需读取的在线训练架构 / 单进程内异步采样、特征线程与训练；已有 raw cache，但缺少后续文本与编码特征复用。

Source: baseline/vanilla_baseline/api.py and runtime/async_pipeline.py. The default async path already has seed/sampler/feature threads and bounded queues. It is not a fully serial or cache-free implementation. SqlFeatureFetcher owns a bounded raw block cache. GloveTextEmbedder is constructed without the later text-cache options; no encoded-feature cache is configured. The repeated SQL-tables cylinder denotes the same database, not another copy. The initial schema statistics configure the static feature encoder. Recommendation branch expansion and error channels are omitted for clarity.

## 23 (c) v1｜单进程优化：窗口复用与分层缓存 / 在单进程线程流水线中合并请求，并复用 raw、text 与 encoded 特征；训练按序消费。

Source: relconnector/api.py, runtime/async_pipeline.py, features/sql.py and features/assembler.py. This depicts the current legacy/v1 architecture, including later direct-text/thread refinements; it is not a claim that every component was present in the historical performance snapshot. Feature windows do not cross epochs. Requested node rows are deduplicated and restored using inverse mappings. Encoded-cache hits reuse requested encoded rows; missing rows follow raw/text/encoding paths. Encoder workers are threads sharing process-local caches, not v2 assembly processes. Persisted schema/topology artifacts are initialization caches, not a per-text disk cache.

## 24 (d) v2｜多进程与自适应缓存训练架构 / assemble 成为供给瓶颈后，v2 将采样、组装、训练分进程执行。

Source: relconnector_v2/process_api.py, runtime/process_pipeline.py, features/process_worker.py, features/shared_text.py, features/shared_encoded.py, features/disk_text.py. The seed feeder is a thread in the main process; its input stream is represented at the left. Sampling is a child process. Assembly workers own SQL readers, raw caches, text encoder and local state. Shared text vectors have reservation-based coordination; the adaptive local index stores only text-to-slot metadata. Encoded-feature cache is optionally shared through a manager or local to each worker; HM's local 4 GiB is an explicit tested profile, not an automatic learned choice. Disk text persistence and initialization artifacts are separate stores. CPU topology is resident; do not claim a single shared physical graph allocation. CPU initialization includes budgeted exact numerical statistics. Training may use ready-order or strict-order with an epoch gate. Queue byte caps / in-flight windows impose backpressure. Diagram excludes error-reporting paths and checkpoint persistence for readability.