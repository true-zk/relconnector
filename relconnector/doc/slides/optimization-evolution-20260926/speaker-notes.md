# RelConnector 演讲备注

## 01 · 有限内存下的关系图训练

本演示围绕有限内存下在线训练的优化机制与证据展开。主对照为 vanilla、v1、v2。batch 仅解释传统全量特征物化方式，不计入比较实验。建议讲解时长20–30分钟。

来源（项目 doc/）：optimization-evolution-report-20260926.md
数据为报告快照，不是本次重跑结果。

## 02 · 研究问题：特征放不进内存，训练如何持续推进？

图拓扑依然需要放进内存，因此本项目不是任意规模图的完全外存训练。vanilla已有基本混合SQL、raw缓存与线程执行器；所谓未优化是相对后续优化而言。

来源（项目 doc/）：optimization-evolution-report-20260926.md
数据为报告快照，不是本次重跑结果。

## 03 · 优化沿着关键路径逐层推进

早期v1被复制为cache_baseline，随后接受正确性修订，再继续动态特征优化。v2具有独立命名空间。源码中未提交的修改也属于当前工作区，因此不能将这些状态称为固定发布tag。

来源（项目 doc/）：optimization-evolution-report-20260926.md
数据为报告快照，不是本次重跑结果。

## 04 · GPU 等待特征，成为最明确的优化信号

这些比例来自历史报告，不代表当前v2。异步worker累计操作时间有并行与嵌套，不能相加得到总时长。重点解释为何先优化CPU供给而不是GNN kernel。

来源（项目 doc/）：bottleneck-analysis-and-optimization.md
数据为报告快照，不是本次重跑结果。

## 05 · 第一轮优化：让 SQL 真正只取需要的行

无显式主键表对外暴露rowid-1，查询输入使用node ID+1。后续离线工具建立显式连续node ID。该修复后来回补到vanilla，正式对照应重跑修订后的版本。

来源（项目 doc/）：feature-pipeline-optimization-plan.md；cache-baseline-correctness-report.md
数据为报告快照，不是本次重跑结果。

## 06 · 去重的是特征生产，不是训练样本

早期采用4 batch、512 MiB feature window。扩大窗口可以降低重复工作，也会增加首批延迟和临时内存。跨epoch预取允许提前准备下一轮，但fetch窗口本身仍需按epoch拆分。

来源（项目 doc/）：feature-pipeline-optimization-plan.md
数据为报告快照，不是本次重跑结果。

## 07 · 第一轮收益取决于数据的重复结构

Salt raw覆盖率98.70%，Stack仅4.86%。Stack文本缓存发生约4390万次淘汰，工作集超过缓存容量。历史v1九项完成，Ratebeer四小时只完成15686/20743。

来源（项目 doc/）：gpu-feature-optimization-experiment.md
数据为报告快照，不是本次重跑结果。

## 08 · 正确性修订为后续性能比较建立共同契约

已记录固定fixture四版本逐值一致及真实F1推荐60组特征一致，但不能外推所有任务和调度都严格等价。current vanilla回补了SQL问题，历史实现必须使用归档。ready顺序会改变优化器轨迹。

来源（项目 doc/）：cache-baseline-correctness-report.md
数据为报告快照，不是本次重跑结果。

## 09 · 第二轮 v1：把复用提升到最终静态特征

encoded cache整块持有TensorFrame，按tensor storage、ID和索引成本计费。有界LRU、oversized rejection及二次访问准入帮助保护热点维表。

来源（项目 doc/）：adaptive-feature-optimization-report.md；../relconnector/features/cache.py
数据为报告快照，不是本次重跑结果。

## 10 · 降低执行开销，但不默认“并行越多越快”

direct与official固定文本逐值相同。Salt/Ratebeer实验同时改变direct、worker与encoded缓存，无法拆分单项贡献。初始化缓存持久化schema统计量和纯拓扑，冷暖差异必须分开计费。

来源（项目 doc/）：adaptive-feature-optimization-report.md
数据为报告快照，不是本次重跑结果。

## 11 · v2：围绕特征生产重设进程边界

v2独立命名空间，不导入原训练实现。拓扑tensor共享，prepared tensors通过PyTorch multiprocessing传输。Assembly负责从读取到组装，而不是机械复制v1线程边界。多进程增加模型、连接和临时内存，必须全进程树测量。

来源（项目 doc/）：optimization-evolution-report-20260926.md
数据为报告快照，不是本次重跑结果。

## 12 · 调度要同时控制顺序、供给与资源

当前默认ready，矩阵初始活跃1个并可扩展4个。队列高水位为75%。epoch预取不意味着可把不同epoch合入fetch_many。消费者等待数据和临时对象还需纳入内存预算。

来源（项目 doc/）：optimization-evolution-report-20260926.md
数据为报告快照，不是本次重跑结果。

## 13 · v2 的复用范围：从 worker 内扩展到跨运行

shared reservation协调同一miss由一个worker编码。已发布向量槽不可变，支持本地只存位置索引。磁盘缓存不是提前全表物化，也不是零存储成本；访问广泛后可能覆盖大量文本。旧文档不持久化embedding的限制已发生演进。

来源（项目 doc/）：optimization-evolution-report-20260926.md
数据为报告快照，不是本次重跑结果。

## 14 · 磁盘缓存：模型调用归零，不等于总耗时归零

adaptive磁盘激活阈值：DB>=256MiB、模型输入>=4096、模型时间>=1秒、内存命中<=10%。F1保持内存路径，磁盘未激活。这里两个worker，ready顺序；不做数值逐位等价结论。

来源（项目 doc/）：v2-disk-text-cache-experiment-20260925.md
数据为报告快照，不是本次重跑结果。

## 15 · HM：高命中缓存，被整块 RPC 传输拖慢

短程等预算对照支持通信归因：shared206.0s、只传行183.5s、local2GiB157.7s、local4GiB152.8s。完整端到端2.87x包含冷暖初始化差异，pipeline2.62x仍包含作用域/容量改变。矩阵HM仍显式特配local4GiB。

来源（项目 doc/）：v2-hm-regression-20260925.md
数据为报告快照，不是本次重跑结果。

## 16 · Salt：99.93% 命中，仍被共享查询拖慢

本地位置索引每worker估算32MiB，向量不重复复制，reservation不变。F1未激活；Arxiv低复用激活后回退。最终完整Salt还包含共享查询批量化，不能视为单独位置索引的因果倍数。ready等待仍约64.7%。

来源（项目 doc/）：v2-salt-regression-20260926.md
数据为报告快照，不是本次重跑结果。

## 17 · Stack：从共享查询开销，到长跑资源耗尽

短测后测继承少量向量并受同机负载影响，5.54x不是严格单因素。后续长跑已有77339个被索引分片，资源逼近vm.max_map_count；隔离复现mmap分配失败及线程无法启动。修复后映射保持1024，完整训练模型输入0/6942。

来源（项目 doc/）：v2-stack-regression-20260926.md
数据为报告快照，不是本次重跑结果。

## 18 · Amazon：资源回收也需要匹配磁盘工作负载

默认自适应checkpoint阈值通常64MiB–4GiB，参考索引4倍并受剩余空间1/16约束。触发值不是绝对硬上限。固定64MiB在机械盘上导致零散脏页刷出和写锁等待，候选弃用。续跑保留/合并6584936条向量。

来源（项目 doc/）：v2-amazon-regression-20260926.md
数据为报告快照，不是本次重跑结果。

## 19 · 当前结果：部分任务显著改善，仍有回归与缺口

历史v1是早期缓存版，不是当前v1。v2各任务不同时间修复完成，HM local4GiB，其余通常shared2GiB；Arxiv/Stack暖磁盘。报告快照8任务10条成功记录，HM/Salt各保留旧结果。不可计算batch对照。

来源（项目 doc/）：optimization-evolution-report-20260926.md §8
数据为报告快照，不是本次重跑结果。

## 20 · 有限内存验收：必须统计整个训练系统

RUSAGE_SELF源码位于relconnector_v2/benchmark.py。相加RSS会重复计共享页，需PSS/USS和cgroup。默认raw4GiB、encoded2GiB、text512MiB、plan2GiB、ready16GiB，raw/local预算按worker分摊但模型和临时内存额外。不能据此声称32GiB硬限制已验收。

来源（项目 doc/）：optimization-evolution-report-20260926.md
数据为报告快照，不是本次重跑结果。

## 21 · 下一轮实验：把机制证据变成严格系统结论

报告中Ratebeer完整历史每轮20743batch，而矩阵max10000，即使达到上限也只是截断工作量。需多次重复、波动与进程树资源，不能将短测或约2%单次差值外推。HM encoded特配应披露或改为可验证运行时策略。

来源（项目 doc/）：optimization-evolution-report-20260926.md
数据为报告快照，不是本次重跑结果。

## 22 · 核心贡献：让有限内存下的特征供给更高效

收束时不强调一个全面加速倍数。项目最有说服力的贡献是明确瓶颈、设计符合内存限制的复用/并行机制，以及对多进程长跑退化的诊断修复。后续用严格实验确认普适性。

来源（项目 doc/）：optimization-evolution-report-20260926.md
数据为报告快照，不是本次重跑结果。

## 23 · 附录 A｜结果数值与工作量

历史v1 wall分别Arxiv2855、Avito1195.8、Event966.1、F115.9、HM1743.1、Salt2832.6、Stack6428.7、Trial260.2秒。比值按各自wall/batches计算。JSONL的wall保留全精度后取三位比值；表只展示两位wall。报告链接中有完整来源。

来源（项目 doc/）：optimization-evolution-report-20260926.md
数据为报告快照，不是本次重跑结果。

## 24 · 附录 B｜来源、复现与讲解备注

完整路径均相对项目relconnector/。源码实际位于relconnector_v2/relconnector_v2/。工作区快照包括未提交修改，不对应固定发布tag。PPT以原生文本框、矩形和条形图形构建，均可编辑；附PDF便于预览分享。

来源（项目 doc/）：optimization-evolution-report-20260926.md
数据为报告快照，不是本次重跑结果。