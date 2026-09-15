# Online Training

## 数据流

    task seed stream -> topology sampling -> sampled SQL features -> tensor assembly -> trainer

稳定契约依次为 SeedBatch、SamplePlan、FeatureBatch、PreparedBatch。每个 batch 带 BatchKey(epoch, batch)，组件不感知队列，executor 负责生产与消费顺序。

- connector/：可替换 reader；pandas 分块 SQL 与 Connector-X Arrow stream。
- graph/：分块扫描 PK/FK/time，构建常驻内存 CSC，不读取业务特征。
- task/：从本地任务表分块打乱 train 行，每 epoch 覆盖一次；同一实体不同时间点的任务行仍是不同训练样本。
- sampling/：通过 pyg-lib 从 CSC 独立生成结构计划。
- features/：训练前生成共享 stype/统计量，训练中执行稀疏 ID 合并查询、密集块读取、LRU 缓存及 TensorFrame/GloVe 组装。
- training/：entity/link GraphSAGE 模型与单 batch 优化，不访问 SQL。
- runtime/：同步与线程式生产者/消费者调度。

## 替换组件

OnlineRelBenchModel.prepare() 返回 OnlineTrainingSession；其 components 是 RuntimeComponents 数据类。可用 dataclasses.replace 替换 seeds、sampler、fetcher、assembler 或 trainer，然后将新组件交给 SyncExecutor.run 或 AsyncPipelineExecutor.run。接口 Protocol 位于 runtime/contracts.py 和 features/contracts.py；模型直接接收 HeteroData。

## 内存与并发

- 缓存上限由 feature_cache_bytes 控制，目前缓存 decoded DataFrame，而非编码后的 tensor。
- seed/plan/ready 三个 FIFO 队列分别有字节预算，每队列最多 64 个 item。超大 item 明确报错，不默许超限；应减小 batch/fanout 或增加预算。
- 预算只约束排队 payload 和缓存，不是进程 RSS 硬上限。还要预留图构建临时数组、各 worker 的一个在途 batch、SQL/编码临时值、GloVe 权重、模型和优化器内存。
- 异步生产可提前到下一 epoch，但训练严格按 FIFO 消费。取消会关闭并清空队列，传播异常并 join 工作线程；不能中断第三方内部永久阻塞的调用，外层任务 timeout 负责终止整个 worker 进程组。
- pyg-lib 当前没有 generator 参数。为避免全局 RNG 污染，线程版本保护采样 RNG 和训练调用，因此采样与训练不会同时执行；SQL/GloVe 特征准备仍可与训练重叠。真正的并行采样需要后续多进程共享拓扑实现。

## 当前约束

图 ID 必须为零起始连续整数，时间截断后必须保留连续前缀；无显式 PK 表使用 rowid - 1。不满足时显式拒绝，不进行隐式重编号。CSC 邻居按时间排序；节点和 seed 时间统一为 UNIX 秒，兼容 pandas 微秒精度。

训练前通过有界 SQL 扫描确定列 stype、类别词表、数值统计和时间统计，并生成 fingerprint。baseline 与 online 使用同一个 TensorFrameFeatureSchema、TensorFrame converter、HeteroEncoder、GraphSAGE、seed source 和 pyg-lib sampler；online 只对采样行执行 GloVe，baseline 则在训练前物化全表 TensorFrame。空节点类型保持相同 schema。严格对比时固定 torch_num_threads=1，以消除 pyg-lib 多线程采样的跨进程非确定性。

当前只实现 train；没有 predict/val/test、多进程 executor、全表特征物化或持久化采样缓存。
