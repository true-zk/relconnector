# Baseline

该目录保存全内存朴素实现：

    SQLite -> pandas/Connector-X -> RelBench Database -> full TensorFrame/HeteroData
           -> NeighborLoader/LinkNeighborLoader -> GraphSAGE

- dataset.py：从本地 catalog 还原数据集和任务，不下载 RelBench。
- feature_store.py：使用共享 TensorFrameFeatureSchema 和 GloVe，在训练前物化所有节点特征。
- graph_builder.py、sampler.py、models/、trainer.py：保留原始 eager 实现供认知实验。
- benchmark 的严格对比入口使用与 online 相同的 SqlSeedReader、PygLibNeighborSampler、TensorFrameBatchAssembler 和 OnlineTrainer，仅将特征来源替换为全量内存 store。

实验入口在 benchmark.baseline.BaselineExperiment 和 benchmark.runner。CLI 默认关闭物化缓存；只有显式传 --cache-materialization 才写 TensorFrame 缓存，不能与冷物化实验混用。

严格 benchmark 会分别记录 sampling、内存 feature_fetch、batch_assemble 和 train_step。双方共享 schema fingerprint、模型初始化 seed 和逐 batch 训练 seed；时间转换前统一规范为纳秒。旧 NeighborLoader 路径不用于严格性能归因。
