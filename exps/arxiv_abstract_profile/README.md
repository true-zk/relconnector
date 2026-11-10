# Arxiv `papers.Abstract` 分阶段实验

本目录独立记录 RelConnector v2 的真实摘要编码路径。`run_profile.py` 用原实现分别计时 `frame_prepare`、`frame_convert`、文本映射、缓存查询、分词、词向量查找、平均池化和向量恢复；`try_packed.py` 比较紧凑 token 原型与原实现。

## 运行

在仓库根目录：

```bash
.venv/bin/python exps/arxiv_abstract_profile/run_profile.py --per-bucket 128
.venv/bin/python exps/arxiv_abstract_profile/run_profile.py --per-bucket 128 --cache-kind shared
.venv/bin/python exps/arxiv_abstract_profile/try_packed.py
.venv/bin/python exps/arxiv_abstract_profile/try_gpu_pool.py
```

数据来自 `exps/data/relbench/rel-arxiv.sqlite`，模型使用本地缓存的 `sentence-transformers/average_word_embeddings_glove.6B.300d`。脚本不会下载数据或模型，也不会修改生产代码。结果分别写入 `results.json`、`results_shared.json`、`packed_results.json` 和 `gpu_results.json`。最后一个脚本需要 CUDA GPU。

## 设计

- 从 `Paper_ID % 11 = 0` 的非空 `Abstract` 中，按原 GloVe tokenizer 的 token 数分桶。去重、固定随机种子并在各桶内蓄水池抽样。正常桶每次 128 条，重复 3 次，批次之间不重叠；稀有桶保留不满 128 条的样本。
- 仅建 `Abstract` 一列的真实 TensorFrame schema，隔离其他列。一次冷缓存编码后立即对同一个 DataFrame 重跑，验证缓存命中和向量逐元素完全相同。
- 比较本地 64 MiB 缓存和 v2 工作进程所用的共享内存缓存；磁盘缓存关闭。单进程、CPU 单线程，TensorFrame 文本批量 256，原直接编码内部批量 8。
- 原型继续使用相同 tokenizer 和 GloVe 权重，把各句 token ID 串接为一维数组，由 `torch.nn.functional.embedding_bag(mode="mean")` 直接求均值。先比较纯编码，再比较完整 TensorFrame 冷路径。结果以 `rtol=1e-4, atol=1e-5` 校验。
- GPU 对比在同一进程内测 CPU 池化、已预装张量的 GPU 池化、CPU 查词后 GPU 池化、GPU 查词加池化，以及 CPU 紧凑 token。GPU 完整路径包含输入和输出搬运，不包含一次性的 GloVe 权重上卡。

计时器有包含关系：`frame_convert` 包含 `map_text_embedded`，后者又包含缓存和模型阶段。不可把这些数字直接相加。抽样时的 SQL 与预先计算分桶 token 数不计入转换时间。`REPORT.md` 汇总结果与边界。
