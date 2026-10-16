# Arxiv 五组消融：首次短程试跑（2026-09-28）

五组均完成；自动工作量、worker diagnostics、关闭缓存的保留/写入检查通过。

## 协议

- Dataset/task：rel-arxiv / paper-citation。RTX 4090，torch 2.7.1+cu128。
- 每组 2 轮 × 16 batches，共 32 batches / 16,384 examples；不是完整 epoch。
- batch size=512，fanout=128,128，channels=128，seed=42，ready-order。
- 各组一次，按 manifest 中顺序顺序执行；固定活跃 worker 数，禁用自适应 worker。
- raw/encoded/text 总 payload 预算 4096/2048/512 MiB，encoded scope=shared。禁用的缓存预算不重分配。
- 每组独立空应用缓存目录；OS 页缓存未控制；持久缓存构建成本计入。
- 不比较训练 loss 等价性：ready-order 可能改变更新顺序；这不是训练质量实验。

## 实测

| 配置 | benchmark 秒 | pipeline 秒 | ready 等待秒 | 文本模型输入数 | disk 活跃 worker |
| --- | ---: | ---: | ---: | ---: | ---: |
| 完整 v2（固定 4 worker） | 172.99 | 130.00 | 124.16 | 494,136 | 4 |
| 1 worker，保留两类缓存 | 200.01 | 157.48 | 151.51 | 494,136 | 1 |
| 关闭普通特征缓存 | 170.21 | 134.75 | 128.99 | 494,136 | 4 |
| 关闭文本缓存 | 103.54 | 61.53 | 55.74 | 1,736,574 | 0 |
| 关闭两类复用缓存 | 129.65 | 92.58 | 86.25 | 3,141,067 | 0 |

benchmark 时间由 benchmark 内部记录，包含 prepare 和 train，不含 Python 导入/CLI 前置启动；进程总墙钟时间在 execution.jsonl 中。pipeline 时间包含进程启动、首批等待与收尾，不能称为稳态吞吐时间。

当前 epoch duration 从该 epoch 第一个 batch 可训练时才开始，漏掉首批等待；不将其用于本表。主进程 peak_rss_mb 未作为总内存结果使用。

## 可以观察到什么

1. 本次固定 4 worker 的完整组 pipeline 为 130.00 秒，单 worker 为 157.48 秒，点估计约 1.21×。该结果包含启动/缓存构建，不能替代 worker 扩展曲线。
2. 有文本缓存时，关闭普通特征缓存的 pipeline 为 134.75 秒，完整组为 130.00 秒，差距较小；无文本缓存时，保留普通特征缓存为 61.53 秒，双关为 92.58 秒。两层存在收益覆盖/交互的线索，不能把 speedup 相加。
3. 本次所有保留文本缓存的组都只编码 494,136 个文本输入，比双关的 3,141,067 少；但完整组比 no_text_cache 更慢。减少文本编码并未在这个短程冷启动负载上转化为时间收益。
4. 带文本缓存的组实际启用了 disk；完整组与 no_feature_cache 的四个 worker 都有 active 报告。因此接下来应优先隔离 disk、内存共享访问和本地索引成本，不能直接归因于其中某一项。

这些是单次短程观测，不提供显著性或稳定加速结论。也不据此修改默认策略。

## 下一轮最有信息量的实验

1. 同配置增加 no_disk_text_cache，与 full/no_text_cache 对比，确定 RAM 文本复用与首次持久化成本。
2. 增长至 3–5 个完整 epoch 或足够稳定的固定训练窗口，分别记录首次构建和后续复用；做至少 3 次独立重复。
3. 对普通特征缓存拆 no_raw_cache/no_encoded_cache；对内存文本层拆 no_local_text_index。
4. F1 strict smoke 最大 mean-loss 差异约 5.6e-6，未满足初步 1e-6 检查，需逐 batch 输入和训练轨迹复核后再写严格数值等价主张。

## 产物

- 原始数据与完整命令：`benchmarks/ablation-20260928-arxiv-short/manifest.json` 及各配置目录。
- 自动汇总：`benchmarks/ablation-20260928-arxiv-short/summary.csv`、`summary.json`。
- 执行器：`relconnector_v2/relconnector_v2/run_ablation.py`。
- 核验/汇总器：`relconnector_v2/relconnector_v2/summarize_ablation.py`。
- 设计协议：`doc/v2-ablation-design-20260928.md`。
