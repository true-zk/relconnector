# relconnector_v2 GPU 实验记录（2026-09-25）

本文记录持久化文本 embedding 缓存加入前的实验状态。新增缓存实现及后续对照
见 `doc/v2-disk-text-cache-experiment-20260925.md`。

## 目的

验证独立的三阶段 v2 流水线在本机 RTX 4090 上的 GPU 训练链路，并使用当前
可用的高效文本路径：本地磁盘 Hugging Face GloVe 模型（`local_files_only=True`）、
`direct` GloVe tokenize/lookup/pool、进程间 shared text cache、shared encoded
cache，以及持久化的 v2 schema/topology initialization cache。

v2 当前没有硬盘 embedding cache 开关；设计文档明确写明暂不恢复硬盘 embedding
cache。因此本文中的“磁盘缓存 text 版本”指本地磁盘模型缓存 + 运行期共享缓存，
不是把每条 text embedding 写回磁盘。

## 环境与口径

- Python 3.12.9，PyTorch 2.7.1+cu128，CUDA build 12.8。
- GPU：`cuda:0`，NVIDIA GeForce RTX 4090 24 GiB。
- `assembly_processes=4`（F1 使用 2），`torch_num_threads=1`。
- batch size 512，fanout `128,128`，channels 128，seed 42。
- `training_order=ready`，`feature_cache=4096 MiB`，`text_cache=512 MiB`，
  `encoded_cache=2048 MiB`，`window=4/512 MiB`。
- Arxiv 实验关闭 `cross_epoch_prefetch`，因为当前 fetcher 拒绝跨 epoch feature
  window；见下方失败记录。

## 成功结果

| 数据集/任务 | epochs x batches | 完成样本 | wall time | mean loss | peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| rel-f1/driver-dnf | 2 x 4 | 4096 | 25.75 s | 0.3700862 | 2.13 GiB |
| rel-arxiv/paper-citation | 1 x 2 | 1024 | 75.06 s | 0.6560243 | 2.39 GiB |

结果文件：

- `benchmarks/v2-f1-gpu-direct-shared-20260925.jsonl`
- `benchmarks/v2-arxiv-paper-citation-gpu-direct-shared-1x2-20260925.jsonl`

F1 的 v2 pipeline wall time 为 12.93 s，训练阶段 epoch 0/1 分别为 2.86 和
8.51 batches/s。共享 text cache 最终记录 1590 hits、1838 misses、796 次
reservation wait；shared encoded cache 记录 1520 hit rows，命中率 2.21%。

Arxiv 2-batch 运行中，Assembly worker 1 完成了全部两个 batch；文本 occurrences
为 295,234，unique texts 为 295,231，shared text cache 仅 10 hits，说明该任务
文本重复率极低。`papers` 的 `frame_convert` 为 50.49 s，其中 Abstract 文本映射
25.80 s，是端到端瓶颈；训练 epoch 本身仅约 1.00 s，GPU 主要在等待 CPU Assembly。

## 未完成尝试与结论

1. Arxiv `2 x 50`、默认跨 epoch 预取：失败，报错
   `feature windows cannot cross epoch boundaries`。这是 v2 当前实现边界，不是
   数据损坏。本轮随后已在 Assembly worker 边界按 epoch 拆分 window，并增加
   `test_cross_epoch_window_is_split_before_fetch` 回归测试；实验结果仍保留关闭预取
   的原始口径，不在修复后静默改写。
2. Arxiv `2 x 50`、关闭跨 epoch 预取：已启动但按首个 batch 的 Assembly 速度估算
   总耗时过长，主动停止，没有写入结果 JSONL。该结果说明当前 v2 在高基数、低文本
   重复的 Arxiv 上仍受 CPU 文本编码/Frame converter 限制，多进程尚未让 GPU 满载。

## 后续建议

- 在更长的多 epoch smoke 中复验本轮新增的跨 epoch window 分组修复。
- Arxiv 优先优化 `frame_convert` 和文本映射，单纯增大 shared text cache 无法解决
  低重复率任务。
- 在 shared cache 命中率较高的 F1/Salt 上继续做 1/2/4 worker 扩展曲线；Arxiv
  使用 1-2 batch smoke 作为回归测试，避免当前 2×50 长跑占用数小时。
