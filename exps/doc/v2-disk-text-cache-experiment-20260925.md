# v2 持久化文本 embedding 缓存实验（2026-09-25）

## 实现

v2 在规范化文本之后、调用 GloVe 之前查询持久化缓存。SQLite 保存文本哈希到
float32 向量 shard 的索引；worker 使用 reservation 避免跨进程同时编码同一
miss，写入后同一批、后续 epoch 和后续训练进程都可复用。缓存命名空间包含数据库
版本、文本模型身份、编码路径和 embedding 维度。

默认 `adaptive` 模式先观察编码量、模型耗时和内存缓存命中率：数据库至少
256 MiB、模型处理至少 4096 条文本且耗时至少 1 秒、内存命中率不超过 10%
时启用磁盘缓存。已经启动的其他 worker 会在下一次查询时附着到新建的缓存。
`on` 和 `off` 可以强制打开或关闭。未达到阈值时的暂存向量有上限；退出时
刷新尚未写满的缓冲区。

## 环境与口径

- RTX 4090，Python 3.12.9，PyTorch 2.7.1+cu128。
- Arxiv `paper-citation`：2 epoch × 50 batch，batch size 512，fanout
  `128,128`，2 个 Assembly worker，GPU `cuda:0`，seed 42。
- F1 `driver-circuit-compete`：2 个完整 train epoch，每 epoch 6 batch。
- 各组共用同一初始化缓存目录。Arxiv cold 从新的 embedding 缓存开始；warm
  和 off 在初始化缓存均已就绪后运行。
- wall time 包含进程启动、数据准备、组装和训练。ready 顺序可因进程调度略有
  差异，因而 loss 并不逐位相同。

## 2 × 50 Arxiv 对照

| 模式 | wall time | 模型编码文本数 | 磁盘命中 | 新写入 | mean loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| adaptive cold | 396.6 s | 533,625 | 49,629 | 533,065 | 0.538111 |
| adaptive warm | 190.5 s | 0 | 573,956 | 0 | 0.538980 |
| off | 204.7 s | 574,119 | 0 | 0 | 0.539127 |

warm 相比 off 快约 6.9%，并将模型编码调用降为零。cold/warm 的 2.08 倍
wall-time 差异还包含首次构建初始化缓存和首次写入 embedding 的成本，不能全部
归因于避免 GloVe 编码。warm 仍有约 249 CPU-s 的 `frame_convert` 和约
137 CPU-s 的 shared text cache lookup（两个 worker 汇总，计时存在嵌套），
因此端到端加速小于编码次数的降幅。

结果文件：

- `benchmarks/v2-arxiv-disk-cold-2x50-20260925.jsonl`
- `benchmarks/v2-arxiv-disk-warm-2x50-20260925.jsonl`
- `benchmarks/v2-arxiv-disk-off-2x50-20260925.jsonl`

## F1 全量训练集

`adaptive` 运行 2 epoch、12 batch、5298 个样本，用时 29.8 秒。
两个 worker 的 `disk.active` 都为 `false`，没有建立 F1 embedding 文件。
结果在 `benchmarks/v2-f1-disk-adaptive-full-20260925.jsonl`。

## Arxiv 全量训练集

在上述 2 × 50 对照生成的缓存基础上，运行 2 个完整 train epoch，未限制
batch 数。每轮 1044 batch、534,233 个样本；合计 2088 batch、1,068,466
个样本，用时 1322.2 秒，mean loss 0.536284。epoch 0 用时 715.2 秒，
epoch 1 用时 505.2 秒。

起始缓存有 533,065 条向量，结束时有 719,105 条。两个 worker 合计模型编码
186,040 条，恰好等于新增的 186,040 条缓存记录；本次运行中没有发现对同一
文本的重复编码。其中 epoch 0 编码 184,724 条，epoch 1 只编码 1,316 条
此前未见的新文本。结果在
`benchmarks/v2-arxiv-disk-full-2epoch-20260925.jsonl`。

## 同步执行落盘验证

用 F1 的同一个 64 样本 batch 连续启动两次 `sync` 训练，强制打开磁盘缓存并
关闭内存与 encoded feature 缓存。第一次编码并写入 538 条，第二次从磁盘
读取 575 次，模型编码 0 次，验证了同步入口训练结束时会刷新未满的写入缓冲区。

## 异步双线程验证

F1、`encode_workers=2`、每轮 2 batch、强制打开磁盘缓存。连续启动两次训练：
第一次编码并写入 770 条；第二次磁盘命中 1138 次、模型编码 0 次。
SQLite 连接与缓存缓冲区在多个编码线程之间串行保护，等待其他线程发布结果
时会释放锁。

最终代码在进程模式下又运行 Arxiv 1 epoch × 2 batch 热缓存回归：54.9 秒、
两个 worker 合计模型编码 0 条、磁盘命中 295,221 次、写入 0 条。结果在
`benchmarks/v2-arxiv-disk-final-process-smoke-20260925.jsonl`。
