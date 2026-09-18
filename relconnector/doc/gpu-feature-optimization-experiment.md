# GPU Feature Optimization Experiment

## 目标

验证当前 `relconnector` 的 rowid 修复、节点去重、跨 batch feature window 和
GloVe LRU 对真实 GPU 训练的影响。训练语义和模型参数保持与首轮实验一致。

## 对照与配置

- 全量物化 batch baseline：`benchmarks/gpu-real-baseline.jsonl`
- 历史 vanilla：`benchmarks/gpu-real-online.jsonl`
- 当前 latest：`benchmarks/gpu-feature-optimized.jsonl`
- 数据库：未迁移的 `data/relbench/`，保持历史数据库大小和 mtime
- 参数：1 epoch，batch 512，fanout 128x128，channels 128，cuda:0
- runtime：async，单 Torch CPU 线程，单任务 timeout 14400 秒
- cache：raw 4096 MiB，feature window 4 batch/512 MiB，text 512 MiB

## 执行顺序

先运行 Salt 和 Stack 两个代表性任务，再按相同配置扩展到全部十个数据库：

1. `rel-salt/item-incoterms`
2. `rel-stack/user-badge`
3. `rel-amazon/item-churn`
4. `rel-ratebeer/beer_ratings-total_score`
5. `rel-hm/user-churn`
6. `rel-avito/searchstream-click`
7. `rel-arxiv/paper-citation`
8. `rel-event/user-attendance`
9. `rel-trial/eligibilities-adult`
10. `rel-f1/driver-circuit-compete`

```bash
../.venv/bin/python -m benchmark.online_runner \
  --implementation latest \
  --dataset rel-salt --task item-incoterms \
  --epochs 1 --batch-size 512 --num-neighbors 128,128 --channels 128 \
  --device cuda:0 --torch-num-threads 1 --executor async \
  --feature-cache-mb 4096 \
  --feature-window-batches 4 --feature-window-mb 512 \
  --text-embedding-cache-mb 512 --task-timeout 14400 \
  --output benchmarks/gpu-feature-optimized.jsonl --resume

../.venv/bin/python -m benchmark.online_runner \
  --implementation latest \
  --dataset rel-stack --task user-badge \
  --epochs 1 --batch-size 512 --num-neighbors 128,128 --channels 128 \
  --device cuda:0 --torch-num-threads 1 --executor async \
  --feature-cache-mb 4096 \
  --feature-window-batches 4 --feature-window-mb 512 \
  --text-embedding-cache-mb 512 --task-timeout 14400 \
  --output benchmarks/gpu-feature-optimized.jsonl --resume
```

## 验收口径

- 每个数据库使用与历史 batch/vanilla 实验相同的选定任务和完整训练 batch 数。
- 对比 loss 差异，但不因性能实验改变数值容差。
- 输出时间、RSS、CUDA、重复率、SQL amplification、cache、window 和 queue 指标。
- timeout 时保留最后完成的 `BatchKey` 和累计 fetch 指标。

## 实验结果

### 十库汇总

latest 对十个数据库均已实际运行。九项完成完整 epoch，Ratebeer 在固定 4 小时
预算内完成 15686/20743 batch（75.6%）后 timeout。

| 数据集 | Batch wall | Vanilla wall | Latest wall | Batch RSS | Vanilla RSS | Latest RSS | Latest 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Amazon | 2541.4s | >14400s | 13898.2s | 103.82GiB | 4.23GiB | 7.23GiB | 4954/4954 |
| Arxiv | 271.2s | >14400s | 2855.0s | 7.33GiB | 4.29GiB | 8.15GiB | 1044/1044 |
| Avito | 754.0s | >14400s | 1195.8s | 17.39GiB | 4.26GiB | 4.51GiB | 4322/4322 |
| Event | 1218.2s | 1019.8s | 966.1s | 31.93GiB | 3.91GiB | 4.24GiB | 38/38 |
| F1 | 15.9s | 18.2s | 15.9s | 2.29GiB | 2.44GiB | 2.57GiB | 6/6 |
| HM | 657.0s | >14400s | 1743.1s | 14.35GiB | 3.89GiB | 5.26GiB | 7486/7486 |
| Ratebeer | 3496.1s | >14400s | >14400s | 20.29GiB | 3.86GiB | 8.74GiB | 15686/20743 |
| Salt | 1102.0s | 14132.8s | 2832.6s | 17.73GiB | 5.15GiB | 9.29GiB | 3170/3170 |
| Stack | 857.2s | 7831.8s | 6428.7s | 19.50GiB | 3.67GiB | 8.16GiB | 6614/6614 |
| Trial | 419.5s | 326.4s | 260.2s | 22.04GiB | 2.92GiB | 4.50GiB | 459/459 |

`>14400s` 表示在 4 小时预算内未完成。timeout 项的 RSS 是终止前观测峰值，
不是完整 epoch 峰值。

### 32GiB 目标环境

- Amazon batch baseline 峰值为 103.82GiB，latest 完整训练峰值为 7.23GiB。
  latest 减少 96.59GiB（93.0%），内存需求缩小 14.35x；全量物化方案无法在
  32GiB 主机运行，而 latest 可以。
- Event batch baseline 峰值为 31.93GiB，尚未计入操作系统和其他进程开销，
  在 32GB 主机上没有可用余量；latest 为 4.24GiB，减少 27.69GiB（86.7%）。
- 十项 latest 的观测峰值均低于 10GiB。Ratebeer 虽然因计算吞吐 timeout，
  但 4 小时内峰值仅 8.74GiB，没有 OOM 或无界缓存增长。

### 历史 timeout 回归

历史 vanilla 的五个 timeout 中：

- Amazon、Arxiv、Avito、HM 已完成完整 epoch。
- Amazon 在限制前约 502 秒完成；Arxiv、Avito、HM 分别用 47.6、19.9、
  29.1 分钟。
- Ratebeer 仍 timeout，完成 75.6%。其 duplicate factor 仅 1.105，虽然
  row-cache coverage 达 20.2%、SQL amplification 为 1.001，但在线编码吞吐
  仍不足；这是下一轮最明确的性能目标。

### rel-salt/item-incoterms

已完成 3170/3170 batch：

| 指标 | Batch baseline | Vanilla online | Latest online |
| --- | ---: | ---: | ---: |
| Process wall time | 1102.0s | 14132.8s | 2832.6s |
| Train phase | 948.6s | 14088.2s | 2788.2s |
| Feature materialize | 89.8s | N/A | N/A |
| Feature fetch | 866.5s | 1169.6s | 261.8s |
| Batch assemble | 0.9s | 12917.6s | 2525.2s |
| Peak RSS | 17.73GiB | 5.15GiB | 9.29GiB |
| CUDA reserved peak | 22.81GiB | 22.81GiB | 22.81GiB |
| Loss | 0.0378852 | 0.0389236 | 0.0377724 |

Latest 相对 vanilla 的端到端加速为 4.99x；相对 batch baseline 仍慢 2.57x，
但 RSS 少 8.44GiB（47.6%）。batch baseline 的 feature fetch 是驻留内存的
TensorFrame 切片，不是 SQL 读取；异步 operation 时间也会重叠，不能相加为 wall time。

Latest 的 duplicate factor 为 2.114，raw cache 行覆盖率 98.70%，SQL
amplification 1.032；794 个 feature window 平均包含 3.992 个 batch。

### rel-stack/user-badge

已完成 6614/6614 batch：

| 指标 | Batch baseline | Vanilla online | Latest online |
| --- | ---: | ---: | ---: |
| Process wall time | 857.2s | 7831.8s | 6428.7s |
| Train phase | 415.9s | 7772.3s | 6368.8s |
| Feature materialize | 355.2s | N/A | N/A |
| Feature fetch | 61.7s | 694.9s | 648.1s |
| Batch assemble | 2.9s | 7075.6s | 5717.3s |
| Peak RSS | 19.50GiB | 3.67GiB | 8.16GiB |
| CUDA reserved peak | 1.10GiB | 1.10GiB | 1.10GiB |
| Loss | 0.14145304 | 0.14145565 | 0.14144955 |

Latest 相对 vanilla 的端到端加速为 1.22x；相对 batch baseline 仍慢 7.50x，
但 RSS 少 11.34GiB（58.1%）。

Latest 的 node duplicate factor 只有 1.011，raw cache 行覆盖率 4.86%，SQL
amplification 1.002；1655 个 feature window 平均包含 3.996 个 batch。节点级
去重收益很小，但文本层仍将 147.36M occurrences 降为 44.35M model inputs。

## 结论

1. **在线方案解决了全量物化的容量上限。** Amazon 从 103.82GiB 降至
   7.23GiB，并完成完整 epoch；Event 从 31.93GiB 降至 4.24GiB。
2. **优化有效，但收益高度依赖数据重复结构。** 五个 vanilla timeout 中四个
   已完成；Salt 的端到端时间减少 79.96%，Stack 减少 17.92%。
3. **Salt 的主要收益来自节点和文本复用。** 节点输入减少约 52.7%，GloVe 实际
   model inputs 仅 146617；SQL 查询放大接近 1。
4. **Stack 的节点复用很低。** 主要收益来自文本去重和缓存，而不是 raw block
   cache；512MiB 文本缓存发生 43.90M 次淘汰。
5. **latest 的内存是有界的。** 十项峰值均低于 10GiB；缓存收益低的 Amazon
   和 Ratebeer 也没有随 batch 数量出现无界增长。
6. **batch baseline 通常仍是速度上界，但不总是。** Event 和 Trial 的 latest
   端到端时间分别比 batch baseline 少 20.7% 和 38.0%；其余完成项中，latest
   为 batch 的 1.0x 到 10.5x。该时间代价换来了大幅降低的主机内存需求。
7. **Ratebeer 的计算瓶颈仍未解决。** SQL amplification 已接近 1，但完整
   epoch 预计约 5.3 小时；继续增加 SQL/raw cache 不会解决主要问题。
8. **数值等价性尚未完全验收。** 以 batch baseline 为参照，Stack latest loss
   相差约 `2.47e-5` 相对值；Salt latest 相差约 0.30%，虽然明显小于 vanilla
   的 2.74%，仍不满足严格逐值一致，需要独立定位。

## 后续决策

- 保留当前优化作为 latest：九项完成、所有任务峰值低于 10GiB，且 Amazon
  证明了在目标 32GiB 环境中相对全量物化方案的可运行性。
- 下一轮优先增加自适应策略：低重复表绕过 `torch.unique`，并根据历史重复率决定
  window 大小，避免 Stack、Amazon、Ratebeer 这类数据支付无效去重成本。
- Ratebeer 优先分析 `batch_assemble` 和文本列编码，目标是让完整 epoch 进入
  4 小时预算；SQL 侧不再作为首要方向。
- 单独开展数值一致性实验，固定 SamplePlan 和编码结果，逐层比较 TensorFrame、
  模型输入、forward 和 loss；不与下一轮性能改动混合。
- 单独修复已知 target leakage，不与本轮性能结果混合；涉及 Salt、Avito、
  Trial 和 Ratebeer 的 loss 仅用于本轮同配置性能对照。
