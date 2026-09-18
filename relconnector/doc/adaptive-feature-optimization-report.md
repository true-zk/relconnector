# 动态特征生产优化实施报告


## 实施范围

本轮只修改 latest、benchmark 和测试，`cache_baseline`、`vanilla_baseline`、
`batch_baseline` 保持冻结。所有版本继续读取同一份官方 `.stypes.json`；未按列名
修改 stype，也未删除 URL、邮箱、电话等字段。

## 已实现

1. 线程安全细粒度遥测
   - SQL execute/decode/concat/reindex、跨 batch unique；
   - DataFrame prepare、逐表逐列 mapper、converter、inverse gather；
   - 文本 normalize/cache/tokenize/lookup/pool/expand；
   - raw/encoded/text cache、动态策略和四层 queue。
2. 初始化持久化
   - schema statistics 使用 JSON typed codec；
   - topology 只保存 tensor 和基础容器，并以 `weights_only=True` 加载；
   - key 包含数据库 path/size/mtime、cutoff、隐藏列、官方 stype 和算法参数；
   - 临时文件加原子 rename，命中前仍校验官方 stype artifact。
3. encoded TensorFrame cache
   - 以 `(table, node_id)` 查询、整块持有 TensorFrame；
   - 只编码 miss 并恢复原始顺序；
   - 按 tensor storage、ID tensor 和索引成本计费；
   - byte-bounded LRU、oversized rejection；容量充足时直接准入，接近预算后切换二次访问准入。
4. 同语义文本快路径
   - 直接调用 SentenceTransformers 已加载的官方 tokenizer、
     WordEmbeddings 和 mean Pooling；
   - 模块结构不匹配时回退 `SentenceTransformer.encode`；
   - 完整文本 LRU 使用分块连续 Tensor slab，不再为每个 cache entry 持有独立
     Tensor storage；容量未满时直接准入，出现压力后才切换二次访问准入。
5. 有界并行流水线
   - sampler -> SQL fetch -> fetched queue -> 1/2 encode workers ->
     ordered ready queue -> trainer；
   - 输入和输出均按 retained bytes 限制；
   - round-robin worker queue 保证慢窗口下仍按 BatchKey FIFO；
   - 没有 assembler factory 时自动退回单 worker。
6. 动态窗口策略
   - 根据 duplicate factor 和 raw-cache coverage 的 EWMA 决策；
   - 每 8 个窗口最多变化一级，具有冷却和阈值滞回；
   - 用户配置的 batch/byte 预算始终是硬上限。

## 已验证结果

### 正确性

- 主测试集（含新增 encoded-cache、artifact 和乱序并行测试）44 项通过。
- data materialization 6 项通过。
- `ruff check .` 与 `pyright` 通过。
- 固定 F1 两 batch：单/双 encode worker loss 均为
  `0.5770172774791718`，差值为 0。
- 文本覆盖空值、OOV、标点、大小写和长文本：direct 与 official 输出
  `torch.equal=True`，最大绝对误差为 0。
- cache baseline worker 仍可运行，证明新增 latest 配置已正确过滤。

### 性能与资源

- 256 条、20 轮文本微基准：
  - official encode：0.2123s
  - direct tokenizer/lookup/pool：0.1334s
  - 该局部路径约 1.59x；direct 内部按 64 条分块以限制长文本 padding 峰值。
- F1 CPU 22 batch 端到端消融：
  - official + 单 worker + 无 encoded cache：train 5.4020s；
  - direct + 双 worker + slab/encoded cache + adaptive：train 5.4846s；
  - loss 完全一致；该训练主导的小任务退化约 1.5%，因此默认保留单 worker，
    双 worker 作为显式开关，待大型 feature-bound workload 验证后再调整。
- F1 优化版 encoded cache：13685 hit rows、20090 miss rows，命中率
  40.5%，实际持有 2.46MiB。
- 细粒度 operation telemetry 开/关：5.3315s 对 5.3065s，单次测量开销
  约 0.47%，低于 2% 目标。
- F1 初始化 cache 冷启动为 schema/topology 均 miss，第二次均 hit；节点数和
  schema fingerprint 完全一致。
- Salt CPU 50-batch 短回归（batch 128、fanout 32x32）：
  - official + 单 worker + 无 encoded cache：train 49.658s；
  - direct + 双 worker + encoded cache + adaptive：train 47.204s；
  - loss 均为 `2.196388840675354`，短回归改善约 4.9%；
  - encoded hit rows 156301，命中率 17.8%，峰值 RSS 5.44GiB（对照
    4.30GiB）；
  - 策略观察到低窗口重复率后将 window 从 4 降至 3。
- Salt 初始化首次 prepare 29.87s，命中 schema/topology 后 5.66s，减少约
  81%；这是短任务上的冷/暖结果，不等同于 Event 的历史 15 分钟收益。

## 两轮 GPU 验证

GPU 恢复后补跑了固定采样规模的 2 epoch 对照，配置为 batch 512、fanout
128x128、channels 128、A30。

F1 不设置 `max_batches` 的完整两轮共 6 batch/2706 examples：epoch 0 为
1.003s，epoch 1 为 0.300s，loss 从 0.7054 降至 0.4379。

| workload | 配置 | epoch 0 | epoch 1 | train 总时长 |
| --- | --- | ---: | ---: | ---: |
| Salt 2x50 | official/单 worker/无 encoded | 130.78s | 82.79s | 213.58s |
| Salt 2x50 | direct/双 worker/encoded（二次准入） | 97.25s | 43.75s | 140.99s |
| Ratebeer 2x20 | official/单 worker/无 encoded | 40.96s | 16.81s | 57.77s |
| Ratebeer 2x20 | direct/双 worker/encoded/adaptive | 36.77s | 16.06s | 52.83s |

- Salt 总训练时间改善 34.0%，encoded cache 累计命中率 55.2%。
- Ratebeer 总训练时间改善 8.5%，encoded cache 累计命中率 27.9%。
- Ratebeer 首次 prepare 154.90s，初始化 cache 命中后 5.76s。
- Salt 优化版峰值 RSS 17.15GiB，Ratebeer 8.91GiB，均低于 32GiB。
- Ratebeer 优化版复跑总时长 52.26s，短回归收益可复现。

Ratebeer 优化/消融聚合 loss 相差 `1.35e-6`；优化版自身复跑差约
`2.00e-5`，后者更大，符合 CUDA 非确定性波动，但仍未满足历史 `1e-7`
严格比较条件。CPU 固定输入和单/双 worker 对照仍逐值一致。

这些是每轮只取 20/50 batch 的跨 epoch 短回归。`max_batches` 会让各 epoch
使用各自 shuffle 后的前 N 个 batch，不能直接外推完整 epoch。

## 尚未完成的正式验收

尚不能宣称 Ratebeer 完整 2 epoch 低于既定时限，也未完成
Amazon/Stack/Avito 的两轮对照。后续应扩大采样规模并分别报告每轮 wall time，
同时继续追踪 CUDA loss 非确定性，不放宽既有严格阈值。

实验文件：

- `benchmarks/adaptive-f1-ablation.json`
- `benchmarks/adaptive-f1-latest-v2.json`
- `benchmarks/adaptive-f1-no-operation-telemetry.json`
- `benchmarks/adaptive-f1-cache-smoke.json`
- `benchmarks/adaptive-salt-ablation-50.json`
- `benchmarks/adaptive-salt-latest-50.json`
- `benchmarks/adaptive-f1-gpu-2epoch-smoke.json`
- `benchmarks/adaptive-f1-gpu-latest-full-2epoch.json`
- `benchmarks/adaptive-salt-gpu-ablation-2epoch-50.json`
- `benchmarks/adaptive-salt-gpu-latest-2epoch-50.json`
- `benchmarks/adaptive-ratebeer-gpu-ablation-2epoch-20.json`
- `benchmarks/adaptive-ratebeer-gpu-latest-v2-2epoch-20.json`
- `benchmarks/adaptive-ratebeer-gpu-latest-v2-repeat.json`
- `benchmarks/adaptive-salt-gpu-latest-2epoch-50.json`
- `benchmarks/adaptive-ratebeer-gpu-ablation-2epoch-20.json`
- `benchmarks/adaptive-ratebeer-gpu-latest-v2-2epoch-20.json`
- `benchmarks/adaptive-ratebeer-gpu-latest-v2-repeat.json`
