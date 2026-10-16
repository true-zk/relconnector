# v2 分组消融设计与执行协议

## 研究问题与分组

本设计以当前 v2 实现为准。附件中的图示简化条件不作为已实现逻辑：disk
启用仍使用数据库文件大小和编码暖机条件；adaptive local 文本层保存共享槽位索引，
不是完整文本向量副本。保留 Direct GloVe、读取策略、模型和采样算法。

三个主因素：P = assembly 多 worker；F = 普通特征复用缓存（raw block + encoded
TensorFrame）；T = 文本复用缓存（adaptive local index + shared vectors + disk）。
Encoded TensorFrame 也可能包含文本向量，因此 F 与 T 有重叠收益，不能将速度收益相加。
子图/batch buffer 是消费型队列，属于流水线机制，不属于本轮 F/T 消融。

## 第一轮：5 个核心配置

| runner variant | workers | F | T | 解释 |
| --- | ---: | --- | --- | --- |
| full | 4 | 开 | 开 | 受控参照：v2，但固定活跃 worker 数 |
| one_worker | 1 | 开 | 开 | 去掉 assembly 多 worker，仍保留异步流水线 |
| no_feature_cache | 4 | 关 | 开 | 普通特征缓存的条件贡献 |
| no_text_cache | 4 | 开 | 关 | 文本缓存在 encoded cache 存在时的条件贡献 |
| no_reuse_cache | 4 | 关 | 关 | 复用缓存的联合贡献，并识别 F/T 遮蔽 |

这不是完整 P×F×T 因子设计；若要识别 worker 与缓存的交互，补 1 worker 下
F-off、T-off、F/T-off 三组，形成 8 组。不要把 one_worker 称为“无流水线”。
所有组关闭自适应 worker 调度，避免“同上限、不同实际活跃数”的混淆。

配对解释（比较时间差或时间比，不能相加各组 speedup）：

- full vs no_feature_cache：T 存在时 F 的收益。
- full vs no_text_cache：F 存在时 T 的收益。
- no_feature_cache vs no_reuse_cache：无 F 时 T 的收益。
- no_text_cache vs no_reuse_cache：无 T 时 F 的收益。
- 若 full vs no_text_cache 差异小，而 no_feature_cache vs no_reuse_cache 差异大，
  说明文本缓存收益被 encoded cache 部分覆盖，不代表文本缓存无效。

## 代码开关与含义

| 目标 | 相对参照的 CLI 改动 | 不能声称的结论 |
| --- | --- | --- |
| 单 worker | `--assembly-processes 1 --initial-active-processes 1` | 不是单进程串行训练 |
| 关闭 F | `--feature-cache-mb 0 --encoded-cache-mb 0` | 不关闭 OS page cache、初始化缓存 |
| 关闭 T | `--text-cache-mb 0 --disk-text-cache-mode off` | 不关闭窗口内去重或 encoded cache 中已有的文本特征 |
| 只关 raw | `--feature-cache-mb 0` | 不关闭密度感知查询 |
| 只关 encoded | `--encoded-cache-mb 0` | 当前 raw fetch 仍在 encoded lookup 前，不可推断 SQL 自动减少 |
| 只关 local index | `--text-cache-scope shared` | 不应用 `local`，后者切换为独立本地向量缓存 |
| 只关 disk | `--disk-text-cache-mode off` | adaptive 在参照组从未激活时，该对照不构成有效 disk 消融 |
| 去掉跨 batch 窗口 | `--window-batches 1` | 保留批内去重和查询策略，并同时改变调度粒度 |

关闭容量意味着不保存跨请求复用内容，仍可能有空缓存对象、lookup 计数和批内去重。
初始化 schema/topology artifact 保留，这是另一种缓存，不包含在“no_reuse_cache”名称中。

## 第二轮：仅拆分有证据的模块

1. F：full/no_raw_cache/no_encoded_cache/no_feature_cache。
2. T：full/no_local_text_index/no_disk_text_cache/no_text_cache。
   local 与 disk 的独立效应可先固定另一层状态再对比；强制 disk on 的配置另行命名。
3. 窗口：4 vs 1，报告实际窗口大小分布、SQL 次数、读行数、唯一节点数。
4. 不把 SQL 密度选择、inverse 恢复、去重等直接全部关闭。尤其 inverse 和部分
   唯一节点处理是正确性路径；独立测试读取策略需要专门策略开关和输入等价验证。

## 固定条件

- 默认主实验沿用 ready；所有组使用同一模式。strict 用于功能验证和必要的配对复核。
  ready 在不同组中可能产生不同更新顺序，因此不要求最终 loss 逐值相同。
- 同一 task、模型、采样 fanout、batch size、seed、训练批数及 epochs。
- 固定 worker 数及初始活跃数，禁用 adaptive workers。
- 固定池总预算：raw 4096 MiB、encoded 2048 MiB、text vectors 512 MiB；
  raw 和 local encoded 在进程池内分摊，共享预算不随 worker 数增加。
  adaptive 本地索引元数据及进程开销不包含在以上 payload 预算中。
- 关闭某类缓存后不将释放的预算重新分配给其他缓存。
- 统一 encoded scope=shared，不静默继承 HM 的历史特例。
  若后续采用各数据集调优配置，必须单独标注并在数据集内保持不变。
- 每 worker 的 torch/BLAS 线程数为 1；同机实验顺序运行，避免 CPU/IO 争用。
- cross-epoch prefetch、queue 和 inflight 预算固定，不与缓存一起消融。
- 3 次性能重复使用同一训练 seed、改变配置执行顺序。训练 seed 泛化是另一个实验。

## 冷暖状态与实验长度

runner 为每个配置/重复创建独立 cache-dir：应用缓存从空开始，后续 epoch 允许自然复用。
不读取既有磁盘文本缓存，不删除用户历史缓存。OS page cache 不主动清空，不能称为
完全冷磁盘实验。配置顺序使用可记录的随机种子打乱。

第一轮建议 Arxiv、HM、Salt：3 epochs × 至多 200 batches × 3 repeats × 5 configs。
这是有训练批数上限的 pilot，不是完整 epoch。F1 只用于开关与正确性 smoke。
按观测延长：若共享/local/disk 从未激活，先报告未激活，不从零收益推断机制无效；
若命中率仍持续变化或时间不稳定，延长到完整 epoch/更多 epoch。
disk 机制验证可补 Stack/Ratebeer，分别报告构建成本与后续复用收益。

计时分别记录 benchmark duration（含 prepare）、pipeline duration、逐 epoch duration
和 ready_wait。pipeline duration 包含启动/收尾，不等于纯稳态训练时间。
代码核对发现逐 epoch duration 从该 epoch 第一个 batch 出队后才开始，遗漏首批等待；
epoch ready_wait 却包含这段等待，所以可能大于 epoch duration。不可将 epoch duration
当完整 epoch 时间，亦不可用两者相除得到等待占比。本轮主结果采用 benchmark/pipeline
duration；后续正式逐 epoch 对比需要统一 epoch 边界计时。不能仅因是第二轮就称为稳态。
短 F1 测试以功能验收为目标，不比较其加速比。

## 必须检查的计数器

- 各组 completed_batches/examples 与预期一致，所有 worker 报告齐全。
- F-off：raw insertions/current_bytes=0，encoded inserted_rows/current_bytes=0。
- T-off：shared 对象为空、disk 未启用，文本编码仍实际发生。
- full：说明 raw/encoded/text 各层实际命中与是否激活，而不只写“配置已开”。
- local：是否进入 active、访问/命中、shared lookup 的变化。
- disk：实际启用、读写/命中、编码次数、磁盘字节；构建与复用分别解释。
- 共享 cache 的全局快照可能在多个 worker 中重复出现，不能直接相加；
  worker-local 操作计数则按其定义汇总。
- 主进程 peak_rss_mb 不能作为整个 v2 的内存消融结论。正式测内存需增加进程树
  PSS 或隔离 cgroup 峰值，并明确页缓存计费方式；当前 runner 不提供此项测量。
- 不把多 worker 的累计 operation time 堆叠成端到端墙钟时间。

## 运行方式

在 `/home/zk/relconnector/exps` 运行。输出目录必须不存在，避免旧结果混入。
runner 保存 manifest（代码 revision、dirty 状态、完整命令、执行顺序）和每组日志、原始 JSON。
遇到失败或超时立即停止矩阵；已成功结果保留。没有按 dataset/task 粗粒度 resume。

```bash
PYTHONPATH=relconnector_v2 ../.venv/bin/python -m relconnector_v2.run_ablation \
  --dataset rel-arxiv --task paper-citation \
  --output-dir benchmarks/ablation-arxiv-plan --dry-run

PYTHONPATH=relconnector_v2 ../.venv/bin/python -m relconnector_v2.run_ablation \
  --dataset rel-arxiv --task paper-citation \
  --output-dir benchmarks/ablation-arxiv-pilot \
  --epochs 3 --max-batches 200 --repeats 3
```

选第二轮配置使用 `--variants full no_raw_cache no_encoded_cache no_feature_cache`。
补 worker/cache 交互使用 `--workers 1 --variants no_feature_cache no_text_cache no_reuse_cache`。
这仍需与同条件、同资源、相同冷暖协议的 4-worker 运行配对。

完成后用下面命令生成 summary.csv/summary.json，核验关闭缓存的遥测、worker 报告和工作量：

```bash
PYTHONPATH=relconnector_v2 ../.venv/bin/python -m relconnector_v2.summarize_ablation \
  benchmarks/ablation-arxiv-pilot
```

汇总检查不替代特征张量等价验证。共享缓存全局快照没有在汇总中重复相加。

## 本次实际执行

F1 smoke 使用 2 epochs × 2 batches、batch size 16、fanout 4,4、channels 16、
seed 42、strict，五个核心配置各运行一次。结果与有效性核验另存
`v2-ablation-smoke-20260928.md`。其结论仅限于开关可执行性及短程一致性。

另进行 Arxiv 短程五组试跑：2 epochs × 16 batches，batch size 512，fanout 128,128，
channels 128，seed 42，ready，各组一次；报告见 `v2-ablation-arxiv-short-20260928.md`。
这是截断工作负载，不能替代上面建议的长窗口、重复实验或训练质量验证。
