# 100 batch 快速实验方法

## 1. 目的

本方法用于在不进行全量训练的情况下，快速比较 RelConnector v2 的数据供给、缓存和多进程优化策略。实验固定每个数据集每个 epoch 最多使用 100 个 batch，并重复 5 个 epoch，主要回答以下问题：

- 新策略是否降低平均 batch 训练时间；
- 首轮付出的缓存构建成本能否在后续 epoch 回收；
- 不同策略在完全相同输入上的相对收益；
- 主进程和完整多进程系统的内存占用如何变化；
- 是否存在只在某些数据集有效、在其他数据集退化的策略。

该方法适合算法筛选、消融实验和回归检查。单次十库运行属于快速筛选；严格的性能结论必须使用同场配对和多次重复。

## 实验级别与“干净”的定义

### 快速筛选

快速筛选可以只运行一次候选策略，用于发现明显收益、回归、OOM 或磁盘膨胀。它不能证明候选策略优于历史结果，因为历史运行的机器负载、代码版本、缓存状态和运行顺序可能不同。

### 严格配对实验

用于决定优化是否有效或写入论文时，必须同时满足：

1. **同输入**：基线和候选使用完全相同的 replay fingerprint；
2. **同代码**：在同一个冻结源码快照中运行，策略开关是两者之间唯一预期差异；
3. **同进程边界**：每个策略启动独立进程，内存缓存不会跨 run 继承；
4. **同应用缓存状态**：每个 run 开始前清空该实验的文本 embedding 缓存；schema/拓扑都使用相同的预热状态；
5. **同资源环境**：使用同一张 GPU、同一组 CPU/NUMA 资源、相同线程数，运行期间没有其他 GPU 计算任务；
6. **同监测开销**：基线和候选都启用相同的 RSS/PSS 采样；
7. **平衡顺序**：采用候选→基线、基线→候选的交替顺序，抵消温度、频率和 OS 页缓存的先后影响；
8. **独立重复**：推荐 4 次配对重复，报告均值、标准差、每对差值和置信区间。

这里的“同一次任务”表示由同一个实验调度器依次执行，**不是同时并发运行**。runner 等前一个 benchmark 主进程和全部子进程退出、GPU 资源释放、待测缓存清理完成后，才启动下一个 profile，因此基线和候选不会互相抢占 GPU 或 CPU。

当前 runner 使用固定随机种子打乱策略顺序。对于 `off roi_disk` 和 `--repeats 4`，两种策略各有两次先运行、两次后运行，形成平衡顺序。若修改 profile 集合或重复次数，必须检查 `execution.json` 中的实际顺序是否仍然平衡。

严格实验不能拿本次候选结果与数天前的历史基线直接作配对结论。历史图只能作为背景参照。

## 2. 标准实验配置

| 配置项 | 标准值 |
|---|---|
| 数据集 | RelBench 全部 10 个数据集 |
| epoch 数 | 5 |
| 每个 epoch 最大 batch 数 | 100 |
| batch size | 512 |
| 邻居采样 | `128,128` |
| 模型 channels | 128 |
| 随机种子 | 42 |
| assembly worker | 4 |
| 初始活跃 worker | 4 |
| worker 自适应伸缩 | 关闭 |
| 训练顺序 | `strict` |
| 跨 epoch 预取 | 关闭 |
| GPU | `cuda:0`，运行前记录占用情况 |
| Torch CPU 线程 | 1 |
| OMP/MKL/OpenBLAS 线程 | 1 |

### Torch CPU 线程为 1 的含义

`--torch-num-threads 1` 限制的是**每个进程内 PyTorch CPU 算子的 intra-op 线程池**，不是让整个 v2 只使用一个 CPU 核。

v2 仍有主进程、sampling 进程和 4 个 assembly worker 并行运行，因此整棵进程树可以同时占用多个 CPU 核。`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1` 和 `OPENBLAS_NUM_THREADS=1` 同时限制每个进程内部的原生计算库，避免 4 个 worker 各自再启动大量线程，造成 CPU 过度订阅和较大的运行波动。

该设置不限制 CUDA kernel 使用 GPU，也不表示 CPU 编码只能串行完成；并行度主要来自多进程 worker。选择 1 线程的目的是让缓存和 worker 策略对比更稳定。如果论文目标是寻找绝对最高吞吐，还应另做 `torch_num_threads=1/2/4` 的线程敏感性实验，但同一组策略对比中必须固定该值。

“100 batch”是上限，不是强制补齐。数据集本身不足 100 个 batch 时，必须使用真实 batch 数：

- F1：每个 epoch 6 个 batch，5 个 epoch 共 30 个；
- Event：每个 epoch 38 个 batch，5 个 epoch 共 190 个；
- 其余八个数据集：每个 epoch 100 个 batch，共 500 个。

所有均 batch 指标必须除以实际完成的 batch 数，不能统一除以 500。

## 3. 固定输入方法

### 3.1 先生成固定回放

每个数据集先用 CPU 运行一次 `prepare-replay-only`，物化最多 100 个 batch。回放内容包括种子节点、邻居、标签和负样本。回放 manifest 保存输入 fingerprint 和实际 batch 数。

不同策略必须使用同一个 replay fingerprint。若 fingerprint 不一致，该组结果不能作为配对对照。

### 3.2 五个 epoch 重复同一组 batch

正式实验的 5 个 epoch 均读取同一份固定回放。这种设计有意保持输入不变，用于隔离缓存跨 epoch 复用收益，避免正常随机采样导致每轮输入集合变化。

固定回放实验不用于评价随机采样质量或最终模型精度。它只用于性能分析。

### 3.3 严格训练顺序

使用 `training-order=strict` 并关闭跨 epoch 预取，保证：

- 各策略消费 batch 的顺序一致；
- 上一个 epoch 完成后才开始下一个 epoch；
- 后续 epoch 的缓存收益不会与跨轮提前计算混在一起。

## 4. 缓存冷热边界

### 4.1 可以预热的内容

回放准备阶段生成的 schema 和拓扑初始化产物可以在正式运行中复用。这样可以把实验重点放在 batch 采样后的特征获取、编码、组装和训练供给上。

初始化准备耗时必须单独记录，不能计入“训练阶段平均 batch 时间”。

### 4.2 每个策略必须冷启动的内容

每次策略运行前和结束后都必须删除并验证该实验拥有的 `cache/text_embeddings` 目录不存在。双向清理可以覆盖第一轮启动、正常完成、策略异常和上一次残留等情况。不同策略之间不得共享以下内容：

- 内存文本缓存状态；
- 磁盘文本 embedding block；
- 策略运行过程中生成的磁盘索引；
- 前一个策略留下的 worker 本地缓存。

每个策略必须启动独立 benchmark 进程，不能在同一个 trainer 中切换策略。

runner 还必须等待该 benchmark 所在 process group 的所有子进程退出。若正常退出后仍发现 sampling、assembly、manager 或 resource tracker 进程存活，该轮标为失败，清理残留进程后才允许继续，不能让孤儿 worker 影响下一轮。

以下内容有意保留，并且基线与候选必须完全一致：

- `cache/schema`：已经生成的特征 schema 和统计信息；
- 拓扑初始化缓存：固定图结构和索引；
- `replay`：完全相同的 100 个采样 batch；
- 数据库原始文件：只读、不复制、不修改。

这些内容不是待测缓存。保留它们的目的是让两次训练都从相同的准备状态开始；将它们也清空会重新引入数分钟的 schema/拓扑初始化噪声。

### 4.3 操作系统页缓存

标准实验不全局清空 Linux OS page cache，因为这需要系统级权限，也会影响同机其他任务。OS 页缓存也属于真实多 epoch 训练中的系统行为。严格配对通过 4 次交替顺序平衡先后影响，报告中必须注明“应用缓存冷启动，OS 页缓存未清空”。若研究目标本身是磁盘冷读性能，应另设专门的冷页缓存实验，不能与本方法的结果混在一起。

## 5. 对照设计

至少保留以下两组：

1. `off`：关闭目标缓存，作为相同固定输入的基线；
2. 待测策略：例如 `roi_disk`。

如需分析缓存组件，可增加：

| profile | 含义 |
|---|---|
| `off` | 关闭待测缓存 |
| `legacy` | 原始缓存策略 |
| `cost_ram` | 成本策略，仅内存缓存 |
| `cost_disk_only` | 成本策略，仅磁盘文本缓存 |
| `cost_disk` | 成本策略，内存和磁盘缓存 |
| `roi_disk` | ROI 动态准入策略 |

策略运行顺序必须平衡，降低机器负载、OS 页缓存和运行顺序造成的系统误差。严格实验推荐每个策略进行 4 次独立重复，并报告均值、标准差、配对差值和置信区间。单次 5×100 结果只能用于快速筛选。

## 6. 时间指标

### 6.1 主指标：训练阶段平均 batch 时间

计算公式：

```text
training_s_per_batch = sum(epoch_wall_duration_s) / completed_batches
```

该指标包含训练阶段首批等待、数据供给、GPU 训练和 epoch 内协调，不包含独立的回放准备与冷初始化。

### 6.2 冷轮和热轮

分别报告：

```text
cold_epoch_s = epoch_1_wall_duration_s
warm_epoch_mean_s = mean(epoch_2 ... epoch_5)
warm_reduction_pct = 1 - warm_epoch_mean_s / cold_epoch_s
```

冷轮反映 worker 启动、首次编码和缓存写入成本；热轮反映跨 epoch 缓存命中后的稳定性能。

只报告后四轮时间会隐藏首次构建成本，因此必须同时报告五轮总时间、首轮时间和后四轮均值。

### 6.3 正式 run 的端到端平均 batch 时间

计算公式：

```text
run_e2e_s_per_batch = benchmark_duration_s / completed_batches
```

该指标包含正式 benchmark 进程内的模型准备和诊断回收，但仍不包含单独的 replay 准备阶段。它可以作为辅助指标，不能与包含完整冷拓扑初始化的历史结果直接等同。

### 6.4 相对变化

同一固定回放下，正数表示待测策略更快：

```text
reduction_pct = 100 * (1 - strategy_time / off_time)
```

只有 replay fingerprint、实际 batch 数、epoch 数和模型配置都相同时，才能将该百分比视为配对结果。

## 7. 内存指标

多进程 v2 不能只记录主进程 RSS。标准实验同时记录三种口径：

1. **主进程峰值 RSS**：benchmark 内通过 `RUSAGE_SELF` 记录，用于与旧实验同口径比较；
2. **进程树 RSS 求和峰值**：每 0.5 秒扫描同一 process group，求所有进程 RSS 之和；该值会重复计算共享页，只作为上界；
3. **进程树 PSS 求和峰值**：每 1 秒读取 `/proc/<pid>/smaps_rollup`，共享页按比例分摊，是多进程实际物理内存的主要指标。

报告中不得用主进程 RSS 代表整个 v2 系统内存。应以整树 PSS 为主要结论，同时保留主进程 RSS 用于和旧图对齐。

## 8. 标准运行命令

从项目目录 `/home/zk/relconnector/relconnector` 执行：

```bash
PYTHONPATH=relconnector_v2 \
  .venv/bin/python -m relconnector_v2.run_cache_iteration \
  --output-dir benchmarks/<experiment-name> \
  --profiles off roi_disk \
  --epochs 5 \
  --max-batches 100 \
  --repeats 4 \
  --device cuda:0
```

上面的命令是严格配对实验。若只做单次快速筛选，可显式使用 `--repeats 1`。

只运行 ROI 策略：

```bash
PYTHONPATH=relconnector_v2 \
  .venv/bin/python -m relconnector_v2.run_cache_iteration \
  --output-dir benchmarks/<experiment-name> \
  --profiles roi_disk \
  --epochs 5 \
  --max-batches 100 \
  --device cuda:0
```

只运行指定数据集可增加：

```text
--datasets rel-amazon rel-ratebeer
```

输出目录必须不存在，避免新旧结果混写。

## 9. 运行前检查

运行前记录以下状态：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv
df -h /home/zk/relconnector
```

要求：

- GPU 显存足以完成模型和最大 batch；
- 严格实验开始前目标 GPU 上没有其他计算进程；运行期间出现其他 GPU 任务时，该轮作废；
- CPU 没有持续的高负载竞争任务；如使用 `taskset` 或 `numactl`，所有策略必须使用相同绑定；
- 磁盘至少保留实验缓存预算之外的安全余量；
- 全部策略使用相同代码快照和依赖环境。

## 10. 结果完整性检查

每个数据集/策略必须满足：

- `status == ok`；
- `completed_batches == actual_batches_per_epoch × 5`；
- 恰好有 5 条 epoch 记录；
- 每轮 batch 数一致；
- categorical input validation 覆盖全部 batch；
- 固定输入 fingerprint 与对照相同；
- 结果文件只有一条正式记录，不能混入重跑数据；
- 没有 timeout、OOM、worker 异常退出或磁盘保护触发。

若任何一项不满足，该运行必须标为失败，不能进入均值和加速比计算。

## 11. 标准结果表

每个数据集至少输出以下字段：

| 字段 | 含义 |
|---|---|
| dataset | 数据集名称 |
| batches_per_epoch | 每轮实际 batch 数 |
| total_batches | 五轮实际总 batch 数 |
| training_s_per_batch | 训练阶段平均 batch 时间 |
| run_e2e_s_per_batch | 正式进程端到端平均 batch 时间 |
| cold_epoch_s | 首轮耗时 |
| warm_epoch_mean_s | 后四轮平均耗时 |
| reduction_vs_off_pct | 相对相同 replay 全关缓存的变化 |
| main_peak_rss_gib | 主进程峰值 RSS |
| tree_peak_pss_gib | 整树峰值 PSS |
| tree_peak_rss_sum_gib | 整树 RSS 求和上界 |
| disk_write_blocks | 磁盘缓存写入 block 数 |
| disk_hit_blocks | 磁盘缓存命中 block 数 |
| disk_roi_rejections | ROI 拒绝写入数 |

汇总时同时给出逐库结果和总体结果。总体结果优先使用总训练时间之和进行加权：

```text
weighted_s_per_batch = sum(training_wall_s) / sum(completed_batches)
```

不能只报告十个百分比的算术平均，因为百分比回归没有下界，且不同数据集的实际 batch 数不同。

## 12. 结果解释边界

固定 100-batch 实验会放大以下成本：

- 多进程启动与模型加载；
- 首轮缓存写入；
- 小数据集的固定协调开销；
- 大型拓扑的冷初始化准备。

因此：

- 它适合判断策略方向和跨 epoch 复用；
- 不适合直接替代完整数据集的最终吞吐结果；
- 不应与 epoch 数、缓存状态或初始化边界不同的历史图直接计算论文加速比；
- 若快速实验与完整训练结论冲突，应使用相同缓存状态的完整训练复测。

## 13. 清理与保留

每个数据集完成后立即删除该实验拥有的派生 `cache` 目录。完成分析和生成 fingerprint 后，删除体积较大的 `replay` 数据。

必须保留：

- 根目录 `manifest.json` 和 `status.json`；
- `summary.json`、`summary.csv` 和报告；
- 每个正式 run 的 `result.jsonl`、`execution.json` 和 `run.log`；
- 冻结的源码快照及 SHA256；
- replay fingerprint、实际 batch 数和完整运行参数。

不得删除数据库原始数据，也不得把实验 cache 写入长期共享的 `data/relbench/.relconnector-v2-cache`。
