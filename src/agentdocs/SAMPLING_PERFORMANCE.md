# RelX 一跳采样性能对照（2026-11-19）

## 测量范围

对本地 10 个 RelBench SQLite 库各取一条真实外键关系，读取来源表最多 100,000 行，按实际外键构建单关系图。`rel-event` 和 `rel-f1` 的选中关系不足 100,000 行。每个图按入度选出最多 256 个候选目标节点；每批固定随机抽取 128 个 seed，时间边界取所选记录时间的第 75 百分位，`fanout=16`。每组运行 64 批、重复 7 次，报告每批耗时中位数。图构建、SQL 读取、特征处理和训练都不计时。

运行机器为 Intel Xeon E5-2630 v4（20 个逻辑 CPU）；原生扩展使用脚本中的 `-O2` 编译选项。pyg-lib 对照使用本地 `0.5.0+pt27cu128` 和 PyTorch `2.7.1+cu128`。

“旧 Python”指迁移前的 `NativeOneHopSampler`：**邻居筛选和随机抽取已经在同一个 C++ `sample_neighbors()` 中**，但 Python 逐边组装节点和边。新实现调用 C++ `sample_one_hop()` 完成抽取与组装。两者在每个库的首批结果逐数组校验一致。下表均使用 4 个 C++ worker。

| 数据集 | 关系 | 旧 Python 毫秒/批 | 当前 C++ 毫秒/批 | 加速比 |
| --- | --- | ---: | ---: | ---: |
| rel-amazon | review → product | 0.935 | 0.297 | 3.15× |
| rel-arxiv | citations → papers | 0.960 | 0.310 | 3.09× |
| rel-avito | VisitStream → AdsInfo | 0.396 | 0.146 | 2.71× |
| rel-event | event_interest → events | 0.605 | 0.190 | 3.18× |
| rel-f1 | results → drivers | 0.832 | 0.278 | 2.99× |
| rel-hm | transactions → article | 0.975 | 0.314 | 3.10× |
| rel-ratebeer | beer_ratings → users | 0.924 | 0.305 | 3.03× |
| rel-salt | salesdocumentitem → salesdocument | 0.784 | 0.263 | 2.98× |
| rel-stack | votes → posts | 0.894 | 0.285 | 3.13× |
| rel-trial | conditions_studies → studies | 0.616 | 0.179 | 3.45× |

10 个比值的几何均值为 **3.08×**。`fanout=1`、4 worker 时仅为 **1.32×**（几何均值）：输出边少时，Python 逐边组装所省时间也少，固定原生调用开销占比更高。

独立运行线程对照：4 worker 在 8 个库上快于 1 worker；`rel-avito` 和 `rel-trial` 使用 1 worker 更快。8 worker 在这组负载下没有胜过 4 worker。线程数应由工作量决定，不宜固定越大越好。

额外用同一批单关系图、相同 `fanout` 和时间条件测了 `torch.ops.pyg.hetero_neighbor_sample` 的一跳算子。其输入 Torch 张量在计时前准备；返回 Torch 张量，不转换成 RelX 数组。当前 RelX C++ 接口平均每批 0.26 ms，pyg-lib 算子平均每批 0.37 ms。**两者输出协议和包装成本不同，这不是多跳、完整模型或总训练时间的结论。**

这些图只含每库的一条外键关系和部分真实记录，不能代表完整数据库的度分布、关系数和训练耗时。若将 3.08× 的采样加速代入整批训练，仍须先测采样占比；例如采样只占 10% 时，按 Amdahl 公式总加速约 1.07×。当前 `rel-event`、`rel-ratebeer` 等库的部分外键存为 `REAL`，原生图键接口尚不支持，需要在完整图基准前处理。

## 下一步性能工作

1. 在真实训练批次记录 seed 读取、采样、特征收集、文本处理、模型前后向和队列等待的耗时、吞吐及峰值内存，先找总耗时最大的阶段。
2. C++ 采样直接在 CSC 的可见区间选中至多 `fanout` 个偏移；当前先复制所有可见邻居，稀疏采样时是 `O(degree)`。当 `fanout` 接近可见度数时，再使用整段复制的路径。
3. 复用 worker 线程，按 batch 实际工作量选线程数；直接写预分配的输出缓冲区，减少当前“邻居向量 → 批次向量 → NumPy”多次分配与复制。
4. 多跳阶段在同一张完整异构图上，对齐 pyg-lib 的时间、去重、seed 归属和输出协议，再比较 pyg-lib 与 RelX 原生内核的准确性、吞吐和内存。通过 Python 调用 pyg-lib 的 `torch.ops` 已经执行原生代码，无需为了形式再从自定义 C++ 调用一次。
5. 若端到端瓶颈落在特征处理，将稳定的特征快照放入列式缓存，并在原生 batch 处理中收集数值/类别等简单特征；文本处理和模型计算分别计时后再决定迁移。

复现命令：

```bash
PYTHONPATH=src .venv/bin/python scripts/bench_sampling_10.py --workers 4 --output exps/benchmarks/relx-sampling-10.json
PYTHONPATH=src .venv/bin/python scripts/bench_sampling_10.py --workers 4 --compare-pyg --output exps/benchmarks/relx-sampling-10-pyg.json
```
