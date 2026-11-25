# rel-f1 多跳批次性能对照（2026-11-20）

**适用范围：**本报告测量的是全表特征预载的性能原型。该原型违反 RelX 正式训练必须按 batch 从源数据库读取未缓存特征的约束，因此下列加速比不得用于描述最终端到端训练。目标在线路径见 [ONLINE_FEATURE_PIPELINE.md](ONLINE_FEATURE_PIPELINE.md)。

## 工作负载与计时边界

使用本地 `exps/data/relbench/rel-f1.sqlite` 的 `driver-dnf` 训练集，共 11,411 个 seed，分为 90 个 batch（最大 128 个 seed/批）。RelX 完整图含 9 张业务表、26 种正反向关系、45 个特征。两跳 fanout 为 `[16, 8]`，2 个采样 worker、2 个特征 worker、每个特征 worker 内 1 个原生 gather worker，最多 4 个在途批次。每种方法先预热，再交替运行 3 次，取耗时中位数。

机器为 Intel Xeon E5-2630 v4（20 个逻辑 CPU）；扩展用 `scripts/build_relx_native.py` 的 `-O2` 编译。PyTorch 为 `2.7.1+cu128`，pyg-lib 为 `0.5.0+pt27cu128`，以下计时均在 CPU 上运行。

图构建、SQL 读入、特征状态拟合与原始列快照准备不计入批次耗时；模型前后向传播也不计入。样本、采样器、特征列和拟合状态相同。首批的图与所有特征输出逐项对照一致。Python 基线和 C++ 流水线都使用 RelX 的同一个 C++ 多跳采样器，目的是隔离**特征处理与流水线调度**的影响。

| 批次路径 | 90 批中位耗时 | 相对 C++ 流水线 |
| --- | ---: | ---: |
| Python 串行逐批变换 Pandas 特征 | 6.040 s | 12.44× 较慢 |
| Python 线程调度 + Python 特征变换 | 8.389 s | 17.28× 较慢 |
| 旧 Python 线程调度 + C++ 特征缓存 | 0.533 s | 1.10× 较慢 |
| C++ 采样→特征有界流水线 | **0.486 s** | 1.00× |

Python 特征变换的两条路径每批执行数值标准化、类别映射、时间处理、多类别解析、向量转换和文本整理；C++ 路径在准备阶段将可预计算的列编码一次，再由原生 worker 收集。因此 12.44× 主要反映特征缓存与一次性编码的收益，**不能解释为 C++ 采样快了 12.44×**。旧 Python 线程调度已经使用同一个 C++ 特征缓存；把调度队列移到 C++ 后，该对照只快约 1.10×。两种 Python 线程路径都不是 v2 的多进程 SQL/TensorFrame 实现。

## pyg-lib 采样单独对照

在同一完整图、同一批 seed、相同 CSC 和边时间、`uniform`/无放回条件下，直接比较两种原始采样调用；采样结果都不做特征处理：

| 原始两跳采样接口 | 90 批中位耗时 |
| --- | ---: |
| RelX C++ `sample_multi_hop` | 0.480 s |
| pyg-lib `hetero_neighbor_sample` | **0.138 s** |

当前 C++ 采样耗时为 pyg-lib 的 **3.47×**，即 pyg-lib 明显更快。首批都返回 12,178 条边；由于随机数算法不同，采样到的节点分别为 6,585 和 5,822 个，输出体积不完全相同。pyg-lib 此处只计原始算子，不含 v2 的 Python 包装、SQL 特征读取、TensorFrame 编码和多进程通信。因而这不是 v2 完整训练速度的比值。

本次总 batch 耗时 0.486 秒，而单独 C++ 采样已耗时约 0.480 秒，说明当前负载几乎由采样主导。下一步应先减少 `sample_multi_hop` 对每种关系、每一跳重复创建线程和中间邻居向量的开销，再测相同任务的完整训练吞吐及进程峰值内存。不能仅凭当前受控实验宣称新版本比 v2 完整流程快。

原始测量值分别在 `exps/benchmarks/relx-f1-cpp-pipeline-20261120.json` 和 `exps/benchmarks/relx-f1-cpp-sampler-20261120.json`。复现命令：

```bash
PYTHONPATH=src .venv/bin/python scripts/bench_rel_f1_pipeline.py --batches 90 --repeats 3 --output exps/benchmarks/relx-f1-cpp-pipeline-20261120.json
PYTHONPATH=src .venv/bin/python scripts/bench_rel_f1_sampler.py --batches 90 --repeats 3 --output exps/benchmarks/relx-f1-cpp-sampler-20261120.json
```
