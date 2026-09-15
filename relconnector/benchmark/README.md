# Benchmark

该目录是实验与观测层，不实现模型训练算法。

- telemetry.py：独立于训练包的阶段/操作计时装饰器、RSS、系统内存、CUDA allocated/reserved。
- wrappers.py：在线组件的外部测量包装器。
- baseline.py、online_training.py：组装并测量两套训练实现。
- online.py：仅测图构建/采样/SQL 特征读取的数据路径实验。
- runner.py、online_runner.py：逐任务子进程隔离、失败记录、timeout、resume。
- cases.py、process.py：共享发现/续跑与进程测量；持续排空输出，仅保留有界尾部；超时/中断清理进程组。
- result.py、metadata.py：结果契约及实验配置。
- report.py：单文件阶段汇总；compare.py：两次实验的配对比较及语义差异检查。

## 运行

```bash
python -m benchmark.runner --reader pandas --epochs 1 \
  --task-timeout 600 --output benchmarks/baseline-all.jsonl
python -m benchmark.online_runner --reader pandas --epochs 1 \
  --task-timeout 600 --output benchmarks/online-all.jsonl
python -m benchmark.report benchmarks/online-all.jsonl
python -m benchmark.compare benchmarks/baseline-all.jsonl benchmarks/online-all.jsonl
```

两个 runner 都支持 --dataset、--task 多次传入，默认遍历本地库的所有任务。--resume 只跳过成功项；务必用相同配置续跑，不同配置使用不同输出文件。不要同时跑正式对比实验，也不要将冷启动、暖 OS 页缓存、物化缓存混为同一口径。

online_runner 还可设置 --executor sync/async、--feature-cache-mb、--seed-queue-mb、--plan-queue-mb、--ready-queue-mb、--text-batch-size、--text-model-path。首轮建议较小 batch 和 fanout，再按实测资源提高预算。

## 指标口径

- telemetry.overall：训练编排的 wall time；process.duration_s 额外包含进程导入、启动与退出。
- phases：baseline 的读库/物化/训练阶段；online 的 prepare/train。
- operations：online 分开计 seed_read、sampling、feature_fetch、batch_assemble、train_step；baseline 为 sample_and_prepare、h2d、forward、backward、optimizer_step。
- 异步阶段会重叠，操作时间不能直接求和当总 wall time；采样测量包含等待 RNG 保护锁的时间。
- RSS 和系统内存通过 /proc 轮询，可能漏掉短暂尖峰；系统内存还受其他进程影响。父进程测得的 peak_rss_mb 仅是 worker 主进程，不含 loader 子进程；默认 num_workers=0。
- CUDA 时长采用同步墙钟测量，会影响吞吐；不要当作无干扰 profiler。CUDA 峰值是 PyTorch allocator，不是全 GPU 驱动内存。
- operation count/total/mean 全量累计，分位数最多保留 4096 个 reservoir 样本；phase 最多保留 4096 条，并记录丢弃条数。操作/阶段名称应是固定集合，不为每个 batch 创建新名字。

## 公平比较

结果包含 experiment.config、reader、device、executor、编码/模型/采样标识、数据库路径/大小/mtime；比较同时检查实际 batch/样本数。数据库内容应只读不变，正式发布可另外记录内容 hash。

baseline 与 online 现在共享 TensorFrameFeatureSchema、全局统计量、GloVe、HeteroEncoder/GraphSAGE、seed 顺序、pyg-lib sampler 和逐 batch RNG。baseline 从训练前完整物化的 TensorFrame store 取特征，online 从 SQL 取原始行后按需生成同样的 TensorFrame。compare 只有在 schema fingerprint、模型、采样、线程数、工作量和 loss 均匹配时才显示 strictly-comparable。正式配对实验应同时使用 --executor sync --torch-num-threads 1。

## 独立测量

```python
from benchmark import TelemetryRecorder, timed


@timed("connector_query")
def query(reader):
    return reader.read_query("SELECT COUNT(*) FROM drivers")


with TelemetryRecorder().activate() as recorder:
    result = query(reader)
report = recorder.report()
```

仅导入计时器或 connector 不会加载两套训练栈；torch 未导入时计时器只测 CPU 资源。训练代码不反向导入本目录。
