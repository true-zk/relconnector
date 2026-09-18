# relconnector

面向内存受限场景的关系数据库图训练实验项目。SQLite 是业务数据源；在线训练只常驻图拓扑，按采样结果读取节点特征，不生成特征快照或预采样训练集。

## 目录与依赖

- [data/](data/README.md)：纯离线下载与 SQLite 落库，保存任务 split、schema、时间配置。
- [relconnector/](relconnector/README.md)：在线 reader、图索引、采样、特征缓存、组装、训练和 executor。
- [baseline/](baseline/README.md)：保留全量读库和 TensorFrame 物化的朴素实现。
- [benchmark/](benchmark/README.md)：通用计时/内存工具、实验编排、子进程运行与结果比较。
- [doc/](doc/README.md)：架构讨论、实施计划、验证记录和 GPU 实验报告。

训练实现不依赖 benchmark；benchmark 调用两套实现。benchmark/ 是代码，benchmarks/ 是实验产物，data/relbench/ 是数据库。安装包含运行包、版本化正确性契约及离线工具；不包含 SQLite 或实验产物。

## 环境

当前使用 Python 3.12、torch 2.7.1+cu128、pyg-lib 0.5.0+pt27cu128、RelBench 3.0.1、pandas 3.0.5。

```bash
uv pip install --python ../.venv/bin/python -e '.[connectorx,training,dev]'
# pyg-lib wheel 必须与 torch/CUDA 版本匹配。
```

## 在线训练

首次使用旧数据库，请先运行 `python -m data.update_catalog_metadata`，再运行
`python -m data.initialize_stypes`。训练只读已固化的官方 stype；文件缺失或过期会拒绝启动。
`baseline/cache_baseline` 是缓存优化后的基线，支持 `--implementation cache`。

```python
from relconnector import OnlineRelBenchModel, OnlineTrainingConfig

model = OnlineRelBenchModel(
    dataset="rel-f1",
    task="driver-dnf",
    config=OnlineTrainingConfig(
        epochs=2,
        batch_size=16,
        num_neighbors=(4, 4),
        channels=16,
        device="cpu",
        torch_num_threads=1,
        executor="sync",
        feature_cache_bytes=16 * 1024**2,
    ),
)
result = model.train()
```

文本使用本地缓存的 sentence-transformers/average_word_embeddings_glove.6B.300d。在线训练不会下载模型；可通过 text_model_path 指向已下载目录。训练期间不下载 RelBench 数据或 task manifest。

## 实验入口

所有命令在项目根目录运行。先用 data/ 准备数据库，再运行 benchmark：

```bash
../.venv/bin/python -m benchmark.runner --dataset rel-f1 --task driver-dnf \
  --epochs 1 --max-batches 2 --device cpu --output benchmarks/baseline-smoke.jsonl
../.venv/bin/python -m benchmark.online_runner --dataset rel-f1 --task driver-dnf \
  --epochs 1 --max-batches 2 --device cpu --output benchmarks/online-smoke.jsonl
../.venv/bin/python -m benchmark.compare \
  benchmarks/baseline-smoke.jsonl benchmarks/online-smoke.jsonl
```

比较工具并排列出耗时和 RSS，同时检查 reader、executor、特征 schema fingerprint、TensorFrame/HeteroEncoder、采样策略、数据版本、训练工作量和 loss。baseline 与 online 现在共享这些训练语义；严格配置下实体与推荐任务的配对 smoke 得到了完全相同的 loss。历史 hash/Linear 编码结果不能与当前结果混用。

## 验证

```bash
../.venv/bin/ruff check .
../.venv/bin/ruff format --check .
../.venv/bin/pyright
../.venv/bin/python -m unittest discover -s data/tests -v
../.venv/bin/python -m unittest discover -s tests -v
```

已完成 F1 实体与推荐任务的两轮短程 CPU 训练。未重跑全 64 任务完整 epoch，未验证 CUDA 吞吐和显存；当前沙箱的进程退出可能报告 /proc/.../comm 权限限制，需与测试断言及 worker 结果分别判断。
