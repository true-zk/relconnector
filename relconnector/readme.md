# relconnector

本项目将 RelBench 的数据准备与训练运行时拆开：

- [`data/`](data/README.md)：下载 RelBench 数据库、task manifest 和
  train/val/test split，并完整保存到本地 SQLite。
- [`relconnector/connector/`](relconnector/connector/README.md)：通过 pandas 或
  Connector-X 从 SQL 重建 `Database`。
- [`relconnector/pipeline/`](relconnector/pipeline/)：本地图物化、pyg-lib
  邻居采样、PyG GraphSAGE 训练及时间/内存监控。

训练阶段不会调用 `RelBenchDataset.get_db()`，也不会从 Hugging Face 下载
`db.zip`。默认 GloVe 文本编码器首次初始化时仍需下载其模型。

## 训练环境

pyg-lib 必须与 PyTorch 和 CUDA ABI 匹配。当前已验证的组合是：

```text
torch 2.7.1+cu128
pyg-lib 0.5.0+pt27cu128
relbench 3.0.1
torch-geometric 2.8.0.post1
pytorch-frame 0.3.0
```

安装命令：

```bash
cd /workspace/relconnector

uv pip install --python ../.venv/bin/python torch==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

uv pip install --python ../.venv/bin/python pyg_lib \
  -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

uv pip install --python ../.venv/bin/python -e '.[connectorx,training]'
```

## 简单训练接口

```python
from relconnector.pipeline import RelBenchModel, TrainingConfig

model = RelBenchModel(
    dataset="rel-f1",
    task="driver-dnf",
    model="graphsage",
    reader="pandas",
    config=TrainingConfig(epochs=1),
)
result = model.train()
print(result.telemetry)
```

默认使用 `GloveTextEmbedder`。entity prediction 和 recommendation task 均支持。

## Baseline benchmark

正式 baseline 使用 pandas 全量读库、无图物化缓存、GloVe 文本编码、
pyg-lib 采样和 PyG 训练。每个 task 在独立子进程运行，避免 RSS、CUDA allocator
和 Python 对象跨任务污染：

```bash
../.venv/bin/python -m relconnector.pipeline.benchmark \
  --reader pandas \
  --epochs 1 \
  --output benchmarks/baseline-all.jsonl
```

快速验证：

```bash
../.venv/bin/python -m relconnector.pipeline.benchmark \
  --dataset rel-f1 \
  --epochs 1 \
  --max-batches 1 \
  --batch-size 16 \
  --num-neighbors 4,4 \
  --channels 16 \
  --device cpu \
  --no-text \
  --output benchmarks/rel-f1-smoke.jsonl
```

长任务中断后可复用同一输出文件继续执行；成功任务会被跳过，失败任务会重跑：

```bash
../.venv/bin/python -m relconnector.pipeline.benchmark \
  --reader pandas \
  --epochs 1 \
  --output benchmarks/baseline-all.jsonl \
  --resume
```

每条 JSONL 对应一个 task，记录：

- 阶段耗时：`database_open`、`task_read`、`database_read`、
  `text_encoder_init`、`graph_build`、`loader_build`、`model_build`、
  `train_epoch_N`。
- batch 操作耗时：`sampling`、`h2d`、`forward`、`backward`、
  `optimizer_step`，包含 count、total、mean、p50、p95、p99 和吞吐。
- 资源：进程 RSS、系统已用内存、CUDA allocated/reserved 的 before、after 和
  peak。总体 CUDA peak 使用 PyTorch allocator 的 peak counter。
- 环境、训练配置、loss、batch 数和样本数。

## 测试

```bash
../.venv/bin/python -m unittest discover -s data/tests -v
../.venv/bin/python -m unittest discover -s tests -v
../.venv/bin/python -m compileall -q data relconnector tests
../.venv/bin/pyright
```
