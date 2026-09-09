# RelBench 数据下载与 SQLite 落库

`data/` 是独立的数据准备工具，不属于 `relconnector` 运行时包。它负责：

- 从 Hugging Face 下载一个或全部 RelBench 数据集；
- 加载完整关系表以及可选的全部任务 target；
- 保存为独立 SQLite 文件；
- 保存主键、外键、时间列、任务 split、`val_timestamp`、`test_timestamp`
  和 task manifest 配置等 catalog；
- 通过临时文件写入和原子替换避免留下半成品数据库。

生成的数据默认位于：

```text
data/relbench/<dataset>.sqlite
```

## 环境

```bash
cd /workspace/relconnector
uv pip install --python ../.venv/bin/python -r data/requirements.txt
```

代理和令牌通过环境变量传入，不写入代码：

```bash
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=.byted.org
export HF_TOKEN=YOUR_HUGGING_FACE_TOKEN
```

脚本也兼容已有环境中的 `HF_TOEKN`，但新配置应使用标准变量 `HF_TOKEN`。

## 下载单个数据集

```bash
../.venv/bin/python -m data.download \
  --dataset rel-ratebeer \
  --task beer_ratings-total_score \
  --revision REVISION_SHA \
  --output data/relbench/rel-ratebeer.sqlite
```

重复传入 `--task` 可保存多个任务；使用 `--all-tasks` 保存该数据集的全部任务。不传任务
参数时只保存关系数据库本身。

## 下载全部数据集

默认发现 RelBench 官方 v1 和 v2-extra 仓库中的所有数据集，并保存全部任务：

```bash
../.venv/bin/python -m data.download_all
```

常用控制参数：

```bash
# 只查看数据集清单
../.venv/bin/python -m data.download_all --dry-run

# 只下载指定数据集
../.venv/bin/python -m data.download_all --dataset rel-f1

# 只下载关系表，不下载任务 target
../.venv/bin/python -m data.download_all --without-tasks

# 覆盖已有数据库；默认行为是跳过，便于断点续跑
../.venv/bin/python -m data.download_all --force
```

单个数据集失败时脚本默认继续，并在结束时以非零状态退出和汇总失败数。
`--fail-fast` 可改为首次失败即停止。

## 模块职责

- `source.py`：RelBench 下载和对象转换。
- `models.py`：下载侧数据契约。
- `encoding.py`：写入前的值编码。
- `writers.py`：数据库 writer 基类、SQLite 实现及注册工厂。
- `discovery.py`：Hugging Face 数据集发现。
- `download.py`：单数据集编排与 CLI。
- `download_all.py`：全部数据集批量编排与 CLI。

读取数据库的代码位于 `relconnector/connector/`，不属于本目录。

已有 SQLite 可通过以下命令只更新 task manifest 元数据，无需重写数据表：

```bash
../.venv/bin/python -m data.update_catalog_metadata
```
