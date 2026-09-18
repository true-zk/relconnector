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

## 显式 node ID 契约

新写入的无自然主键 data table 会自动增加
`__relconnector_node_id__ INTEGER PRIMARY KEY`，值严格为 `0..N-1`。已有数据库先做只读检查：

```bash
../.venv/bin/python -m data.initialize_node_ids
```

确认报告后再迁移。迁移会事务性重建无主键表，因此需要接近目标表大小的临时空间：

```bash
../.venv/bin/python -m data.initialize_node_ids --apply --vacuum
```

也可以显式传入一个或多个 SQLite 路径。工具会拒绝 rowid 有缺口的表，避免改变旧采样器使用的逻辑 node ID。

为保留旧实现的严格对照，可先复制数据库再迁移：

```bash
mkdir -p data/relbench-explicit-node-id
cp --reflink=auto data/relbench/*.sqlite data/relbench-explicit-node-id/
../.venv/bin/python -m data.initialize_node_ids \
  --database-dir data/relbench-explicit-node-id --apply --vacuum
```

## 官方 stype 初始化与目标列修复

新下载流程会在 SQLite 写入后生成 `<dataset>.stypes.json`。现有数据库先刷新任务
metadata，再生成 stype，顺序不能反过来（catalog 更新会让 artifact 过期）：

```bash
../.venv/bin/python -m data.update_catalog_metadata --database-dir data/relbench
../.venv/bin/python -m data.initialize_stypes --database-dir data/relbench
# 或单库
../.venv/bin/python -m data.initialize_stypes --database data/relbench/rel-f1.sqlite
```

metadata 刷新只使用本地 Hugging Face manifest，不读取训练标签全表。修复 autocomplete
隐含隐藏目标列，并对已证实的 Ratebeer external 目标列遗漏应用限定任务的勘误。
不以列名/字符串内容更改 URL、邮箱、UPC 等 stype。

stype 初始化固定 RandomState(42)、表名字典序、test_timestamp 截断后的数据库视图；
按 pandas.sample 的无放回抽样方式选最多 1000 行，调用 PyTorch Frame 官方推断，
并沿用 RelBench 的 embedding -> multicategorical 转换。仅读取这些节点的特征。
为与 pandas 抽样一致，ID 排列临时内存是 O(N)，特征内存只与样本规模相关。

artifact 记录样本 ID、依赖版本、数据库路径/大小/mtime、cutoff 和校验指纹；
运行时必须匹配，缺失/过期/损坏时明确报错。迁移 node ID 后需重新生成 artifact。
工具依赖 pytorch-frame（包含在项目 training 环境中），不生成完整 TensorFrame 或 embedding。
