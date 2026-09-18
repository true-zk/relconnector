# cache_baseline 冻结与正确性修订结果

## 1. 本轮交付

按本轮授权先完成版本冻结、再回补正确性问题。没有实施动态策略算法、
encoded TensorFrame cache、token 执行重写或多 worker 优化。

- `baseline/cache_baseline/`：保留原 latest 的节点去重、推荐分支合并、四 batch
  feature window、raw block LRU 和 GloVe 整句 LRU。
- `--implementation cache`：独立 benchmark adapter，不导入 latest 的训练组件。
- `relbench_compat/`：小型、版本化的 task/stype 数据正确性契约；不含训练实现。
- `experiment.correctness_version=relbench-correctness-v1`：新结果标识；比较工具
  将旧结果与新结果识别为不同正确性版本。

## 2. 修复前快照

修复前的 latest、batch、vanilla、benchmark 和离线代码共 119 个文件已保存：

- `baseline/archives/pre-correctness-20260916.tar.gz`
- `baseline/archives/pre-correctness-20260916.sha256.json`
- `baseline/archives/catalogs-before-correctness-20260916.json`

压缩包 SHA256：`aab3bd5ac361c21a4c2b254137bc345b144915dcac4c21b314437c68a7de2614`。
逐文件校验已通过。历史三组 GPU JSONL 未改写；当前基线目录按用户授权接受
正确性回补，所以复现修复前的源码须使用归档。

## 3. 已修复的问题

| 问题 | 根因和修复 | 覆盖版本 |
| --- | --- | --- |
| autocomplete 目标泄漏 | catalog 更新只复制 remove_columns，漏掉官方 hidden_columns 隐含添加的目标列；离线写入、更新和运行时均补齐 | 全部 |
| Ratebeer external 目标泄漏 | 上游 external manifest 未排除直接作为 label 的 total_score；应用明确限定任务/实体/列/kind 的勘误 | 全部 |
| stype 与官方抽样不一致 | 旧版取按 ID 排序的前 1000 行；现离线固化官方随机抽样及推断结果，训练读取同一 artifact | 全部 |
| object dtype 未还原 | reader 的 nullable/string dtype 被保留，影响官方类型推断；按 catalog 转回 object | 全部 reader 及离线抽样 |
| vanilla rowid 查询退化 | WHERE/ORDER BY 使用 rowid-1；现使用裸 rowid 和节点 ID +1，SELECT 保留零基映射 | vanilla；其余在线版本原有修复保留 |
| 基线默认路径错误 | 拷贝增加一层目录后 parents 下标未调整 | batch、vanilla、cache |
| batch benchmark 导入/类型混用 | 入口引用不存在的 relconnector.baseline 路径，同时混用 latest 和 vanilla contracts | batch benchmark |
| GloVe 缓存实际 storage 超出计费 | 缓存单行 Tensor view 会保留整批 backing storage；改为缓存独立 clone | latest、cache |
| SQL amplification 分母失真 | 常量特征节点无需 SQL，却计入待查询行数；增加 featureless_rows 并从分母排除 | latest、cache |

`text_embedding_cache_bytes` 仍表示 Tensor payload 预算，不包含字符串键和 Python
容器开销，也不是进程 RSS 硬限制。此次 clone 修复解决额外持有整批 Tensor storage。

### 目标隐藏的边界

原始目录与显式 node ID 目录各有 64 个任务，已刷新全部任务 metadata。
其中每份目录 21 个 autocomplete 任务补齐目标隐藏，另修复 Ratebeer external
任务的 total_score。forecast 的同名历史观测列不因名字相同而删除。

Ratebeer 勘误是防止直接 label 泄漏的正确性修复，不是对字符串 stype 的人工干预。
URL、邮箱、电话、UPC、邮编等仍按官方推断处理。

### stype 的可复现定义

`data.initialize_stypes` 生成数据库旁的 `<dataset>.stypes.json`：

1. 数据库视图采用 `dataset.get_db()` 默认的 test_timestamp 截断范围。
2. 表顺序固定为名字典序，随机状态固定为 NumPy RandomState(42)。
3. 使用与 pandas.sample 相同的 choice(replace=False) 选择最多 1000 行。
4. SQL 仅读取被选节点的特征，恢复 catalog dtype，调用官方 infer_df_stype。
5. 沿用 RelBench embedding -> multicategorical 转换；不按列名人工改型。
6. 排除后加的合成 node ID；任务隐藏列在消费 artifact 时排除。

官方 get_stype_proposal 本身没有固定 RNG 或表顺序；本项目固定这两项以便复现。
小数据上与官方函数在相同种子、顺序、数据视图下逐表对照通过。

每表特征读取最多 1000 行；为精确复用 pandas 抽样，ID 排列的临时内存为 O(N)，
不是严格 O(1000)。没有全量读取大表业务特征，也没有生成完整 dense embedding。
两份数据目录共生成 20 个 artifact，每份目录全部文件合计约 0.88MiB。

artifact 包含数据库路径/大小/mtime、cutoff、依赖版本、样本 ID 和 checksum。
正式模型 API 与 batch benchmark 要求有效 artifact；缺失、过期、损坏均失败。
低层 builder 保留 require_stypes=False 以支持独立小型 fixture；正式入口显式设为 True。

## 4. 验证结果

- 运行包测试：43 项通过，其中新增 10 项覆盖本轮缺陷。
- 离线数据测试：6 项通过。
- Ruff、Pyright、compileall 通过。
- 基线导入检查：batch、vanilla、cache 均不导入 latest。
- 原始/显式 node ID 两目录的 20 个库、128 个任务视图检查通过：三个在线版本
  的 stype projection 与隐藏列一致；batch 使用相同 vanilla schema 契约。
- 四版本固定 fixture：相同 TensorFrame、CPU 单步 loss、optimizer 更新后的
  model state 均逐值一致。
- 真实 F1 推荐 SamplePlan：四版本 60 组 TensorFrame 特征、节点和边顺序一致；
  rtol=0、atol=0 且缺失值位置一致。可重跑 `python -m benchmark.feature_parity`。

GPU 回归使用 A30、batch 512、fanout 128x128、channels 128、1 CPU thread：
F1 推荐任务为完整 6 batch，Salt autocomplete 为前 8 batch。四版本运行入口、
特征隐藏和训练更新均经过实际执行；GPU 结果见文末产物。

| 任务 | Batch loss | Vanilla loss | Cache loss | Latest loss |
| --- | ---: | ---: | ---: | ---: |
| rel-f1 | 0.892001512322 | 0.892027439284 | 0.891995716709 | 0.892046897838 |
| rel-salt | 1.504982978106 | 1.504981309175 | 1.504981815815 | 1.504984661937 |

以上是短回归结果，不作吞吐提升结论。四版本工作量、schema 指纹、executor 和
正确性版本均核对一致。

### 数值一致性边界

固定 CPU fixture 和上述真实 F1 特征输入通过严格对照。独立 GPU 训练进程仍有
末位 loss 差异，本轮没有放宽原来的 `1e-7` 比较容差，也没有将其宣称为已解决。
GPU 原子归约是可能原因，但没有据此认定全部历史漂移来自 GPU；Salt 完整 epoch
的数值一致性仍需独立复验。

本轮没有重跑十库全 epoch 性能矩阵。旧报告的时间、内存、loss 仅代表修复前版本；
特别是目标泄漏修复和 stype 改变后，旧 loss 不能用于评价新模型。

## 5. 运行方式与后续边界

已有数据库先刷新 metadata 再初始化 stype：

```bash
../.venv/bin/python -m data.update_catalog_metadata --database-dir data/relbench
../.venv/bin/python -m data.initialize_stypes --database-dir data/relbench
../.venv/bin/python -m benchmark.online_runner --implementation cache \
  --dataset rel-f1 --task driver-circuit-compete --device cuda:0 \
  --executor async --output benchmarks/cache-corrected.jsonl
```

本地两份目录已完成初始化。后续修改 catalog、迁移 node ID 或移动数据库后需重新
生成 artifact。新实验使用新文件，避免旧成功记录导致 resume 跳过修订后的测试。

动态算法留待下一轮：在此正确性修订版上按表观测重复率、cache reuse、编码成本和
queue 等待，探索窗口大小、缓存准入和编码缓存策略。本轮未加入这些性能变化。

结果文件：

- `benchmarks/correctness-contract-audit.json`
- `benchmarks/correctness-real-feature-parity.json`
- `benchmarks/correctness-{batch,vanilla,cache,latest}.jsonl`（async 回归）
- `benchmarks/correctness-batch-sync.jsonl`（单独保留的 sync 入口验证）
