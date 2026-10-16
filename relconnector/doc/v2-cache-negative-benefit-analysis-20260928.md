# v2 缓存负收益归因：Stack / Amazon / Trial

分析时间：2026-09-28。本报告使用 all50-r2 的原始 worker 遥测和该轮 source/ 代码快照。读取时九个数据集完成，Event 尚在运行。未改动训练实现，也未并行启动新的性能测试。

## 结论

负收益确实发生在已测短程负载，且有可定位的实现原因。Trial/Stack 的主要额外成本来自低收益的 disk 文本缓存；Amazon 完整组 disk 未激活，主要问题是共享文本缓存访问、容量饱和及未激活 disk 的持续预热维护。普通 encoded 缓存对低复用/低编码成本表的维护还会叠加开销。不能统称为缓存尚未预热，也不能将所有缓存一并判为无效。

## 端到端与流水线口径

| 数据集 | 完整 pipeline 秒 | 关 F | 关 T | 关 F/T | 完整 benchmark 秒 | 关 F/T benchmark 秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rel-stack | 217.61 | 215.56 | 70.59 | 67.52 | 324.36 | 177.13 |
| rel-trial | 72.05 | 74.28 | 34.17 | 33.01 | 181.76 | 138.13 |
| rel-amazon | 127.25 | 340.19 | 112.34 | 112.08 | 503.79 | 585.55 |

F=raw+encoded，T=内存文本+disk。pipeline 包含启动/等待/收尾，不是纯稳态时间。Amazon 的“关缓存更快”适用于 pipeline；该次 benchmark 总时间反而是完整组更快。Amazon 初始化差为约 473.47 秒（双关）与 376.54 秒（完整），OS 缓存及执行顺序影响尚未隔离。不能混用这两种口径。

## 缓存命中与真正覆盖的工作

| 数据集 | raw 命中 | encoded 行命中率 | shared 文本命中率 | disk 活跃 worker | disk 命中 / 写入 |
| --- | ---: | ---: | ---: | ---: | ---: |
| rel-stack | 0 | 0.581% | 19.03% | 4 | 77,243 / 874,923 |
| rel-trial | 0 | 0.000% | 21.04% | 4 | 0 / 213,876 |
| rel-amazon | 0 | 15.804% | 15.38% | 0 | 0 / 0 |

Encoded 和 shared 的全局计数取已返回的最末进度快照（请求数最大），不跨 worker 重复求和。Disk 的 hits/misses/writes 是各实例本地计数，按 worker 合并；entries/size 是共享文件状态，不能直接求和。

## 操作计时证据

**下表是各 worker 累计经过时间，包含等待，不是 CPU 利用时间，也不是端到端墙钟分解。checkpoint 被包含在外层 disk 操作内，不应重复相加。**

| 数据集/配置 | 文本模型 | shared lookup | disk lookup | disk publish | disk observe | encoded lookup+put | checkpoint（包含项） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rel-stack/full | 168.82 | 23.50 | 216.75 | 341.39 | 2.50 | 9.60 | 101.75 |
| rel-stack/no_text_cache | 186.89 | 0.00 | 0.05 | 0.00 | 0.08 | 9.51 | 0.00 |
| rel-stack/no_reuse_cache | 187.25 | 0.00 | 0.05 | 0.00 | 0.07 | 0.33 | 0.00 |
| rel-trial/full | 55.26 | 2.50 | 52.83 | 96.23 | 1.29 | 5.49 | 28.22 |
| rel-trial/no_text_cache | 55.94 | 0.00 | 0.02 | 0.00 | 0.02 | 5.12 | 0.00 |
| rel-trial/no_reuse_cache | 54.84 | 0.00 | 0.02 | 0.00 | 0.02 | 0.18 | 0.00 |
| rel-amazon/full | 311.81 | 45.79 | 0.72 | 0.00 | 20.41 | 12.58 | 0.00 |
| rel-amazon/no_text_cache | 332.56 | 0.00 | 0.05 | 0.00 | 0.08 | 11.59 | 0.00 |
| rel-amazon/no_reuse_cache | 347.33 | 0.00 | 0.06 | 0.00 | 0.09 | 0.47 | 0.00 |

### Trial：付出大量持久化成本，没有获得磁盘复用

- 完整组 disk 写入 213,876 条，命中 0；raw 和 encoded 也都是零命中。
- 关文本缓存后模型累计耗时约 55.94 秒，完整组约 55.26 秒；尽管编码输入数有所降低，省下的模型时间很小。重复项可能偏向便宜文本，当前没有逐文本计算成本遥测，不能直接将其断言为空字符串或某个指定字段。
- 完整组额外 disk lookup/publish 累计约 149.06 worker 秒，checkpoint 约 28.22 秒（已包含）。
- 完整与双关 pipeline 差约 39.05 秒，ready 等待差约 38.51 秒；变化主要位于数据供给等待。

### Stack：低复用、缓存饱和，disk 管理成本明显超过编码节省

- encoded 命中约 0.58%；昂贵表 postHistory 仅约 0.026%。raw 不命中。
- shared 已存满 447,392 条向量，出现约 57.4 万次探测失败/满载拒绝；disk 命中约 8.41%，写入约 87.5 万条。
- 完整组 disk lookup/publish 累计约 558.14 worker 秒，其中 checkpoint 累计约 101.75 秒。相对 no_text_cache，模型时间仅少约 18.07 worker 秒。
- 关闭普通缓存仍为约 215.56 秒，与完整组约 217.61 秒接近；关闭文本缓存降至约 70.59 秒，双关为约 67.52 秒。主因落在文本缓存组。
- 完整与双关 pipeline 差约 150.09 秒，ready 等待差约 149.61 秒，供给路径解释与总时间一致。

### Amazon：完整组没有 disk IO，仍存在昂贵的 RAM lookup 和预热维护

- 完整组四个 worker 的 disk active 都为 false，不能把完整组负收益归因于磁盘写入。
- shared 命中约 15.38%，容量 447,392 已满，full_rejections 约 157.7 万。局部快速索引没有激活。
- shared lookup 累计 45.79 worker 秒；disk observe 另有 20.41 worker 秒。代码显示后者在 disk 尚未激活时仍逐条 clone 向量并维护最多 8192 条的 warmup OrderedDict。
- 完整组相对 no_text_cache 节省模型累计时间约 20.75 秒，低于已观测的 shared lookup 与 observe 成本。shared 发布/等待并没有全部单独计时，不能将剩余差额强行精确配平。
- encoded 全局命中率约 15.80%，但 customer 命中约 34.75%，product 约 1.39%，review 约 0.267%；按行命中率掩盖了昂贵 review 转换仍然大量发生。
- no_text_cache 与双关 pipeline 分别约 112.34/112.08 秒，本次普通缓存没有可辨识的明显净收益。
- no_feature_cache 却触发了全部四个 worker 的 disk（约 180.5 万写入），pipeline 达 340.19 秒。该消融改变了进入文本层的请求分布和启停条件，不能将 340→127 的差额全部解释为 encoded 命中所节省的计算。

## 实现上的根因

1. **Disk 决策把低 RAM 命中误当成持久化收益证据。** 当前使用数据库大小≥256 MiB、编码数≥4096、编码时间≥1 秒、累计 RAM 命中≤10% 触发；但低命中也可能意味着近乎一次性流式访问，不能推导未来 disk 高命中。没有成本收益比较、训练剩余长度估计或激活后负收益回退。
2. **启用之后的写入在供给关键路径上。** lookup 的 miss 会 BEGIN IMMEDIATE 并登记 reservations；publish 再事务更新索引与删除 reservations；还可能触发同步 TRUNCATE checkpoint。多个 worker 共享一个 SQLite 文件，WAL 并不使这些写事务并行。记录中的 disk lookup 包含 miss 登记、锁等待与 checkpoint，不能简单理解为读取向量耗时。
3. **两层做重复的缺失协调。** 文本 miss 先在 shared 做 reservation，之后 `_resolve_missing` 默认仍 coordinated=True，进入 disk reservation；在共享层已拥有保留槽位的键上，存在冗余协调。共享层溢出、其他运行/进程等场景仍需保证去重，不能一刀切删掉 disk 协调。
4. **共享文本缓存饱和后缺少向量淘汰/替换。** 发布向量在当前实现中不可变，满载后继续哈希、加锁、探测至 max_probe=32，随后 uncached 编码。adaptive local index 只减少高热查找，不改变全局向量缓存准入与淘汰。
5. **Encoded adaptive 主要调节准入，不会自动绕过低收益查找。** 即使 Trial 全零命中，仍经 Manager RPC、node id 转换、准入计数与部分写入。查找又在 raw SQL fetch 之后，命中主要避免转换而不是 SQL 读取。
6. **实验缓存位于机械盘。** 本机 /home 对应 sda1，块设备 ROTA=1；checkpoint 和 SQLite 索引写入对该存储敏感。这是环境放大因素，尚未做 SSD 对照，不能宣称换盘就一定解决。

## 为什么不能只归咎于 50 batches

- 每组新建应用缓存且不跨运行复用，确实衡量了首次构建成本，不能据此否定长期/跨运行持久复用。
- 每个 epoch 重新打乱 seed blocks，再截取前 50 batches；第二轮不是重复第一轮的同一批请求。因此不能把第二轮自然视为相同工作集的稳态。
- 但 Trial 在已测两轮 disk 命中为零，且缓存控制器没有按该事实回退，是这次真实发生的负收益；延长 epoch 不能代替修复控制策略。
- 所有数据集使用同一 order_seed=1729，导致相同变体顺序（双关最先，full 最后）；没有重复运行，OS 页缓存、冷热和顺序效应未消除。小于数个百分点的差异不应下结论，2–3 倍回退则有独立计时证据支持。
- 关 F/关 T/双关仍保留窗口内去重、密度感知读取、流水线和 OS 页缓存，不是逐 ID 读库或完全无复用的 Vanilla。

## 下一步，按验证信息量排序

1. **分离 RAM 与 disk。** Stack/Trial/Arxiv 对 full、no_disk_text_cache、no_text_cache 做相同配置重复对照；先验证关 disk 保留 RAM 能否恢复性能。不将旧单次数据与新代码单次数据直接作为严格配对。
2. **Amazon 单独隔离未激活观察成本。** full vs no_disk_text_cache：两者在旧 full 的实际 active=false 条件下主要隔离 warmup/存在性观察；再比 no_text_cache 分离共享 lookup。若重复运行 disk 激活行为变化，必须重新按实际状态解释。
3. **试运行式启停。** 按表/列估计被避免的编码时间，减去 lookup、锁/IPC、复制和写入成本；负收益时绕过/停止准入，并允许低频重新探测。不能仅依据缓存命中率或总文本数。
4. **Disk 写入策略。** 仅对预计复用的昂贵文本持久化、分批/后台写入、避免双重 reservation；采用有界写缓冲和落后时限流，避免将开销转成无界内存。激活后监测 disk 自身命中与净收益，必要时停止继续写入。
5. **固定策略归因 + 自适应策略效果分别报告。** 做 F 的独立贡献时固定文本策略；另外保留当前自适应矩阵展示实际策略交互。
6. **重复与长程。** 随机化配对顺序、至少 3 次重复；独立报告冷构建、同 run 后续 epoch 和进程重启后的持久复用。不要预设预热后一定更快。

## 证据文件

- 原始结果：`benchmarks/ablation-20260928-all50-r2/rel-*/r1-*/result.jsonl`。
- 提取指标：`benchmarks/ablation-20260928-all50-r2/cache-diagnosis/metrics.json`。
- 对应实现：该实验 `source/features/disk_text.py`、`shared_text.py`、`shared_encoded.py`、`text.py`、`process_worker.py`、`task/seeds.py`。
