# Amazon 磁盘缓存写入回退（2026-09-26）

## 现场

Stack 完整结果通过后，Amazon 于 14:29 启动。前约 8 分钟在进行首次 schema 数值统计，CPU 持续工作，属于初始化，并非死锁。进入训练后，pipeline 约 1,743 秒仅完成 568 batch，近期约 4–6 秒/batch，多个 worker 处于磁盘 I/O 等待。

现场磁盘缓存 WAL 从 55 GiB 继续增长至 60,458,758,752 字节，SQLite 主索引约 1 GiB；文件系统可用空间下降至约 83 GiB。采样栈显示 worker 在 SQLite reservation commit、分片映射及原始 SQL 读取；映射数保持约 6,300，未复发 Stack 映射上限问题。在线 PASSIVE 和 TRUNCATE checkpoint 都返回 busy，原有自动 checkpoint 没有及时回收持续写入的日志。

自动 checkpoint 是 PASSIVE，只推进当时可完成的部分，不阻止并发写入；TRUNCATE 会协调当前读写并在成功后将 WAL 缩至零。依据：[SQLite PRAGMA 文档](https://www.sqlite.org/pragma.html#pragma_wal_checkpoint)。

## 修复和恢复

- 停止本轮 Amazon，保留已编码 embedding；诊断文件位于 `benchmarks/amazon-regression-20260926/`。此中断运行不计入完整矩阵成绩。
- 禁用连接的 PASSIVE 自动 checkpoint，写事务提交后检查 WAL 大小，达到自适应阈值时尝试 TRUNCATE；SQLite 的 busy timeout 继续限制锁等待，若有并发 checkpoint 忙则由后续写入重试。预算是触发阈值，不承诺外部长期持有读锁时的绝对硬上限。
- 每个 writer 将向量连续写入预分配 8,192 行的独占分片，避免每次发布少量向量都新建文件。SQLite 只发布已写完的行；固定分片尺寸让已存在的只读 mmap 能看到后续新增行。保持既有文件格式及命名空间兼容，保留跨进程预约以抑制重复编码。
- 空闲时执行 checkpoint，107.83 秒内将旧 WAL 从 60,458,758,752 字节缩至 0。quick_check 为 ok，保留 5,131,864 条 embedding；确认旧 worker PID 全部退出后，按其 owner 释放 174 条未完成预约。文件系统可用空间回到约 138 GiB。

## 验证

15 项磁盘缓存测试通过（10.22 秒），包括并发预约/发布与 checkpoint、LRU 驱逐、分片追加后旧 reader mmap 的可见性、cache 关闭后的返回值独立性、跨 epoch 不重复模型编码。Ruff 和 Pyright 通过。

真实 Amazon 诊断采用独立空文本缓存，复用 schema/topology 初始化缓存，2 epoch × 100 batch；最终候选结果见下文。随后使用保留的原缓存恢复完整矩阵，应明确属于已有部分磁盘 embedding 的续跑，不能将后测总差异全部归因于代码优化。


## 64 MiB 固定阈值候选被弃用

空文本缓存短测暴露了机械盘上的另一个问题：强制小 WAL checkpoint 频繁刷出稀疏脏页，执行 checkpoint 的 worker 阻塞在磁盘写入，其余三者等待 SQLite 写锁。虽将 WAL 控制在约 64 MiB，但出现多次约 100 秒的无 batch 输出区间；该候选约 20 分钟只完成 164/200 batch，已停止，不作为成功性能结果。

后续候选改为无数据集特判的自适应 checkpoint 预算：以索引文件大小的 4 倍为目标，通常至少 64 MiB，最多 4 GiB，并约束为当前磁盘剩余空间的 1/16（极低空间时最低触发值 1 MiB）。每秒最多更新一次预算。这样大索引允许积累更多修改后再 checkpoint，减少机械盘上的零散刷盘；预算是回收触发阈值，仍需记录 checkpoint busy 与实际 WAL 峰值。

并发测试同时发现首次创建空数据库时 PRAGMA journal_mode=WAL 可能立即返回 SQLITE_BUSY，绕过 SQLite busy timeout。已加入针对 BUSY/LOCKED 的有截止时间重试，其他错误仍原样抛出。16 项缓存测试通过（9.31 秒），Ruff 通过。新候选使用另一份空文本缓存运行相同 2×100 batch 短测，结果如下。


## 自适应候选验证及续跑

独立空文本缓存的 2×100 batch 已完整通过，总耗时 676.0115 秒，四个 worker 均正常收到结束信号退出。实际 WAL 观测峰值 1,448,933,992 字节（约 1.35 GiB），60 次 checkpoint 尝试均未返回 busy，退出后 WAL 为 0。16 项缓存测试、Ruff 和 Pyright 通过。此短测只验证资源回收和执行完成，不代表完整 Amazon 的最终加速比。

短测缓存包含 3,368,981 条 embedding。确认所有写进程退出并核对元数据完全一致后，将引用分片硬链接至主缓存，再事务性合并索引：原有 5,131,864 条，新增 1,453,072 条，合计 6,584,936 条。合并后 checkpoint 成功，记录见 `benchmarks/amazon-regression-20260926/cache-merge.json`。完整矩阵继续使用这份部分热缓存，最终比较必须保留该说明。


## 第二轮读取开销（18:30–18:38 诊断）

第二轮仍持续推进，但高 WAL 时部分区间约 2.9 秒/batch。四个 worker 的 48 次短时栈采样中，30 次位于 SQLite 索引结果迭代，其余多为向量分片映射/取值和共享索引；同期磁盘 I/O pressure 很低，worker CPU 持续工作。该采样不是精确墙钟耗时分解。

18:33 自动 checkpoint 将约 4 GiB WAL 回收后，分钟进度从约 20 多个 batch 回升到约 60 个；因 batch 内容不同，这只是相关证据。进一步使用可自动清理的 `/dev/shm` 独立数据库，固定相同的 4096 个命中查询：空 WAL 稳态约 0.0193 秒，256 MiB WAL 约 0.0219 秒，1 GiB 约 0.0277 秒，2 GiB 约 0.0421–0.0434 秒；checkpoint 后回到约 0.0195 秒。此隔离实验不写入或 checkpoint 当前 benchmark 缓存。原始记录 `benchmarks/amazon-regression-20260926/wal-read-cost.json`；真实缓存只读页缓存大小探针见 `index-read-probes.jsonl`（2 MiB 与 64 MiB 在当时 WAL 约 306 MB 时差异较小）。

SQLite 官方也说明读取性能随 WAL 增大而下降，checkpoint 频度需要在读写间取舍：[WAL Performance Considerations](https://www.sqlite.org/wal.html)。当前索引大小/空闲磁盘预算解决了大量写入的频繁刷盘问题，但未区分后续读取为主的阶段；下一候选应基于近期读写比例调整 WAL 预算，并验证重新进入大量未命中文本时能够恢复写入预算，避免复发固定小阈值造成的长停顿。当前完整 Amazon 进程仍使用原已验证策略，未热修改代码或强制 checkpoint。


## 读取阶段候选（18:44）

已加入可逆的读写阶段判断，无数据集特判。每 30 秒评估近期磁盘请求数、命中率及共享 WAL 正向增长；连续两个窗口各至少 4,096 个请求、命中率至少 99%、WAL 增长不超过 4 MiB/s 时，预算上限降至 512 MiB，仍受索引大小和空闲磁盘原预算约束。命中率降到 95% 以下（有足够样本）或共享 WAL 增长超过 8 MiB/s 时，恢复原写入预算；因此某 worker 高命中但其他 worker 大量写入时也不会长期保留较小预算。全命中读取路径每秒最多检查一次 checkpoint，解决不再写入时旧 WAL 一直保留的问题。

19 项磁盘缓存测试通过（9.56 秒），包括读写阶段进入/退出、跨进程写入增长、低样本保护、WAL 回收后重新观察、无新写入时读取触发回收，以及原并发预约/发布和跨 epoch 复用测试。Ruff、指定项目解释器的 Pyright 通过。候选不热修改当前已加载的 Amazon worker；完整 Amazon 成绩将保留旧预算策略，随后新启动的任务使用新候选。此处尚无新策略的真实完整性能结果，不将单元测试或隔离 WAL 实验等同于端到端加速验证。


## 完整 Amazon 结果与诊断收集修复（19:58）

Amazon 两轮训练成功完成：每轮 4,954 batch、2,536,014 样本，合计 9,908 batch、5,072,028 样本，总耗时 14,699.6339 秒，1.4836 秒/batch；历史 v1 为 13,898.2 秒/4,954 batch，按相同工作量折算加速 1.891×。两轮训练阶段 6,306.80/8,277.52 秒，平均 loss 0.512911/0.507733。第二轮较慢仍与索引读取开销一致，新增读写阶段 WAL 策略未热加载进本轮 Amazon。

该次结果的 assembly worker 诊断全部缺失。核对收尾代码发现：主进程只等待 10 秒收集报告，超时后不报错，随后逐个 join、terminate；大缓存的诊断可能涉及全索引计数和分片文件统计。因此保留训练步数、损失、耗时，但模型编码数/磁盘命中写入计数标记 unknown，不能解释为 0；缺失报告的具体内部阻塞点没有退出时栈证据。

已修复为等待全部 worker 报告，默认期限 300 秒；到期明确报错并列出缺失 worker，不再静默输出不完整报告。等待时输出 collecting_diagnostics 阶段进度。9 项流水线测试通过（79.72 秒），包含超过旧 10 秒期限的延迟报告、超时明确失败、空 epoch、错误传递与顺序约束；Ruff 和指定解释器的 Pyright 通过。

Ratebeer 自动启动后仍在首次 schema 初始化，尚无训练 batch；停止该次初始化后从第 10 项恢复，应用诊断修复及读写阶段 WAL 策略。完整矩阵的比较脚本已将缺失 worker 指标显示为 unknown。9 项完成子集总耗时 30,797.58 秒，对历史 v1 同工作量 60,391.20 秒，约 1.961×；仍待 Ratebeer，不能作为十项最终结论。
