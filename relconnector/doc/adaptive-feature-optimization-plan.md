# 动态特征生产优化实施计划

## 范围与约束

以修正后的 cache_baseline 为对照，仅修改 latest、benchmark 和测试。
保持官方 stype、任务隐藏列、GloVe 300 维语义、采样 RNG 和训练 FIFO。
实现观测、初始化复用、编码缓存、文本执行和动态流水线；不按数据集名称硬编码，
不持久化完整特征或预采样数据。用户已授权按瓶颈分析报告施工。

## 实施顺序

1. 独立轻量观测模块：按表/列线程安全累计，支持关闭。SQL execute/decode/merge、
   prepare/converter/gather、text lookup/tokenize/pool 分开记录。周期 progress
   包含 operation/cache/policy/queue，timeout 保留。
2. 原子保存 schema statistics 和纯拓扑。key 包括数据库版本、cutoff、stype
   artifact、hidden columns、依赖/算法版本、影响统计的配置。失效重算；缓存命中
   不得绕过官方 stype artifact 的正确性检查。
3. 有界 CPU 编码特征缓存：以 table/node ID 索引不可变 TensorFrame 小段，仅编码
   miss，gather 恢复重复节点/推荐分支。存已取到的稀疏行，避免全量物化大 block。
   计入实际持有 storage 和索引成本，准入跟踪表有界，支持关闭/固定 LRU 消融。
4. 文本执行优化：复用 GloVe 已有词向量表及官方 tokenizer，不重复缓存整个词表。
   连续向量 slab、二次访问准入、可切换直接 token lookup/pooling；空文本/OOV/
   重复/标点/大小写/长文本与官方对照。模型结构不兼容时回退，绝不修改 stype。
5. 动态策略：按表 EWMA 成本/重复率/缓存复用反馈，有界探索、冷却和滞回；控制
   编码缓存准入、去重方式、窗口大小及 SQL sparse/dense 选择。用户硬字节预算
   优先，可配置 static/adaptive。不把低复用时扩大缓存作为默认答案。
6. fetch/encode 分离，支持 1/2 encode worker。使用有界输入/按序输出通道，限制
   在途窗口；禁止无限收集 Future，避免首批慢导致乱序内存膨胀。独立 converter，
   缓存并发保护，采样/训练 RNG 保留原契约。
7. benchmark 增加独立开关、cache 基线配置适配、冷/暖启动和消融对照。旧历史性能
   结果不作新版本加速比的分母。

## 验证与完成口径

- 既有 49 项测试；新增缓存失效/storage、命中缺失混合顺序、多分支、空节点、
  策略改变后数值不变、乱序/取消/超大项/字节边界测试。
- 固定 SamplePlan/TensorFrame/CPU 更新对照；GPU loss 保留既有容差与漂移说明。
- 正常 batch/fanout 短回归及消融，代表性 workload：Ratebeer、Amazon、Stack、
  Salt、Avito、Event。区分准备/训练 wall time 与 worker 服务时间。
- 有收益后执行 Ratebeer 完整 epoch（4h 上限）；观测开销<2%、RSS<32GiB 且保留
  系统余量、Ratebeer<4h 是实验目标，未达标必须如实报告，退化策略调回合适默认。
- 多 sampler/GPU kernel 仅在新遥测显示关键路径转移后推进，不能改变采样语义。
