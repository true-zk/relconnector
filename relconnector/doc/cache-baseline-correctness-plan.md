# cache_baseline 冻结与正确性修复计划

本轮先冻结当前 latest 为独立 cache_baseline，再向四版本回补正确性问题。
用户已授权修复旧基线；动态策略算法、性能并行化留待后续。

1. 修复前源码及 SHA256 清单存入 baseline/archives，复制 latest 并改为独立导入。
2. 修复 autocomplete 隐藏目标列：兼顾官方 remove_columns，保留 forecast 合法特征；
   修复离线更新工具和已有 catalog。
3. 固定种子、表顺序、输入视图、依赖版本，以有界 SQL 抽样固化官方 stype proposal。
   四版本共享 artifact；官方小数据逐表对照；陈旧 artifact 必须失败；不全量读大库。
4. 修复 vanilla rowid 索引、默认路径、batch benchmark 导入和版本类型混用；
   核查缓存 storage 所有权、字节预算和数据一致性。
5. 单元测试、四版本固定输入/CPU 单步更新对照、真实 schema 检查、GPU 短回归和静态检查。

历史 JSONL 不改写，新实验标记正确性修订版本。cache_baseline 保留本轮之前的
缓存和窗口策略，仅回补已证实 bug。Salt 浮点漂移须明确验证边界，不能宣称无证据修复。
