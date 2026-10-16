# v2 五组消融：F1 GPU smoke 实测

2026-09-28，RTX 4090 / torch 2.7.1+cu128。五组全部完成，缓存开关断言通过；严格 loss 一致性未通过 1e-6 检查。

2 epochs × 2 batches，batch size 16，fanout 4,4，channels 16，seed 42，strict。每组独立空应用缓存，OS 页缓存未控制。

| 配置 | 批数 | 样本数 | mean loss | benchmark 秒 | 实际文本编码输入数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 4 | 64 | 0.456542193890 | 22.95 | 515 |
| no_feature_cache | 4 | 64 | 0.456536628306 | 16.09 | 515 |
| no_reuse_cache | 4 | 64 | 0.456536673009 | 23.33 | 678 |
| no_text_cache | 4 | 64 | 0.456542149186 | 22.25 | 575 |
| one_worker | 4 | 64 | 0.456536628306 | 22.87 | 515 |

平均 loss 的最大差异为 5.57e-06，超过初步检查阈值 1e-6。不能将其直接归为舍入误差，也不据此认定训练质量退化；需要逐 batch 特征/采样/参数轨迹复核。未事后放宽阈值来宣称等价。

已断言：所有组 worker diagnostics 齐全；F-off 的 raw/encoded 写入与保留字节为零；T-off 的 shared 为空、disk 未启用、文本缓存命中及 local 向量保留字节为零。

局限：只有四批，四个 assembly workers 未必均有工作；不能据此评价多 worker 扩展性。disk 在所有组均未激活，raw 缓存也无写入，不能验证它们的性能贡献。一次短跑，初始化/进程启动占比高，时间仅作执行记录，不形成加速结论。

原始结果：`benchmarks/ablation-20260928-f1-smoke/`，含 manifest、execution.jsonl、每组 result.jsonl/run.log，以及 verified-summary.json。
