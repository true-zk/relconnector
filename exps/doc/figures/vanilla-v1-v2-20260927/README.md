# Vanilla / v1 / v2 十库对比图

v1 表示单进程优化版本，v2 表示多进程升级版本；历史原始字段名保留以便追溯。

2 行 × 5 列，每个数据集一个子图。每个子图包含 Time 与 RSS 两组柱，每组顺序为 Vanilla、v1、v2；版本配色在全部子图中一致。Time 使用左轴（端到端秒 / batch，包含准备），RSS 使用右轴（GiB）；均为从零开始的线性轴，RSS 轴十库共用相同范围，Time 轴各库独立。不同指标柱高不能直接比较。

Vanilla 五项超时显示 TO，不用 14400 秒或未经测量的完成耗时代替。v1 Ratebeer 为超时前吞吐估计，以 ~ 与点纹柱标识。超时 RSS 为截至终止的观测峰值。

v2 RSS 为 RUSAGE_SELF 主进程峰值，斜线标识，不含采样 / 组装 / manager，不能当作系统总内存或用此图断言整体内存节省。

耗时使用已有端到端秒 / batch 口径，历史一轮与 v2 两轮、冷热缓存状态不同；Event 是热初始化运行，Trial 使用已存在文本缓存；Ratebeer v2 每轮截断 10000 batch。不是严格同条件配对实验。无误差棒，因为没有独立重复实验。

数据来源：doc/slides/bottleneck-review-20260927/performance-data.json 和 rss-data.csv（历史报告与选定 v2 实测记录），未使用任何内存估算。

输出：PNG 预览/高分辨率，PDF/SVG 矢量，plotted-data.csv 逐项数值，plot_comparison.py 可复现脚本。未修改 PPT。
