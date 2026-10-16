# Vanilla / v1 / v2 十库对比图

v1 表示单进程优化版本，v2 表示多进程升级版本；历史原始字段名保留以便追溯。

2 行 × 5 列，每个数据集一个子图。每个子图包含 Time 细柱与 RSS 三个点，两组顺序均为 Vanilla、v1、v2；版本配色在全部子图中一致。Time 使用左轴（端到端秒 / batch，包含准备），RSS 使用右轴（GiB）；均为从零开始的线性轴，RSS 轴十库共用相同范围，Time 轴各库独立。耗时柱高与 RSS 点高度不能直接比较；RSS 标尺保持 0–11.5 GiB，不通过放大坐标范围压缩差异。

Vanilla 五项超时以及 v1 Ratebeer 均画为顶部截断的高柱，并标 TO。截断柱为不按比例的超时状态符号，高度不是实测耗时或数值下界。v1 Ratebeer 原始吞吐估计仍保留在 CSV，但本图不将其视为完成时间。超时 RSS 为截至终止的观测峰值。

v2 RSS 为 RUSAGE_SELF 主进程峰值，空心圆点标识，不含采样 / 组装 / manager，不能当作系统总内存或用此图断言整体内存节省。

耗时使用已有端到端秒 / batch 口径，历史一轮与 v2 两轮、冷热缓存状态不同；Event 是热初始化运行，Trial 使用已存在文本缓存；Ratebeer v2 每轮截断 10000 batch。不是严格同条件配对实验。无误差棒，因为没有独立重复实验。

数据来源：doc/slides/bottleneck-review-20260927/performance-data.json 和 rss-data.csv（历史报告与选定 v2 实测记录），未使用任何内存估算。

输出：PNG 预览/高分辨率，PDF/SVG 矢量，plotted-data.csv 逐项数值，plot_comparison.py 可复现脚本。未修改 PPT。
