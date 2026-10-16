# 论文图式参考后的第三版

本次实际阅读了以下论文的图页：

- BGL, NSDI 2023，Figure 10–12（PDF 第 11 页）：https://www.usenix.org/system/files/nsdi23-liu-tianfeng.pdf 。参考其规则的小面板、固定颜色编码、直接标注数值及明确的坐标单位。
- Legion, USENIX ATC 2023，Figure 8（PDF 第 10 页）：https://www.usenix.org/system/files/atc23-sun.pdf 。参考其将主要性能指标与辅助资源指标分别呈现，以及明确标记未完成的实验。

新图为自主绘制，并非复制论文图表：仍为两行五列、十个数据集；每个数据集内部上方为耗时细柱，下方为较矮的 RSS 点图，横向版本位置对齐，只有耗时区域占主要面积。共享图例，去掉大标题和双纵轴，减少重复装饰。v2 以暖色突出，灰/蓝区分 Vanilla/v1。

## 测量口径

- 时间是含初始化的端到端秒/batch；各数据集时间轴独立且从零开始。不是纯训练算子时间。
- 所有 RSS 点图统一使用 0–12 GiB 线性范围。保留真实值，不通过修改数据制造“RSS变化小”的结论。
- v2 空心圆表示仅主进程 RSS；不含采样、assembly、manager，因此不能作为总进程树内存下降的依据。
- Vanilla 五项及 v1 Ratebeer 超时均以高柱、截断符号和 TO 标示，截断柱高度仅为状态符号，不是完成时间，也不是数值下界。原始 v1 Ratebeer 部分运行吞吐估计留在 CSV，未当作完成耗时画柱。
- 历史运行与 v2 的 epoch 数、缓存冷热条件不同；Event 为热初始化，Trial 有既有缓存，v2 Ratebeer 截断。不能据图作严格配对实验的因果结论。
- RSS 超时项为截至终止的观测峰值；没有独立重复实验，因此未绘制误差棒。

输入仍为历史性能汇总 performance-data.json / rss-data.csv，未使用内存估算；未修改 PPT。

运行：`python doc/figures/vanilla-v1-v2-20260927/revision-3/plot_comparison.py`
