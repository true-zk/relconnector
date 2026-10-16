# RelConnector：瓶颈驱动的优化复盘

按用户提供旧稿重新组织的 19 页演示稿。未改写原附件。

- `RelConnector-瓶颈驱动优化复盘.pptx`：正文、架构图、流程和表格为 PowerPoint 可编辑对象。
- `RelConnector-瓶颈驱动优化复盘.pdf`：已渲染检查的预览。
- `speaker-notes.md`：逐页讲稿、证据和比较口径，亦嵌入PPT备注。
- `performance-data.json`、`rss-data.csv`：图表数据。
- `assets/`：高分辨率 PNG 图表及 SVG 矢量图；图表可通过脚本再生成。
- `build_slides.py`：生成源码，需要 python-pptx、matplotlib、numpy 和 Noto CJK 字体。当前本地环境：`PYTHONPATH=/tmp/relconnector-pptx-deps python build_slides.py`。

结构：1 封面主线；2 Vanilla问题；3–4 v1优化及收益；5 新瓶颈；6 v2架构；7–13 启发式、配置选择、资源与正确性；14 实验总结；15–17 性能图与证据；18–19 RSS图与内存结论。

耗时图仅含Vanilla、v1、v2；RSS图加入Batch。超时条为明确标注的下界，Ratebeer v1为超时吞吐估算。Event热初始化训练和冷初始化优化分开解释。v2 RSS只有主进程，使用斜线单独标识，不据此计算总内存节省。

采用2026-09-27报告中每个任务最新有效结果，非统一最终代码的配对实验。HM本地缓存为实验选定配置，并未误写为自动决策。历史性能/RSS来源为 `doc/gpu-feature-optimization-experiment.md`；v2来源为 `benchmarks/v2-performance-review-20260927/comparison.json` 指向的原始结果。
