# 十库参考与当前v2结果

上方十库为历史端到端秒/batch及历史RSS；底部单独展示最新Ratebeer与Stack配对流水线ms/batch。不同口径不替换、不直接相除。v2 RSS仅主进程，TO柱是状态符号。

`plot_comparison.py`调用统一图表生成器；`_historical_plot.py`仅提供历史图布局，不作为当前交付的独立重建入口。精确新结果来自`doc/targeted-batch-results-20260927.json`，历史数值在`plotted-data.csv`中。
