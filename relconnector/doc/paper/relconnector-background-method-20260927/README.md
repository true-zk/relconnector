# RelConnector 论文片段

本目录给出背景、动机与方法的中文初稿，不替代完整实验及相关工作章节。

- `RelConnector-背景动机与方法.pdf`：阅读版本。
- `RelConnector-背景动机与方法.docx`：可编辑版本，含引用链接和六张插图。
- `manuscript.md`：正文源文件。
- `research-and-evidence.md`：文献研读、实测数据与当前代码的映射；包含不可直接宣称的结论。
- `figures/`：六张图，每张包含 PNG、PDF、SVG；后两者为矢量格式。
- `figure-data.csv`：61条实际绘图数值，附单位和报告来源。
- `build_figures.py`、`build_document.py`：绘图和排版脚本。

正文七章：关系学习背景；容量与吞吐动机；整体架构；离线初始化；采样计划与需求复用；自适应特征组装；方法总结与实验边界。

绘图依赖 Python、NumPy、Matplotlib；Word 导出依赖 python-docx，中文字体为 Noto Sans CJK SC。运行两个 build 脚本后，可用 LibreOffice 将 Word 导出为 PDF。本次使用的辅助依赖位于 `/tmp/relconnector-paper-deps`，不修改项目运行环境。

数据全部取自已有报告，未重跑训练。历史 v1 在正文统一命名为 v1。论文背景依据原始论文和作者实现；ReDeLEx 加载路径固定到提交 `3d1b392d90de6f6bf8acbac2d539d6fbaf3f32fa`。冷/热初始化、历史诊断与当前结果、多进程服务时间与墙钟时间均分开说明。
