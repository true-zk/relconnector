# RelConnector 背景、动机与方法：整合修订版

阅读：`RelConnector-背景动机与方法-修订版.pdf`；编辑：同名DOCX或`manuscript.md`。保持原交付路径。

本轮将新增优化并回原方法：第6.4节统一解释HM与Ratebeer的编码缓存传输/位置，第6.5节统一解释持久化文本向量与Stack分片布局，第6.6节解释特征复制的有效改动和负结果。当前定向验证移到第7.1节，不再作为方法章节末尾的追加说明。公式与交叉引用已重排。

文档含14个Word原生公式、11张表、3张图。图3用配对点图同时呈现流水线平均、就绪等待和整次运行摊销；新流水线结果为Ratebeer1104.6→980.2 ms/batch（−11.26%），Stack428.6→370.6（−13.52%）。每配置两次，64/200批；Stack离线整理144.43秒另计。未替换十库历史完整结果，未生成新RSS结论。

- 证据索引：`research-and-evidence.md`。
- 原有数据：`evidence-data.csv`；当前精确结果：`targeted-batch-results.json`；图3数据：`figures/fig03-v2-current-per-batch.csv`。
- 图形：`figures/`中PNG/PDF/SVG。
- 重建：依次运行`build_figures.py`、`build_document.py`，再用LibreOffice将DOCX导出PDF。
- 依赖：`PYTHONPATH=/tmp/relconnector-paper-deps:/tmp/relconnector-slides-deps`。
- 图3与PPT共用`doc/targeted-batch-results-20260927.json`，不手填新的十库估计数。

ZIP包含本目录的正文、成品、图表及重建脚本；PPT共用生成器另外收录于`shared-build/`，在完整项目中按其原路径运行。
