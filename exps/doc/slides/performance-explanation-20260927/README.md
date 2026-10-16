# 优化复盘：方法与当前 v2 结果整合版

当前入口：`RelConnector-优化复盘-按数据类型解释.pptx` 及同名 PDF，24页（20页主线＋4页架构附录）。保持原交付路径，替代此前30页的追加式组织。

主线：Vanilla问题 → v1单进程复用 → assemble瓶颈 → v2三阶段分进程 → 离线初始化 → 采样需求 → 文本/编码/持久化缓存 → 资源与正确性 → 实验与内存边界。

- 第10页：HM请求行投影与Ratebeer本地缓存选择合为“编码缓存的传输粒度与位置”。
- 第11–12页：持久化文本复用与Stack离线分片整理前后相接，解释同一条缓存访问路径。
- 第14页：行选择保留、投影融合及SQL前移归档，以端到端证据说明取舍。
- 第17页：最新配对图，Ratebeer流水线1104.6→980.2 ms/batch（−11.26%），Stack428.6→370.6（−13.52%）。
- 第18页：十库历史参考图＋当前定向结果条带，明确两种时间口径；未捏造新的十库完整成绩。
- 第19页：RSS点图；没有新作业内存数据，保留历史记录和v2仅主进程的限制。

不再保留重复的追加优化页、两组重复性能图或以主进程RSS概括整个作业内存的均值页。四种完整架构图在附录保留。

重建：`PYTHONPATH=/tmp/relconnector-paper-deps:/tmp/relconnector-slides-deps python build_update.py`，随后LibreOffice导出同名PDF。`build_integrated_deck.py`负责章节整合；`build_integrated_figures.py`从统一数据源生成当前配对图、十库参考图和RSS图。`update_targeted_optimizations.py`保留为重建兼容入口，不再追加两页。

精确数值：`../../targeted-batch-results-20260927.json`。页序及证据：`integrated-manifest.json`。当前预览：`preview/integrated-*.png`，旧page/type/targeted预览对应历史页序。

每配置两次重复；Ratebeer64批、Stack200批。流水线平均值包含内部启动、等待与排空，非独立batch延迟；Stack离线整理144.43秒另计。此处没有运行新训练，也没有将短测比例外推到完整矩阵。
