"""Render the Chinese report from recorded measurements and official references."""

import json
from pathlib import Path


OUT = Path(__file__).resolve().parent
local = json.loads((OUT / "local-stats.json").read_text())
hub = json.loads((OUT / "official-hub.json").read_text())
paper = json.loads((OUT / "paper-reference.json").read_text())
datasets = local["datasets"]
repos = {r["repo"]: r for r in hub["repositories"]}

BACKGROUND = {
    "rel-amazon": "Amazon 电商评论数据，围绕客户、商品和带时间的评分/评论建立关系，可用于流失、消费价值和商品推荐研究。",
    "rel-arxiv": "arXiv 学术出版数据，连接论文、作者、研究类别和引用关系，可用于研究方向、发文量、引用与共同引用预测。",
    "rel-avito": "Avito 分类广告数据，包含用户、广告、搜索以及曝光、点击、访问和电话请求行为，面向广告点击与用户访问预测。",
    "rel-event": "活动推荐数据，连接用户、活动、好友关系、出席记录和兴趣反馈，面向出席次数、重复参加和忽略邀请预测。",
    "rel-f1": "Formula 1 赛车数据，连接比赛、赛道、车手、车队、排位赛、比赛结果与积分排名，面向完赛名次、前三名和退赛预测。",
    "rel-hm": "H&M 时尚零售数据，由客户、商品和带时间的购买交易组成，面向购买推荐、客户流失和商品销售预测。",
    "rel-ratebeer": "RateBeer 啤酒评价数据，连接用户、啤酒、酿酒商、场所和多维评分/评价，面向评分、活跃度与偏好推荐研究。",
    "rel-salt": "SAP SALT 企业销售数据，以订单主表、订单明细、客户和地址为核心，面向工厂、发货点、付款条件等业务字段补全。",
    "rel-stack": "Stack Exchange 问答社区数据，包含用户、帖子、评论、投票、徽章与帖子关联，面向互动、投票和推荐预测。",
    "rel-trial": "ClinicalTrials.gov 临床试验数据，连接研究、结果、不良事件、入组条件、资助方、疾病和研究机构，面向试验结果及相关预测。",
}


def n(value):
    return f"{value:,}"


def size(value):
    if value < 2**20:
        return f"{value / 2**10:.2f} KiB"
    if value < 2**30:
        return f"{value / 2**20:.2f} MiB"
    return f"{value / 2**30:.3f} GiB"


def ratio(count, total):
    return f"{count}/{total} ({100 * count / total:.2f}%)"


lines = []


def add(text=""):
    lines.append(text)


def table(headers, rows):
    add("| " + " | ".join(headers) + " |")
    add("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        add("| " + " | ".join(map(str, row)) + " |")
    add()


for d in datasets:
    repo = repos[d["source_repo"]]
    assert repo["revision"] == d["snapshot"]
    remote = {f["path"]: f["size"] for f in repo["parquet_files"]}
    prefix = d["dataset"] + "/db/"
    assert {p for p in remote if p.startswith(prefix)} == {
        prefix + t["table"] + ".parquet" for t in d["tables"]}
    for t in d["tables"]:
        assert remote[prefix + t["table"] + ".parquet"] == t["parquet_bytes"]

add("# RelBench 十个数据集背景与规模统计")
add()
add("统计日期：2026-09-28。统计对象为本项目 `data/relbench/` 中的十个 SQLite 数据库；所有数据库均以只读方式访问。")
add()
add("## 1. 结论与统计口径")
add()
nt = sum(len(d["tables"]) for d in datasets)
nc = sum(d["columns"] for d in datasets)
nr = sum(d["rows"] for d in datasets)
tx = sum(d["text_count"] for d in datasets)
add(f"本地十库共有 **{nt} 张原始关系表、{n(nc)} 列、{n(nr)} 行**；现有 stype 配置识别出 **{tx} 个文本特征列，占 {100*tx/nc:.2f}%**。完整 SQLite 文件共 **{size(sum(d['sqlite_bytes'] for d in datasets))}**，原始关系表 Parquet 文件共 **{size(sum(d['parquet_bytes'] for d in datasets))}**。")
add()
add("本地 SQLite 每张原始关系表的行数及字段集合，均与本地缓存的官方 Parquet 一致；该缓存版本号与本次在线查询的官方仓库当前版本一致，且全部 Parquet 文件大小逐个匹配。这里验证的是表结构、行数、版本和文件大小，未进行全部单元格逐值比对或文件内容哈希复核。")
add()
add("[RelBench 官网](https://star-project.stanford.edu/relbench/) 当前列出 11 个基准数据库；本项目使用其中公开分发的十个，不含需要授权访问的 rel-mimic。官网首页没有逐表规模、统一存储大小或文本列比例，因此进一步核对官网链接的两篇论文及官方 Hugging Face 仓库。")
add()
add("- **表数**：仅统计 catalog 中 `table_kind=data` 的原始关系表；任务标签表在附录单列，5 张 `_relconnector_*` 管理表不算数据集业务表。")
add("- **列数**：各关系表字段数之和，包含原有主键、外键、时间和目标字段；跨表同名字段分别计数。扣除本项目新增的 `__relconnector_node_id__`；不扣除原有 ID。它不是训练实际输入特征数。")
add("- **行数**：各表 `COUNT(*)` 相加，不进行 join，不去重，也不等同于独立用户数。完整行数包含测试时间之后的数据；截止行数按每表声明的时间列 `<= test_timestamp` 计算，无时间列的表保留全量，有时间列但时间为空的行不进入截止视图。")
add("- **文本列（主要口径）**：现有 `.stypes.json` 中标为 `text_embedded` 的列；占比 = 文本列数 / 原始总列数。该配置使用 PyTorch Frame，在测试截止视图每表最多抽样 1000 行推断，种子为 42；不等于人工语义标注，也不是对全量数据重新推断。它可能包含名称、地址、URL 等列。")
add("- **SQLite TEXT（补充口径）**：SQLite 声明类型为 `TEXT` 的原始列，包括字符串类别、ISO 时间、JSON 等存储编码，不等同于需要文本编码器处理的列。未被 stype 识别的列保留在总列数分母中，并在第 5 节列明。")
add("- **大小**：SQLite 为 `.sqlite` 文件的逻辑字节数，包含关系表、任务标签表、索引和管理表；Parquet 为官方 `db/*.parquet` 压缩文件字节数之和，只含关系表。均不包含训练缓存、embedding、WAL、任务 Parquet 或 stype JSON，也不是运行内存；二者不可直接当作相同内容的压缩率。GiB = 2^30 bytes，MiB = 2^20 bytes。")
add("- **文本占比限制**：以下是列数量占比，不是文本字节占比、非空文本单元格占比或编码耗时占比；未对全文 UTF-8 字节数做扫描求和。")
add()
add("## 2. 十库总览")
add()
table(["数据集", "关系表", "总列", "完整总行", "测试截止总行", "文本列/总列", "SQLite TEXT/总列"], [
    [d["dataset"], len(d["tables"]), n(d["columns"]), n(d["rows"]), n(d["cutoff_rows"]), ratio(d["text_count"], d["columns"]), ratio(d["sql_text_count"], d["columns"])] for d in datasets])
table(["数据集", "SQLite 完整文件", "SQLite 字节数", "关系表 Parquet", "Parquet 字节数", "测试截止时间"], [
    [d["dataset"], size(d["sqlite_bytes"]), n(d["sqlite_bytes"]), size(d["parquet_bytes"]), n(d["parquet_bytes"]), d["cutoff"].replace("T", " ")] for d in datasets])
add("## 3. 与官网及官方论文核对")
add()
assert all(d["cutoff_rows"] == paper["datasets"][d["dataset"]]["rows"] for d in datasets)
assert all(len(d["tables"]) == paper["datasets"][d["dataset"]]["tables"] for d in datasets)
add("**核对结论：十库的表数、测试截止总行数全部与论文精确一致。** 六库总列数也一致；四库不同：Event 为 129/128、RateBeer 为 174/221、Stack 为 50/52、Trial 为 136/140（本地/论文）。本地列数均与当前官方 Parquet 的字段数一致，因此应采用当前版本的实测列数，并保留这四处历史差异。")
add()
add("历史参考为 [RelBench v1 论文 Table 1](https://arxiv.org/html/2407.20060v1#S2.T1) 与 [RelBench v2 论文 Table 1](https://arxiv.org/html/2602.12606v2#S3.T1)。两表均明确按截至测试时间的可用行计数，因此这里使用本地截止行数比较。差值 = 本地 - 论文。")
add()
table(["数据集", "表数 本地/论文", "列数 本地/论文", "本地截止行", "论文截止行", "行数差", "相对差"], [
    [d["dataset"], f"{len(d['tables'])}/{paper['datasets'][d['dataset']]['tables']}", f"{d['columns']}/{paper['datasets'][d['dataset']]['columns']}", n(d["cutoff_rows"]), n(paper["datasets"][d["dataset"]]["rows"]), f"{d['cutoff_rows'] - paper['datasets'][d['dataset']]['rows']:+,}", f"{100*(d['cutoff_rows']/paper['datasets'][d['dataset']]['rows']-1):+.3f}%"] for d in datasets])
add("当前官方发布文件是本地数据版本的直接核对基准；论文是历史规模参考。上述四处列数差异不能由时间截断解释，但也不能据此判定本地下载不完整，因为本地与当前官方文件的表结构、行数和大小已匹配。具体历史变更原因需比较论文当时的数据版本，本文不推定。")
add()
add("官方在线版本及文件列表（本次核对保存于 `official-hub.json`）：")
add()
for repo in hub["repositories"]:
    add(f"- [{repo['repo']}](https://huggingface.co/datasets/{repo['repo']}/tree/{repo['revision']})：`{repo['revision']}`，本次读取 {len(repo['parquet_files'])} 个关系表 Parquet 文件记录。")
add()
add("官方论文未提供与本报告同口径的文本列比例，也未给出这里的 SQLite 大小。因而不能声称这两项已经与论文数值一致；Parquet 文件大小则已通过官方 API 逐文件验证。")
add()
add("## 4. 数据集背景与逐表明细")
add()
for i, d in enumerate(datasets, 1):
    name = d["dataset"]
    add(f"### 4.{i} {name}")
    add()
    add(BACKGROUND[name] + f" [官方数据说明](https://huggingface.co/datasets/{d['source_repo']}/blob/{d['snapshot']}/{name}/README.md)。")
    add()
    add(f"合计：{len(d['tables'])} 表，{n(d['columns'])} 列，{n(d['rows'])} 行；文本列 {ratio(d['text_count'], d['columns'])}；SQLite {size(d['sqlite_bytes'])}，关系表 Parquet {size(d['parquet_bytes'])}。")
    add()
    table(["表名", "原始列数", "完整行数", "截止行数", "文本列数（占比）", "SQLite TEXT 列", "Parquet 大小"], [
        [f"`{t['table']}`", t["columns"], n(t["rows"]), n(t["cutoff_rows"]), f"{len(t['text_columns'])} ({100*len(t['text_columns'])/t['columns']:.2f}%)", len(t["sql_text_columns"]), size(t["parquet_bytes"])] for t in d["tables"]])
    add("文本特征列清单（未列出的表没有 `text_embedded` 列）：")
    add()
    for t in d["tables"]:
        if t["text_columns"]:
            add(f"- `{t['table']}`：" + "、".join(f"`{c}`" for c in t["text_columns"]) + "。")
    add()
add("## 5. stype 缺失列与内部 ID")
add()
add("下列原始列没有出现在现有 stype 配置中，未计入文本列分子；非空行数按 SQLite 全表精确统计。即使全量存在非空值，抽样推断仍可能未给出类型，不能把“未识别”直接解释成该列没有数据。")
add()
table(["数据集", "表", "未识别列", "全量非空行数"], [
    [d["dataset"], t["table"], col, n(t["untyped_nonnull"][col])] for d in datasets for t in d["tables"] for col in t["untyped_columns"]])
table(["数据集", "原始列总数", "SQLite 关系表物理列总数", "新增内部 ID 列数"], [
    [d["dataset"], d["columns"], d["sqlite_columns"], d["sqlite_columns"]-d["columns"]] for d in datasets])
add("## 附录 A. 任务标签表")
add()
add("以下表由预测任务生成，存放在同一 SQLite 文件中，因此占据文件空间，但不计入前文原始关系表的行列总数。列数按实际 SQLite 物理列统计，包含 `__target_row_id__`、`split` 等辅助字段；行数合并训练、验证和测试拆分。管理表未逐项展示。")
add()
table(["数据集", "任务表数", "任务表列数之和", "任务表行数之和", "关系表+任务表物理列数", "关系表+任务表总行数"], [
    [d["dataset"], len(d["tasks"]), sum(t["columns"] for t in d["tasks"]), n(sum(t["rows"] for t in d["tasks"])), d["sqlite_columns"]+sum(t["columns"] for t in d["tasks"]), n(d["rows"]+sum(t["rows"] for t in d["tasks"]))] for d in datasets])
for d in datasets:
    add(f"### {d['dataset']} 任务表")
    add()
    table(["任务标签表", "物理列数", "行数"], [[f"`{t['table']}`", t["columns"], n(t["rows"])] for t in d["tasks"]])
add("## 附录 B. 复现与证据")
add()
add(f"本地统计完成时间（UTC）：`{local['collected_at']}`。官方 API 核对时间（UTC）：`{hub['checked_at']}`。")
add()
add("本地数据库记录的 RelBench 版本为 3.0.1；stype artifact 记录 RelBench 3.0.1、PyTorch Frame 0.3.0、pandas 3.0.5、NumPy 2.5.3。统计前确认 stype 记录的数据库大小及 mtime 与本地文件一致，并在统计后再次确认数据库未改变。")
add()
add("以下文件均位于报告同级的 `dataset-background-20260928/` 目录：")
add()
add("- `local-stats.json`：所有本地逐表结果、文本列名、stype、源仓库、时间范围及精确字节数。")
add("- `rel-*.json`：各数据库独立的同口径统计结果。")
add("- `official-hub.json`：在线查询的官方仓库版本与文件大小记录。")
add("- `paper-reference.json`：从官方论文 Table 1 抄录的比较基准及出处。")
add("- `collect_stats.py`、`verify_official.py`、`render_report.py`：采集、在线核对和报告生成脚本。")
add()
add("在项目目录 `/home/zk/relconnector/relconnector` 执行：")
add()
add("```bash\n.venv/bin/python doc/dataset-background-20260928/collect_stats.py\n.venv/bin/python doc/dataset-background-20260928/verify_official.py\n.venv/bin/python doc/dataset-background-20260928/render_report.py\n```")
add()
add("生成器会检查官方版本、完整文件列表及大小，若官方更新导致与本地快照不符则中止，以免输出过期的“一致”结论。")
add()
report = OUT.parent / "dataset-background-20260928.md"
report.write_text("\n".join(lines))
print(report)
print(f"Verified {nt} relation tables; report has {len(lines)} lines.")
