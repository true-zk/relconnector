# RelBench 十个数据集背景与规模统计

统计日期：2026-09-28。统计对象为本项目 `data/relbench/` 中的十个 SQLite 数据库；所有数据库均以只读方式访问。

## 1. 结论与统计口径

本地十库共有 **73 张原始关系表、702 列、145,333,317 行**；现有 stype 配置识别出 **174 个文本特征列，占 24.79%**。完整 SQLite 文件共 **32.704 GiB**，原始关系表 Parquet 文件共 **11.905 GiB**。

本地 SQLite 每张原始关系表的行数及字段集合，均与本地缓存的官方 Parquet 一致；该缓存版本号与本次在线查询的官方仓库当前版本一致，且全部 Parquet 文件大小逐个匹配。这里验证的是表结构、行数、版本和文件大小，未进行全部单元格逐值比对或文件内容哈希复核。

[RelBench 官网](https://star-project.stanford.edu/relbench/) 当前列出 11 个基准数据库；本项目使用其中公开分发的十个，不含需要授权访问的 rel-mimic。官网首页没有逐表规模、统一存储大小或文本列比例，因此进一步核对官网链接的两篇论文及官方 Hugging Face 仓库。

- **表数**：仅统计 catalog 中 `table_kind=data` 的原始关系表；任务标签表在附录单列，5 张 `_relconnector_*` 管理表不算数据集业务表。
- **列数**：各关系表字段数之和，包含原有主键、外键、时间和目标字段；跨表同名字段分别计数。扣除本项目新增的 `__relconnector_node_id__`；不扣除原有 ID。它不是训练实际输入特征数。
- **行数**：各表 `COUNT(*)` 相加，不进行 join，不去重，也不等同于独立用户数。完整行数包含测试时间之后的数据；截止行数按每表声明的时间列 `<= test_timestamp` 计算，无时间列的表保留全量，有时间列但时间为空的行不进入截止视图。
- **文本列（主要口径）**：现有 `.stypes.json` 中标为 `text_embedded` 的列；占比 = 文本列数 / 原始总列数。该配置使用 PyTorch Frame，在测试截止视图每表最多抽样 1000 行推断，种子为 42；不等于人工语义标注，也不是对全量数据重新推断。它可能包含名称、地址、URL 等列。
- **SQLite TEXT（补充口径）**：SQLite 声明类型为 `TEXT` 的原始列，包括字符串类别、ISO 时间、JSON 等存储编码，不等同于需要文本编码器处理的列。未被 stype 识别的列保留在总列数分母中，并在第 5 节列明。
- **大小**：SQLite 为 `.sqlite` 文件的逻辑字节数，包含关系表、任务标签表、索引和管理表；Parquet 为官方 `db/*.parquet` 压缩文件字节数之和，只含关系表。均不包含训练缓存、embedding、WAL、任务 Parquet 或 stype JSON，也不是运行内存；二者不可直接当作相同内容的压缩率。GiB = 2^30 bytes，MiB = 2^20 bytes。
- **文本占比限制**：以下是列数量占比，不是文本字节占比、非空文本单元格占比或编码耗时占比；未对全文 UTF-8 字节数做扫描求和。

## 2. 十库总览

| 数据集 | 关系表 | 总列 | 完整总行 | 测试截止总行 | 文本列/总列 | SQLite TEXT/总列 |
| --- | --- | --- | --- | --- | --- | --- |
| rel-amazon | 3 | 15 | 23,218,245 | 15,000,713 | 6/15 (40.00%) | 8/15 (53.33%) |
| rel-arxiv | 6 | 21 | 2,733,846 | 2,146,112 | 6/21 (28.57%) | 10/21 (47.62%) |
| rel-avito | 8 | 42 | 24,653,915 | 20,679,117 | 2/42 (4.76%) | 6/42 (14.29%) |
| rel-event | 5 | 129 | 44,822,992 | 41,328,337 | 6/129 (4.65%) | 12/129 (9.30%) |
| rel-f1 | 9 | 67 | 97,606 | 74,063 | 13/67 (19.40%) | 21/67 (31.34%) |
| rel-hm | 3 | 37 | 16,931,173 | 16,664,809 | 12/37 (32.43%) | 18/37 (48.65%) |
| rel-ratebeer | 13 | 174 | 16,874,623 | 13,787,005 | 55/174 (31.61%) | 66/174 (37.93%) |
| rel-salt | 4 | 31 | 4,748,942 | 4,257,145 | 8/31 (25.81%) | 13/31 (41.94%) |
| rel-stack | 7 | 50 | 5,399,818 | 4,247,264 | 15/50 (30.00%) | 25/50 (50.00%) |
| rel-trial | 15 | 136 | 5,852,157 | 5,434,924 | 51/136 (37.50%) | 94/136 (69.12%) |

| 数据集 | SQLite 完整文件 | SQLite 字节数 | 关系表 Parquet | Parquet 字节数 | 测试截止时间 |
| --- | --- | --- | --- | --- | --- |
| rel-amazon | 15.821 GiB | 16,988,098,560 | 6.924 GiB | 7,435,026,130 | 2016-01-01 00:00:00 |
| rel-arxiv | 613.87 MiB | 643,686,400 | 165.16 MiB | 173,186,055 | 2023-01-01 00:00:00 |
| rel-avito | 1.399 GiB | 1,501,802,496 | 404.56 MiB | 424,210,227 | 2015-05-14 00:00:00 |
| rel-event | 1.394 GiB | 1,496,756,224 | 113.16 MiB | 118,655,479 | 2012-11-29 00:00:00 |
| rel-f1 | 5.94 MiB | 6,225,920 | 1.17 MiB | 1,224,756 | 2010-01-01 00:00:00 |
| rel-hm | 2.139 GiB | 2,297,167,872 | 179.02 MiB | 187,716,297 | 2020-09-14 00:00:00 |
| rel-ratebeer | 6.204 GiB | 6,661,517,312 | 2.480 GiB | 2,663,144,896 | 2020-01-01 00:00:00 |
| rel-salt | 575.07 MiB | 603,009,024 | 59.72 MiB | 62,622,085 | 2020-07-01 00:00:00 |
| rel-stack | 2.764 GiB | 2,967,822,336 | 995.33 MiB | 1,043,682,616 | 2021-01-01 00:00:00 |
| rel-trial | 1.815 GiB | 1,949,143,040 | 642.46 MiB | 673,669,581 | 2021-01-01 00:00:00 |

## 3. 与官网及官方论文核对

**核对结论：十库的表数、测试截止总行数全部与论文精确一致。** 六库总列数也一致；四库不同：Event 为 129/128、RateBeer 为 174/221、Stack 为 50/52、Trial 为 136/140（本地/论文）。本地列数均与当前官方 Parquet 的字段数一致，因此应采用当前版本的实测列数，并保留这四处历史差异。

历史参考为 [RelBench v1 论文 Table 1](https://arxiv.org/html/2407.20060v1#S2.T1) 与 [RelBench v2 论文 Table 1](https://arxiv.org/html/2602.12606v2#S3.T1)。两表均明确按截至测试时间的可用行计数，因此这里使用本地截止行数比较。差值 = 本地 - 论文。

| 数据集 | 表数 本地/论文 | 列数 本地/论文 | 本地截止行 | 论文截止行 | 行数差 | 相对差 |
| --- | --- | --- | --- | --- | --- | --- |
| rel-amazon | 3/3 | 15/15 | 15,000,713 | 15,000,713 | +0 | +0.000% |
| rel-arxiv | 6/6 | 21/21 | 2,146,112 | 2,146,112 | +0 | +0.000% |
| rel-avito | 8/8 | 42/42 | 20,679,117 | 20,679,117 | +0 | +0.000% |
| rel-event | 5/5 | 129/128 | 41,328,337 | 41,328,337 | +0 | +0.000% |
| rel-f1 | 9/9 | 67/67 | 74,063 | 74,063 | +0 | +0.000% |
| rel-hm | 3/3 | 37/37 | 16,664,809 | 16,664,809 | +0 | +0.000% |
| rel-ratebeer | 13/13 | 174/221 | 13,787,005 | 13,787,005 | +0 | +0.000% |
| rel-salt | 4/4 | 31/31 | 4,257,145 | 4,257,145 | +0 | +0.000% |
| rel-stack | 7/7 | 50/52 | 4,247,264 | 4,247,264 | +0 | +0.000% |
| rel-trial | 15/15 | 136/140 | 5,434,924 | 5,434,924 | +0 | +0.000% |

当前官方发布文件是本地数据版本的直接核对基准；论文是历史规模参考。上述四处列数差异不能由时间截断解释，但也不能据此判定本地下载不完整，因为本地与当前官方文件的表结构、行数和大小已匹配。具体历史变更原因需比较论文当时的数据版本，本文不推定。

官方在线版本及文件列表（本次核对保存于 `official-hub.json`）：

- [stanford-star/relbench-v1](https://huggingface.co/datasets/stanford-star/relbench-v1/tree/d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600)：`d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600`，本次读取 50 个关系表 Parquet 文件记录。
- [stanford-star/relbench-v2-extra](https://huggingface.co/datasets/stanford-star/relbench-v2-extra/tree/071cb21e698ad91537d43ca6282411a6d0e78256)：`071cb21e698ad91537d43ca6282411a6d0e78256`，本次读取 23 个关系表 Parquet 文件记录。

官方论文未提供与本报告同口径的文本列比例，也未给出这里的 SQLite 大小。因而不能声称这两项已经与论文数值一致；Parquet 文件大小则已通过官方 API 逐文件验证。

## 4. 数据集背景与逐表明细

### 4.1 rel-amazon

Amazon 电商评论数据，围绕客户、商品和带时间的评分/评论建立关系，可用于流失、消费价值和商品推荐研究。 [官方数据说明](https://huggingface.co/datasets/stanford-star/relbench-v1/blob/d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600/rel-amazon/README.md)。

合计：3 表，15 列，23,218,245 行；文本列 6/15 (40.00%)；SQLite 15.821 GiB，关系表 Parquet 6.924 GiB。

| 表名 | 原始列数 | 完整行数 | 截止行数 | 文本列数（占比） | SQLite TEXT 列 | Parquet 大小 |
| --- | --- | --- | --- | --- | --- | --- |
| `customer` | 2 | 1,850,193 | 1,850,193 | 1 (50.00%) | 1 | 25.84 MiB |
| `product` | 6 | 506,012 | 506,012 | 3 (50.00%) | 4 | 196.68 MiB |
| `review` | 7 | 20,862,040 | 12,644,508 | 2 (28.57%) | 3 | 6.707 GiB |

文本特征列清单（未列出的表没有 `text_embedded` 列）：

- `customer`：`customer_name`。
- `product`：`brand`、`title`、`description`。
- `review`：`review_text`、`summary`。

### 4.2 rel-arxiv

arXiv 学术出版数据，连接论文、作者、研究类别和引用关系，可用于研究方向、发文量、引用与共同引用预测。 [官方数据说明](https://huggingface.co/datasets/stanford-star/relbench-v2-extra/blob/071cb21e698ad91537d43ca6282411a6d0e78256/rel-arxiv/README.md)。

合计：6 表，21 列，2,733,846 行；文本列 6/21 (28.57%)；SQLite 613.87 MiB，关系表 Parquet 165.16 MiB。

| 表名 | 原始列数 | 完整行数 | 截止行数 | 文本列数（占比） | SQLite TEXT 列 | Parquet 大小 |
| --- | --- | --- | --- | --- | --- | --- |
| `authors` | 3 | 143,691 | 143,691 | 2 (66.67%) | 2 | 4.24 MiB |
| `categories` | 2 | 53 | 53 | 0 (0.00%) | 0 | 2.27 KiB |
| `citations` | 3 | 1,595,687 | 1,223,361 | 0 (0.00%) | 1 | 6.14 MiB |
| `paperAuthors` | 3 | 616,585 | 452,251 | 0 (0.00%) | 1 | 3.21 MiB |
| `paperCategories` | 3 | 155,061 | 133,060 | 0 (0.00%) | 1 | 877.73 KiB |
| `papers` | 7 | 222,769 | 193,696 | 4 (57.14%) | 5 | 150.70 MiB |

文本特征列清单（未列出的表没有 `text_embedded` 列）：

- `authors`：`Name`、`ORCID`。
- `papers`：`arXiv_Code`、`Title`、`Abstract`、`DOI`。

### 4.3 rel-avito

Avito 分类广告数据，包含用户、广告、搜索以及曝光、点击、访问和电话请求行为，面向广告点击与用户访问预测。 [官方数据说明](https://huggingface.co/datasets/stanford-star/relbench-v1/blob/d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600/rel-avito/README.md)。

合计：8 表，42 列，24,653,915 行；文本列 2/42 (4.76%)；SQLite 1.399 GiB，关系表 Parquet 404.56 MiB。

| 表名 | 原始列数 | 完整行数 | 截止行数 | 文本列数（占比） | SQLite TEXT 列 | Parquet 大小 |
| --- | --- | --- | --- | --- | --- | --- |
| `AdsInfo` | 6 | 5,960,558 | 5,960,558 | 1 (16.67%) | 1 | 196.17 MiB |
| `Category` | 4 | 68 | 68 | 0 (0.00%) | 0 | 3.69 KiB |
| `Location` | 4 | 3,512 | 3,512 | 0 (0.00%) | 0 | 43.60 KiB |
| `PhoneRequestsStream` | 4 | 302,974 | 244,435 | 0 (0.00%) | 1 | 5.14 MiB |
| `SearchInfo` | 8 | 2,579,289 | 1,987,156 | 1 (12.50%) | 2 | 39.33 MiB |
| `SearchStream` | 7 | 9,254,702 | 7,107,277 | 0 (0.00%) | 1 | 89.24 MiB |
| `UserInfo` | 5 | 98,250 | 98,250 | 0 (0.00%) | 0 | 944.57 KiB |
| `VisitStream` | 4 | 6,454,562 | 5,277,861 | 0 (0.00%) | 1 | 73.72 MiB |

文本特征列清单（未列出的表没有 `text_embedded` 列）：

- `AdsInfo`：`Title`。
- `SearchInfo`：`SearchQuery`。

### 4.4 rel-event

活动推荐数据，连接用户、活动、好友关系、出席记录和兴趣反馈，面向出席次数、重复参加和忽略邀请预测。 [官方数据说明](https://huggingface.co/datasets/stanford-star/relbench-v1/blob/d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600/rel-event/README.md)。

合计：5 表，129 列，44,822,992 行；文本列 6/129 (4.65%)；SQLite 1.394 GiB，关系表 Parquet 113.16 MiB。

| 表名 | 原始列数 | 完整行数 | 截止行数 | 文本列数（占比） | SQLite TEXT 列 | Parquet 大小 |
| --- | --- | --- | --- | --- | --- | --- |
| `event_attendees` | 4 | 11,245,010 | 8,430,002 | 0 (0.00%) | 2 | 1.08 MiB |
| `event_interest` | 6 | 15,398 | 14,978 | 0 (0.00%) | 1 | 151.50 KiB |
| `events` | 110 | 3,137,972 | 2,459,811 | 4 (3.64%) | 5 | 109.18 MiB |
| `user_friends` | 2 | 30,386,403 | 30,386,403 | 0 (0.00%) | 0 | 2.01 MiB |
| `users` | 7 | 38,209 | 37,143 | 2 (28.57%) | 4 | 756.43 KiB |

文本特征列清单（未列出的表没有 `text_embedded` 列）：

- `events`：`city`、`state`、`zip`、`country`。
- `users`：`locale`、`location`。

### 4.5 rel-f1

Formula 1 赛车数据，连接比赛、赛道、车手、车队、排位赛、比赛结果与积分排名，面向完赛名次、前三名和退赛预测。 [官方数据说明](https://huggingface.co/datasets/stanford-star/relbench-v1/blob/d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600/rel-f1/README.md)。

合计：9 表，67 列，97,606 行；文本列 13/67 (19.40%)；SQLite 5.94 MiB，关系表 Parquet 1.17 MiB。

| 表名 | 原始列数 | 完整行数 | 截止行数 | 文本列数（占比） | SQLite TEXT 列 | Parquet 大小 |
| --- | --- | --- | --- | --- | --- | --- |
| `circuits` | 8 | 77 | 77 | 4 (50.00%) | 4 | 10.37 KiB |
| `constructor_results` | 5 | 12,290 | 9,408 | 0 (0.00%) | 1 | 111.24 KiB |
| `constructor_standings` | 7 | 13,051 | 10,170 | 0 (0.00%) | 1 | 140.01 KiB |
| `constructors` | 4 | 211 | 211 | 3 (75.00%) | 3 | 8.28 KiB |
| `drivers` | 7 | 857 | 857 | 5 (71.43%) | 6 | 37.90 KiB |
| `qualifying` | 7 | 9,815 | 4,082 | 0 (0.00%) | 1 | 99.09 KiB |
| `races` | 7 | 1,101 | 820 | 1 (14.29%) | 3 | 24.29 KiB |
| `results` | 15 | 26,080 | 20,323 | 0 (0.00%) | 1 | 428.61 KiB |
| `standings` | 7 | 34,124 | 28,115 | 0 (0.00%) | 1 | 336.25 KiB |

文本特征列清单（未列出的表没有 `text_embedded` 列）：

- `circuits`：`circuitRef`、`name`、`location`、`country`。
- `constructors`：`constructorRef`、`name`、`nationality`。
- `drivers`：`driverRef`、`code`、`forename`、`surname`、`nationality`。
- `races`：`name`。

### 4.6 rel-hm

H&M 时尚零售数据，由客户、商品和带时间的购买交易组成，面向购买推荐、客户流失和商品销售预测。 [官方数据说明](https://huggingface.co/datasets/stanford-star/relbench-v1/blob/d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600/rel-hm/README.md)。

合计：3 表，37 列，16,931,173 行；文本列 12/37 (32.43%)；SQLite 2.139 GiB，关系表 Parquet 179.02 MiB。

| 表名 | 原始列数 | 完整行数 | 截止行数 | 文本列数（占比） | SQLite TEXT 列 | Parquet 大小 |
| --- | --- | --- | --- | --- | --- | --- |
| `article` | 25 | 105,542 | 105,542 | 10 (40.00%) | 14 | 6.05 MiB |
| `customer` | 7 | 1,371,980 | 1,371,980 | 2 (28.57%) | 3 | 85.62 MiB |
| `transactions` | 5 | 15,453,651 | 15,187,287 | 0 (0.00%) | 1 | 87.35 MiB |

文本特征列清单（未列出的表没有 `text_embedded` 列）：

- `article`：`prod_name`、`product_type_name`、`product_group_name`、`graphical_appearance_name`、`colour_group_name`、`perceived_colour_value_name`、`perceived_colour_master_name`、`department_name`、`section_name`、`detail_desc`。
- `customer`：`club_member_status`、`postal_code`。

### 4.7 rel-ratebeer

RateBeer 啤酒评价数据，连接用户、啤酒、酿酒商、场所和多维评分/评价，面向评分、活跃度与偏好推荐研究。 [官方数据说明](https://huggingface.co/datasets/stanford-star/relbench-v2-extra/blob/071cb21e698ad91537d43ca6282411a6d0e78256/rel-ratebeer/README.md)。

合计：13 表，174 列，16,874,623 行；文本列 55/174 (31.61%)；SQLite 6.204 GiB，关系表 Parquet 2.480 GiB。

| 表名 | 原始列数 | 完整行数 | 截止行数 | 文本列数（占比） | SQLite TEXT 列 | Parquet 大小 |
| --- | --- | --- | --- | --- | --- | --- |
| `availability` | 11 | 117,424 | 117,424 | 1 (9.09%) | 2 | 3.18 MiB |
| `beer_ratings` | 14 | 14,344,128 | 11,847,969 | 2 (14.29%) | 3 | 2.138 GiB |
| `beer_styles` | 6 | 157 | 157 | 3 (50.00%) | 3 | 56.11 KiB |
| `beer_upcs` | 3 | 249,348 | 249,348 | 1 (33.33%) | 1 | 4.03 MiB |
| `beers` | 20 | 1,191,662 | 751,524 | 5 (25.00%) | 6 | 222.45 MiB |
| `brewers` | 30 | 50,013 | 50,013 | 16 (53.33%) | 17 | 8.15 MiB |
| `countries` | 5 | 251 | 251 | 2 (40.00%) | 3 | 8.49 KiB |
| `favorites` | 5 | 212,273 | 121,769 | 0 (0.00%) | 1 | 4.13 MiB |
| `place_ratings` | 12 | 385,707 | 343,179 | 1 (8.33%) | 2 | 91.24 MiB |
| `place_types` | 2 | 8 | 8 | 1 (50.00%) | 1 | 1.80 KiB |
| `places` | 59 | 86,061 | 86,061 | 22 (37.29%) | 24 | 14.14 MiB |
| `states` | 4 | 622 | 622 | 1 (25.00%) | 1 | 13.21 KiB |
| `users` | 3 | 236,969 | 218,680 | 0 (0.00%) | 2 | 3.45 MiB |

文本特征列清单（未列出的表没有 `text_embedded` 列）：

- `availability`：`postal_code`。
- `beer_ratings`：`comments`、`language`。
- `beer_styles`：`name`、`description`、`srm_range`。
- `beer_upcs`：`upc`。
- `beers`：`name`、`name_plain`、`description`、`tags`、`admin_note`。
- `brewers`：`name`、`description`、`name_plain`、`address`、`city`、`zip_code`、`website`、`facebook`、`twitter`、`email`、`phone`、`created_at`、`created_by`、`out_of_business_date`、`hours`、`admin_note`。
- `countries`：`name`、`code`。
- `place_ratings`：`comments`。
- `place_types`：`name`。
- `places`：`name`、`name_plain`、`description`、`address`、`city`、`city_plain`、`postal_code`、`phone`、`website`、`facebook`、`twitter`、`scene`、`staff`、`parking`、`music`、`events`、`noise`、`seating`、`food_details`、`hours`、`created_at`、`phone_area_code`。
- `states`：`State`。

### 4.8 rel-salt

SAP SALT 企业销售数据，以订单主表、订单明细、客户和地址为核心，面向工厂、发货点、付款条件等业务字段补全。 [官方数据说明](https://huggingface.co/datasets/stanford-star/relbench-v2-extra/blob/071cb21e698ad91537d43ca6282411a6d0e78256/rel-salt/README.md)。

合计：4 表，31 列，4,748,942 行；文本列 8/31 (25.81%)；SQLite 575.07 MiB，关系表 Parquet 59.72 MiB。

| 表名 | 原始列数 | 完整行数 | 截止行数 | 文本列数（占比） | SQLite TEXT 列 | Parquet 大小 |
| --- | --- | --- | --- | --- | --- | --- |
| `address` | 3 | 1,788,887 | 1,788,887 | 2 (66.67%) | 2 | 9.25 MiB |
| `customer` | 2 | 139,607 | 139,607 | 0 (0.00%) | 0 | 1.71 MiB |
| `salesdocument` | 13 | 500,908 | 411,966 | 3 (23.08%) | 7 | 8.34 MiB |
| `salesdocumentitem` | 13 | 2,319,540 | 1,916,685 | 3 (23.08%) | 4 | 40.42 MiB |

文本特征列清单（未列出的表没有 `text_embedded` 列）：

- `address`：`COUNTRY`、`REGION`。
- `salesdocument`：`SALESDOCUMENTTYPE`、`BILLINGCOMPANYCODE`、`TRANSACTIONCURRENCY`。
- `salesdocumentitem`：`SALESDOCUMENTITEM`、`SALESDOCUMENTITEMCATEGORY`、`PRODUCT`。

### 4.9 rel-stack

Stack Exchange 问答社区数据，包含用户、帖子、评论、投票、徽章与帖子关联，面向互动、投票和推荐预测。 [官方数据说明](https://huggingface.co/datasets/stanford-star/relbench-v1/blob/d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600/rel-stack/README.md)。

合计：7 表，50 列，5,399,818 行；文本列 15/50 (30.00%)；SQLite 2.764 GiB，关系表 Parquet 995.33 MiB。

| 表名 | 原始列数 | 完整行数 | 截止行数 | 文本列数（占比） | SQLite TEXT 列 | Parquet 大小 |
| --- | --- | --- | --- | --- | --- | --- |
| `badges` | 6 | 590,833 | 463,463 | 1 (16.67%) | 2 | 9.13 MiB |
| `comments` | 7 | 794,597 | 623,967 | 2 (28.57%) | 4 | 121.30 MiB |
| `postHistory` | 10 | 1,486,886 | 1,175,368 | 4 (40.00%) | 6 | 533.61 MiB |
| `postLinks` | 5 | 103,969 | 77,337 | 0 (0.00%) | 1 | 2.20 MiB |
| `posts` | 10 | 415,913 | 333,893 | 4 (40.00%) | 6 | 296.65 MiB |
| `users` | 7 | 333,784 | 255,360 | 4 (57.14%) | 5 | 19.48 MiB |
| `votes` | 5 | 1,673,836 | 1,317,876 | 0 (0.00%) | 1 | 12.96 MiB |

文本特征列清单（未列出的表没有 `text_embedded` 列）：

- `badges`：`Name`。
- `comments`：`UserDisplayName`、`Text`。
- `postHistory`：`UserDisplayName`、`RevisionGUID`、`Text`、`Comment`。
- `posts`：`OwnerDisplayName`、`Title`、`Tags`、`Body`。
- `users`：`DisplayName`、`Location`、`WebsiteUrl`、`AboutMe`。

### 4.10 rel-trial

ClinicalTrials.gov 临床试验数据，连接研究、结果、不良事件、入组条件、资助方、疾病和研究机构，面向试验结果及相关预测。 [官方数据说明](https://huggingface.co/datasets/stanford-star/relbench-v1/blob/d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600/rel-trial/README.md)。

合计：15 表，136 列，5,852,157 行；文本列 51/136 (37.50%)；SQLite 1.815 GiB，关系表 Parquet 642.46 MiB。

| 表名 | 原始列数 | 完整行数 | 截止行数 | 文本列数（占比） | SQLite TEXT 列 | Parquet 大小 |
| --- | --- | --- | --- | --- | --- | --- |
| `conditions` | 2 | 3,973 | 3,973 | 1 (50.00%) | 1 | 82.15 KiB |
| `conditions_studies` | 4 | 440,543 | 408,422 | 0 (0.00%) | 1 | 4.24 MiB |
| `designs` | 15 | 272,521 | 249,093 | 5 (33.33%) | 13 | 7.70 MiB |
| `drop_withdrawals` | 6 | 440,547 | 381,199 | 2 (33.33%) | 3 | 4.37 MiB |
| `eligibilities` | 15 | 273,160 | 249,730 | 5 (33.33%) | 13 | 186.84 MiB |
| `facilities` | 6 | 453,233 | 453,233 | 5 (83.33%) | 5 | 14.45 MiB |
| `facilities_studies` | 4 | 1,867,226 | 1,798,765 | 0 (0.00%) | 1 | 16.34 MiB |
| `interventions` | 2 | 3,462 | 3,462 | 1 (50.00%) | 1 | 68.31 KiB |
| `interventions_studies` | 4 | 179,738 | 171,771 | 0 (0.00%) | 1 | 2.03 MiB |
| `outcome_analyses` | 23 | 254,420 | 225,846 | 9 (39.13%) | 14 | 11.75 MiB |
| `outcomes` | 12 | 476,790 | 411,933 | 9 (75.00%) | 10 | 86.39 MiB |
| `reported_event_totals` | 7 | 435,129 | 383,064 | 0 (0.00%) | 3 | 3.94 MiB |
| `sponsors` | 3 | 53,241 | 53,241 | 2 (66.67%) | 2 | 1.47 MiB |
| `sponsors_studies` | 5 | 425,014 | 391,462 | 0 (0.00%) | 2 | 4.67 MiB |
| `studies` | 28 | 273,160 | 249,730 | 12 (42.86%) | 24 | 298.13 MiB |

文本特征列清单（未列出的表没有 `text_embedded` 列）：

- `conditions`：`mesh_term`。
- `designs`：`observational_model`、`primary_purpose`、`time_perspective`、`masking_description`、`intervention_model_description`。
- `drop_withdrawals`：`period`、`reason`。
- `eligibilities`：`minimum_age`、`maximum_age`、`population`、`criteria`、`gender_description`。
- `facilities`：`name`、`city`、`state`、`zip`、`country`。
- `interventions`：`mesh_term`。
- `outcome_analyses`：`non_inferiority_description`、`param_type`、`ci_upper_limit_na_comment`、`p_value_description`、`method`、`method_description`、`estimate_description`、`groups_description`、`other_analysis_description`。
- `outcomes`：`outcome_type`、`title`、`description`、`time_frame`、`population`、`units`、`units_analyzed`、`dispersion_type`、`param_type`。
- `sponsors`：`name`、`agency_class`。
- `studies`：`target_duration`、`acronym`、`baseline_population`、`brief_title`、`official_title`、`source`、`biospec_retention`、`biospec_description`、`source_class`、`baseline_type_units_analyzed`、`detailed_descriptions`、`brief_summaries`。

## 5. stype 缺失列与内部 ID

下列原始列没有出现在现有 stype 配置中，未计入文本列分子；非空行数按 SQLite 全表精确统计。即使全量存在非空值，抽样推断仍可能未给出类型，不能把“未识别”直接解释成该列没有数据。

| 数据集 | 表 | 未识别列 | 全量非空行数 |
| --- | --- | --- | --- |
| rel-event | events | user_id | 1,158 |
| rel-ratebeer | beer_ratings | availability_id | 107,716 |
| rel-ratebeer | places | currency | 2 |
| rel-trial | studies | is_ppsd | 4 |
| rel-trial | studies | fdaaa801_violation | 5 |

| 数据集 | 原始列总数 | SQLite 关系表物理列总数 | 新增内部 ID 列数 |
| --- | --- | --- | --- |
| rel-amazon | 15 | 16 | 1 |
| rel-arxiv | 21 | 24 | 3 |
| rel-avito | 42 | 45 | 3 |
| rel-event | 129 | 132 | 3 |
| rel-f1 | 67 | 67 | 0 |
| rel-hm | 37 | 38 | 1 |
| rel-ratebeer | 174 | 175 | 1 |
| rel-salt | 31 | 31 | 0 |
| rel-stack | 50 | 50 | 0 |
| rel-trial | 136 | 136 | 0 |

## 附录 A. 任务标签表

以下表由预测任务生成，存放在同一 SQLite 文件中，因此占据文件空间，但不计入前文原始关系表的行列总数。列数按实际 SQLite 物理列统计，包含 `__target_row_id__`、`split` 等辅助字段；行数合并训练、验证和测试拆分。管理表未逐项展示。

| 数据集 | 任务表数 | 任务表列数之和 | 任务表行数之和 | 关系表+任务表物理列数 | 关系表+任务表总行数 |
| --- | --- | --- | --- | --- | --- |
| rel-amazon | 8 | 40 | 50,380,233 | 56 | 73,598,478 |
| rel-arxiv | 4 | 20 | 1,862,283 | 44 | 4,596,129 |
| rel-avito | 6 | 30 | 7,337,175 | 75 | 31,991,090 |
| rel-event | 6 | 30 | 117,327 | 162 | 44,940,319 |
| rel-f1 | 6 | 30 | 51,783 | 97 | 149,389 |
| rel-hm | 4 | 20 | 29,156,912 | 58 | 46,088,085 |
| rel-ratebeer | 8 | 40 | 18,123,909 | 215 | 34,998,532 |
| rel-salt | 8 | 40 | 9,463,160 | 71 | 14,212,102 |
| rel-stack | 5 | 25 | 8,224,192 | 75 | 13,624,010 |
| rel-trial | 9 | 45 | 2,085,038 | 181 | 7,937,195 |

### rel-amazon 任务表

| 任务标签表 | 物理列数 | 行数 |
| --- | --- | --- |
| `target_item_churn` | 5 | 2,880,545 |
| `target_item_ltv` | 5 | 3,052,991 |
| `target_review_rating` | 5 | 20,862,040 |
| `target_user_churn` | 5 | 5,470,060 |
| `target_user_item_purchase` | 5 | 5,858,664 |
| `target_user_item_rate` | 5 | 4,217,705 |
| `target_user_item_review` | 5 | 2,568,168 |
| `target_user_ltv` | 5 | 5,470,060 |

### rel-arxiv 任务表

| 任务标签表 | 物理列数 | 行数 |
| --- | --- | --- |
| `target_author_category` | 5 | 289,439 |
| `target_author_publication` | 5 | 289,439 |
| `target_paper_citation` | 5 | 883,774 |
| `target_paper_paper_cocitation` | 5 | 399,631 |

### rel-avito 任务表

| 任务标签表 | 物理列数 | 行数 |
| --- | --- | --- |
| `target_ad_ctr` | 5 | 8,682 |
| `target_searchinfo_isuserloggedon` | 5 | 2,579,289 |
| `target_searchstream_click` | 5 | 4,315,120 |
| `target_user_ad_visit` | 5 | 152,724 |
| `target_user_clicks` | 5 | 128,633 |
| `target_user_visits` | 5 | 152,727 |

### rel-event 任务表

| 任务标签表 | 物理列数 | 行数 |
| --- | --- | --- |
| `target_event_interest_interested` | 5 | 15,398 |
| `target_event_interest_not_interested` | 5 | 15,398 |
| `target_user_attendance` | 5 | 23,210 |
| `target_user_ignore` | 5 | 23,210 |
| `target_user_repeat` | 5 | 3,441 |
| `target_users_birthyear` | 5 | 36,670 |

### rel-f1 任务表

| 任务标签表 | 物理列数 | 行数 |
| --- | --- | --- |
| `target_driver_circuit_compete` | 5 | 2,703 |
| `target_driver_dnf` | 5 | 12,679 |
| `target_driver_position` | 5 | 8,712 |
| `target_driver_top3` | 5 | 2,667 |
| `target_qualifying_position` | 5 | 9,815 |
| `target_results_position` | 5 | 15,207 |

### rel-hm 任务表

| 任务标签表 | 物理列数 | 行数 |
| --- | --- | --- |
| `target_item_sales` | 5 | 5,699,268 |
| `target_transactions_price` | 5 | 15,453,651 |
| `target_user_churn` | 5 | 3,983,823 |
| `target_user_item_purchase` | 5 | 4,020,170 |

### rel-ratebeer 任务表

| 任务标签表 | 物理列数 | 行数 |
| --- | --- | --- |
| `target_beer_churn` | 5 | 2,642,980 |
| `target_beer_ratings_total_score` | 5 | 14,343,239 |
| `target_brewer_dormant` | 5 | 130,903 |
| `target_user_beer_favorite` | 5 | 2,641 |
| `target_user_beer_liked` | 5 | 158,786 |
| `target_user_churn` | 5 | 403,009 |
| `target_user_count` | 5 | 403,009 |
| `target_user_place_liked` | 5 | 39,342 |

### rel-salt 任务表

| 任务标签表 | 物理列数 | 行数 |
| --- | --- | --- |
| `target_item_incoterms` | 5 | 2,319,540 |
| `target_item_plant` | 5 | 2,319,540 |
| `target_item_shippoint` | 5 | 2,319,540 |
| `target_sales_group` | 5 | 500,908 |
| `target_sales_incoterms` | 5 | 500,908 |
| `target_sales_office` | 5 | 500,908 |
| `target_sales_payterms` | 5 | 500,908 |
| `target_sales_shipcond` | 5 | 500,908 |

### rel-stack 任务表

| 任务标签表 | 物理列数 | 行数 |
| --- | --- | --- |
| `target_post_post_related` | 5 | 6,339 |
| `target_post_votes` | 5 | 2,771,171 |
| `target_user_badge` | 5 | 3,889,034 |
| `target_user_engagement` | 5 | 1,534,825 |
| `target_user_post_comment` | 5 | 22,823 |

### rel-trial 任务表

| 任务标签表 | 物理列数 | 行数 |
| --- | --- | --- |
| `target_condition_sponsor_run` | 5 | 41,072 |
| `target_eligibilities_adult` | 5 | 273,160 |
| `target_eligibilities_child` | 5 | 273,160 |
| `target_site_sponsor_run` | 5 | 733,741 |
| `target_site_success` | 5 | 193,764 |
| `target_studies_enrollment` | 5 | 271,866 |
| `target_studies_has_dmc` | 5 | 234,467 |
| `target_study_adverse` | 5 | 50,029 |
| `target_study_outcome` | 5 | 13,779 |

## 附录 B. 复现与证据

本地统计完成时间（UTC）：`2026-09-28T03:46:10.672858+00:00`。官方 API 核对时间（UTC）：`2026-09-28T03:41:42.263198+00:00`。

本地数据库记录的 RelBench 版本为 3.0.1；stype artifact 记录 RelBench 3.0.1、PyTorch Frame 0.3.0、pandas 3.0.5、NumPy 2.5.3。统计前确认 stype 记录的数据库大小及 mtime 与本地文件一致，并在统计后再次确认数据库未改变。

以下文件均位于报告同级的 `dataset-background-20260928/` 目录：

- `local-stats.json`：所有本地逐表结果、文本列名、stype、源仓库、时间范围及精确字节数。
- `rel-*.json`：各数据库独立的同口径统计结果。
- `official-hub.json`：在线查询的官方仓库版本与文件大小记录。
- `paper-reference.json`：从官方论文 Table 1 抄录的比较基准及出处。
- `collect_stats.py`、`verify_official.py`、`render_report.py`：采集、在线核对和报告生成脚本。

在项目目录 `/home/zk/relconnector/relconnector` 执行：

```bash
.venv/bin/python doc/dataset-background-20260928/collect_stats.py
.venv/bin/python doc/dataset-background-20260928/verify_official.py
.venv/bin/python doc/dataset-background-20260928/render_report.py
```

生成器会检查官方版本、完整文件列表及大小，若官方更新导致与本地快照不符则中止，以免输出过期的“一致”结论。
