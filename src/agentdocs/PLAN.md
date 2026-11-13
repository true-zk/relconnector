# RelX 端到端流程与第一阶段设计

## 目标与本轮边界

RelX 面向不同关系型数据库和不同关系表格学习模型。用户最终通过一份运行 YAML 指定数据库、任务及可选覆盖项；内部仍由连接、任务校验和预处理模块分别处理。数据库内不需要 RelX 专用 catalog 或 task metadata 表。第一阶段完成训练前的准备流程，并为后续训练和推理提供稳定的数据契约。

完整流程：

```text
connect(database_url, reader)
    → discover_schema(db)
    → validate_task(task_yaml, schema)
    → prepare_graph(db, schema, task)
    → plan_features(db, schema, task, feature_config)
    → PreparedDataset
    → [后续] seed sampling → batch feature transform → model training
    → [后续] validation / inference
```

第一阶段的实施范围是连接、结构发现、任务校验、构图和特征处理策略/统计量准备。连接、基于 Inspector 的表结构发现、全局 YAML 解析和任务校验已实现；构图与特征处理策略仍按本文档实施。训练时的种子采样、按批次读取并转换特征、模型训练，以及推理执行，暂不实现。`prepare` 可以拟合仅由训练数据得到的统计量；这与训练时对每个 batch 执行 `transform` 是两个阶段。

现状：连接与 schema 发现已支持 SQLAlchemy 方言和组合主外键；[`task/spec.py`](../relx/task/spec.py) 已描述三种 split 规则。v2 的[构图器](../../exps/relconnector_v2/relconnector_v2/graph/builder.py)依赖连续整数 ID、`rowid` 和 `julianday()`，不能直接作为跨数据库实现搬入 `src/relx`。

## 公共接口草案

完整训练流程的目标入口（训练阶段完成后）：

```python
run = relx.run("churn.yaml")
report = run.report
```

当前已实现的入口和下一步编排形式：

```python
config = relx.load_config("churn.yaml")
with relx.connect(config.database.location, reader=config.database.reader) as db:
    schema = relx.discover_schema(
        db,
        schema_name=config.database.schema_name,
        overrides=config.schema,
    )
    task = relx.validate_task(
        config.task, schema, hidden_columns=config.hidden_columns, reader=db
    )
    # 后续阶段：prepared = relx.prepare(db, schema, task, config.features)
```

用 `relx.load_config` 解析一次运行的全局 YAML；`relx.conn` 和 `relx.task` 保留为子包。`prepare` 返回可检查的 `PreparedDataset`，至少含 `DatabaseSchema`、`ValidatedTask`、`GraphIndex`、`FeaturePlan`、数据快照标识及本地 artifact 指纹。模型是否需要图由后续模型适配器声明；准备接口允许按需跳过图，不要求所有模型使用图批次。

## 1. 数据库连接：`conn/`

### 职责与接口

`conn` 只负责连接和读取，不识别任务、列的语义类型，也不执行结构发现或建图。

```python
class DatabaseReader(Protocol):
    def read_query(self, query: SqlQuery, *, params: Mapping[str, object] = ...) -> pd.DataFrame: ...
    def iter_query(
        self, query: SqlQuery, *, params: Mapping[str, object] = ..., batch_size: int
    ) -> Iterator[pd.DataFrame]: ...
    def close(self) -> None: ...
```

`DatabaseHandle` 持有数据库 URL、SQLAlchemy `Engine`/连接资源、选择的 reader 与 dialect 信息；preprocess 使用同一 `Engine` 执行元数据反射，但反射逻辑留在 `preprocess`。`SqlQuery` 可以先接受 SQLAlchemy Core 表达式和显式参数，避免把值拼进 SQL、手写跨数据库标识符转义。表名先经过已发现 schema 校验，再交给 dialect 编译。

连接地址使用 `dialect+driver://...`。默认使用 SQLAlchemy reader，因为它和结构反射可共享数据库驱动；ConnectorX 是同一读取接口下的可选高吞吐实现，是否支持特定数据库、类型和分块模式由适配器显式检测，不支持时给出清晰错误或回退到 SQLAlchemy。不能默认认为 ConnectorX 的连接与 SQLAlchemy 反射处在同一个事务快照。

连接对象必须支持上下文管理和关闭；分块迭代提前停止时也应释放游标/连接。默认只读，数据库凭据不写入 task YAML、日志或本地 artifact。远程数据库可能在 prepare 期间变化：优先在可用的只读一致性快照内准备；若 reader 无法共享快照，要求用户提供稳定数据源/快照标识，或声明结果只保证尽力一致。

**验收：**SQLite 和至少一种服务端数据库在同一接口下完成带参数的整表读取、分块读取、异常及提前终止时的资源释放；可选 ConnectorX 与默认 reader 的列值/空值行为有对照测试。

## 2. 表结构发现：`preprocess/schema.py`

### 数据契约

使用 SQLAlchemy `Inspector` 获得表、列、主键、外键；如某方言的反射缺项，在 `preprocess/schema_adapters/` 放专用补丁。`conn/` 不增加 `schemas()` 等发现方法。统一结构建议为：

```text
DatabaseSchema: tables: Mapping[TableRef, TableSchema]
TableRef:       table_name, schema_name | None
TableSchema:    ref, columns, primary_key: tuple[column, ...] | None,
                identity_key: tuple[column, ...] | None,
                foreign_keys: tuple[ForeignKey, ...]
ForeignKey:     source_columns: tuple[column, ...],
                target_table: TableRef, target_columns: tuple[column, ...]
ColumnSchema:   name, ordinal, raw_sql_type, nullable
```

保留数据库 schema/namespace，避免不同 schema 下的同名表冲突。组合主外键在结构层完整表示；不再因当前构图器只支持单列而丢弃或误表示。数据库物理类型只用于反射和读取解码；`NUMERIC`、`TEXT`、时间等**特征语义类型**由第 4 步单独决定。

输入可限定要发现的表，默认不把任务表或数据库全部表无条件当作图节点。数据库没有声明外键时允许在用户配置中补充关系；覆盖项需通过表/列存在性检查。`primary_key` 保留数据库声明的主键，`identity_key` 是后续构图使用的有效键，默认等于主键。对于无主键表，必须提供稳定唯一键配置，不能使用 SQLite `rowid` 作为通用规则；覆盖键的唯一性与空值需在后续数据级校验中确认。未能反射的类型、视图或跨 schema 引用给出明确诊断。

**验收：**同一 `DatabaseSchema` 在 SQLite 与 PostgreSQL（或另一服务端数据库）上正确描述普通键、组合键、同名不同 schema 表和可空列；反射结果不依赖 RelX 专用数据库表。

**MySQL 实测（2026-11-13）：**在隔离的 MySQL 5.7 实例中导入 `rel-f1` 的 15 张业务/任务表（不含旧 catalog），共 149,389 行。使用 `relx.discover_schema()` 对照 SQLite 源库，列名、主键、外键、可空性和行数均一致；49 个整数列映射为 `BIGINT`、14 个实数列映射为 `DOUBLE`、34 个文本列映射为 `TEXT`。240,095 个实数值逐值相同。可通过项目的 `mysql` 可选依赖安装 PyMySQL；`tests/test_relx_mysql_integration.py` 提供基于 `RELX_TEST_MYSQL_REL_F1_URL` 的只读回归检查。数据导入由测试脚本显式重建主外键；若仅用 `pandas.to_sql()` 写入表格，数据库不会自动获得这些关系约束。

## 3. 任务校验：`task/`

用户只维护一份运行 YAML。`task` 部分定义预测样本的语义；`database`、`schema`、`hidden_columns`、`features`、`model`、`training`、`report` 是平级配置。连接串可以通过环境变量提供，避免把凭据写进文件。以下格式已由 `relx.load_config()` 解析；模型和训练参数当前只解析保存，训练流程尚未实现：

```yaml
database:
  url_env: RELX_DATABASE_URL
task:
  name: driver-dnf
  type: binary_classification
  table: target_driver_dnf
  entity_column: driverId
  label_column: did_not_finish
  time_column: date
  split:
    type: column
    column: split
    values: {train: train, validation: val, test: test}
hidden_columns: []
features:
  columns: []
model:
  name: default
training:
  batch_size: 1024
  epochs: 20
  lr: 0.001
  seed: 42
report:
  directory: runs/driver-dnf
```

任务表是样本来源，每行至少可确定样本身份、实体引用、标签（测试行可为空）和可选预测时间。`entity_column` 是任务表中的外键列；如果它恰好存在一条指向有效实体键的外键，就从已发现的 schema 推断实体表和实体键，不要求用户重复填写。缺少外键时在平级 `schema.foreign_keys` 覆盖项声明关系；同一列有多条候选关系时可选填 `task.entity_table` 指定目标。推荐任务有源、目标两侧实体，`task.destination: {table, key}` 指定目标实体；当前要求其标签为 JSON ID 数组。任务表自身的主键默认作为样本 ID；没有稳定主键时由 `schema.identity_keys` 补充。时间列是任务样本的预测时间；业务表各自的时间列属于后续预处理配置。

`task.split` 支持三种来源，优先采用用户显式规则：

| 类型 | 配置和语义 |
| --- | --- |
| `column` | 任务表中的划分列与 train/validation/test 值映射；同一任务表的常见形式。 |
| `sql_query` | 用只读 SQL 查询返回任务表样本主键；可以只指定 test，余下有标签的样本再自动划分 train/validation，也可以分别指定三组查询。结果必须唯一、互斥，默认要求覆盖全部样本；应通过只读数据库账号执行。 |
| `default` | 省略 `split` 时采用：标签为空的样本为 test；有标签的样本划分 train/validation。带预测时间时按时间顺序取末段作验证，不带时间时按固定随机种子划分；默认验证比例为 20%、随机种子为 42，均可覆盖。测试集无标签时只输出预测和验证指标。 |

`sql_query` 是按行选择样本，查询列名必须与任务表主键列名一致；不允许通过查询结果重新定义实体、标签或预测时间。它不可保证跨数据库方言通用。若数据库把训练、验证、测试放在不同物理表中，未来可为 `task.table` 增加按 split 指定来源表的形式，并要求列语义一致；当前只支持单任务表。默认规则的“空标签即测试”是明确的用户约定；如果空标签还表示脏数据，用户应显式指定划分并处理缺失标签。空测试集允许生成验证报告，但不声称存在测试指标。

校验分两层：配置解析检查 YAML 形状、枚举和必填项；`validate_task(task, schema)` 检查表列存在、实体外键唯一且目标键有效、split 规则完整。传入 reader 时，数据级检查分块读取任务列，验证样本键唯一、实体键可解析、标签类型、时间可解析、split 查询互斥/覆盖及各组样本量。目前会在内存中保存任务键、标签和 split 映射，内存消耗随任务样本数增长；后续大规模数据需要磁盘索引或数据库端计算。

任务表默认只提供样本与标签，不进入图特征。标签、split 和任务时间自动进入 `ValidatedTask.hidden_columns`；若任务表也承载业务特征，则只能在明确排除这些列后使用。其他表中的答案或衍生列由平级 `hidden_columns` 排除；不能靠数据集名称（例如 Ratebeer）写特殊分支。用于拟合的标签只来自 train split；特征表统计量来自训练时间可见的数据。默认允许 RelBench 式的无标签、跨实体特征统计；若任务要求归纳式设置，应可配置仅扫描训练实体可见行。验证和测试标签不参与拟合。无时间信息时不得声称具备时间泄漏防护。

输出 `ValidatedTask`，包含解析后的表/列引用、确定的 split 规则、最终排除列集合、时间策略和数据检查报告；后续准备和训练只消费它，不反复解释 YAML。

**验收：**错误字段、缺失表列、目标泄漏、错误关系映射和无效 split 都有定位到表/列的错误；有效任务可在不读取全部任务行的情况下通过结构校验。

## 4. 构图与特征处理策略：`preprocess/`

### 4.1 构图

`GraphBuilder` 消费 `DatabaseSchema`、`ValidatedTask` 和数据库读取接口。每张参与建图的表形成一种节点；每条外键关系形成正向边和反向边。原始主键可以是整数、字符串、UUID 或组合键：先建立稳定的 `(TableRef, business_key) → dense_node_id` 映射，再用外键匹配目标节点。任务表默认是 seed/label 来源；只有显式选择时才作为图节点。

策略需明确：空外键不建边；悬空外键是跳过并计数还是报错由配置决定；组合外键按完整键匹配；重复主键报错；不依赖 SQL 查询结果的自然顺序。时间列与任务时间分开定义，按统一时区/精度转换；有准备阶段 cutoff 时只纳入该 cutoff 前可见的节点和边；如果图包含更晚的历史，必须保留节点/边时间，供后续采样按每个 seed 的时间再次过滤。边的时间语义由来源表及配置决定，不能直接照搬 SQLite `julianday()`。大表分块扫描，图可以先实现内存索引，但必须报告节点/边数和峰值估算，之后允许替换为磁盘索引。

输出与数据库无关的 `GraphIndex`：节点类型及数量、双向边、业务键映射、可选时间数组、构建报告。图中不存业务特征，也不存任务标签。图结构对模型中立，不在此阶段决定 GraphSAGE、RelGNN 或其他模型。

### 4.2 特征语义类型与处理器

为每个候选列生成 `FeatureSpec(table, column, semantic_type, processor, params)`。先排除主外键、任务标签和隐藏列；其余列的类型按 **用户显式配置 > 受控样本推断 > 明确报错/要求指定** 确定。物理 SQL 类型不是语义类型：`VARCHAR` 可能是类别、文本、时间或序列化多类别值。默认推断与 RelBench 的 `get_stype_proposal()` 保持语义一致：在候选列的有界样本上调用 PyTorch Frame 的类型推断；样本策略和随机种子记录在 `FeaturePlan` 中。用户配置优先于推断。

| 语义类型 | 默认准备阶段 | 后续批次处理/模型侧默认行为 |
| --- | --- | --- |
| 数值 `numerical` | 仅用训练可见行计算均值、标准差和必要分位数 | PyTorch Frame 的数值映射；RelBench 默认 `LinearEncoder` |
| 类别 `categorical` | 统计类别词表/频数，规定缺失值和未知类别 | 类别索引；RelBench 默认 `EmbeddingEncoder` |
| 多类别 `multicategorical` | 定义分隔/解析规则，统计元素词表 | 多值索引；RelBench 默认 `MultiCategoricalEmbeddingEncoder` |
| 时间 `timestamp` | 确定时区、解析规则和时间范围统计量 | 时间映射；RelBench 默认 `TimestampEncoder` |
| 文本 `text` | 确定清洗、嵌入器配置、输出维数与缓存策略 | 用指定文本嵌入器生成向量，再交给模型编码；未指定嵌入器时明确要求配置 |
| 已有向量 `embedding` | 校验固定维数、空值规则 | 保留向量；RelBench 默认 `LinearEmbeddingEncoder` |

这里的“默认与 RelBench 一致”指候选列过滤、语义类型/统计量，以及接入 PyTorch Frame 时的默认映射和编码器。它不意味着替用户自动选择一种文本模型；RelBench 的文本嵌入也需要显式提供配置。具体对齐以当前安装的 RelBench 3.0.1 / PyTorch Frame 实现为参考，避免把实验代码中的 GloVe 路径误当通用默认。

特征配置是运行 YAML 中可选的平级 `features` 部分；未列出的列使用默认推断和默认处理器。示例为拟议格式：

```yaml
features:
  columns:
    - {table: users, column: age, semantic_type: numerical, processor: relbench_numerical}
    - table: users
      column: bio
      semantic_type: text
      processor: relbench_text
      params: {embedder: my_text_encoder}
    - {table: orders, column: amount, semantic_type: numerical, processor: log_amount}
```

文本嵌入器也通过显式注册名或 Python 对象提供，YAML 只描述选择与参数。不同模型可以复用同一个语义 `FeaturePlan`，再由模型适配层决定最终编码器。

### 4.3 自定义特征处理类

支持用户继承一个简洁的基类，按列覆盖默认处理方式：

```python
class FeatureProcessor(ABC):
    semantic_type: str

    def fit(self, values: Iterable[pd.Series], context: FitContext) -> FeatureState:
        """可选：只用训练可见数据生成词表/统计量。"""

    @abstractmethod
    def transform(self, values: pd.Series, state: FeatureState) -> FeatureBlock:
        """后续实现：将一批原始值转为有 dtype/shape/缺失标记的特征。"""

    def output_spec(self, state: FeatureState) -> FeatureOutputSpec:
        """声明输出维数、dtype、空值约定，供后续模型适配。"""
```

数值、类别、多类别、时间、文本等默认处理器分别实现此契约。自定义处理器可以覆盖 `fit`、`transform`，并声明输出规格；例如对金额做对数变换、使用自己的文本向量器。`fit` 状态与配置保存在本地 artifact，`transform` 只能使用该状态，不能在验证/测试批次上重新拟合。处理器不得默默查询标签或读取未来数据。

注册方式建议同时支持 Python 对象和显式注册名：

```python
relx.register_feature_processor("log_amount", LogAmountProcessor)
prepared = relx.prepare(
    db, task,
    feature_processors={("orders", "amount"): LogAmountProcessor()},
)
```

YAML 只引用已注册名称和参数，不从 YAML 任意导入 Python 类。处理器输出使用 RelX 的 `FeatureBlock` 契约；将它转成 TensorFrame、PyG 或其他模型输入留给后续模型适配层，因此自定义处理器不必继承某个模型的类。需要持久化状态的处理器应提供稳定的处理器 ID、参数和状态序列化方法；无法安全序列化的状态只在当前进程复用。

本阶段产出 `FeaturePlan` 和可选 `FeatureState`/统计量；**不执行训练 batch 的特征查询、`transform`、TensorFrame 组装或模型前向**。若某处理器需要 `fit`，准备阶段可分块扫描训练可见数据，不能全表物化。保存的本地 artifact 记录表结构、任务、时间窗口、处理器身份及参数、数据快照标识的指纹；这只是缓存有效性信息，不恢复数据库内的 catalog version。

**验收：**同一列可以由默认处理器或注册的自定义处理器生成不同 `FeatureSpec`；数值/类别/文本等策略和输出规格可检查；标签列不会进入特征计划；统计量只依赖训练可见数据；构图支持非连续业务键并报告悬空外键。

## 建议目录与实施顺序

```text
src/relx/
  api.py                  # 未来的 run / prepare 编排
  config/                 # 全局 YAML 解析、各模块配置对象
  core/schema.py          # TableRef、ColumnSchema、ForeignKey、DatabaseSchema
  conn/                   # SQLAlchemy / 可选 ConnectorX 读取器
  task/                   # YAML、TaskSpec、ValidatedTask、任务校验
  preprocess/
    schema.py             # 结构反射与覆盖配置
    graph.py              # 节点 ID 映射和拓扑构建
    features/
      plan.py             # FeatureSpec、FeaturePlan、输出规格
      processors.py       # 基类和默认处理器契约
      registry.py         # 显式注册自定义处理器
      statistics.py       # 训练可见数据的统计量准备
  sampler/                # 后续阶段
  assembler/              # 后续批次特征处理与组装
  trainer/                # 后续模型训练
```

1. 先固定 `core` 的表标识、组合键和 `DatabaseReader` 契约；实现 SQLAlchemy 连接与读取，保留现有 SQLite 读取器作兼容适配。
2. 迁移 schema 发现到 SQLAlchemy Inspector，增加多数据库对照测试；删除调用方对 `discover_sqlite_schema()` 的依赖，再决定是否保留 SQLite 专用优化。
3. 扩展 YAML 和 `ValidatedTask`；先完成纯结构校验、split/泄漏规则，再增加必要的数据级检查。
4. 实现通用业务键映射和构图；先覆盖单列/组合键、空/悬空外键及时间 cutoff，再处理大图的持久化优化。
5. 定义 `FeatureProcessor`、注册表、默认处理器与 `FeaturePlan`；最后增加训练数据统计量及本地 artifact。
6. 第一阶段结束时提供 `relx.prepare()` 和可检查报告；采样、按批次特征处理、模型训练/推理作为下一阶段单独设计与实现。

每一步完成后，保证 `src/relx` 不导入 `exps` 包；旧 v2 代码只作为行为参考。测试优先验证数据库适配、组合键/时间/泄漏边界，以及默认与自定义处理器的契约，不以复制旧实现作为验收标准。
