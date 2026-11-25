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

当前单跳基线的训练入口（更多模型待接入）：

```python
run = relx.run("churn.yaml")
report = run.report
```

当前已实现的准备入口：

```python
import relx

config = relx.load_config("churn.yaml")
prepared = relx.prepare(config)
print(prepared.report)
```

用 `relx.load_config` 解析一次运行的全局 YAML；`relx.conn` 和 `relx.task` 保留为子包。当前 `prepare` 返回可检查的 `PreparedDataset`，包含 `DatabaseSchema`、`ValidatedTask`、`GraphIndex`、`FeaturePlan` 和准备报告。数据快照标识、本地 artifact 指纹及按模型需求跳过构图仍待后续实现。

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

用户只维护一份运行 YAML。`task` 部分定义预测样本的语义；`database`、`schema`、`graph`、`hidden_columns`、`features`、`model`、`training`、`report` 是平级配置。连接串可以通过环境变量提供，避免把凭据写进文件。以下格式已由 `relx.load_config()` 解析；模型和训练参数当前只解析保存，训练流程尚未实现：

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
graph:
  exclude_tables:
    - target_driver_circuit_compete
    - target_driver_position
    - target_driver_top3
    - target_qualifying_position
    - target_results_position
  time_columns:
    constructor_results: date
    constructor_standings: date
    qualifying: date
    races: date
    results: date
    standings: date
  static_tables: [circuits, constructors, drivers]
  timezone: UTC
  time_boundary: at_or_before
hidden_columns: []
features:
  columns: []
model:
  name: one_hop_baseline
  params: {channels: 32, fanout: 10, workers: 4, text_mode: hash}
training:
  batch_size: 1024
  epochs: 20
  lr: 0.001
  seed: 42
report:
  directory: runs/driver-dnf
```

`relx.run("run.yaml")` 当前可运行分类或回归的单跳实体任务，并写入 `report.json`；`predict_test()` 返回测试任务行的预测。此基线模型的 `text_mode: hash` 是显式选择的轻量文本适配方式，**不是** RelBench 的文本嵌入默认实现。多标签、推荐任务及其他模型仍待后续适配。

任务表是样本来源，每行至少可确定样本身份、实体引用、标签（测试行可为空）和可选预测时间。`entity_column` 是任务表中的外键列；如果它恰好存在一条指向有效实体键的外键，就从已发现的 schema 推断实体表和实体键，不要求用户重复填写。缺少外键时在平级 `schema.foreign_keys` 覆盖项声明关系；同一列有多条候选关系时可选填 `task.entity_table` 指定目标。推荐任务有源、目标两侧实体，`task.destination: {table, key}` 指定目标实体；当前要求其标签为 JSON ID 数组。任务表自身的主键默认作为样本 ID；没有稳定主键时由 `schema.identity_keys` 补充。时间列是任务样本的预测时间；业务表各自的时间列属于后续预处理配置。

`task.split` 支持三种来源，优先采用用户显式规则：

| 类型 | 配置和语义 |
| --- | --- |
| `column` | 任务表中的划分列与 train/validation/test 值映射；同一任务表的常见形式。 |
| `sql_query` | 用只读 SQL 查询返回任务表样本主键；可以只指定 test，余下有标签的样本再自动划分 train/validation，也可以分别指定三组查询。结果必须唯一、互斥，默认要求覆盖全部样本；应通过只读数据库账号执行。 |
| `default` | 省略 `split` 时采用：标签为空的样本为 test；有标签的样本划分 train/validation。带预测时间时按时间顺序取末段作验证，不带时间时按固定随机种子划分；默认验证比例为 20%、随机种子为 42，均可覆盖。测试集无标签时只输出预测和验证指标。 |

`sql_query` 是按行选择样本，查询列名必须与任务表主键列名一致；不允许通过查询结果重新定义实体、标签或预测时间。它不可保证跨数据库方言通用。若数据库把训练、验证、测试放在不同物理表中，未来可为 `task.table` 增加按 split 指定来源表的形式，并要求列语义一致；当前只支持单任务表。默认规则的“空标签即测试”是明确的用户约定；如果空标签还表示脏数据，用户应显式指定划分并处理缺失标签。空测试集允许生成验证报告，但不声称存在测试指标。

校验分两层：配置解析检查 YAML 形状、枚举和必填项；`validate_task(task, schema)` 检查表列存在、实体外键唯一且目标键有效、split 规则完整。传入 reader 时，数据级检查分块读取任务列，验证样本键唯一、实体键可解析、标签类型、时间可解析、split 查询互斥/覆盖及各组样本量。目前会在内存中保存任务键、标签和 split 映射，内存消耗随任务样本数增长；后续大规模数据需要磁盘索引或数据库端计算。

任务表默认只提供样本与标签，不进入图特征。标签、split 和任务时间自动进入 `ValidatedTask.hidden_columns`；若任务表也承载业务特征，则只能在明确排除这些列后使用。其他表中的答案或衍生列由平级 `hidden_columns` 排除；不能靠数据集名称（例如 Ratebeer）写特殊分支。用于拟合的标签只来自 train split；特征表统计量来自训练时间可见的数据。默认允许 RelBench 式的无标签、跨实体特征统计；若任务要求归纳式设置，应可配置仅扫描训练实体可见行。验证和测试标签不参与拟合。无时间信息时不得声称具备时间泄漏防护。若旧数据库还保存 RelX/v2 的内部 catalog 表，也需要在 `graph.exclude_tables` 中列出；新项目不会依赖这些表。

输出 `ValidatedTask`，包含解析后的表/列引用、确定的 split 规则、最终排除列集合、时间策略和数据检查报告；后续准备和训练只消费它，不反复解释 YAML。

**验收：**错误字段、缺失表列、目标泄漏、错误关系映射和无效 split 都有定位到表/列的错误；有效任务可在不读取全部任务行的情况下通过结构校验。

## 4. 构图与特征处理策略：`preprocess/`

### 4.1 构图

`GraphBuilder` 消费 `DatabaseSchema`、`ValidatedTask`、`GraphConfig` 和数据库读取接口。`select_graph_tables()` 自动排除当前任务表，再应用 `graph.exclude_tables` 排除同库中的其他标签表或不需要的表；不存在的表、任务实体表和推荐目标实体表不能被排除。每张参与建图的表形成一种节点；每条外键关系形成正向边和反向边。原始主键可以是整数、字符串、UUID 或组合键：先建立稳定的 `(TableRef, business_key) → dense_node_id` 映射，再用外键匹配目标节点。任务表默认是 seed/label 来源。

策略需明确：空外键不建边；悬空外键当前跳过并计数；组合外键按完整键匹配；重复主键报错；不依赖 SQL 查询结果的自然顺序。时间列与任务时间分开定义。当前 `graph.time_columns` 标记有记录可用时间的表，`graph.static_tables` 是用户确认始终可用的表；有预测时间的任务要求每张入图表都属于二者之一。无时区的值按 `graph.timezone` 解读，统一为 UTC 微秒。边的可用时间取两个端点可用时间的较晚者；`at_or_before`（默认）允许时间早于或等于 seed 预测时刻，`strict_before` 则要求严格早于。C++ 索引保留全部节点/边及其时间，`visible_neighbors()` 按时间过滤；采样器执行相同规则。大表分块扫描，图目前是内存索引；峰值估算和磁盘索引留待后续。

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
  sample_limit: 1000
  sample_seed: 42
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

    def output_spec(self, params: Mapping[str, object]) -> FeatureOutputSpec:
        """声明输出维数、dtype、空值约定，供后续模型适配。"""
```

数值、类别、多类别、时间、文本等默认处理器分别实现此契约。自定义处理器可以覆盖 `fit`、`transform`，并声明输出规格；例如对金额做对数变换、使用自己的文本向量器。`fit` 状态与配置保存在本地 artifact，`transform` 只能使用该状态，不能在验证/测试批次上重新拟合。处理器不得默默查询标签或读取未来数据。

当前使用 `prepare()` 传入实例；YAML 中的名字从这份映射解析：

```python
prepared = relx.prepare(
    "run.yaml",
    feature_processors={
        "log_amount": LogAmountProcessor(),
        ("orders", "amount"): LogAmountProcessor(),
    },
)
```

YAML 只引用调用方显式提供的名称和参数，不从 YAML 任意导入 Python 类。处理器输出使用 RelX 的 `FeatureBlock` 契约；将它转成 TensorFrame、PyG 或其他模型输入留给后续模型适配层，因此自定义处理器不必继承某个模型的类。需要持久化状态的处理器应提供稳定的处理器 ID、参数和状态序列化方法；无法安全序列化的状态只在当前进程复用。

当前已产出 `FeaturePlan`、`FeatureOutputSpec` 和自定义 `FeatureProcessor` 接口。未显式指定语义类型的列通过稳定主键顺序分块扫描，并用固定种子的有界优先级样本调用 PyTorch Frame 的 `infer_df_stype()`；时间列优先采用 `graph.time_columns`，文本列的输出规格标记仍需模型适配器。`prepare()` 不拟合统计量、不执行训练 batch 的特征查询、`transform`、TensorFrame 组装或模型前向。后续若某处理器需要 `fit`，只能使用训练可见数据，不能全表物化；已保留 `FeatureState`/`FitContext` 契约。未来保存的本地 artifact 应记录表结构、任务、时间窗口、处理器身份及参数、数据快照标识的指纹；这只是缓存有效性信息，不恢复数据库内的 catalog version。

**当前验收：**同一列可以由默认处理路径或传入的自定义处理器生成不同 `FeatureSpec`；数值/类别/文本等策略和输出规格可检查；标签列不会进入特征计划；构图支持非连续业务键并报告悬空外键。训练可见统计量的拟合留待特征执行阶段验证。

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
    graph_scope.py        # 选择建图表
    graph_time.py         # 时间规则与时间归一化
    graph_builder.py      # Arrow 批次与 C++ 拓扑构建编排
    features/
      contracts.py        # FeatureSpec、FeaturePlan、处理器接口
      defaults.py         # 默认处理器输出规格
      planner.py          # 候选列和受控样本类型推断
  sampling/contracts.py   # 后续采样器输出协议
  batch/contracts.py      # 后续批次准备协议
  sampler/                # 后续采样执行
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

## 下一步实施顺序（2026-11-19）

连接、schema 发现、全局配置解析和任务校验已有实现。安装与打包暂缓；下一阶段先固定数据准备的语义，再优化计算路径：

1. **核对图的数据范围与时间语义。**平级 `graph.exclude_tables`、`time_columns`、`static_tables` 与预测时刻边界已实现；当前任务表自动排除。`rel-f1` 的 train/val/test 样本已通过 SQL 对照，下一步扩大数据集覆盖并在 MySQL 上复核。
2. **补全 C++ `GraphIndex`。**Arrow 分块读取、单列/组合业务键映射、正反向 CSC、节点/边时间及可见性过滤已实现；下一步处理指向非身份唯一键的外键、大图内存控制与构建统计。以小规模 Python 参考算法验证边界语义。
3. **实现 `FeaturePlan` 与处理器契约。**候选列过滤、语义类型、显式覆盖、受控样本推断及自定义处理器接口已完成。离线 `FeatureStore` 已按训练可见时间拟合数值、类别和时间状态；自定义处理器在批次边界运行。后续需对齐更多 RelBench 处理器、持久化状态及更严格的归纳式拟合策略。
4. **组合 `PreparedDataset` 和训练闭环。**`relx.prepare(config)` 已返回 schema、任务、GraphIndex、FeaturePlan 和准备报告；`relx.run(config)` 已完成单跳或多跳实体任务的采样、批次特征处理、训练、验证、测试预测和 JSON 报告。后续仍需推荐/多标签任务与更多模型适配器，并测量完整训练性能。
5. **补齐安装与公共入口。**`relx.prepare()` 已有源码入口；在准备流程和数据接口稳定后整理打包配置，验证安装后的导入和调用。

当前任务校验会在内存中保留每个样本的 split 映射；大任务表在上述流程正确后，再改为磁盘索引或数据库端划分，避免影响本阶段图与特征契约的确定。

### RelX 自有 C++ 数据准备核心

这是对 `exps/relconnector_v2` 的重构迁移：保留任务、构图、采样、特征读取和训练的业务语义，重新设计通用数据结构及接口。AGNES 只提供“C++ 扩展由 Python 调用”的接口形式参考；不移植其构图、采样算法或数据布局。

v2 的 `graph/builder.py` 构建无特征 CSC 图，但要求保留行的 ID 是从零开始的连续整数，使用 SQLite 的 `julianday()` 处理时间，外键路径也以单列整数为中心。这些限制在 RelX 中需要由通用键映射、数据库无关的时间归一化和组合外键支持替代。v2 的 `sampling/pyg_lib.py` 已调用 pyg-lib 原生算子，因此新 C++ 采样器应在语义完整和性能测试后再替换，不能假定重写就更快。

分层接口：`config`/`task` 由 Python 负责配置和校验；读取层向 `cpp/relx/` 的 C++ 构建器提供 Arrow RecordBatch，优先通过现有 ConnectorX 读取器的 Arrow stream 传递，避免先转换为 Pandas。构图和采样核心不处理数据库方言。当前 pybind11 绑定暴露构图、实验性快照流水线以及在线 `NativeOnlinePipeline`。正式训练路径由 Python 调用 pyg-lib 采样，然后由 C++ worker 按 batch 查询 SQLite 或 MySQL 并处理默认特征；模型及反向传播仍在 Python/PyTorch 中。

训练循环通过 `next_batch()` 接收完整 batch。旧全表特征快照仅是性能原型：准备阶段由 Python 读取一次原始特征，并注册到共享的 C++ 列存储，由原生线程按采样 ID 收集；它违反正式训练必须按 batch 回源数据库的约束。当前 SQLite 和 MySQL 在线路径由原生 worker 在每批 pyg-lib 采样后，以预编译参数 SQL 查询原始特征，再处理并返回完整 batch。`connectorx-cpp` 的直接读取已验证可行，但其当前错误与参数接口尚不能直接用于正式 worker；因此在线路径分别使用 SQLite C 客户端和 MySQL C 客户端。PostgreSQL 尚需独立原生适配器。

原生在线读取接口需要负责参数化查询、标识符转义、NULL/时间/文本类型映射、分块扫描、连接池或每 worker 独立连接，以及一致的只读快照。ConnectorX 的公开 Python API 输出 Arrow stream；仓库中还有以 Rust `cdylib` 实现的 `connectorx-cpp` C ABI，但在用于生产前需核对编译分发、错误传播、内存所有权和 SQL 参数绑定。该 ABI 当前在部分读取错误上会 panic，不能未经封装直接作为稳定的 RelX 接口。SQL 方言相关的查询生成仍需隔离，构图和采样核心不得依赖 `julianday()` 或其他特定数据库函数。直接读库减少了 Python/Pandas 中间转换，但数据库查询及网络延迟可能主导总耗时；用 v2 的 SQL 读取基准衡量收益。

**阶段进度（2026-11-19）：**`conn.iter_arrow_query()` 已能从 ConnectorX 直接提供 Arrow RecordBatch；C++ 扩展已实现组合业务键、正反向 CSC、UTC 微秒时间数组与 `visible_neighbors()` 过滤。SQLite 小库通过 ConnectorX/Pandas 两条读取路径验证。C++ 项目源码位于 `cpp/relx/`，测试位于 `tests/`，`src/relx/` 只保留项目 Python 代码和原生模块接口。基于 ConnectorX 提交 `c52b20c` 的独立 `connectorx-cpp` SQLite 构建与 C++ 查询验证成功；验证代码和第三方构建产物保留在被 Git 忽略的 `build/`。上游 C ABI 目前只接收 SQL 字符串，部分读取错误通过 panic 传播；正式在线读取需要先建立安全的错误返回与参数化查询接口，或改用相应数据库客户端。后续仍需非身份唯一键目标外键及大规模内存测试。

`rel-f1` SQLite 源库的只读构图已验证：显式排除旧 `_relconnector_*` catalog 表与其他任务表后，得到 9 张业务表、26 种正反向关系；旧 catalog 表只存在于实验数据，不应作为图节点。`tests/test_relx_f1_graph_compare.py` 用 5 个真实任务样本（train/val/test）和两种时间边界，将 `results`、`qualifying`、`standings` 的一跳邻居共 30 组结果逐项对照独立 SQL；另对 `qualifying → races` 的双端点时间做 2 组对照，全部一致。同日记录在 `strict_before` 下不可见，在 `at_or_before` 下可见。该检查尚未覆盖 MySQL 构图对照或完整多跳采样结果。

自定义 Python `FeatureProcessor` 在成批数据上运行；默认 GloVe 文本处理已在 C++ worker 中完成，支持内存和可选磁盘缓存。当前原生线程池已有确定的批次随机种子、有界队列、异常传播和停止信号；下一步测量取消延迟及大批次缓冲区内存，不使用多进程。每阶段用 v2 结果做语义对照，并用 `rel-f1` 测量吞吐、内存和准备耗时。

### 推荐的 C++ 迁移顺序

**约束修订（2026-11-20）：**训练 batch 的原始特征必须从关系型数据库按需读取；已命中的部分缓存可以免去对应查询。当前 `FeatureStore` 全表快照原型不符合该约束，不能成为正式训练路径。最终流水线改为 Python 调用 pyg-lib 采样，再由 C++ assembler worker 按 sampled ID 读库、处理特征并回传完整 batch。接口、缓存边界和迁移顺序见 [ONLINE_FEATURE_PIPELINE.md](ONLINE_FEATURE_PIPELINE.md)。此前快照基准仅用于定位计算瓶颈，不代表在线训练加速。

**在线训练路径（2026-11-20，MySQL 更新于 2026-11-23）：**`relx.run()` 已改用 pyg-lib 多跳采样和 `OnlineFeatureStore`；Python 将 sampled ID 提交给 `NativeOnlinePipeline`，C++ worker 按业务键批量查询原数据库、处理默认特征和 GloVe 文本，并按序返回 batch。每个 worker 复用 SQLite 或 MySQL 连接，节点特征缓存有字节上限；完成队列按 batch 数和载荷字节数限流。拟合状态以数据库结果块流式计算，仅保留身份键、图索引、统计量与词表。原生在线读取支持文件型 SQLite 和 MySQL，时间特征需 UTC；自定义处理器的原始列由 C++ 读库，再在批次边界运行用户的 Python `transform()`。RDL GraphSAGE 和 RelGNN backbone 已在 rel-f1 SQLite 上各跑通 1 epoch，MySQL 5.7 隔离实例已跑通小型 RDL 任务。PostgreSQL 原生读取、严格的数据库快照/缓存版本和完整同语义 v2 性能对照仍待完成。

**训练闭环初版历史记录（2026-11-19）：**`SqlEntitySeedSource` 将已校验任务行映射为图节点 ID 和 UTC 预测时间；`NativeOneHopSampler` 调用 C++ `sample_neighbors()`，按 seed 时间过滤并在 C++ 内以 `std::thread` 并行，绑定期间释放 GIL。当时 `FeatureStore` 生成离线原始特征快照，数值/类别/时间状态只用训练时刻可见的行拟合，批次边界执行默认或自定义处理器。`SynchronousBatchIterator.next_batch()`、单跳关系基线模型和旧 `relx.run()` 曾串成训练、验证、测试预测及 JSON 报告；`rel-f1` 的 1 epoch 实跑完成。该阶段批次特征读取与变换由 Python 串行完成，现已由上述在线路径取代。模型默认编码与 RelBench TensorFrame 编码尚未对齐，当前哈希文本编码只属于显式选择的基线模型。

**多跳与快照原型历史记录（2026-11-20）：**`NativeMultiHopSampler` 支持每跳一个 fanout、种子独立子图、同一 seed 内节点去重、逐跳计数和固定 seed 时间的早于等于过滤；`fanout=-1` 时的节点、边和各跳计数已与 pyg-lib 在异构时序小图上逐项对齐。随机策略均为 uniform、无放回，但不同 PRNG 不保证抽到相同节点。当时默认六类特征在准备阶段注册到共享的 C++ 列缓存，`NativeBatchPipeline` 在 C++ 内用采样线程、特征线程池和有界队列衔接两阶段；该原型不用于当前训练。`feature_workers`、`feature_native_workers`、`prefetch_batches` 在 `training` 配置。`multi_hop_baseline` 会从外层向 seed 聚合逐跳边。完整在线训练吞吐和加速仍需测量。

**rel-f1 批次基准（2026-11-20）：**完整 9 表/26 关系图、`driver-dnf` 的 90 个训练 batch、两跳 `[16, 8]` 下，C++ 流水线 0.486 秒；相同采样和特征语义的 Python 逐批变换 6.040 秒，旧 Python 线程调度加原生特征缓存 0.533 秒。pyg-lib 原始两跳采样 0.138 秒，当前 C++ 原始采样 0.480 秒，采样是现阶段主要瓶颈。测量范围、不同随机输出量及不能直接推断 v2 完整训练加速的限制见 [REL_F1_CPP_PIPELINE_BENCHMARK.md](REL_F1_CPP_PIPELINE_BENCHMARK.md)。

```yaml
model:
  name: multi_hop_baseline
  params:
    num_neighbors: [16, 8]
    workers: 4
    channels: 32
    text_mode: hash
training:
  batch_size: 1024
  epochs: 20
  feature_workers: 2
  feature_native_workers: 2
  prefetch_batches: 4
  prefetch_bytes: 268435456
  feature_cache_bytes: 67108864
```

1. **验收图语义并完善原生索引。**业务表时间列、静态表规则、预测边界和 `GraphIndex` 时间数组已实现；`rel-f1` 的 SQLite ConnectorX/Pandas 两条路径得到相同的节点、CSC 和时间数组。下一步对照 MySQL、处理非身份唯一键外键、大图有界内存或分段索引，并统计准备耗时和峰值内存。后续时间采样器必须继续使用节点/边时间过滤。
2. **衡量多跳采样性能。**`SampledSubgraph` 已定义局部边、逐 seed 归属和逐跳计数；C++ 多跳穷举语义已与 pyg-lib 对齐，随机结果在 C++ worker 数量变化时保持一致。下一步在相同图、seed、fanout 和时间过滤条件下，对照 v2/pyg-lib 的每秒 seed 数、批次延迟与峰值内存，再针对热点优化。
3. **优化在线查询。**SQLite 批量参数查询、worker 连接复用、部分节点缓存和 C++ GloVe 已接通；下一步测量 SQL、编码、队列等待和完整训练的分段成本，并针对真实瓶颈优化。
4. **完善原生调度。**`NativeOnlinePipeline` 已有 C++ 多 worker、按 batch 数及结果字节限流、按序消费及错误传播。下一步验证取消延迟和大批次峰值内存；用户自定义 Python 处理器仍需在批次边界单独运行。
5. **扩展数据库适配器。**`NativeFeatureReader` 抽象已建立，SQLite 和 MySQL 使用参数化批量查询。下一步在 rel-f1 MySQL 数据库核对完整训练输出，再接入 PostgreSQL 原生客户端并比较 v2 的完整训练吞吐。ConnectorX 实验 C ABI 仍需安全错误及参数接口才能用于正式在线 worker。
