# Database readers

该包只负责从已落盘的 SQL 数据库读取数据：

- `BaseDatabaseReader` 定义全量查询、表读取、任务 split 读取和 RelBench
  `Database` 重建接口；
- `PandasDatabaseReader` 使用 `pandas.read_sql_query`；
- `ConnectorXDatabaseReader` 使用 Connector-X；
- `create_reader()` 根据名称选择实现；
- `iter_query()` 是后续分批/流式读取扩展点，本阶段只允许单个全量 batch。

模块职责：

- `base.py`：读取器基类和共享读库流程。
- `pandas.py`：pandas 实现。
- `connectorx.py`：Connector-X 实现。
- `factory.py`：实现注册与构造。
- `catalog.py`：数据库 catalog 数据结构。
- `decoding.py`：从 SQL 存储类型恢复 pandas 类型。

```python
from relconnector.connector import create_reader

reader = create_reader("connector-x", "data/relbench/rel-f1.sqlite")
tables = reader.read_all_tables()
database = reader.read_relbench_database()
```
