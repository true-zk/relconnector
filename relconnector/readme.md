# relconnector

项目分为两个独立部分：

- [`data/`](data/README.md)：RelBench 下载与 SQLite 数据准备脚本；
- [`relconnector/connector/`](relconnector/connector/README.md)：项目运行时的
  pandas / Connector-X 数据库读取器。

快速验证：

```bash
cd /workspace/relconnector
../.venv/bin/python -m unittest discover -s data/tests -v
```
