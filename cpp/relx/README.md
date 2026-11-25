# Native graph builder

The C++ project sources live here, outside `src/relx`. The Python package
contains the graph orchestration and native interface; local builds place the
extension binary there.

`GraphBuilder` consumes Arrow RecordBatch objects through the Arrow C Data
interface. ConnectorX can provide these batches without a Pandas conversion;
other readers use the `BaseDatabaseReader.iter_arrow_query()` fallback.

For local development, install the `native` optional dependencies and run
`python scripts/build_relx_native.py` from the repository root. The resulting
extension is placed beside `src/relx/__init__.py` and ignored by Git. Import
`relx.preprocess.build_graph` with `PYTHONPATH=src` to use it.
Native SQLite feature reads also need the SQLite C development header and
library. Set `SQLITE3_INCLUDE_DIR` and, when needed, `SQLITE3_LIBRARY` before
building if they are outside the system compiler paths.

The current index contains node ID maps and forward/reverse CSC topology. It
supports integer, boolean, UTF-8, and binary identity or foreign keys, including
composite keys. It counts null and dangling foreign keys. Timed tables receive
UTC microsecond node times, and an edge becomes available when both endpoints
are available. `visible_neighbors()` applies the configured strict or inclusive
prediction boundary. The selected tables come from `graph.exclude_tables`; the
current task table is omitted.

`GraphIndex.sample_one_hop()` and `sample_multi_hop()` take one cutoff per seed
and build node and edge arrays in C++. Multi-hop sampling keeps seeds disjoint,
deduplicates nodes within each seed, reports counts for every hop, and uses the
seed cutoff at every hop. Neighbor selection runs in C++ threads with the GIL
released. The default time boundary includes records at the seed time.

The current `NativeFeatureStore` prototype holds the six default feature types once for all batch
workers. `NativeBatchPipeline` owns the sampling thread, feature worker threads,
and bounded queues between them. A completed sample goes directly from the C++
sampler to a C++ feature worker; the Python adapter supplies seeds and receives
only completed batches. Results remain in seed order. The native bindings
release the GIL while submitting, waiting, or closing. Text strings are
converted to Python objects when returned because the current model adapter
consumes strings. Custom Python processors use a serialized fallback after the
native batch is returned. No process pool is used. This full feature snapshot
violates RelX's online database training contract and is not the target runtime.
The target assembler must query uncached sampled node features from the source
database inside its worker stage; see `src/agentdocs/ONLINE_FEATURE_PIPELINE.md`.

The current online SQLite path uses `NativeOnlinePipeline`: Python samples with
pyg-lib and submits node IDs; C++ workers use `SqliteFeatureReader` to query
source rows with bound identity keys and transform built-in features. It has
bounded in-flight batches and returns in submission order. Each worker holds a
read-only SQLite connection and issues parameterized batch queries for at most
500 uncached identity keys at a time. An optional byte-bounded worker-local row
cache stores previously read features. The completed-result queue has count
and byte limits. `MySqlFeatureReader` uses one MySQL C client connection per
worker and parameterized derived-table queries for up to 128 uncached keys at
a time. The local build includes it when `mysql_config` is available; set
`MYSQL_CONFIG` to an explicit path or `RELX_WITH_MYSQL=0` to disable it.
`NativeFeatureReader` / `NativeFeatureSession` are the interfaces for further
database adapters. Only UTC timestamp features are supported. Native workers
read raw custom columns as well; their user-defined
Python transforms run at the batch boundary. The C++ GloVe encoder supports
an in-memory LRU and an optional SQLite text disk cache.

The raw CSC arrays retain future edges; samplers use their time arrays to avoid
future-data leakage. Non-identity unique-key foreign targets, a persistent
index, and PostgreSQL native reading are subsequent stages. An isolated
C++ probe successfully read SQLite
through ConnectorX's `connectorx-cpp` C ABI. That upstream ABI does not yet
provide safe error and parameter handling for RelX's online batch reader, so
it is not linked into this extension.
