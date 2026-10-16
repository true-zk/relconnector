# Four training architectures

## (a) Batch｜全量物化训练架构

训练前生成全部特征；训练时从常驻 TensorFrame 切片。虚线框表示执行范围。

Source: benchmark/baseline.py; baseline/batch_baseline/feature_store.py. Full table reads and EagerFeatureStore.materialize precede training. Graph topology is resident. The async baseline reuses sampler/feature threads, but feature fetch is an in-memory TensorFrame slice. The node/edge/time/label information in SamplePlan is preserved during assembly. Feature enc. in the GPU block is the trainable model encoder, distinct from frozen GloVe/statistical preprocessing. Task-seed queue is omitted here for clarity.
中文：全量物化不等于全批量梯度下降；本架构仍采用邻居采样和mini-batch训练。

## (b) Vanilla｜按需读取的在线训练架构

单进程内异步采样、特征线程与训练；已有 raw cache，但缺少后续文本与编码特征复用。

Source: baseline/vanilla_baseline/api.py and runtime/async_pipeline.py. The default async path already has seed/sampler/feature threads and bounded queues. It is not a fully serial or cache-free implementation. SqlFeatureFetcher owns a bounded raw block cache. GloveTextEmbedder is constructed without the later text-cache options; no encoded-feature cache is configured. The repeated SQL-tables cylinder denotes the same database, not another copy. The initial schema statistics configure the static feature encoder. Recommendation branch expansion and error channels are omitted for clarity.

## (c) v1｜单进程优化：窗口复用与分层缓存

在单进程线程流水线中合并请求，并复用 raw、text 与 encoded 特征；训练按序消费。

Source: relconnector/api.py, runtime/async_pipeline.py, features/sql.py and features/assembler.py. This depicts the current legacy/v1 architecture, including later direct-text/thread refinements; it is not a claim that every component was present in the historical performance snapshot. Feature windows do not cross epochs. Requested node rows are deduplicated and restored using inverse mappings. Encoded-cache hits reuse requested encoded rows; missing rows follow raw/text/encoding paths. Encoder workers are threads sharing process-local caches, not v2 assembly processes. Persisted schema/topology artifacts are initialization caches, not a per-text disk cache.

## (d) v2｜多进程与自适应缓存训练架构

assemble 成为供给瓶颈后，v2 将采样、组装、训练分进程执行。

Source: relconnector_v2/process_api.py, runtime/process_pipeline.py, features/process_worker.py, features/shared_text.py, features/shared_encoded.py, features/disk_text.py. The seed feeder is a thread in the main process; its input stream is represented at the left. Sampling is a child process. Assembly workers own SQL readers, raw caches, text encoder and local state. Shared text vectors have reservation-based coordination; the adaptive local index stores only text-to-slot metadata. Encoded-feature cache is optionally shared through a manager or local to each worker; HM's local 4 GiB is an explicit tested profile, not an automatic learned choice. Disk text persistence and initialization artifacts are separate stores. CPU topology is resident; do not claim a single shared physical graph allocation. CPU initialization includes budgeted exact numerical statistics. Training may use ready-order or strict-order with an epoch gate. Queue byte caps / in-flight windows impose backpressure. Diagram excludes error-reporting paths and checkpoint persistence for readability.