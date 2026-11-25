#pragma once

#include "graph_index.h"
#include "native_features.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace relx {

struct NativeBatchResult {
  uint64_t sequence;
  SampledBatch sample;
  FeatureBatch features;
};

class NativeBatchPipeline {
 public:
  NativeBatchPipeline(const GraphIndex& graph, const NativeFeatureStore& features,
                      std::vector<int64_t> fanouts, bool multi_hop,
                      int64_t sampling_workers, int64_t feature_workers,
                      int64_t feature_native_workers, size_t max_inflight);
  ~NativeBatchPipeline();

  NativeBatchPipeline(const NativeBatchPipeline&) = delete;
  NativeBatchPipeline& operator=(const NativeBatchPipeline&) = delete;

  void submit(uint64_t sequence, std::string seed_table,
              std::vector<int64_t> seed_ids, std::vector<int64_t> cutoffs_us,
              uint64_t random_seed);
  void finish();
  std::optional<NativeBatchResult> next();
  void close();

 private:
  struct SeedJob {
    uint64_t sequence;
    std::string seed_table;
    std::vector<int64_t> seed_ids;
    std::vector<int64_t> cutoffs_us;
    uint64_t random_seed;
  };

  struct FeatureJob {
    uint64_t sequence;
    SampledBatch sample;
  };

  void sample_loop();
  void feature_loop();
  void fail(std::exception_ptr error);

  const GraphIndex& graph_;
  const NativeFeatureStore& features_;
  const std::vector<int64_t> fanouts_;
  const bool multi_hop_;
  const int64_t sampling_workers_;
  const int64_t feature_native_workers_;
  const size_t max_inflight_;

  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<SeedJob> seeds_;
  std::deque<FeatureJob> sampled_;
  std::map<uint64_t, NativeBatchResult> completed_;
  std::thread sampler_;
  std::vector<std::thread> feature_threads_;
  std::exception_ptr error_;
  uint64_t submitted_ = 0;
  uint64_t consumed_ = 0;
  bool input_finished_ = false;
  bool sampler_finished_ = false;
  bool stopping_ = false;
};

}  // namespace relx
