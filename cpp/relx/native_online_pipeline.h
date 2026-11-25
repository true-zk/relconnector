#pragma once

#include "online_feature_reader.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

namespace relx {

struct OnlineBatchResult {
  uint64_t sequence;
  FeatureBatch features;
};

class NativeOnlinePipeline {
 public:
  NativeOnlinePipeline(const NativeFeatureReader& reader,
                       size_t workers, size_t max_inflight,
                       size_t max_ready_bytes = 256 * 1024 * 1024);
  ~NativeOnlinePipeline();

  NativeOnlinePipeline(const NativeOnlinePipeline&) = delete;
  NativeOnlinePipeline& operator=(const NativeOnlinePipeline&) = delete;

  void submit(uint64_t sequence,
              std::unordered_map<std::string, std::vector<int64_t>> node_ids);
  void finish();
  void abort();
  std::optional<OnlineBatchResult> next();
  void close();

 private:
  struct Job {
    uint64_t sequence;
    std::unordered_map<std::string, std::vector<int64_t>> node_ids;
  };

  void worker_loop();
  void fail(std::exception_ptr error);
  static size_t payload_bytes(const FeatureBatch& batch);

  const NativeFeatureReader& reader_;
  const size_t max_inflight_;
  const size_t max_ready_bytes_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<Job> jobs_;
  std::map<uint64_t, std::pair<FeatureBatch, size_t>> completed_;
  size_t ready_bytes_ = 0;
  std::vector<std::thread> workers_;
  std::exception_ptr error_;
  uint64_t submitted_ = 0;
  uint64_t consumed_ = 0;
  bool input_finished_ = false;
  bool stopping_ = false;
};

}  // namespace relx
