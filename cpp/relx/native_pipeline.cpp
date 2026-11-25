#include "native_pipeline.h"

#include <stdexcept>
#include <utility>

namespace relx {

NativeBatchPipeline::NativeBatchPipeline(
    const GraphIndex& graph, const NativeFeatureStore& features,
    std::vector<int64_t> fanouts, bool multi_hop, int64_t sampling_workers,
    int64_t feature_workers, int64_t feature_native_workers,
    size_t max_inflight)
    : graph_(graph), features_(features), fanouts_(std::move(fanouts)),
      multi_hop_(multi_hop), sampling_workers_(sampling_workers),
      feature_native_workers_(feature_native_workers),
      max_inflight_(max_inflight) {
  if (fanouts_.empty() || sampling_workers < 1 || feature_workers < 1 ||
      feature_native_workers < 1 || max_inflight == 0) {
    throw std::invalid_argument("Pipeline fanouts and worker counts must be positive");
  }
  if (!multi_hop_ && fanouts_.size() != 1) {
    throw std::invalid_argument("One-hop pipeline needs exactly one fanout");
  }
  for (const auto fanout : fanouts_) {
    if (fanout < -1) throw std::invalid_argument("Fanout must be >= -1");
  }
  try {
    sampler_ = std::thread(&NativeBatchPipeline::sample_loop, this);
    for (int64_t index = 0; index < feature_workers; ++index) {
      feature_threads_.emplace_back(&NativeBatchPipeline::feature_loop, this);
    }
  } catch (...) {
    close();
    throw;
  }
}

NativeBatchPipeline::~NativeBatchPipeline() { close(); }

void NativeBatchPipeline::submit(uint64_t sequence, std::string seed_table,
                                 std::vector<int64_t> seed_ids,
                                 std::vector<int64_t> cutoffs_us,
                                 uint64_t random_seed) {
  if (seed_ids.size() != cutoffs_us.size()) {
    throw std::invalid_argument("Seed IDs and cutoffs must have equal length");
  }
  std::unique_lock<std::mutex> lock(mutex_);
  ready_.wait(lock, [&] {
    return stopping_ || error_ || input_finished_ ||
           submitted_ - consumed_ < max_inflight_;
  });
  if (error_) std::rethrow_exception(error_);
  if (stopping_ || input_finished_) {
    throw std::runtime_error("Pipeline is closed for input");
  }
  if (sequence != submitted_) {
    throw std::invalid_argument("Batch sequence must be contiguous from zero");
  }
  seeds_.push_back({sequence, std::move(seed_table), std::move(seed_ids),
                    std::move(cutoffs_us), random_seed});
  ++submitted_;
  ready_.notify_all();
}

void NativeBatchPipeline::finish() {
  std::lock_guard<std::mutex> lock(mutex_);
  input_finished_ = true;
  ready_.notify_all();
}

std::optional<NativeBatchResult> NativeBatchPipeline::next() {
  std::unique_lock<std::mutex> lock(mutex_);
  ready_.wait(lock, [&] {
    return stopping_ || error_ || completed_.count(consumed_) ||
           (input_finished_ && consumed_ == submitted_);
  });
  if (error_) std::rethrow_exception(error_);
  if (stopping_) return std::nullopt;
  const auto found = completed_.find(consumed_);
  if (found == completed_.end()) return std::nullopt;
  NativeBatchResult result = std::move(found->second);
  completed_.erase(found);
  ++consumed_;
  ready_.notify_all();
  return result;
}

void NativeBatchPipeline::close() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
    ready_.notify_all();
  }
  if (sampler_.joinable()) sampler_.join();
  for (auto& worker : feature_threads_) {
    if (worker.joinable()) worker.join();
  }
  feature_threads_.clear();
}

void NativeBatchPipeline::fail(std::exception_ptr error) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!error_) error_ = error;
  stopping_ = true;
  ready_.notify_all();
}

void NativeBatchPipeline::sample_loop() {
  try {
    while (true) {
      SeedJob job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] {
          return stopping_ || !seeds_.empty() || input_finished_;
        });
        if (stopping_ || seeds_.empty()) break;
        job = std::move(seeds_.front());
        seeds_.pop_front();
      }
      SampledBatch sample = multi_hop_
          ? graph_.sample_multi_hop(job.seed_table, job.seed_ids,
                                    job.cutoffs_us, fanouts_, job.random_seed,
                                    sampling_workers_)
          : graph_.sample_one_hop(job.seed_table, job.seed_ids,
                                  job.cutoffs_us, fanouts_.front(),
                                  job.random_seed, sampling_workers_);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) break;
        sampled_.push_back({job.sequence, std::move(sample)});
        ready_.notify_all();
      }
    }
  } catch (...) {
    fail(std::current_exception());
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    sampler_finished_ = true;
    ready_.notify_all();
  }
}

void NativeBatchPipeline::feature_loop() {
  try {
    while (true) {
      FeatureJob job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] {
          return stopping_ || !sampled_.empty() || sampler_finished_;
        });
        if (stopping_ || sampled_.empty()) break;
        job = std::move(sampled_.front());
        sampled_.pop_front();
      }
      auto gathered = features_.gather(job.sample.node_ids,
                                       feature_native_workers_);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) break;
        completed_.emplace(job.sequence, NativeBatchResult{
            job.sequence, std::move(job.sample), std::move(gathered)});
        ready_.notify_all();
      }
    }
  } catch (...) {
    fail(std::current_exception());
  }
}

}  // namespace relx
