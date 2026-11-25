#include "native_online_pipeline.h"

#include <stdexcept>
#include <utility>

namespace relx {

NativeOnlinePipeline::NativeOnlinePipeline(
    const NativeFeatureReader& reader, size_t workers, size_t max_inflight,
    size_t max_ready_bytes)
    : reader_(reader), max_inflight_(max_inflight),
      max_ready_bytes_(max_ready_bytes) {
  if (workers == 0 || max_inflight == 0 || max_ready_bytes == 0) {
    throw std::invalid_argument("Online worker and queue limits must be positive");
  }
  try {
    for (size_t index = 0; index < workers; ++index) {
      workers_.emplace_back(&NativeOnlinePipeline::worker_loop, this);
    }
  } catch (...) {
    close();
    throw;
  }
}

NativeOnlinePipeline::~NativeOnlinePipeline() { close(); }

void NativeOnlinePipeline::submit(
    uint64_t sequence,
    std::unordered_map<std::string, std::vector<int64_t>> node_ids) {
  std::unique_lock<std::mutex> lock(mutex_);
  ready_.wait(lock, [&] {
    return stopping_ || error_ || input_finished_ ||
           submitted_ - consumed_ < max_inflight_;
  });
  if (error_) std::rethrow_exception(error_);
  if (stopping_ || input_finished_) {
    throw std::runtime_error("Online pipeline is closed for input");
  }
  if (sequence != submitted_) {
    throw std::invalid_argument("Online batch sequence must be contiguous");
  }
  jobs_.push_back({sequence, std::move(node_ids)});
  ++submitted_;
  ready_.notify_all();
}

void NativeOnlinePipeline::finish() {
  std::lock_guard<std::mutex> lock(mutex_);
  input_finished_ = true;
  ready_.notify_all();
}

void NativeOnlinePipeline::abort() {
  std::lock_guard<std::mutex> lock(mutex_);
  stopping_ = true;
  ready_.notify_all();
}

std::optional<OnlineBatchResult> NativeOnlinePipeline::next() {
  std::unique_lock<std::mutex> lock(mutex_);
  ready_.wait(lock, [&] {
    return stopping_ || error_ || completed_.count(consumed_) ||
           (input_finished_ && consumed_ == submitted_);
  });
  if (error_) std::rethrow_exception(error_);
  if (stopping_) return std::nullopt;
  auto found = completed_.find(consumed_);
  if (found == completed_.end()) return std::nullopt;
  OnlineBatchResult result{consumed_, std::move(found->second.first)};
  ready_bytes_ -= found->second.second;
  completed_.erase(found);
  ++consumed_;
  ready_.notify_all();
  return result;
}

void NativeOnlinePipeline::close() {
  abort();
  for (auto& worker : workers_) {
    if (worker.joinable()) worker.join();
  }
  workers_.clear();
}

void NativeOnlinePipeline::fail(std::exception_ptr error) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!error_) error_ = error;
  stopping_ = true;
  ready_.notify_all();
}

size_t NativeOnlinePipeline::payload_bytes(const FeatureBatch& batch) {
  size_t bytes = 0;
  for (const auto& [table, columns] : batch) {
    bytes += table.capacity();
    for (const auto& [name, block] : columns) {
      bytes += name.capacity() + sizeof(block);
      bytes += block.floats.capacity() * sizeof(float);
      bytes += block.integers.capacity() * sizeof(int64_t);
      bytes += block.missing.capacity();
      bytes += block.offsets.capacity() * sizeof(int64_t);
      for (const auto& value : block.text) bytes += sizeof(value) + value.capacity();
      for (const auto& value : block.raw) {
        bytes += sizeof(value);
        if (const auto* string = std::get_if<std::string>(&value)) {
          bytes += string->capacity();
        } else if (const auto* blob = std::get_if<RawBlob>(&value)) {
          bytes += blob->data.capacity();
        }
      }
    }
  }
  return bytes;
}

void NativeOnlinePipeline::worker_loop() {
  try {
    auto session = reader_.open_session();
    while (true) {
      Job job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] {
          return stopping_ || !jobs_.empty() || input_finished_;
        });
        if (stopping_ || jobs_.empty()) return;
        job = std::move(jobs_.front());
        jobs_.pop_front();
      }
      auto features = session->fetch(job.node_ids);
      const auto bytes = payload_bytes(features);
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] {
          return stopping_ || job.sequence == consumed_ ||
                 (bytes <= max_ready_bytes_ &&
                  ready_bytes_ <= max_ready_bytes_ - bytes);
        });
        if (stopping_) return;
        ready_bytes_ += bytes;
        completed_.emplace(job.sequence,
                           std::make_pair(std::move(features), bytes));
        ready_.notify_all();
      }
    }
  } catch (...) {
    fail(std::current_exception());
  }
}

}  // namespace relx
