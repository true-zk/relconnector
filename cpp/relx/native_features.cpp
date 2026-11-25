#include "native_features.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace relx {
namespace {

struct GatherJob {
  const FeatureColumn* column;
  const std::vector<int64_t>* ids;
  NativeFeatureBlock* output;
  size_t start;
  size_t stop;
};

void gather_range(const GatherJob& job) {
  const auto& column = *job.column;
  const auto& ids = *job.ids;
  auto& output = *job.output;
  for (size_t index = job.start; index < job.stop; ++index) {
    const auto id = ids[index];
    if (id < 0 || static_cast<size_t>(id) >= column.size()) {
      throw std::out_of_range("Feature node ID is out of range");
    }
    const auto row = static_cast<size_t>(id);
    if (column.kind == FeatureKind::kCategorical) {
      output.integers[index] = column.integer[row];
    } else if (column.kind == FeatureKind::kMulticategorical) {
      const auto start = static_cast<size_t>(column.offsets[row]);
      const auto stop = static_cast<size_t>(column.offsets[row + 1]);
      std::copy(column.integer.begin() + start,
                column.integer.begin() + stop,
                output.integers.begin() + output.offsets[index]);
    } else if (column.kind == FeatureKind::kEmbedding) {
      const auto width = static_cast<size_t>(column.width);
      std::copy_n(column.embedding.begin() + row * width, width,
                  output.floats.begin() + index * width);
    } else if (column.kind == FeatureKind::kText) {
      output.text[index] = column.text[row];
    } else if (column.kind == FeatureKind::kNumeric) {
      const auto value = column.numeric[row];
      const bool missing = !std::isfinite(value);
      output.missing[index] = static_cast<uint8_t>(missing);
      output.floats[index] = static_cast<float>(
          ((missing ? column.mean : value) - column.mean) / column.std);
    } else {
      const auto value = column.integer[row];
      const bool missing = value == std::numeric_limits<int64_t>::min();
      output.missing[index] = static_cast<uint8_t>(missing);
      output.floats[index] = missing
          ? 0.0F
          : static_cast<float>(
                (static_cast<double>(value) - column.mean) / column.std);
    }
  }
}

}  // namespace

void NativeFeatureStore::add(const std::string& table,
                             const std::string& column,
                             FeatureColumn values) {
  if (table.empty() || column.empty()) {
    throw std::invalid_argument("Feature table and column must not be empty");
  }
  if (!columns_[table].emplace(column, std::move(values)).second) {
    throw std::invalid_argument("Duplicate native feature: " + table + "." +
                                column);
  }
}

void NativeFeatureStore::add_numeric(const std::string& table,
                                     const std::string& column,
                                     std::vector<double> values, double mean,
                                     double std) {
  if (!std::isfinite(mean) || !std::isfinite(std) || std <= 0.0) {
    throw std::invalid_argument("Numeric feature mean/std must be finite; std > 0");
  }
  FeatureColumn feature;
  feature.kind = FeatureKind::kNumeric;
  feature.numeric = std::move(values);
  feature.mean = mean;
  feature.std = std;
  add(table, column, std::move(feature));
}

void NativeFeatureStore::add_categorical(const std::string& table,
                                         const std::string& column,
                                         std::vector<int64_t> values) {
  FeatureColumn feature;
  feature.kind = FeatureKind::kCategorical;
  feature.integer = std::move(values);
  add(table, column, std::move(feature));
}

void NativeFeatureStore::add_timestamp(const std::string& table,
                                       const std::string& column,
                                       std::vector<int64_t> values,
                                       double mean, double std) {
  if (!std::isfinite(mean) || !std::isfinite(std) || std <= 0.0) {
    throw std::invalid_argument("Timestamp feature mean/std must be finite; std > 0");
  }
  FeatureColumn feature;
  feature.kind = FeatureKind::kTimestamp;
  feature.integer = std::move(values);
  feature.mean = mean;
  feature.std = std;
  add(table, column, std::move(feature));
}

void NativeFeatureStore::add_multicategorical(
    const std::string& table, const std::string& column,
    std::vector<int64_t> values, std::vector<int64_t> offsets) {
  if (offsets.empty() || offsets.front() != 0 ||
      offsets.back() != static_cast<int64_t>(values.size()) ||
      !std::is_sorted(offsets.begin(), offsets.end())) {
    throw std::invalid_argument("Invalid multicategorical offsets");
  }
  FeatureColumn feature;
  feature.kind = FeatureKind::kMulticategorical;
  feature.integer = std::move(values);
  feature.offsets = std::move(offsets);
  add(table, column, std::move(feature));
}

void NativeFeatureStore::add_embedding(const std::string& table,
                                       const std::string& column,
                                       std::vector<float> values,
                                       int64_t width) {
  if (width <= 0 || values.size() % static_cast<size_t>(width) != 0) {
    throw std::invalid_argument("Embedding width must divide the value count");
  }
  FeatureColumn feature;
  feature.kind = FeatureKind::kEmbedding;
  feature.embedding = std::move(values);
  feature.width = width;
  add(table, column, std::move(feature));
}

void NativeFeatureStore::add_text(const std::string& table,
                                  const std::string& column,
                                  std::vector<std::string> values) {
  FeatureColumn feature;
  feature.kind = FeatureKind::kText;
  feature.text = std::move(values);
  add(table, column, std::move(feature));
}

FeatureBatch NativeFeatureStore::gather(
    const std::unordered_map<std::string, std::vector<int64_t>>& node_ids,
    int64_t workers) const {
  if (workers < 1) {
    throw std::invalid_argument("Feature workers must be positive");
  }
  FeatureBatch result;
  std::vector<GatherJob> jobs;
  constexpr size_t kChunk = 512;
  for (const auto& [table, ids] : node_ids) {
    const auto found = columns_.find(table);
    if (found == columns_.end()) continue;
    auto& outputs = result[table];
    for (const auto& [name, column] : found->second) {
      auto& output = outputs[name];
      output.kind = column.kind;
      output.rows = static_cast<int64_t>(ids.size());
      output.width = column.width;
      if (column.kind == FeatureKind::kCategorical) {
        output.integers.resize(ids.size());
      } else if (column.kind == FeatureKind::kMulticategorical) {
        output.offsets.reserve(ids.size() + 1);
        output.offsets.push_back(0);
        for (const auto id : ids) {
          if (id < 0 || static_cast<size_t>(id) >= column.size()) {
            throw std::out_of_range("Feature node ID is out of range");
          }
          const auto row = static_cast<size_t>(id);
          output.offsets.push_back(
              output.offsets.back() + column.offsets[row + 1] -
              column.offsets[row]);
        }
        output.integers.resize(static_cast<size_t>(output.offsets.back()));
      } else if (column.kind == FeatureKind::kEmbedding) {
        output.floats.resize(ids.size() * static_cast<size_t>(column.width));
      } else if (column.kind == FeatureKind::kText) {
        output.text.resize(ids.size());
      } else {
        output.floats.resize(ids.size());
        output.missing.resize(ids.size());
      }
      for (size_t start = 0; start < ids.size(); start += kChunk) {
        jobs.push_back({&column, &ids, &output, start,
                        std::min(start + kChunk, ids.size())});
      }
    }
  }
  std::atomic<size_t> next{0};
  std::exception_ptr error;
  std::mutex error_mutex;
  const auto work = [&]() {
    try {
      while (true) {
        const auto index = next.fetch_add(1);
        if (index >= jobs.size()) break;
        gather_range(jobs[index]);
      }
    } catch (...) {
      std::lock_guard<std::mutex> guard(error_mutex);
      if (!error) error = std::current_exception();
    }
  };
  const auto count = std::min<size_t>(
      {static_cast<size_t>(workers), jobs.size(),
       std::max(1U, std::thread::hardware_concurrency())});
  std::vector<std::thread> threads;
  threads.reserve(count > 0 ? count - 1 : 0);
  try {
    for (size_t index = 1; index < count; ++index) {
      threads.emplace_back(work);
    }
  } catch (...) {
    for (auto& thread : threads) thread.join();
    throw;
  }
  work();
  for (auto& thread : threads) thread.join();
  if (error) std::rethrow_exception(error);
  return result;
}

}  // namespace relx
