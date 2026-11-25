#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace relx {

enum class FeatureKind {
  kNumeric, kCategorical, kTimestamp, kMulticategorical, kEmbedding, kText,
  kRaw, kTextEmbedding
};

struct RawBlob { std::string data; };
using RawValue = std::variant<std::monostate, int64_t, double,
                              std::string, RawBlob>;

struct FeatureColumn {
  FeatureKind kind;
  std::vector<double> numeric;
  std::vector<int64_t> integer;
  std::vector<int64_t> offsets;
  std::vector<float> embedding;
  std::vector<std::string> text;
  int64_t width = 1;
  double mean = 0.0;
  double std = 1.0;

  size_t size() const {
    switch (kind) {
      case FeatureKind::kNumeric: return numeric.size();
      case FeatureKind::kEmbedding:
        return embedding.size() / static_cast<size_t>(width);
      case FeatureKind::kText: return text.size();
      case FeatureKind::kMulticategorical: return offsets.size() - 1;
      default: return integer.size();
    }
  }
};

struct NativeFeatureBlock {
  FeatureKind kind;
  std::vector<float> floats;
  std::vector<int64_t> integers;
  std::vector<uint8_t> missing;
  std::vector<int64_t> offsets;
  std::vector<std::string> text;
  std::vector<RawValue> raw;
  int64_t rows = 0;
  int64_t width = 1;
};

using FeatureBatch =
    std::unordered_map<std::string,
                       std::unordered_map<std::string, NativeFeatureBlock>>;

class NativeFeatureStore {
 public:
  void add_numeric(const std::string& table, const std::string& column,
                   std::vector<double> values, double mean, double std);
  void add_categorical(const std::string& table, const std::string& column,
                       std::vector<int64_t> values);
  void add_timestamp(const std::string& table, const std::string& column,
                     std::vector<int64_t> values, double mean, double std);
  void add_multicategorical(const std::string& table,
                            const std::string& column,
                            std::vector<int64_t> values,
                            std::vector<int64_t> offsets);
  void add_embedding(const std::string& table, const std::string& column,
                     std::vector<float> values, int64_t width);
  void add_text(const std::string& table, const std::string& column,
                std::vector<std::string> values);

  FeatureBatch gather(
      const std::unordered_map<std::string, std::vector<int64_t>>& node_ids,
      int64_t workers) const;

 private:
  void add(const std::string& table, const std::string& column,
           FeatureColumn values);

  std::unordered_map<std::string,
                     std::unordered_map<std::string, FeatureColumn>> columns_;
};

}  // namespace relx
