#pragma once

#include "glove_model.h"
#include "native_features.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace relx {

struct SqlBlob { std::string data; };
using SqlKey = std::variant<int64_t, double, std::string, SqlBlob>;

struct SqlFeatureSpec {
  std::string column;
  FeatureKind kind;
  double mean = 0.0;
  double std = 1.0;
  int64_t width = 1;
  std::unordered_map<std::string, int64_t> vocab;
  std::shared_ptr<const GloveModel> glove;
};

struct SqlTableSpec {
  std::string name;
  std::string schema;
  std::vector<std::string> identity;
  std::vector<std::vector<SqlKey>> keys;
  std::vector<SqlFeatureSpec> features;
};

struct SqlTableCache {
  std::unordered_map<int64_t, size_t> positions;
  std::unordered_map<std::string, NativeFeatureBlock> blocks;
  size_t bytes = 0;
};

NativeFeatureBlock feature_block_for(const SqlFeatureSpec& spec, size_t rows);
void copy_feature_row(const SqlFeatureSpec& spec,
                      const NativeFeatureBlock& source, size_t row,
                      NativeFeatureBlock& target);
size_t feature_cache_bytes(const SqlTableCache& cache);
using ReadMissingRows = std::function<void(
    const std::vector<int64_t>&,
    std::unordered_map<std::string, NativeFeatureBlock>&)>;
void fetch_sql_table(const SqlTableSpec& table, const std::vector<int64_t>& ids,
                     FeatureBatch& output, SqlTableCache* cache,
                     const ReadMissingRows& read_missing);
void trim_sql_caches(std::unordered_map<std::string, SqlTableCache>& caches,
                     const std::string& current, size_t limit);

}  // namespace relx
