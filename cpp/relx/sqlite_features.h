#pragma once

#include "online_feature_reader.h"
#include "online_sql_features.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace relx {

using SqliteFeatureSpec = SqlFeatureSpec;
using SqliteTableSpec = SqlTableSpec;

class SqliteFeatureReader : public NativeFeatureReader {
 public:
  explicit SqliteFeatureReader(std::string database_path);

  void add_table(std::string name, std::string schema,
                 std::vector<std::string> identity,
                 std::vector<std::vector<SqlKey>> keys);
  void add_feature(const std::string& table, SqliteFeatureSpec feature);
  void set_glove_model(const std::string& directory,
                       size_t cache_bytes = 0,
                       const std::string& disk_cache_path = "");
  void set_row_cache_bytes(size_t bytes) { row_cache_bytes_ = bytes; }
  size_t row_cache_bytes() const { return row_cache_bytes_; }
  std::unique_ptr<NativeFeatureSession> open_session() const override;
  FeatureBatch fetch(
      const std::unordered_map<std::string, std::vector<int64_t>>& node_ids
  ) const;

  const std::string& database_path() const { return database_path_; }
  const std::unordered_map<std::string, SqliteTableSpec>& tables() const {
    return tables_;
  }

 private:
  std::string database_path_;
  std::unordered_map<std::string, SqliteTableSpec> tables_;
  std::shared_ptr<const GloveModel> glove_;
  size_t row_cache_bytes_ = 0;
};

}  // namespace relx
