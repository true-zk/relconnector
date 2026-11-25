#pragma once

#include "online_feature_reader.h"
#include "online_sql_features.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace relx {

struct MySqlConnectionConfig {
  std::string host;
  unsigned int port = 3306;
  std::string user;
  std::string password;
  std::string database;
  std::string unix_socket;
};

class MySqlFeatureReader final : public NativeFeatureReader {
 public:
  explicit MySqlFeatureReader(MySqlConnectionConfig config);

  void add_table(std::string name, std::string schema,
                 std::vector<std::string> identity,
                 std::vector<std::vector<SqlKey>> keys);
  void add_feature(const std::string& table, SqlFeatureSpec feature);
  void set_glove_model(const std::string& directory,
                       size_t cache_bytes = 0,
                       const std::string& disk_cache_path = "");
  void set_row_cache_bytes(size_t bytes) { row_cache_bytes_ = bytes; }
  size_t row_cache_bytes() const { return row_cache_bytes_; }
  const MySqlConnectionConfig& connection() const { return connection_; }
  const std::unordered_map<std::string, SqlTableSpec>& tables() const {
    return tables_;
  }

  std::unique_ptr<NativeFeatureSession> open_session() const override;
  FeatureBatch fetch(
      const std::unordered_map<std::string, std::vector<int64_t>>& node_ids
  ) const;

 private:
  MySqlConnectionConfig connection_;
  std::unordered_map<std::string, SqlTableSpec> tables_;
  std::shared_ptr<const GloveModel> glove_;
  size_t row_cache_bytes_ = 0;
};

}  // namespace relx
