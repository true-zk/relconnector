#include "sqlite_features.h"

#include "sqlite_feature_query.h"

#include <sqlite3.h>

#include <memory>
#include <stdexcept>
#include <utility>

namespace relx {
namespace {

class SqliteSession final : public NativeFeatureSession {
 public:
  explicit SqliteSession(const SqliteFeatureReader& reader)
      : reader_(reader), db_(nullptr, sqlite3_close) {
    sqlite3* raw = nullptr;
    const int status = sqlite3_open_v2(
        reader.database_path().c_str(), &raw,
        SQLITE_OPEN_READONLY | SQLITE_OPEN_NOMUTEX, nullptr);
    db_.reset(raw);
    if (status != SQLITE_OK) {
      throw std::runtime_error("Open SQLite feature database: " +
          std::string(raw ? sqlite3_errmsg(raw) : "unknown error"));
    }
  }

  FeatureBatch fetch(const std::unordered_map<std::string,
                     std::vector<int64_t>>& node_ids) override {
    FeatureBatch result;
    for (const auto& [name, ids] : node_ids) {
      const auto found = reader_.tables().find(name);
      if (found != reader_.tables().end()) {
        SqliteTableCache* cache = nullptr;
        if (reader_.row_cache_bytes()) cache = &caches_[name];
        fetch_sqlite_table(db_.get(), found->second, ids, result, cache);
        if (cache) trim_sql_caches(caches_, name, reader_.row_cache_bytes());
      }
    }
    return result;
  }

 private:
  const SqliteFeatureReader& reader_;
  std::unique_ptr<sqlite3, decltype(&sqlite3_close)> db_;
  std::unordered_map<std::string, SqliteTableCache> caches_;
};

}  // namespace

SqliteFeatureReader::SqliteFeatureReader(std::string database_path)
    : database_path_(std::move(database_path)) {
  if (database_path_.empty()) throw std::invalid_argument("SQLite path is empty");
}

void SqliteFeatureReader::add_table(
    std::string name, std::string schema, std::vector<std::string> identity,
    std::vector<std::vector<SqlKey>> keys) {
  if (identity.empty()) throw std::invalid_argument("Feature identity is empty");
  for (const auto& key : keys) {
    if (key.size() != identity.size()) {
      throw std::invalid_argument("Feature identity key width differs");
    }
  }
  const std::string label = schema.empty() ? name : schema + "." + name;
  if (!tables_.emplace(label, SqliteTableSpec{
          std::move(name), std::move(schema), std::move(identity),
          std::move(keys), {}}).second) {
    throw std::invalid_argument("Duplicate feature table: " + label);
  }
}

void SqliteFeatureReader::add_feature(const std::string& table,
                                      SqliteFeatureSpec feature) {
  auto found = tables_.find(table);
  if (found == tables_.end()) throw std::invalid_argument("Unknown table: " + table);
  if (feature.column.empty() || feature.std <= 0.0 || feature.width <= 0) {
    throw std::invalid_argument("Invalid feature specification");
  }
  for (const auto& prior : found->second.features) {
    if (prior.column == feature.column) {
      throw std::invalid_argument("Duplicate feature column: " + feature.column);
    }
  }
  if (feature.kind == FeatureKind::kTextEmbedding) {
    if (!glove_) throw std::invalid_argument("Set GloVe model before text features");
    feature.glove = glove_;
    feature.width = glove_->width();
  }
  found->second.features.push_back(std::move(feature));
}

void SqliteFeatureReader::set_glove_model(
    const std::string& directory, size_t cache_bytes,
    const std::string& disk_cache_path) {
  glove_ = std::make_shared<GloveModel>(directory, cache_bytes, disk_cache_path);
}

std::unique_ptr<NativeFeatureSession> SqliteFeatureReader::open_session() const {
  return std::make_unique<SqliteSession>(*this);
}

FeatureBatch SqliteFeatureReader::fetch(
    const std::unordered_map<std::string, std::vector<int64_t>>& node_ids) const {
  return open_session()->fetch(node_ids);
}

}  // namespace relx
