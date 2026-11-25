#include "mysql_features.h"

#include "mysql_feature_query.h"

#include <mysql.h>

#include <mutex>
#include <stdexcept>
#include <utility>

namespace relx {
namespace {

void initialize_client() {
  static std::once_flag once;
  std::call_once(once, [] {
    if (mysql_library_init(0, nullptr, nullptr) != 0) {
      throw std::runtime_error("Initialize MySQL client library failed");
    }
  });
}

class MySqlSession final : public NativeFeatureSession {
 public:
  explicit MySqlSession(const MySqlFeatureReader& reader) : reader_(reader) {
    initialize_client();
    db_ = mysql_init(nullptr);
    if (!db_) throw std::runtime_error("Initialize MySQL connection failed");
    const unsigned int timeout = 30;
    mysql_options(db_, MYSQL_OPT_CONNECT_TIMEOUT, &timeout);
    mysql_options(db_, MYSQL_OPT_READ_TIMEOUT, &timeout);
    mysql_options(db_, MYSQL_SET_CHARSET_NAME, "utf8mb4");
    const auto& config = reader_.connection();
    const unsigned int protocol = config.unix_socket.empty()
        ? MYSQL_PROTOCOL_TCP : MYSQL_PROTOCOL_SOCKET;
    if (mysql_options(db_, MYSQL_OPT_PROTOCOL, &protocol) != 0) {
      const std::string error = mysql_error(db_);
      mysql_close(db_);
      db_ = nullptr;
      mysql_thread_end();
      throw std::runtime_error("Set MySQL transport protocol: " + error);
    }
    if (!mysql_real_connect(
            db_, config.host.empty() ? nullptr : config.host.c_str(),
            config.user.c_str(), config.password.c_str(),
            config.database.c_str(), config.port,
            config.unix_socket.empty() ? nullptr : config.unix_socket.c_str(),
            0)) {
      const std::string error = mysql_error(db_);
      mysql_close(db_);
      db_ = nullptr;
      mysql_thread_end();
      throw std::runtime_error("Connect to MySQL feature database: " + error);
    }
    if (mysql_query(db_, "SET time_zone = '+00:00'") != 0) {
      const std::string error = mysql_error(db_);
      mysql_close(db_);
      db_ = nullptr;
      mysql_thread_end();
      throw std::runtime_error("Set MySQL session time zone: " + error);
    }
  }

  ~MySqlSession() override {
    if (db_) mysql_close(db_);
    mysql_thread_end();
  }

  FeatureBatch fetch(const std::unordered_map<std::string,
                     std::vector<int64_t>>& node_ids) override {
    FeatureBatch result;
    for (const auto& [name, ids] : node_ids) {
      const auto found = reader_.tables().find(name);
      if (found == reader_.tables().end()) continue;
      SqlTableCache* cache = nullptr;
      if (reader_.row_cache_bytes()) cache = &caches_[name];
      fetch_mysql_table(db_, found->second, ids, result, cache);
      if (cache) trim_sql_caches(caches_, name, reader_.row_cache_bytes());
    }
    return result;
  }

 private:
  const MySqlFeatureReader& reader_;
  MYSQL* db_ = nullptr;
  std::unordered_map<std::string, SqlTableCache> caches_;
};

}  // namespace

MySqlFeatureReader::MySqlFeatureReader(MySqlConnectionConfig config)
    : connection_(std::move(config)) {
  if (connection_.database.empty()) {
    throw std::invalid_argument("MySQL database name is required");
  }
}

void MySqlFeatureReader::add_table(
    std::string name, std::string schema, std::vector<std::string> identity,
    std::vector<std::vector<SqlKey>> keys) {
  if (identity.empty()) throw std::invalid_argument("Feature identity is empty");
  for (const auto& key : keys) {
    if (key.size() != identity.size()) {
      throw std::invalid_argument("Feature identity key width differs");
    }
  }
  const std::string label = schema.empty() ? name : schema + "." + name;
  if (!tables_.emplace(label, SqlTableSpec{
          std::move(name), std::move(schema), std::move(identity),
          std::move(keys), {}}).second) {
    throw std::invalid_argument("Duplicate feature table: " + label);
  }
}

void MySqlFeatureReader::add_feature(const std::string& table,
                                     SqlFeatureSpec feature) {
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

void MySqlFeatureReader::set_glove_model(
    const std::string& directory, size_t cache_bytes,
    const std::string& disk_cache_path) {
  glove_ = std::make_shared<GloveModel>(directory, cache_bytes, disk_cache_path);
}

std::unique_ptr<NativeFeatureSession> MySqlFeatureReader::open_session() const {
  return std::make_unique<MySqlSession>(*this);
}

FeatureBatch MySqlFeatureReader::fetch(
    const std::unordered_map<std::string, std::vector<int64_t>>& node_ids) const {
  return open_session()->fetch(node_ids);
}

}  // namespace relx
