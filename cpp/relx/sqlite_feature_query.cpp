#include "sqlite_feature_query.h"

#include "sqlite_feature_values.h"

#include <sqlite3.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace relx {
namespace {

using Statement = std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)>;

std::string quote(const std::string& name) {
  if (name.empty()) throw std::invalid_argument("SQL identifier is empty");
  std::string result = "\"";
  for (const char character : name) {
    result += character;
    if (character == '"') result += '"';
  }
  return result + '"';
}

std::string query_for(const SqliteTableSpec& table, size_t rows) {
  std::string sql = "WITH requested(ord";
  for (size_t key = 0; key < table.identity.size(); ++key) {
    sql += ", k" + std::to_string(key);
  }
  sql += ") AS (VALUES ";
  for (size_t row = 0; row < rows; ++row) {
    if (row) sql += ", ";
    sql += "(?";
    for (size_t key = 0; key < table.identity.size(); ++key) sql += ", ?";
    sql += ")";
  }
  sql += ") SELECT requested.ord";
  for (const auto& feature : table.features) {
    sql += ", source." + quote(feature.column);
  }
  sql += " FROM requested JOIN ";
  if (!table.schema.empty()) sql += quote(table.schema) + ".";
  sql += quote(table.name) + " AS source ON ";
  for (size_t key = 0; key < table.identity.size(); ++key) {
    if (key) sql += " AND ";
    sql += "source." + quote(table.identity[key]) +
           " = requested.k" + std::to_string(key);
  }
  return sql + " ORDER BY requested.ord";
}

void bind_value(sqlite3* db, sqlite3_stmt* statement, int parameter,
                const SqlKey& key) {
  int status = SQLITE_MISUSE;
  if (const auto* value = std::get_if<int64_t>(&key)) {
    status = sqlite3_bind_int64(statement, parameter, *value);
  } else if (const auto* value = std::get_if<double>(&key)) {
    status = sqlite3_bind_double(statement, parameter, *value);
  } else if (const auto* value = std::get_if<std::string>(&key)) {
    status = sqlite3_bind_text(statement, parameter, value->data(),
                               static_cast<int>(value->size()), SQLITE_TRANSIENT);
  } else if (const auto* value = std::get_if<SqlBlob>(&key)) {
    status = sqlite3_bind_blob(statement, parameter, value->data.data(),
                               static_cast<int>(value->data.size()), SQLITE_TRANSIENT);
  }
  if (status != SQLITE_OK) throw std::runtime_error(sqlite3_errmsg(db));
}

void read_chunk(sqlite3* db, const SqliteTableSpec& table,
                const std::vector<int64_t>& unique, size_t start, size_t stop,
                std::unordered_map<std::string, NativeFeatureBlock>& blocks) {
  const auto sql = query_for(table, stop - start);
  sqlite3_stmt* raw = nullptr;
  if (sqlite3_prepare_v2(db, sql.c_str(), -1, &raw, nullptr) != SQLITE_OK) {
    throw std::runtime_error("Prepare batched feature query for " + table.name +
                             ": " + sqlite3_errmsg(db));
  }
  Statement statement(raw, sqlite3_finalize);
  int parameter = 1;
  for (size_t position = start; position < stop; ++position) {
    if (sqlite3_bind_int64(statement.get(), parameter++,
                           static_cast<int64_t>(position)) != SQLITE_OK) {
      throw std::runtime_error(sqlite3_errmsg(db));
    }
    for (const auto& key : table.keys[static_cast<size_t>(unique[position])]) {
      bind_value(db, statement.get(), parameter++, key);
    }
  }
  size_t expected = start;
  int status = SQLITE_ROW;
  while ((status = sqlite3_step(statement.get())) == SQLITE_ROW) {
    const auto ordinal = sqlite3_column_int64(statement.get(), 0);
    if (ordinal != static_cast<int64_t>(expected)) {
      throw std::runtime_error("Missing or duplicate feature row on " + table.name);
    }
    for (size_t index = 0; index < table.features.size(); ++index) {
      const auto& feature = table.features[index];
      append_feature(db, statement.get(), static_cast<int>(index + 1),
                     feature, blocks.at(feature.column));
    }
    ++expected;
  }
  if (status != SQLITE_DONE) throw std::runtime_error(sqlite3_errmsg(db));
  if (expected != stop) {
    throw std::runtime_error("Feature row disappeared from " + table.name);
  }
}

}  // namespace

void fetch_sqlite_table(sqlite3* db, const SqliteTableSpec& table,
                        const std::vector<int64_t>& ids, FeatureBatch& output,
                        SqliteTableCache* cache) {
  fetch_sql_table(table, ids, output, cache,
      [db, &table](const std::vector<int64_t>& missing,
                   std::unordered_map<std::string, NativeFeatureBlock>& blocks) {
        const int variables = sqlite3_limit(db, SQLITE_LIMIT_VARIABLE_NUMBER, -1);
        const size_t width = table.identity.size() + 1;
        if (variables < static_cast<int>(width)) {
          throw std::runtime_error("SQLite variable limit is too small for identity key");
        }
        const size_t chunk = std::min<size_t>(500, variables / width);
        for (size_t start = 0; start < missing.size(); start += chunk) {
          read_chunk(db, table, missing, start,
                     std::min(start + chunk, missing.size()), blocks);
        }
      });
}

}  // namespace relx
