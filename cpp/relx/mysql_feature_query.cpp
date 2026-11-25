#include "mysql_feature_query.h"

#include "mysql_feature_values.h"

#include <mysql.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace relx {
namespace {

using Statement = std::unique_ptr<MYSQL_STMT, decltype(&mysql_stmt_close)>;
using Metadata = std::unique_ptr<MYSQL_RES, decltype(&mysql_free_result)>;

std::string quote(const std::string& name) {
  if (name.empty()) throw std::invalid_argument("SQL identifier is empty");
  std::string result = "`";
  for (const char character : name) {
    result += character;
    if (character == '`') result += '`';
  }
  return result + '`';
}

std::string query_for(const SqlTableSpec& table, size_t rows) {
  std::string sql = "SELECT requested.ord";
  for (const auto& feature : table.features) {
    sql += ", source." + quote(feature.column);
  }
  sql += " FROM (";
  for (size_t row = 0; row < rows; ++row) {
    if (row) sql += " UNION ALL ";
    sql += "SELECT " + std::to_string(row) + " AS ord";
    for (size_t key = 0; key < table.identity.size(); ++key) {
      sql += ", ? AS k" + std::to_string(key);
    }
  }
  sql += ") AS requested JOIN ";
  if (!table.schema.empty()) sql += quote(table.schema) + ".";
  sql += quote(table.name) + " AS source ON ";
  for (size_t key = 0; key < table.identity.size(); ++key) {
    if (key) sql += " AND ";
    sql += "source." + quote(table.identity[key]) +
           " = requested.k" + std::to_string(key);
  }
  return sql + " ORDER BY requested.ord";
}

void bind_keys(MYSQL_STMT* statement, const SqlTableSpec& table,
               const std::vector<int64_t>& missing, size_t start,
               size_t stop) {
  const size_t count = (stop - start) * table.identity.size();
  std::vector<MYSQL_BIND> bindings(count);
  std::vector<unsigned long> lengths(count);
  size_t index = 0;
  for (size_t position = start; position < stop; ++position) {
    for (const auto& key : table.keys[static_cast<size_t>(missing[position])]) {
      auto& bind = bindings[index];
      if (const auto* value = std::get_if<int64_t>(&key)) {
        bind.buffer_type = MYSQL_TYPE_LONGLONG;
        bind.buffer = const_cast<int64_t*>(value);
      } else if (const auto* value = std::get_if<double>(&key)) {
        bind.buffer_type = MYSQL_TYPE_DOUBLE;
        bind.buffer = const_cast<double*>(value);
      } else if (const auto* value = std::get_if<std::string>(&key)) {
        bind.buffer_type = MYSQL_TYPE_STRING;
        bind.buffer = const_cast<char*>(value->data());
        lengths[index] = static_cast<unsigned long>(value->size());
        bind.length = &lengths[index];
        bind.buffer_length = lengths[index];
      } else if (const auto* value = std::get_if<SqlBlob>(&key)) {
        bind.buffer_type = MYSQL_TYPE_BLOB;
        bind.buffer = const_cast<char*>(value->data.data());
        lengths[index] = static_cast<unsigned long>(value->data.size());
        bind.length = &lengths[index];
        bind.buffer_length = lengths[index];
      }
      ++index;
    }
  }
  if (mysql_stmt_param_count(statement) != count ||
      mysql_stmt_bind_param(statement, bindings.data()) != 0 ||
      mysql_stmt_execute(statement) != 0) {
    throw std::runtime_error("Execute MySQL feature query: " +
                             std::string(mysql_stmt_error(statement)));
  }
}

void read_chunk(MYSQL* db, const SqlTableSpec& table,
                const std::vector<int64_t>& missing, size_t start,
                size_t stop,
                std::unordered_map<std::string, NativeFeatureBlock>& blocks) {
  const auto sql = query_for(table, stop - start);
  Statement statement(mysql_stmt_init(db), mysql_stmt_close);
  if (!statement) throw std::runtime_error("Initialize MySQL statement failed");
  if (mysql_stmt_prepare(statement.get(), sql.c_str(), sql.size()) != 0) {
    throw std::runtime_error("Prepare MySQL feature query for " + table.name +
                             ": " + mysql_stmt_error(statement.get()));
  }
  bind_keys(statement.get(), table, missing, start, stop);
  Metadata metadata(mysql_stmt_result_metadata(statement.get()), mysql_free_result);
  if (!metadata) throw std::runtime_error("MySQL feature query has no metadata");
  const size_t columns = mysql_num_fields(metadata.get());
  if (columns != table.features.size() + 1) {
    throw std::runtime_error("MySQL feature column count differs");
  }
  const MYSQL_FIELD* fields = mysql_fetch_fields(metadata.get());
  std::vector<MYSQL_BIND> bindings(columns);
  std::vector<std::string> buffers(columns, std::string(4096, '\0'));
  std::vector<unsigned long> lengths(columns);
  using BindFlag = std::remove_pointer_t<decltype(MYSQL_BIND{}.is_null)>;
  auto nulls = std::make_unique<BindFlag[]>(columns);
  auto errors = std::make_unique<BindFlag[]>(columns);
  for (size_t index = 0; index < columns; ++index) {
    bindings[index].buffer_type = MYSQL_TYPE_STRING;
    bindings[index].buffer = buffers[index].data();
    bindings[index].buffer_length = buffers[index].size();
    bindings[index].length = &lengths[index];
    bindings[index].is_null = nulls.get() + index;
    bindings[index].error = errors.get() + index;
  }
  if (mysql_stmt_bind_result(statement.get(), bindings.data()) != 0 ||
      mysql_stmt_store_result(statement.get()) != 0) {
    throw std::runtime_error("Read MySQL feature result: " +
                             std::string(mysql_stmt_error(statement.get())));
  }
  size_t expected = 0;
  while (true) {
    const int status = mysql_stmt_fetch(statement.get());
    if (status == MYSQL_NO_DATA) break;
    if (status != 0 && status != MYSQL_DATA_TRUNCATED) {
      throw std::runtime_error("Fetch MySQL feature row: " +
                               std::string(mysql_stmt_error(statement.get())));
    }
    std::vector<MySqlCell> cells(columns);
    for (size_t index = 0; index < columns; ++index) {
      cells[index].is_null = nulls[index];
      cells[index].type = fields[index].type;
      cells[index].binary = fields[index].charsetnr == 63;
      if (nulls[index]) continue;
      if (lengths[index] > 64 * 1024 * 1024UL) {
        throw std::runtime_error("MySQL feature cell exceeds 64 MiB");
      }
      if (lengths[index] <= buffers[index].size()) {
        cells[index].data.assign(buffers[index].data(), lengths[index]);
      } else {
        cells[index].data.resize(lengths[index]);
        MYSQL_BIND column{};
        column.buffer_type = MYSQL_TYPE_STRING;
        column.buffer = cells[index].data.data();
        column.buffer_length = lengths[index];
        if (mysql_stmt_fetch_column(statement.get(), &column, index, 0) != 0) {
          throw std::runtime_error("Fetch long MySQL feature: " +
                                   std::string(mysql_stmt_error(statement.get())));
        }
      }
    }
    if (cells[0].is_null || std::stoll(cells[0].data) !=
            static_cast<int64_t>(expected)) {
      throw std::runtime_error("Missing or duplicate feature row on " + table.name);
    }
    for (size_t index = 0; index < table.features.size(); ++index) {
      const auto& feature = table.features[index];
      append_mysql_feature(cells[index + 1], feature, blocks.at(feature.column));
    }
    ++expected;
  }
  if (expected != stop - start) {
    throw std::runtime_error("Feature row disappeared from " + table.name);
  }
}

}  // namespace

void fetch_mysql_table(MYSQL* db, const SqlTableSpec& table,
                       const std::vector<int64_t>& ids, FeatureBatch& output,
                       SqlTableCache* cache) {
  fetch_sql_table(table, ids, output, cache,
      [db, &table](const std::vector<int64_t>& missing,
                   std::unordered_map<std::string, NativeFeatureBlock>& blocks) {
        const size_t chunk = std::min<size_t>(128, 65535 / table.identity.size());
        if (!chunk) throw std::runtime_error("MySQL identity key is too wide");
        for (size_t start = 0; start < missing.size(); start += chunk) {
          read_chunk(db, table, missing, start,
                     std::min(start + chunk, missing.size()), blocks);
        }
      });
}

}  // namespace relx
