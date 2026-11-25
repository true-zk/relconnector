#include "sqlite_feature_values.h"
#include "sql_time.h"
#include "sql_number.h"

#include <sqlite3.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace relx {
namespace {

using Statement = std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)>;

std::string cell_text(sqlite3_stmt* row, int column) {
  const auto* data = sqlite3_column_text(row, column);
  const int size = sqlite3_column_bytes(row, column);
  return data ? std::string(reinterpret_cast<const char*>(data), size) : "";
}

double number(sqlite3_stmt* row, int column) {
  const int type = sqlite3_column_type(row, column);
  if (type == SQLITE_NULL) return std::numeric_limits<double>::quiet_NaN();
  if (type == SQLITE_INTEGER || type == SQLITE_FLOAT) {
    return sqlite3_column_double(row, column);
  }
  const auto text = cell_text(row, column);
  char* end = nullptr;
  const double parsed = std::strtod(text.c_str(), &end);
  return end != text.c_str() && *end == '\0'
             ? parsed : std::numeric_limits<double>::quiet_NaN();
}

int64_t category(sqlite3_stmt* row, int column,
                 const std::unordered_map<std::string, int64_t>& vocab,
                 const std::string& json_type = "") {
  const int type = sqlite3_column_type(row, column);
  if (type == SQLITE_NULL || json_type == "null") return 0;
  std::string key;
  if (json_type == "true" || json_type == "false") {
    key = "bool:" + std::string(json_type == "true" ? "True" : "False");
  } else if (type == SQLITE_TEXT) {
    key = "str:" + cell_text(row, column);
  } else if (type == SQLITE_INTEGER) {
    key = "int:" + std::to_string(sqlite3_column_int64(row, column));
  } else if (type == SQLITE_FLOAT) {
    key = "float:" + python_float_token(sqlite3_column_double(row, column));
  } else {
    return 0;
  }
  auto found = vocab.find(key);
  if (found != vocab.end()) return found->second;
  if (key.rfind("int:", 0) == 0 || key.rfind("float:", 0) == 0) {
    const auto separator = key.find(':');
    key = (type == SQLITE_INTEGER ? "int64:" : "float64:") +
          key.substr(separator + 1);
    found = vocab.find(key);
  }
  return found == vocab.end() ? 0 : found->second;
}


void append_json(sqlite3* db, sqlite3_stmt* row, int column,
                 const SqliteFeatureSpec& spec, NativeFeatureBlock& output) {
  if (sqlite3_column_type(row, column) == SQLITE_NULL) {
    if (spec.kind == FeatureKind::kMulticategorical) {
      output.offsets.push_back(static_cast<int64_t>(output.integers.size()));
    } else {
      output.floats.insert(output.floats.end(),
                           static_cast<size_t>(spec.width), 0.0F);
    }
    return;
  }
  const auto raw = cell_text(row, column);
  const auto first = raw.find_first_not_of(" \t\r\n");
  const auto last = raw.find_last_not_of(" \t\r\n");
  if (first == std::string::npos || raw[first] != '[' || raw[last] != ']') {
    throw std::invalid_argument("Feature JSON value must be an array");
  }
  sqlite3_stmt* pointer = nullptr;
  if (sqlite3_prepare_v2(db,
          "SELECT value, type FROM json_each(?1) ORDER BY CAST(key AS INTEGER)",
          -1, &pointer, nullptr) != SQLITE_OK) {
    throw std::runtime_error("SQLite JSON1 is required: " +
                             std::string(sqlite3_errmsg(db)));
  }
  Statement elements(pointer, sqlite3_finalize);
  sqlite3_bind_text(elements.get(), 1, raw.data(),
                    static_cast<int>(raw.size()), SQLITE_TRANSIENT);
  int count = 0;
  int status = SQLITE_ROW;
  while ((status = sqlite3_step(elements.get())) == SQLITE_ROW) {
    if (spec.kind == FeatureKind::kMulticategorical) {
      output.integers.push_back(category(
          elements.get(), 0, spec.vocab, cell_text(elements.get(), 1)));
    } else {
      const double value = number(elements.get(), 0);
      if (!std::isfinite(value)) {
        throw std::invalid_argument("Embedding JSON needs finite numbers");
      }
      output.floats.push_back(static_cast<float>(value));
    }
    ++count;
  }
  if (status != SQLITE_DONE) throw std::runtime_error(sqlite3_errmsg(db));
  if (spec.kind == FeatureKind::kMulticategorical) {
    output.offsets.push_back(static_cast<int64_t>(output.integers.size()));
  } else if (count != spec.width) {
    throw std::invalid_argument("Embedding JSON width differs from feature plan");
  }
}

}  // namespace

void append_feature(sqlite3* db, sqlite3_stmt* row, int column,
                    const SqliteFeatureSpec& spec, NativeFeatureBlock& output) {
  if (spec.kind == FeatureKind::kRaw) {
    const int type = sqlite3_column_type(row, column);
    if (type == SQLITE_NULL) output.raw.emplace_back(std::monostate{});
    else if (type == SQLITE_INTEGER) {
      output.raw.emplace_back(sqlite3_column_int64(row, column));
    } else if (type == SQLITE_FLOAT) {
      output.raw.emplace_back(sqlite3_column_double(row, column));
    } else if (type == SQLITE_BLOB) {
      const auto* data = static_cast<const char*>(sqlite3_column_blob(row, column));
      output.raw.emplace_back(RawBlob{data ? std::string(
          data, static_cast<size_t>(sqlite3_column_bytes(row, column))) : ""});
    } else output.raw.emplace_back(cell_text(row, column));
  } else if (spec.kind == FeatureKind::kNumeric) {
    const double value = number(row, column);
    const bool missing = !std::isfinite(value);
    output.missing.push_back(missing);
    output.floats.push_back(static_cast<float>(
        ((missing ? spec.mean : value) - spec.mean) / spec.std));
  } else if (spec.kind == FeatureKind::kCategorical) {
    output.integers.push_back(category(row, column, spec.vocab));
  } else if (spec.kind == FeatureKind::kTimestamp) {
    const bool missing = sqlite3_column_type(row, column) == SQLITE_NULL;
    const int64_t value = missing ? 0 : parse_utc_time_us(cell_text(row, column));
    output.missing.push_back(missing);
    output.floats.push_back(missing ? 0.0F : static_cast<float>(
        (static_cast<double>(value) - spec.mean) / spec.std));
  } else if (spec.kind == FeatureKind::kText) {
    output.text.push_back(sqlite3_column_type(row, column) == SQLITE_NULL
                              ? "" : cell_text(row, column));
  } else if (spec.kind == FeatureKind::kTextEmbedding) {
    const auto encoded = spec.glove->encode(
        sqlite3_column_type(row, column) == SQLITE_NULL
            ? "" : cell_text(row, column));
    output.floats.insert(output.floats.end(), encoded.begin(), encoded.end());
  } else {
    append_json(db, row, column, spec, output);
  }
}

}  // namespace relx
