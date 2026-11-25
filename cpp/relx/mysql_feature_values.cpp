#include "mysql_feature_values.h"

#include "json_array.h"
#include "sql_time.h"
#include "sql_number.h"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace relx {
namespace {

bool integer_type(enum_field_types type) {
  switch (type) {
    case MYSQL_TYPE_TINY:
    case MYSQL_TYPE_SHORT:
    case MYSQL_TYPE_LONG:
    case MYSQL_TYPE_LONGLONG:
    case MYSQL_TYPE_INT24:
    case MYSQL_TYPE_YEAR:
      return true;
    default:
      return false;
  }
}

bool floating_type(enum_field_types type) {
  return type == MYSQL_TYPE_FLOAT || type == MYSQL_TYPE_DOUBLE ||
         type == MYSQL_TYPE_DECIMAL || type == MYSQL_TYPE_NEWDECIMAL;
}

double number(const std::string& raw) {
  char* end = nullptr;
  const double value = std::strtod(raw.c_str(), &end);
  return end != raw.c_str() && *end == '\0'
      ? value : std::numeric_limits<double>::quiet_NaN();
}

std::string float_token(const std::string& raw) {
  const double value = number(raw);
  if (!std::isfinite(value)) return raw;
  return python_float_token(value);
}

int64_t lookup(const std::unordered_map<std::string, int64_t>& vocab,
               const std::string& prefix, const std::string& text) {
  auto found = vocab.find(prefix + text);
  if (found != vocab.end()) return found->second;
  if (prefix == "int:") found = vocab.find("int64:" + text);
  else if (prefix == "float:") found = vocab.find("float64:" + text);
  return found == vocab.end() ? 0 : found->second;
}

int64_t category(const MySqlCell& cell,
                 const std::unordered_map<std::string, int64_t>& vocab) {
  if (cell.is_null) return 0;
  if (integer_type(cell.type)) return lookup(vocab, "int:", cell.data);
  if (cell.type == MYSQL_TYPE_DECIMAL || cell.type == MYSQL_TYPE_NEWDECIMAL) {
    const auto exact = lookup(vocab, "Decimal:", cell.data);
    if (exact) return exact;
  }
  if (floating_type(cell.type)) {
    return lookup(vocab, "float:", float_token(cell.data));
  }
  return lookup(vocab, "str:", cell.data);
}

int64_t json_category(
    const JsonScalar& value,
    const std::unordered_map<std::string, int64_t>& vocab) {
  switch (value.kind) {
    case JsonScalar::Kind::kNull:
      return 0;
    case JsonScalar::Kind::kBoolean:
      return lookup(vocab, "bool:", value.text == "true" ? "True" : "False");
    case JsonScalar::Kind::kInteger:
      return lookup(vocab, "int:", value.text == "-0" ? "0" : value.text);
    case JsonScalar::Kind::kFloat:
      return lookup(vocab, "float:", float_token(value.text));
    case JsonScalar::Kind::kString:
      return lookup(vocab, "str:", value.text);
  }
  return 0;
}

void append_json(const MySqlCell& cell, const SqlFeatureSpec& spec,
                 NativeFeatureBlock& output) {
  if (cell.is_null) {
    if (spec.kind == FeatureKind::kMulticategorical) {
      output.offsets.push_back(static_cast<int64_t>(output.integers.size()));
    } else {
      output.floats.insert(output.floats.end(),
                           static_cast<size_t>(spec.width), 0.0F);
    }
    return;
  }
  const auto values = parse_json_array(cell.data);
  if (spec.kind == FeatureKind::kMulticategorical) {
    for (const auto& value : values) {
      output.integers.push_back(json_category(value, spec.vocab));
    }
    output.offsets.push_back(static_cast<int64_t>(output.integers.size()));
    return;
  }
  if (values.size() != static_cast<size_t>(spec.width)) {
    throw std::invalid_argument("Embedding JSON width differs from feature plan");
  }
  for (const auto& value : values) {
    if (value.kind != JsonScalar::Kind::kInteger &&
        value.kind != JsonScalar::Kind::kFloat) {
      throw std::invalid_argument("Embedding JSON needs numbers");
    }
    const double parsed = number(value.text);
    if (!std::isfinite(parsed)) {
      throw std::invalid_argument("Embedding JSON needs finite numbers");
    }
    output.floats.push_back(static_cast<float>(parsed));
  }
}

}  // namespace

void append_mysql_feature(const MySqlCell& cell, const SqlFeatureSpec& spec,
                          NativeFeatureBlock& output) {
  if (spec.kind == FeatureKind::kRaw) {
    if (cell.is_null) output.raw.emplace_back(std::monostate{});
    else if (integer_type(cell.type)) {
      output.raw.emplace_back(std::stoll(cell.data));
    } else if (cell.type == MYSQL_TYPE_DECIMAL ||
               cell.type == MYSQL_TYPE_NEWDECIMAL) {
      output.raw.emplace_back(cell.data);
    } else if (floating_type(cell.type)) {
      output.raw.emplace_back(number(cell.data));
    } else if (cell.binary) {
      output.raw.emplace_back(RawBlob{cell.data});
    } else output.raw.emplace_back(cell.data);
  } else if (spec.kind == FeatureKind::kNumeric) {
    const double value = cell.is_null
        ? std::numeric_limits<double>::quiet_NaN() : number(cell.data);
    const bool missing = !std::isfinite(value);
    output.missing.push_back(missing);
    output.floats.push_back(static_cast<float>(
        ((missing ? spec.mean : value) - spec.mean) / spec.std));
  } else if (spec.kind == FeatureKind::kCategorical) {
    output.integers.push_back(category(cell, spec.vocab));
  } else if (spec.kind == FeatureKind::kTimestamp) {
    const int64_t value = cell.is_null ? 0 : parse_utc_time_us(cell.data);
    output.missing.push_back(cell.is_null);
    output.floats.push_back(cell.is_null ? 0.0F : static_cast<float>(
        (static_cast<double>(value) - spec.mean) / spec.std));
  } else if (spec.kind == FeatureKind::kText) {
    output.text.push_back(cell.is_null ? "" : cell.data);
  } else if (spec.kind == FeatureKind::kTextEmbedding) {
    const auto encoded = spec.glove->encode(cell.is_null ? "" : cell.data);
    output.floats.insert(output.floats.end(), encoded.begin(), encoded.end());
  } else {
    append_json(cell, spec, output);
  }
}

}  // namespace relx
