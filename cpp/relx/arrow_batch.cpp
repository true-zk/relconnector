#include "arrow_batch.h"

#include "sql_number.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace relx {
namespace {

template <typename T>
T read_number(const void* buffer, int64_t index) {
  T value;
  std::memcpy(&value, static_cast<const char*>(buffer) + index * sizeof(T),
              sizeof(T));
  return value;
}

bool is_valid(const ArrowArray& array, int64_t index) {
  if (array.buffers[0] == nullptr) return true;
  const auto* bits = static_cast<const uint8_t*>(array.buffers[0]);
  return (bits[index / 8] & (1U << (index % 8))) != 0;
}

std::string length_prefix(std::string_view value) {
  if (value.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("Graph key component is too large");
  }
  uint32_t length = static_cast<uint32_t>(value.size());
  std::string result(sizeof(length), '\0');
  std::memcpy(result.data(), &length, sizeof(length));
  result.append(value);
  return result;
}

std::string numeric_key(double value) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument("Graph key must be finite");
  }
  if (std::trunc(value) == value &&
      value >= -9223372036854775808.0 &&
      value < 9223372036854775808.0) {
    return "n" + std::to_string(static_cast<int64_t>(value));
  }
  return "f" + python_float_token(value);
}

}  // namespace

ArrowBatch::ArrowBatch(const pybind11::object& batch)
    : owners_(batch.attr("__arrow_c_array__")().cast<pybind11::tuple>()),
      schema_(static_cast<ArrowSchema*>(
          PyCapsule_GetPointer(owners_[0].ptr(), "arrow_schema"))),
      array_(static_cast<ArrowArray*>(
          PyCapsule_GetPointer(owners_[1].ptr(), "arrow_array"))) {
  if (!schema_ || !array_ || !schema_->format ||
      std::strcmp(schema_->format, "+s") != 0 ||
      schema_->n_children != array_->n_children) {
    throw std::invalid_argument("Expected an Arrow RecordBatch");
  }
  for (int64_t i = 0; i < schema_->n_children; ++i) {
    const auto* child = schema_->children[i];
    if (!child || !child->name || !array_->children[i]) {
      throw std::invalid_argument("Arrow batch has an unnamed column");
    }
    if (!columns_.emplace(child->name, static_cast<size_t>(i)).second) {
      throw std::invalid_argument("Arrow batch has duplicate column names");
    }
  }
}

std::optional<std::string> ArrowBatch::key(
    const std::vector<std::string>& columns, int64_t row) const {
  if (columns.empty() || row < 0 || row >= length()) {
    throw std::invalid_argument("Invalid graph key request");
  }
  std::string result;
  for (const auto& name : columns) {
    const auto found = columns_.find(name);
    if (found == columns_.end()) {
      throw std::invalid_argument("Arrow batch is missing column " + name);
    }
    auto part = scalar(found->second, row);
    if (!part) return std::nullopt;
    result += length_prefix(*part);
  }
  return result;
}

std::optional<std::string> ArrowBatch::scalar(size_t column, int64_t row) const {
  const auto& values = *array_->children[column];
  const auto* format = schema_->children[column]->format;
  const int64_t index = values.offset + row;
  if (!is_valid(values, index)) return std::nullopt;
  if (!format || values.n_buffers < 2 || !values.buffers[1]) {
    throw std::invalid_argument("Unsupported Arrow graph key column");
  }

  // Integer widths have one canonical representation so an INT FK can match
  // a BIGINT primary key. String and binary keys stay distinct.
  const char kind = format[0];
  const void* data = values.buffers[1];
  switch (kind) {
    case 'c': return "n" + std::to_string(read_number<int8_t>(data, index));
    case 'C': return "n" + std::to_string(read_number<uint8_t>(data, index));
    case 's': return "n" + std::to_string(read_number<int16_t>(data, index));
    case 'S': return "n" + std::to_string(read_number<uint16_t>(data, index));
    case 'i': return "n" + std::to_string(read_number<int32_t>(data, index));
    case 'I': return "n" + std::to_string(read_number<uint32_t>(data, index));
    case 'l': return "n" + std::to_string(read_number<int64_t>(data, index));
    case 'L': return "n" + std::to_string(read_number<uint64_t>(data, index));
    case 'f': return numeric_key(read_number<float>(data, index));
    case 'g': return numeric_key(read_number<double>(data, index));
    case 'b': {
      const auto* bits = static_cast<const uint8_t*>(data);
      return (bits[index / 8] & (1U << (index % 8))) ? "n1" : "n0";
    }
    case 'u':
    case 'z': {
      if (values.n_buffers < 3 || !values.buffers[2]) break;
      const auto* offsets = static_cast<const int32_t*>(data);
      const auto* bytes = static_cast<const char*>(values.buffers[2]);
      return std::string(kind == 'u' ? "s" : "b") +
             std::string(bytes + offsets[index],
                         static_cast<size_t>(offsets[index + 1] - offsets[index]));
    }
    case 'U':
    case 'Z': {
      if (values.n_buffers < 3 || !values.buffers[2]) break;
      const auto* offsets = static_cast<const int64_t*>(data);
      const auto* bytes = static_cast<const char*>(values.buffers[2]);
      return std::string(kind == 'U' ? "s" : "b") +
             std::string(bytes + offsets[index],
                         static_cast<size_t>(offsets[index + 1] - offsets[index]));
    }
  }
  throw std::invalid_argument(std::string("Unsupported Arrow graph key type: ") +
                              format);
}

int64_t ArrowBatch::time_us(const std::string& column, int64_t row) const {
  const auto found = columns_.find(column);
  if (found == columns_.end()) {
    throw std::invalid_argument("Arrow batch is missing time column " + column);
  }
  const auto& values = *array_->children[found->second];
  const auto* format = schema_->children[found->second]->format;
  const int64_t index = values.offset + row;
  if (!is_valid(values, index)) {
    throw std::invalid_argument("Graph time column contains NULL");
  }
  if (!format || std::strcmp(format, "l") != 0 || values.n_buffers < 2 ||
      !values.buffers[1]) {
    throw std::invalid_argument("Graph time column must be int64 UTC microseconds");
  }
  const auto time = read_number<int64_t>(values.buffers[1], index);
  if (time == std::numeric_limits<int64_t>::min()) {
    throw std::invalid_argument("Graph time uses reserved static timestamp");
  }
  return time;
}

}  // namespace relx
