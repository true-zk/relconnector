#include "bindings_sql_common.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <stdexcept>

namespace py = pybind11;
namespace relx::binding {

SqlKey key_value(const py::handle& value) {
  if (py::isinstance<py::bool_>(value)) {
    return static_cast<int64_t>(py::cast<bool>(value));
  }
  if (py::isinstance<py::int_>(value)) return py::cast<int64_t>(value);
  if (py::isinstance<py::float_>(value)) return py::cast<double>(value);
  if (py::isinstance<py::bytes>(value)) {
    return SqlBlob{py::cast<std::string>(value)};
  }
  if (py::isinstance<py::str>(value)) return py::cast<std::string>(value);
  throw std::invalid_argument("Unsupported feature identity key type");
}

FeatureKind feature_kind(const std::string& kind) {
  if (kind == "numerical") return FeatureKind::kNumeric;
  if (kind == "categorical") return FeatureKind::kCategorical;
  if (kind == "timestamp") return FeatureKind::kTimestamp;
  if (kind == "multicategorical") return FeatureKind::kMulticategorical;
  if (kind == "embedding") return FeatureKind::kEmbedding;
  if (kind == "text") return FeatureKind::kText;
  if (kind == "text_glove") return FeatureKind::kTextEmbedding;
  if (kind == "raw") return FeatureKind::kRaw;
  throw std::invalid_argument("Unsupported native feature kind: " + kind);
}

std::vector<std::vector<SqlKey>> table_keys(const py::iterable& rows) {
  std::vector<std::vector<SqlKey>> keys;
  for (const auto& row : rows) {
    std::vector<SqlKey> key;
    for (const auto& value : py::cast<py::iterable>(row)) {
      key.push_back(key_value(value));
    }
    keys.push_back(std::move(key));
  }
  return keys;
}

std::unordered_map<std::string, std::vector<int64_t>> sampled_ids(
    const py::dict& node_ids) {
  std::unordered_map<std::string, std::vector<int64_t>> ids;
  for (const auto& [key, value] : node_ids) {
    const auto array = py::cast<py::array_t<
        int64_t, py::array::c_style | py::array::forcecast>>(value);
    if (array.ndim() != 1) {
      throw std::invalid_argument("Sampled node IDs must be one-dimensional");
    }
    ids.emplace(py::cast<std::string>(key),
                std::vector<int64_t>(array.data(), array.data() + array.size()));
  }
  return ids;
}

}  // namespace relx::binding
