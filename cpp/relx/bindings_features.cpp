#include "bindings_features.h"
#include "bindings_output.h"
#include "native_features.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
namespace {

template <typename T>
std::vector<T> read_vector(const py::array_t<T, py::array::c_style |
                                           py::array::forcecast>& array) {
  if (array.ndim() != 1) {
    throw std::invalid_argument("Native feature values must be one-dimensional");
  }
  return {array.data(), array.data() + array.size()};
}

template <typename T>
py::array_t<T> copy_vector(const std::vector<T>& values) {
  py::array_t<T> output(values.size());
  if (!values.empty()) {
    std::memcpy(output.mutable_data(), values.data(),
                values.size() * sizeof(T));
  }
  return output;
}

}  // namespace

py::dict feature_batch_dict(const relx::FeatureBatch& batch) {
  py::dict tables;
  for (const auto& [table, columns] : batch) {
    py::dict features;
    for (const auto& [name, block] : columns) {
      py::dict value;
      value["missing"] = py::none();
      value["offsets"] = py::none();
      if (block.kind == relx::FeatureKind::kCategorical ||
          block.kind == relx::FeatureKind::kMulticategorical) {
        value["values"] = copy_vector(block.integers);
        if (block.kind == relx::FeatureKind::kMulticategorical) {
          value["offsets"] = copy_vector(block.offsets);
        }
      } else if (block.kind == relx::FeatureKind::kText) {
        py::list strings;
        for (const auto& item : block.text) strings.append(py::str(item));
        value["values"] = py::module_::import("numpy").attr("asarray")(
            strings, py::arg("dtype") = "object");
      } else if (block.kind == relx::FeatureKind::kRaw) {
        py::list raw;
        for (const auto& item : block.raw) {
          if (std::holds_alternative<std::monostate>(item)) raw.append(py::none());
          else if (const auto* number = std::get_if<int64_t>(&item)) {
            raw.append(py::int_(*number));
          } else if (const auto* number = std::get_if<double>(&item)) {
            raw.append(py::float_(*number));
          } else if (const auto* string = std::get_if<std::string>(&item)) {
            raw.append(py::str(*string));
          } else if (const auto* blob = std::get_if<relx::RawBlob>(&item)) {
            raw.append(py::bytes(blob->data));
          }
        }
        value["values"] = py::module_::import("numpy").attr("asarray")(
            raw, py::arg("dtype") = "object");
      } else {
        py::array_t<float> numbers(
            py::array::ShapeContainer{
                static_cast<py::ssize_t>(block.rows),
                static_cast<py::ssize_t>(block.width)});
        if (!block.floats.empty()) {
          std::memcpy(numbers.mutable_data(), block.floats.data(),
                      block.floats.size() * sizeof(float));
        }
        if (!block.missing.empty() ||
            (block.kind != relx::FeatureKind::kEmbedding &&
             block.kind != relx::FeatureKind::kTextEmbedding)) {
          py::array_t<bool> missing(block.missing.size());
          for (size_t row = 0; row < block.missing.size(); ++row) {
            missing.mutable_data()[row] = block.missing[row] != 0;
          }
          value["missing"] = std::move(missing);
        }
        value["values"] = std::move(numbers);
      }
      features[py::str(name)] = std::move(value);
    }
    tables[py::str(table)] = std::move(features);
  }
  return tables;
}

void bind_native_features(py::module_& module) {
  py::class_<relx::NativeFeatureStore>(module, "NativeFeatureStore")
      .def(py::init<>())
      .def("add_numeric", [](relx::NativeFeatureStore& store,
                              const std::string& table,
                              const std::string& column,
                              const py::array_t<double, py::array::c_style |
                                                       py::array::forcecast>& values,
                              double mean, double std) {
        store.add_numeric(table, column, read_vector(values), mean, std);
      })
      .def("add_categorical", [](relx::NativeFeatureStore& store,
                                  const std::string& table,
                                  const std::string& column,
                                  const py::array_t<int64_t, py::array::c_style |
                                                       py::array::forcecast>& values) {
        store.add_categorical(table, column, read_vector(values));
      })
      .def("add_timestamp", [](relx::NativeFeatureStore& store,
                                const std::string& table,
                                const std::string& column,
                                const py::array_t<int64_t, py::array::c_style |
                                                       py::array::forcecast>& values,
                                double mean, double std) {
        store.add_timestamp(table, column, read_vector(values), mean, std);
      })
      .def("add_multicategorical", [](relx::NativeFeatureStore& store,
                                       const std::string& table,
                                       const std::string& column,
                                       const py::array_t<int64_t, py::array::c_style |
                                                            py::array::forcecast>& values,
                                       const py::array_t<int64_t, py::array::c_style |
                                                            py::array::forcecast>& offsets) {
        store.add_multicategorical(table, column, read_vector(values),
                                   read_vector(offsets));
      })
      .def("add_embedding", [](relx::NativeFeatureStore& store,
                                const std::string& table,
                                const std::string& column,
                                const py::array_t<float, py::array::c_style |
                                                         py::array::forcecast>& values) {
        if (values.ndim() != 2) {
          throw std::invalid_argument("Embedding values must be two-dimensional");
        }
        store.add_embedding(
            table, column, {values.data(), values.data() + values.size()},
            values.shape(1));
      })
      .def("add_text", &relx::NativeFeatureStore::add_text)
      .def("gather", [](const relx::NativeFeatureStore& store,
                        const py::dict& node_ids, int64_t workers) {
        std::unordered_map<std::string, std::vector<int64_t>> ids;
        for (const auto& [key, value] : node_ids) {
          const auto array = py::cast<
              py::array_t<int64_t, py::array::c_style |
                                        py::array::forcecast>>(value);
          ids.emplace(py::cast<std::string>(key), read_vector(array));
        }
        relx::FeatureBatch result;
        {
          py::gil_scoped_release release;
          result = store.gather(ids, workers);
        }
        return feature_batch_dict(result);
      }, py::arg("node_ids"), py::arg("workers") = 1);
}
