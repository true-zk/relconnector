#include "bindings_output.h"
#include "bindings_sql_common.h"
#include "sqlite_features.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;
void bind_sqlite_features(py::module_& module) {
  py::class_<relx::NativeFeatureReader>(module, "NativeFeatureReader");
  py::class_<relx::SqliteFeatureReader, relx::NativeFeatureReader>(
      module, "SqliteFeatureReader")
      .def(py::init<std::string>())
      .def("add_table", [](relx::SqliteFeatureReader& reader,
                            std::string name, std::string schema,
                            std::vector<std::string> identity,
                            const py::iterable& rows) {
        reader.add_table(std::move(name), std::move(schema),
                         std::move(identity), relx::binding::table_keys(rows));
      })
      .def("add_feature", [](relx::SqliteFeatureReader& reader,
                              const std::string& table,
                              const std::string& column,
                              const std::string& kind,
                              double mean, double std, int64_t width,
                              std::unordered_map<std::string, int64_t> vocab) {
        reader.add_feature(table, relx::SqliteFeatureSpec{
            column, relx::binding::feature_kind(kind), mean, std, width, std::move(vocab),
            nullptr});
      }, py::arg("table"), py::arg("column"), py::arg("kind"),
           py::arg("mean") = 0.0, py::arg("std") = 1.0,
           py::arg("width") = 1, py::arg("vocab") =
           std::unordered_map<std::string, int64_t>{})
      .def("set_glove_model", &relx::SqliteFeatureReader::set_glove_model,
           py::arg("directory"), py::arg("cache_bytes") = 0,
           py::arg("disk_cache_path") = "")
      .def("set_row_cache_bytes", &relx::SqliteFeatureReader::set_row_cache_bytes)
      .def("fetch", [](const relx::SqliteFeatureReader& reader,
                        const py::dict& node_ids) {
        auto ids = relx::binding::sampled_ids(node_ids);
        relx::FeatureBatch result;
        {
          py::gil_scoped_release release;
          result = reader.fetch(ids);
        }
        return feature_batch_dict(result);
      });
}
