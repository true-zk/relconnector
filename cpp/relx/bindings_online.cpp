#include "bindings_output.h"
#include "native_online_pipeline.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

void bind_native_online_pipeline(py::module_& module) {
  py::class_<relx::NativeOnlinePipeline>(module, "NativeOnlinePipeline")
      .def(py::init<const relx::NativeFeatureReader&, size_t, size_t, size_t>(),
           py::arg("reader"), py::arg("workers"), py::arg("max_inflight"),
           py::arg("max_ready_bytes") = 256 * 1024 * 1024,
           py::keep_alive<1, 2>())
      .def("submit", [](relx::NativeOnlinePipeline& pipeline,
                        uint64_t sequence, const py::dict& node_ids) {
        std::unordered_map<std::string, std::vector<int64_t>> ids;
        for (const auto& [key, value] : node_ids) {
          const auto array = py::cast<py::array_t<
              int64_t, py::array::c_style | py::array::forcecast>>(value);
          if (array.ndim() != 1) {
            throw std::invalid_argument("Sampled node IDs must be one-dimensional");
          }
          ids.emplace(py::cast<std::string>(key),
                      std::vector<int64_t>(array.data(),
                                           array.data() + array.size()));
        }
        py::gil_scoped_release release;
        pipeline.submit(sequence, std::move(ids));
      })
      .def("finish", &relx::NativeOnlinePipeline::finish)
      .def("abort", &relx::NativeOnlinePipeline::abort)
      .def("next", [](relx::NativeOnlinePipeline& pipeline) -> py::object {
        std::optional<relx::OnlineBatchResult> result;
        {
          py::gil_scoped_release release;
          result = pipeline.next();
        }
        if (!result) return py::none();
        py::dict batch;
        batch["sequence"] = result->sequence;
        batch["features"] = feature_batch_dict(result->features);
        return std::move(batch);
      })
      .def("close", [](relx::NativeOnlinePipeline& pipeline) {
        py::gil_scoped_release release;
        pipeline.close();
      });
}
