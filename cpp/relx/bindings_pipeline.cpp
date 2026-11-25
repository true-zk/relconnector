#include "bindings_output.h"
#include "bindings_pipeline.h"
#include "native_pipeline.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

namespace py = pybind11;

void bind_native_pipeline(py::module_& module) {
  py::class_<relx::NativeBatchPipeline>(module, "NativeBatchPipeline")
      .def(py::init<const relx::GraphIndex&, const relx::NativeFeatureStore&,
                    std::vector<int64_t>, bool, int64_t, int64_t, int64_t,
                    size_t>(),
           py::arg("graph"), py::arg("features"), py::arg("fanouts"),
           py::arg("multi_hop"), py::arg("sampling_workers"),
           py::arg("feature_workers"), py::arg("feature_native_workers"),
           py::arg("max_inflight"), py::keep_alive<1, 2>(),
           py::keep_alive<1, 3>())
      .def("submit", [](relx::NativeBatchPipeline& pipeline,
                        uint64_t sequence, const std::string& seed_table,
                        const std::vector<int64_t>& seed_ids,
                        const std::vector<int64_t>& cutoffs_us,
                        uint64_t random_seed) {
        py::gil_scoped_release release;
        pipeline.submit(sequence, seed_table, seed_ids, cutoffs_us,
                        random_seed);
      })
      .def("finish", &relx::NativeBatchPipeline::finish)
      .def("next", [](relx::NativeBatchPipeline& pipeline) -> py::object {
        std::optional<relx::NativeBatchResult> result;
        {
          py::gil_scoped_release release;
          result = pipeline.next();
        }
        if (!result) return py::none();
        py::dict batch;
        batch["sequence"] = result->sequence;
        batch["sample"] = sampled_dict(result->sample);
        batch["features"] = feature_batch_dict(result->features);
        return std::move(batch);
      })
      .def("close", [](relx::NativeBatchPipeline& pipeline) {
        py::gil_scoped_release release;
        pipeline.close();
      });
}
