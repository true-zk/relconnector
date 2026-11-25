#include "graph_index.h"
#include "bindings_features.h"
#include "bindings_output.h"
#include "bindings_online.h"
#include "bindings_pipeline.h"
#include "bindings_sqlite.h"
#ifdef RELX_WITH_MYSQL
#include "bindings_mysql.h"
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace py = pybind11;

namespace {

py::array_t<int64_t> copy_array(const std::vector<int64_t>& values) {
  py::array_t<int64_t> result(values.size());
  if (!values.empty()) {
    std::memcpy(result.mutable_data(), values.data(),
                values.size() * sizeof(int64_t));
  }
  return result;
}

py::array_t<int64_t> copy_edges(const relx::SampledEdges& edges) {
  py::array_t<int64_t> result(
      py::array::ShapeContainer{static_cast<py::ssize_t>(2),
                                static_cast<py::ssize_t>(edges.source.size())});
  if (!edges.source.empty()) {
    std::memcpy(result.mutable_data(0, 0), edges.source.data(),
                edges.source.size() * sizeof(int64_t));
    std::memcpy(result.mutable_data(1, 0), edges.target.data(),
                edges.target.size() * sizeof(int64_t));
  }
  return result;
}

}  // namespace

py::dict sampled_dict(const relx::SampledBatch& sampled) {
  py::dict nodes;
  py::dict owners;
  py::dict edges;
  py::dict node_counts;
  py::dict edge_counts;
  for (const auto& [table, ids] : sampled.node_ids) {
    nodes[py::str(table)] = copy_array(ids);
    owners[py::str(table)] = copy_array(sampled.node_seed_index.at(table));
  }
  for (const auto& [name, values] : sampled.edge_index) {
    edges[py::str(name)] = copy_edges(values);
  }
  for (const auto& [table, values] : sampled.nodes_per_hop) {
    node_counts[py::str(table)] = copy_array(values);
  }
  for (const auto& [name, values] : sampled.edges_per_hop) {
    edge_counts[py::str(name)] = copy_array(values);
  }
  py::dict result;
  result["node_ids"] = std::move(nodes);
  result["node_seed_index"] = std::move(owners);
  result["edge_index"] = std::move(edges);
  result["nodes_per_hop"] = std::move(node_counts);
  result["edges_per_hop"] = std::move(edge_counts);
  return result;
}

PYBIND11_MODULE(_relx_native, module) {
  module.doc() = "RelX's database-independent native graph topology builder";

  py::class_<relx::GraphIndex>(module, "GraphIndex")
      .def_property_readonly("node_counts", [](const relx::GraphIndex& index) {
        return index.node_counts;
      })
      .def("lookup_nodes", [](const relx::GraphIndex& index,
                              const std::string& table,
                              const std::vector<std::string>& key_columns,
                              const py::object& batch) {
        relx::ArrowBatch view(batch);
        std::vector<int64_t> ids;
        {
          py::gil_scoped_release release;
          ids = index.lookup_nodes(table, key_columns, view);
        }
        return copy_array(ids);
      })
      .def("relation_names", [](const relx::GraphIndex& index) {
        std::vector<std::string> names;
        names.reserve(index.relations.size());
        for (const auto& [name, _] : index.relations) names.push_back(name);
        std::sort(names.begin(), names.end());
        return names;
      })
      .def("relation_endpoints", [](const relx::GraphIndex& index,
                                     const std::string& name) {
        const auto found = index.relations.find(name);
        if (found == index.relations.end()) {
          throw std::invalid_argument("Unknown relation: " + name);
        }
        return py::make_tuple(found->second.source, found->second.target);
      })
      .def("node_times", [](const relx::GraphIndex& index,
                            const std::string& table) {
        const auto found = index.node_time.find(table);
        if (found == index.node_time.end()) {
          throw std::invalid_argument("Unknown graph table: " + table);
        }
        return copy_array(found->second);
      })
      .def("visible_neighbors", &relx::GraphIndex::visible_neighbors)
      .def("sample_neighbors", [](const relx::GraphIndex& index,
                                   const std::string& relation,
                                   const std::vector<int64_t>& destinations,
                                   const std::vector<int64_t>& cutoffs_us,
                                   int64_t fanout, uint64_t random_seed,
                                   int64_t workers) {
        std::vector<std::vector<int64_t>> sampled;
        {
          py::gil_scoped_release release;
          sampled = index.sample_neighbors(relation, destinations, cutoffs_us,
                                           fanout, random_seed, workers);
        }
        return sampled;
      }, py::arg("relation"), py::arg("destinations"),
           py::arg("cutoffs_us"), py::arg("fanout"),
           py::arg("random_seed"), py::arg("workers") = 1)
      .def("sample_one_hop", [](const relx::GraphIndex& index,
                                 const std::string& seed_table,
                                 const std::vector<int64_t>& seed_ids,
                                 const std::vector<int64_t>& cutoffs_us,
                                 int64_t fanout, uint64_t random_seed,
                                 int64_t workers) {
        relx::SampledBatch sampled;
        {
          py::gil_scoped_release release;
          sampled = index.sample_one_hop(seed_table, seed_ids, cutoffs_us,
                                         fanout, random_seed, workers);
        }
        return sampled_dict(sampled);
      }, py::arg("seed_table"), py::arg("seed_ids"),
           py::arg("cutoffs_us"), py::arg("fanout"),
           py::arg("random_seed"), py::arg("workers") = 1)
      .def("sample_multi_hop", [](const relx::GraphIndex& index,
                                   const std::string& seed_table,
                                   const std::vector<int64_t>& seed_ids,
                                   const std::vector<int64_t>& cutoffs_us,
                                   const std::vector<int64_t>& fanouts,
                                   uint64_t random_seed, int64_t workers) {
        relx::SampledBatch sampled;
        {
          py::gil_scoped_release release;
          sampled = index.sample_multi_hop(seed_table, seed_ids, cutoffs_us,
                                           fanouts, random_seed, workers);
        }
        return sampled_dict(sampled);
      }, py::arg("seed_table"), py::arg("seed_ids"),
           py::arg("cutoffs_us"), py::arg("fanouts"),
           py::arg("random_seed"), py::arg("workers") = 1)
      .def("relation", [](const relx::GraphIndex& index,
                          const std::string& name) {
        const auto found = index.relations.find(name);
        if (found == index.relations.end()) {
          throw std::invalid_argument("Unknown relation: " + name);
        }
        const auto& relation = found->second;
        py::dict result;
        result["source"] = relation.source;
        result["target"] = relation.target;
        result["colptr"] = copy_array(relation.colptr);
        result["row"] = copy_array(relation.row);
        result["edge_time"] = copy_array(relation.edge_time);
        result["null_keys"] = relation.null_keys;
        result["dangling_keys"] = relation.dangling_keys;
        return result;
      });

  py::class_<relx::GraphBuilder>(module, "GraphBuilder")
      .def(py::init<bool>(), py::arg("inclusive") = true)
      .def("register_table", &relx::GraphBuilder::register_table,
           py::arg("table"), py::arg("timed") = false)
      .def("register_relation", &relx::GraphBuilder::register_relation)
      .def("add_nodes", [](relx::GraphBuilder& builder,
                           const std::string& table,
                           const std::vector<std::string>& key_columns,
                           const py::object& batch) {
        relx::ArrowBatch view(batch);
        py::gil_scoped_release release;
        builder.add_nodes(table, key_columns, view);
      })
      .def("add_edges", [](relx::GraphBuilder& builder,
                           const std::string& relation,
                           const std::string& source,
                           const std::string& target,
                           const std::vector<std::string>& source_key_columns,
                           const std::vector<std::string>& foreign_key_columns,
                           const py::object& batch) {
        relx::ArrowBatch view(batch);
        py::gil_scoped_release release;
        builder.add_edges(relation, source, target, source_key_columns,
                          foreign_key_columns, view);
      })
      .def("finish", &relx::GraphBuilder::finish);

  bind_native_features(module);
  bind_native_pipeline(module);
  bind_sqlite_features(module);
#ifdef RELX_WITH_MYSQL
  bind_mysql_features(module);
#endif
  bind_native_online_pipeline(module);
}
