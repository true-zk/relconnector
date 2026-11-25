#pragma once

#include "online_sql_features.h"

#include <pybind11/pybind11.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace relx::binding {

SqlKey key_value(const pybind11::handle& value);
FeatureKind feature_kind(const std::string& kind);
std::vector<std::vector<SqlKey>> table_keys(const pybind11::iterable& rows);
std::unordered_map<std::string, std::vector<int64_t>> sampled_ids(
    const pybind11::dict& node_ids);

}  // namespace relx::binding
