#pragma once

#include "graph_index.h"
#include "native_features.h"

#include <pybind11/pybind11.h>

pybind11::dict sampled_dict(const relx::SampledBatch& sampled);
pybind11::dict feature_batch_dict(const relx::FeatureBatch& batch);
