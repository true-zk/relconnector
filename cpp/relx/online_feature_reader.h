#pragma once

#include "native_features.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace relx {

class NativeFeatureSession {
 public:
  virtual ~NativeFeatureSession() = default;
  virtual FeatureBatch fetch(
      const std::unordered_map<std::string, std::vector<int64_t>>& node_ids
  ) = 0;
};

class NativeFeatureReader {
 public:
  virtual ~NativeFeatureReader() = default;
  virtual std::unique_ptr<NativeFeatureSession> open_session() const = 0;
};

}  // namespace relx
