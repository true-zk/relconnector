#pragma once

#include <pybind11/pybind11.h>
#include <arrow/c/abi.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace relx {

class ArrowBatch {
 public:
  explicit ArrowBatch(const pybind11::object& batch);

  int64_t length() const { return array_->length; }
  std::optional<std::string> key(
      const std::vector<std::string>& columns, int64_t row) const;
  int64_t time_us(const std::string& column, int64_t row) const;

 private:
  std::optional<std::string> scalar(size_t column, int64_t row) const;

  pybind11::tuple owners_;
  const ArrowSchema* schema_;
  const ArrowArray* array_;
  std::unordered_map<std::string, size_t> columns_;
};

}  // namespace relx
