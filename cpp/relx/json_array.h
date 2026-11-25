#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace relx {

struct JsonScalar {
  enum class Kind { kNull, kBoolean, kInteger, kFloat, kString } kind;
  std::string text;
};

std::vector<JsonScalar> parse_json_array(std::string_view input);

}  // namespace relx
