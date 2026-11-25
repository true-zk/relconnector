#include "sql_number.h"

#include <charconv>
#include <stdexcept>
#include <string>

namespace relx {
namespace {

std::string render(double value, std::chars_format format) {
  char buffer[512];
  const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value, format);
  if (result.ec != std::errc{}) {
    throw std::runtime_error("Format SQL floating-point category failed");
  }
  return {buffer, result.ptr};
}

}  // namespace

std::string python_float_token(double value) {
  const std::string scientific = render(value, std::chars_format::scientific);
  const auto marker = scientific.find('e');
  if (marker == std::string::npos) return scientific;
  const int exponent = std::stoi(scientific.substr(marker + 1));
  if (exponent < -4 || exponent >= 16) return scientific;
  std::string fixed = render(value, std::chars_format::fixed);
  if (fixed.find('.') == std::string::npos) fixed += ".0";
  return fixed;
}

}  // namespace relx
