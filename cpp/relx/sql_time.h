#pragma once

#include <cstdint>
#include <string>

namespace relx {
int64_t parse_utc_time_us(const std::string& value);
}
