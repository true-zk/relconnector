#include "sql_time.h"

#include <cstdio>
#include <ctime>
#include <stdexcept>

namespace relx {

int64_t parse_utc_time_us(const std::string& value) {
  if (value.empty()) throw std::invalid_argument("Timestamp text is empty");
  std::tm date{};
  size_t cursor = 0;
  bool only_time = value.size() > 2 && value[2] == ':';
  if (!only_time) {
    int year = 0, month = 0, day = 0;
    if (std::sscanf(value.c_str(), "%d-%d-%d", &year, &month, &day) != 3 ||
        value.size() < 10 || value[4] != '-' || value[7] != '-') {
      throw std::invalid_argument("Unsupported SQLite timestamp: " + value);
    }
    date.tm_year = year - 1900;
    date.tm_mon = month - 1;
    date.tm_mday = day;
    cursor = 10;
    if (cursor < value.size() && (value[cursor] == 'T' || value[cursor] == ' ')) {
      ++cursor;
    }
  }
  if (only_time || cursor < value.size()) {
    if (only_time) cursor = 0;
    int hour = 0, minute = 0, second = 0;
    if (std::sscanf(value.c_str() + cursor, "%d:%d:%d",
                    &hour, &minute, &second) != 3 ||
        cursor + 8 > value.size() || value[cursor + 2] != ':' ||
        value[cursor + 5] != ':') {
      throw std::invalid_argument("Unsupported SQLite timestamp: " + value);
    }
    date.tm_hour = hour;
    date.tm_min = minute;
    date.tm_sec = second;
    cursor += 8;
  }
  int64_t fraction = 0;
  if (cursor < value.size() && value[cursor] == '.') {
    ++cursor;
    int digits = 0;
    while (cursor < value.size() && value[cursor] >= '0' &&
           value[cursor] <= '9') {
      if (digits < 6) fraction = fraction * 10 + value[cursor] - '0';
      ++digits;
      ++cursor;
    }
    if (digits == 0) throw std::invalid_argument("Invalid timestamp fraction");
    while (digits++ < 6) fraction *= 10;
  }
  int offset_seconds = 0;
  if (cursor < value.size() && (value[cursor] == '+' || value[cursor] == '-')) {
    const int sign = value[cursor++] == '+' ? 1 : -1;
    int hours = 0, minutes = 0;
    if (std::sscanf(value.c_str() + cursor, "%d:%d", &hours, &minutes) != 2 ||
        cursor + 5 != value.size()) {
      throw std::invalid_argument("Invalid timestamp UTC offset");
    }
    offset_seconds = sign * (hours * 3600 + minutes * 60);
    cursor += 5;
  }
  if (cursor < value.size() && value[cursor] == 'Z') ++cursor;
  if (cursor != value.size()) {
    throw std::invalid_argument("Unsupported SQLite timestamp: " + value);
  }
  const auto seconds = only_time
      ? date.tm_hour * 3600 + date.tm_min * 60 + date.tm_sec
      : static_cast<int64_t>(timegm(&date));
  return (seconds - offset_seconds) * 1'000'000 + fraction;
}


}  // namespace relx
