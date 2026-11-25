#include "json_array.h"

#include <cctype>
#include <cstdint>
#include <stdexcept>

namespace relx {
namespace {

void append_utf8(std::string& output, uint32_t codepoint) {
  if (codepoint <= 0x7f) {
    output += static_cast<char>(codepoint);
  } else if (codepoint <= 0x7ff) {
    output += static_cast<char>(0xc0 | (codepoint >> 6));
    output += static_cast<char>(0x80 | (codepoint & 0x3f));
  } else if (codepoint <= 0xffff) {
    output += static_cast<char>(0xe0 | (codepoint >> 12));
    output += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f));
    output += static_cast<char>(0x80 | (codepoint & 0x3f));
  } else {
    output += static_cast<char>(0xf0 | (codepoint >> 18));
    output += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f));
    output += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f));
    output += static_cast<char>(0x80 | (codepoint & 0x3f));
  }
}

class Parser {
 public:
  explicit Parser(std::string_view input) : input_(input) {}

  std::vector<JsonScalar> array() {
    space();
    expect('[');
    space();
    std::vector<JsonScalar> values;
    if (take(']')) {
      finish();
      return values;
    }
    while (true) {
      space();
      values.push_back(scalar());
      space();
      if (take(']')) break;
      expect(',');
    }
    finish();
    return values;
  }

 private:
  void space() {
    while (position_ < input_.size() &&
           std::isspace(static_cast<unsigned char>(input_[position_]))) {
      ++position_;
    }
  }

  void expect(char expected) {
    if (!take(expected)) throw std::invalid_argument("Invalid feature JSON array");
  }

  bool take(char expected) {
    if (position_ < input_.size() && input_[position_] == expected) {
      ++position_;
      return true;
    }
    return false;
  }

  void finish() {
    space();
    if (position_ != input_.size()) {
      throw std::invalid_argument("Trailing data in feature JSON array");
    }
  }

  uint32_t hex4() {
    if (position_ + 4 > input_.size()) {
      throw std::invalid_argument("Incomplete JSON Unicode escape");
    }
    uint32_t value = 0;
    for (int index = 0; index < 4; ++index) {
      const char digit = input_[position_++];
      value <<= 4;
      if (digit >= '0' && digit <= '9') value += digit - '0';
      else if (digit >= 'a' && digit <= 'f') value += digit - 'a' + 10;
      else if (digit >= 'A' && digit <= 'F') value += digit - 'A' + 10;
      else throw std::invalid_argument("Invalid JSON Unicode escape");
    }
    return value;
  }

  std::string string() {
    expect('"');
    std::string result;
    while (position_ < input_.size()) {
      const char character = input_[position_++];
      if (character == '"') return result;
      if (static_cast<unsigned char>(character) < 0x20) {
        throw std::invalid_argument("Control character in feature JSON");
      }
      if (character != '\\') {
        result += character;
        continue;
      }
      if (position_ == input_.size()) break;
      const char escaped = input_[position_++];
      if (escaped == 'u') {
        uint32_t codepoint = hex4();
        if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
          expect('\\');
          expect('u');
          const uint32_t low = hex4();
          if (low < 0xdc00 || low > 0xdfff) {
            throw std::invalid_argument("Invalid JSON surrogate pair");
          }
          codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + low - 0xdc00;
        } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
          throw std::invalid_argument("Unpaired JSON surrogate");
        }
        append_utf8(result, codepoint);
      } else {
        switch (escaped) {
          case '"': result += '"'; break;
          case '\\': result += '\\'; break;
          case '/': result += '/'; break;
          case 'b': result += '\b'; break;
          case 'f': result += '\f'; break;
          case 'n': result += '\n'; break;
          case 'r': result += '\r'; break;
          case 't': result += '\t'; break;
          default: throw std::invalid_argument("Invalid feature JSON escape");
        }
      }
    }
    throw std::invalid_argument("Unterminated feature JSON string");
  }

  JsonScalar scalar() {
    if (position_ >= input_.size()) {
      throw std::invalid_argument("Missing JSON array element");
    }
    if (input_[position_] == '"') {
      return {JsonScalar::Kind::kString, string()};
    }
    for (const auto word : {"true", "false", "null"}) {
      const std::string_view literal(word);
      if (input_.substr(position_, literal.size()) == literal) {
        position_ += literal.size();
        return {literal == "null" ? JsonScalar::Kind::kNull
                    : JsonScalar::Kind::kBoolean, std::string(literal)};
      }
    }
    const size_t start = position_;
    take('-');
    if (take('0')) {
      if (position_ < input_.size() &&
          std::isdigit(static_cast<unsigned char>(input_[position_]))) {
        throw std::invalid_argument("Invalid leading zero in feature JSON");
      }
    } else {
      if (position_ == input_.size() || input_[position_] < '1' ||
          input_[position_] > '9') {
        throw std::invalid_argument("Invalid JSON number");
      }
      do { ++position_; } while (position_ < input_.size() &&
          std::isdigit(static_cast<unsigned char>(input_[position_])));
    }
    bool floating = false;
    if (take('.')) {
      floating = true;
      const size_t digits = position_;
      while (position_ < input_.size() &&
             std::isdigit(static_cast<unsigned char>(input_[position_]))) ++position_;
      if (digits == position_) throw std::invalid_argument("Invalid JSON fraction");
    }
    if (take('e') || take('E')) {
      floating = true;
      if (!take('+')) take('-');
      const size_t digits = position_;
      while (position_ < input_.size() &&
             std::isdigit(static_cast<unsigned char>(input_[position_]))) ++position_;
      if (digits == position_) throw std::invalid_argument("Invalid JSON exponent");
    }
    return {floating ? JsonScalar::Kind::kFloat : JsonScalar::Kind::kInteger,
            std::string(input_.substr(start, position_ - start))};
  }

  std::string_view input_;
  size_t position_ = 0;
};

}  // namespace

std::vector<JsonScalar> parse_json_array(std::string_view input) {
  return Parser(input).array();
}

}  // namespace relx
