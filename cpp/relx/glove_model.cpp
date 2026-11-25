#include "glove_model.h"

#include <sqlite3.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace relx {
namespace {

uint32_t read_u32(std::ifstream& stream) {
  uint32_t value = 0;
  stream.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!stream) throw std::runtime_error("Invalid RelX GloVe vocabulary file");
  return value;
}

std::string read_string(std::ifstream& stream) {
  const auto length = read_u32(stream);
  if (length > 1'000'000) throw std::runtime_error("GloVe token is too long");
  std::string value(length, '\0');
  stream.read(value.data(), length);
  if (!stream) throw std::runtime_error("Invalid RelX GloVe token");
  return value;
}

std::string lowercase(std::string value) {
  for (auto& character : value) {
    character = static_cast<char>(std::tolower(
        static_cast<unsigned char>(character)));
  }
  return value;
}

std::string strip_punctuation(const std::string& token) {
  constexpr const char* punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
  const auto first = token.find_first_not_of(punctuation);
  if (first == std::string::npos) return "";
  const auto last = token.find_last_not_of(punctuation);
  return token.substr(first, last - first + 1);
}

}  // namespace

GloveModel::GloveModel(const std::string& directory, size_t cache_bytes,
                       const std::string& disk_cache_path)
    : cache_limit_(cache_bytes) {
  std::ifstream vocabulary(directory + "/vocab.bin", std::ios::binary);
  if (!vocabulary) throw std::runtime_error("Open RelX GloVe vocabulary failed");
  if (read_u32(vocabulary) != 0x47584c52 || read_u32(vocabulary) != 1) {
    throw std::runtime_error("Unsupported RelX GloVe export version");
  }
  const auto count = read_u32(vocabulary);
  width_ = read_u32(vocabulary);
  lower_case_ = read_u32(vocabulary) != 0;
  const auto stop_count = read_u32(vocabulary);
  if (count == 0 || width_ <= 0 || width_ > 4096) {
    throw std::runtime_error("Invalid RelX GloVe dimensions");
  }
  vocab_.reserve(count);
  for (uint32_t index = 0; index < count; ++index) {
    vocab_.emplace(read_string(vocabulary), index);
  }
  for (uint32_t index = 0; index < stop_count; ++index) {
    stop_words_.insert(read_string(vocabulary));
  }
  std::ifstream weights(directory + "/weights.f32", std::ios::binary | std::ios::ate);
  if (!weights) throw std::runtime_error("Open RelX GloVe weights failed");
  const auto bytes = static_cast<uint64_t>(weights.tellg());
  const auto expected = static_cast<uint64_t>(count) * width_ * sizeof(float);
  if (bytes != expected) throw std::runtime_error("RelX GloVe weights shape differs");
  weights_.resize(static_cast<size_t>(count) * width_);
  weights.seekg(0);
  weights.read(reinterpret_cast<char*>(weights_.data()), bytes);
  if (!weights) throw std::runtime_error("Read RelX GloVe weights failed");
  if (!disk_cache_path.empty()) {
    if (sqlite3_open_v2(disk_cache_path.c_str(), &disk_db_,
            SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
            nullptr) != SQLITE_OK) {
      const std::string error = disk_db_ ? sqlite3_errmsg(disk_db_) : "unknown";
      if (disk_db_) sqlite3_close(disk_db_);
      disk_db_ = nullptr;
      throw std::runtime_error("Open GloVe disk cache: " + error);
    }
    char* error = nullptr;
    const char* schema = "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; "
        "CREATE TABLE IF NOT EXISTS embeddings "
        "(text BLOB PRIMARY KEY, vector BLOB NOT NULL) WITHOUT ROWID;";
    if (sqlite3_exec(disk_db_, schema, nullptr, nullptr, &error) != SQLITE_OK) {
      const std::string message = error ? error : "Unknown GloVe cache error";
      sqlite3_free(error);
      sqlite3_close(disk_db_);
      disk_db_ = nullptr;
      throw std::runtime_error(message);
    }
  }
}

GloveModel::~GloveModel() {
  if (disk_db_) sqlite3_close(disk_db_);
}

std::vector<float> GloveModel::encode(const std::string& text) const {
  if (cache_limit_) {
    std::lock_guard<std::mutex> guard(cache_mutex_);
    const auto found = cache_.find(text);
    if (found != cache_.end()) {
      recent_.splice(recent_.end(), recent_, found->second.position);
      return found->second.value;
    }
  }
  std::vector<float> value;
  if (!disk_lookup(text, value)) {
    value = encode_uncached(text);
    disk_store(text, value);
  }
  remember(text, value);
  return value;
}

std::vector<float> GloveModel::encode_uncached(const std::string& text) const {
  std::vector<float> result(static_cast<size_t>(width_), 0.0F);
  uint64_t count = 0;
  size_t position = 0;
  while (position < text.size()) {
    while (position < text.size() && std::isspace(
        static_cast<unsigned char>(text[position]))) ++position;
    const auto start = position;
    while (position < text.size() && !std::isspace(
        static_cast<unsigned char>(text[position]))) ++position;
    if (start == position) continue;
    std::string token = text.substr(start, position - start);
    if (lower_case_) token = lowercase(std::move(token));
    if (stop_words_.count(token)) continue;
    auto found = vocab_.find(token);
    if (found == vocab_.end()) {
      token = strip_punctuation(token);
      if (stop_words_.count(token)) continue;
      found = vocab_.find(token);
      if (found == vocab_.end()) {
        token = lowercase(std::move(token));
        if (stop_words_.count(token)) continue;
        found = vocab_.find(token);
      }
    }
    if (found == vocab_.end()) continue;
    const float* vector = weights_.data() + static_cast<size_t>(found->second) * width_;
    for (int64_t index = 0; index < width_; ++index) {
      result[static_cast<size_t>(index)] += vector[index];
    }
    ++count;
  }
  if (count) {
    for (auto& value : result) value /= static_cast<float>(count);
  }
  return result;
}

void GloveModel::remember(const std::string& text,
                          const std::vector<float>& value) const {
  const size_t bytes = text.size() + value.size() * sizeof(float) + 96;
  if (bytes > cache_limit_) return;
  std::lock_guard<std::mutex> guard(cache_mutex_);
  if (cache_.count(text)) return;
  while (!recent_.empty() && cache_bytes_ + bytes > cache_limit_) {
    const auto found = cache_.find(recent_.front());
    cache_bytes_ -= found->second.bytes;
    cache_.erase(found);
    recent_.pop_front();
  }
  recent_.push_back(text);
  cache_.emplace(text, CacheEntry{value, std::prev(recent_.end()), bytes});
  cache_bytes_ += bytes;
}

bool GloveModel::disk_lookup(const std::string& text,
                             std::vector<float>& value) const {
  if (!disk_db_) return false;
  std::lock_guard<std::mutex> guard(disk_mutex_);
  sqlite3_stmt* statement = nullptr;
  if (sqlite3_prepare_v2(disk_db_,
          "SELECT vector FROM embeddings WHERE text = ?1", -1,
          &statement, nullptr) != SQLITE_OK) {
    throw std::runtime_error(sqlite3_errmsg(disk_db_));
  }
  sqlite3_bind_blob(statement, 1, text.data(),
                    static_cast<int>(text.size()), SQLITE_TRANSIENT);
  const int status = sqlite3_step(statement);
  if (status == SQLITE_ROW) {
    if (static_cast<size_t>(sqlite3_column_bytes(statement, 0)) !=
        static_cast<size_t>(width_) * sizeof(float)) {
      sqlite3_finalize(statement);
      throw std::runtime_error("GloVe disk cache vector width differs");
    }
    value.resize(static_cast<size_t>(width_));
    std::memcpy(value.data(), sqlite3_column_blob(statement, 0),
                value.size() * sizeof(float));
  }
  sqlite3_finalize(statement);
  if (status != SQLITE_ROW && status != SQLITE_DONE) {
    throw std::runtime_error(sqlite3_errmsg(disk_db_));
  }
  return status == SQLITE_ROW;
}

void GloveModel::disk_store(const std::string& text,
                            const std::vector<float>& value) const {
  if (!disk_db_) return;
  std::lock_guard<std::mutex> guard(disk_mutex_);
  sqlite3_stmt* statement = nullptr;
  if (sqlite3_prepare_v2(disk_db_,
          "INSERT OR IGNORE INTO embeddings (text, vector) VALUES (?1, ?2)",
          -1, &statement, nullptr) != SQLITE_OK) {
    throw std::runtime_error(sqlite3_errmsg(disk_db_));
  }
  sqlite3_bind_blob(statement, 1, text.data(),
                    static_cast<int>(text.size()), SQLITE_TRANSIENT);
  sqlite3_bind_blob(statement, 2, value.data(),
                    static_cast<int>(value.size() * sizeof(float)), SQLITE_TRANSIENT);
  const int status = sqlite3_step(statement);
  sqlite3_finalize(statement);
  if (status != SQLITE_DONE) throw std::runtime_error(sqlite3_errmsg(disk_db_));
}

}  // namespace relx
