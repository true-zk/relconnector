#pragma once

#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct sqlite3;

namespace relx {

class GloveModel {
 public:
  GloveModel(const std::string& directory, size_t cache_bytes = 0,
             const std::string& disk_cache_path = "");
  ~GloveModel();
  std::vector<float> encode(const std::string& text) const;
  int64_t width() const { return width_; }

 private:
  std::vector<float> encode_uncached(const std::string& text) const;
  bool disk_lookup(const std::string& text, std::vector<float>& value) const;
  void disk_store(const std::string& text, const std::vector<float>& value) const;
  void remember(const std::string& text, const std::vector<float>& value) const;

  struct CacheEntry {
    std::vector<float> value;
    std::list<std::string>::iterator position;
    size_t bytes;
  };

  int64_t width_ = 0;
  bool lower_case_ = false;
  std::unordered_map<std::string, uint32_t> vocab_;
  std::unordered_set<std::string> stop_words_;
  std::vector<float> weights_;
  size_t cache_limit_ = 0;
  mutable size_t cache_bytes_ = 0;
  mutable std::mutex cache_mutex_;
  mutable std::list<std::string> recent_;
  mutable std::unordered_map<std::string, CacheEntry> cache_;
  mutable std::mutex disk_mutex_;
  ::sqlite3* disk_db_ = nullptr;
};

}  // namespace relx
