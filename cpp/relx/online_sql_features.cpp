#include "online_sql_features.h"

#include <stdexcept>

namespace relx {

NativeFeatureBlock feature_block_for(const SqlFeatureSpec& spec, size_t rows) {
  NativeFeatureBlock block;
  block.kind = spec.kind;
  block.rows = static_cast<int64_t>(rows);
  block.width = spec.width;
  if (spec.kind == FeatureKind::kMulticategorical) block.offsets.push_back(0);
  return block;
}

void copy_feature_row(const SqlFeatureSpec& feature,
                      const NativeFeatureBlock& source, size_t row,
                      NativeFeatureBlock& target) {
  if (feature.kind == FeatureKind::kCategorical) {
    target.integers.push_back(source.integers.at(row));
  } else if (feature.kind == FeatureKind::kMulticategorical) {
    const auto begin = static_cast<size_t>(source.offsets.at(row));
    const auto end = static_cast<size_t>(source.offsets.at(row + 1));
    target.integers.insert(target.integers.end(),
                           source.integers.begin() + begin,
                           source.integers.begin() + end);
    target.offsets.push_back(static_cast<int64_t>(target.integers.size()));
  } else if (feature.kind == FeatureKind::kText) {
    target.text.push_back(source.text.at(row));
  } else if (feature.kind == FeatureKind::kRaw) {
    target.raw.push_back(source.raw.at(row));
  } else if (feature.kind == FeatureKind::kEmbedding ||
             feature.kind == FeatureKind::kTextEmbedding) {
    const auto begin = row * static_cast<size_t>(feature.width);
    target.floats.insert(target.floats.end(),
                         source.floats.begin() + begin,
                         source.floats.begin() + begin + feature.width);
  } else {
    target.floats.push_back(source.floats.at(row));
    target.missing.push_back(source.missing.at(row));
  }
}

size_t feature_cache_bytes(const SqlTableCache& cache) {
  size_t bytes = cache.positions.size() * (sizeof(int64_t) + 3 * sizeof(size_t));
  for (const auto& [name, block] : cache.blocks) {
    bytes += name.capacity() + sizeof(block);
    bytes += block.floats.capacity() * sizeof(float);
    bytes += block.integers.capacity() * sizeof(int64_t);
    bytes += block.missing.capacity();
    bytes += block.offsets.capacity() * sizeof(int64_t);
    for (const auto& value : block.text) bytes += sizeof(value) + value.capacity();
    for (const auto& value : block.raw) {
      bytes += sizeof(value);
      if (const auto* text = std::get_if<std::string>(&value)) {
        bytes += text->capacity();
      } else if (const auto* blob = std::get_if<RawBlob>(&value)) {
        bytes += blob->data.capacity();
      }
    }
  }
  return bytes;
}

void fetch_sql_table(const SqlTableSpec& table, const std::vector<int64_t>& ids,
                     FeatureBatch& output, SqlTableCache* cache,
                     const ReadMissingRows& read_missing) {
  if (table.features.empty() || ids.empty()) return;
  std::vector<int64_t> unique;
  std::vector<size_t> inverse;
  std::unordered_map<int64_t, size_t> positions;
  unique.reserve(ids.size());
  inverse.reserve(ids.size());
  for (const auto id : ids) {
    if (id < 0 || static_cast<size_t>(id) >= table.keys.size()) {
      throw std::out_of_range("Sampled node ID is outside " + table.name);
    }
    const auto [found, inserted] = positions.emplace(id, unique.size());
    if (inserted) unique.push_back(id);
    inverse.push_back(found->second);
  }
  std::vector<int64_t> missing_ids;
  missing_ids.reserve(unique.size());
  for (const auto id : unique) {
    if (!cache || !cache->positions.count(id)) missing_ids.push_back(id);
  }
  std::unordered_map<std::string, NativeFeatureBlock> fetched;
  for (const auto& feature : table.features) {
    fetched.emplace(feature.column, feature_block_for(feature, missing_ids.size()));
  }
  if (!missing_ids.empty()) read_missing(missing_ids, fetched);
  std::unordered_map<std::string, NativeFeatureBlock> unique_blocks;
  for (const auto& feature : table.features) {
    auto& target = unique_blocks[feature.column];
    target = feature_block_for(feature, unique.size());
    size_t fetched_position = 0;
    for (const auto id : unique) {
      if (cache) {
        const auto hit = cache->positions.find(id);
        if (hit != cache->positions.end()) {
          copy_feature_row(feature, cache->blocks.at(feature.column),
                           hit->second, target);
          continue;
        }
      }
      copy_feature_row(feature, fetched.at(feature.column),
                       fetched_position++, target);
    }
  }
  const std::string label = table.schema.empty()
      ? table.name : table.schema + "." + table.name;
  auto& blocks = output[label];
  for (const auto& feature : table.features) {
    auto& target = blocks[feature.column];
    target = feature_block_for(feature, ids.size());
    const auto& source = unique_blocks.at(feature.column);
    for (const auto position : inverse) {
      copy_feature_row(feature, source, position, target);
    }
  }
  if (cache) {
    cache->positions = std::move(positions);
    cache->blocks = std::move(unique_blocks);
    cache->bytes = feature_cache_bytes(*cache);
  }
}

void trim_sql_caches(std::unordered_map<std::string, SqlTableCache>& caches,
                     const std::string& current, size_t limit) {
  size_t used = 0;
  for (const auto& [_, cache] : caches) used += cache.bytes;
  while (used > limit) {
    auto victim = caches.begin();
    if (victim->first == current && caches.size() > 1) ++victim;
    used -= victim->second.bytes;
    caches.erase(victim);
  }
}

}  // namespace relx
