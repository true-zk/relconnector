#include "graph_index.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace relx {
namespace {

RelationIndex make_csc(const std::string& source, const std::string& target,
                       int64_t destination_count,
                       std::vector<EdgeRecord> edges,
                       int64_t null_keys, int64_t dangling_keys) {
  std::sort(edges.begin(), edges.end(), [](const auto& left, const auto& right) {
    return std::tie(left.destination, left.time, left.source) <
           std::tie(right.destination, right.time, right.source);
  });
  RelationIndex result;
  result.source = source;
  result.target = target;
  result.null_keys = null_keys;
  result.dangling_keys = dangling_keys;
  result.colptr.assign(static_cast<size_t>(destination_count + 1), 0);
  result.row.reserve(edges.size());
  result.edge_time.reserve(edges.size());
  for (const auto& edge : edges) {
    ++result.colptr[static_cast<size_t>(edge.destination + 1)];
    result.row.push_back(edge.source);
    result.edge_time.push_back(edge.time);
  }
  for (size_t i = 1; i < result.colptr.size(); ++i) {
    result.colptr[i] += result.colptr[i - 1];
  }
  return result;
}

uint64_t mix_seed(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

uint64_t relation_seed(const std::string& relation) {
  uint64_t result = 14695981039346656037ULL;
  for (const unsigned char character : relation) {
    result = (result ^ character) * 1099511628211ULL;
  }
  return result;
}

}  // namespace

std::vector<int64_t> GraphIndex::lookup_nodes(
    const std::string& table, const std::vector<std::string>& key_columns,
    const ArrowBatch& batch) const {
  const auto table_it = node_ids.find(table);
  if (table_it == node_ids.end()) {
    throw std::invalid_argument("Unknown graph table: " + table);
  }
  std::vector<int64_t> result;
  result.reserve(static_cast<size_t>(batch.length()));
  for (int64_t row = 0; row < batch.length(); ++row) {
    auto key = batch.key(key_columns, row);
    const auto found = key ? table_it->second.find(*key) : table_it->second.end();
    result.push_back(found == table_it->second.end() ? -1 : found->second);
  }
  return result;
}

std::vector<int64_t> GraphIndex::visible_neighbors(
    const std::string& relation, int64_t destination, int64_t cutoff_us) const {
  const auto found = relations.find(relation);
  if (found == relations.end()) {
    throw std::invalid_argument("Unknown relation: " + relation);
  }
  const auto& index = found->second;
  const auto count = node_counts.at(index.target);
  if (destination < 0 || destination >= count) {
    throw std::out_of_range("Destination node ID is out of range");
  }
  const auto node_time_us = node_time.at(index.target)[destination];
  if (node_time_us != kStaticTime &&
      (inclusive ? node_time_us > cutoff_us : node_time_us >= cutoff_us)) {
    return {};
  }
  const auto start = index.colptr[destination];
  const auto stop = index.colptr[destination + 1];
  const auto first = index.edge_time.begin() + start;
  const auto last = index.edge_time.begin() + stop;
  const auto end = inclusive ? std::upper_bound(first, last, cutoff_us)
                             : std::lower_bound(first, last, cutoff_us);
  return {index.row.begin() + start,
          index.row.begin() + start + (end - first)};
}

std::vector<std::vector<int64_t>> GraphIndex::sample_neighbors(
    const std::string& relation, const std::vector<int64_t>& destinations,
    const std::vector<int64_t>& cutoffs_us, int64_t fanout,
    uint64_t random_seed, int64_t workers) const {
  if (destinations.size() != cutoffs_us.size()) {
    throw std::invalid_argument("destinations and cutoffs_us must have equal length");
  }
  if (fanout < -1 || workers < 1) {
    throw std::invalid_argument("fanout must be >= -1 and workers must be positive");
  }
  if (!relations.count(relation)) {
    throw std::invalid_argument("Unknown relation: " + relation);
  }
  std::vector<std::vector<int64_t>> result(destinations.size());
  if (destinations.empty()) return result;
  std::atomic<size_t> next{0};
  std::exception_ptr error;
  std::mutex error_mutex;
  const auto salt = relation_seed(relation);
  const auto work = [&]() {
    try {
      while (true) {
        const size_t index = next.fetch_add(1);
        if (index >= destinations.size()) break;
        auto neighbors = visible_neighbors(
            relation, destinations[index], cutoffs_us[index]);
        if (fanout >= 0 && static_cast<size_t>(fanout) < neighbors.size()) {
          std::mt19937_64 rng(mix_seed(random_seed ^ salt ^ index));
          for (size_t position = 0; position < static_cast<size_t>(fanout);
               ++position) {
            std::uniform_int_distribution<size_t> draw(
                position, neighbors.size() - 1);
            std::swap(neighbors[position], neighbors[draw(rng)]);
          }
          neighbors.resize(static_cast<size_t>(fanout));
        }
        result[index] = std::move(neighbors);
      }
    } catch (...) {
      std::lock_guard<std::mutex> guard(error_mutex);
      if (!error) error = std::current_exception();
    }
  };
  const auto thread_count = std::min<size_t>(
      {static_cast<size_t>(workers), destinations.size(),
       std::max(1U, std::thread::hardware_concurrency())});
  std::vector<std::thread> threads;
  threads.reserve(thread_count - 1);
  try {
    for (size_t index = 1; index < thread_count; ++index) {
      threads.emplace_back(work);
    }
  } catch (...) {
    for (auto& thread : threads) thread.join();
    throw;
  }
  work();
  for (auto& thread : threads) thread.join();
  if (error) std::rethrow_exception(error);
  return result;
}

SampledBatch GraphIndex::sample_one_hop(
    const std::string& seed_table, const std::vector<int64_t>& seed_ids,
    const std::vector<int64_t>& cutoffs_us, int64_t fanout,
    uint64_t random_seed, int64_t workers) const {
  if (!node_counts.count(seed_table)) {
    throw std::invalid_argument("Unknown graph table: " + seed_table);
  }
  if (seed_ids.size() != cutoffs_us.size()) {
    throw std::invalid_argument("seed_ids and cutoffs_us must have equal length");
  }
  if (fanout < -1 || workers < 1) {
    throw std::invalid_argument("fanout must be >= -1 and workers must be positive");
  }
  const auto seed_count = static_cast<int64_t>(seed_ids.size());
  for (const auto id : seed_ids) {
    if (id < 0 || id >= node_counts.at(seed_table)) {
      throw std::out_of_range("Seed node ID is out of range");
    }
  }

  SampledBatch result;
  result.node_ids.emplace(seed_table, seed_ids);
  auto& seed_owners = result.node_seed_index[seed_table];
  seed_owners.reserve(seed_ids.size());
  for (int64_t seed = 0; seed < seed_count; ++seed) {
    seed_owners.push_back(seed);
  }

  std::vector<std::string> names;
  for (const auto& [name, relation] : relations) {
    if (relation.target == seed_table) names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  for (const auto& name : names) {
    const auto& relation = relations.at(name);
    auto sampled = sample_neighbors(name, seed_ids, cutoffs_us, fanout,
                                    random_seed, workers);
    auto& source_ids = result.node_ids[relation.source];
    auto& source_owners = result.node_seed_index[relation.source];
    auto& edges = result.edge_index[name];
    size_t count = 0;
    for (const auto& neighbors : sampled) count += neighbors.size();
    source_ids.reserve(source_ids.size() + count);
    source_owners.reserve(source_owners.size() + count);
    edges.source.reserve(count);
    edges.target.reserve(count);
    for (size_t seed = 0; seed < sampled.size(); ++seed) {
      for (const auto neighbor : sampled[seed]) {
        edges.source.push_back(static_cast<int64_t>(source_ids.size()));
        edges.target.push_back(static_cast<int64_t>(seed));
        source_ids.push_back(neighbor);
        source_owners.push_back(static_cast<int64_t>(seed));
      }
    }
  }
  return result;
}

void GraphBuilder::register_table(const std::string& table, bool timed) {
  if (table.empty()) throw std::invalid_argument("Table name must not be empty");
  auto [position, inserted] = tables_.try_emplace(table);
  if (!inserted && position->second.timed != timed) {
    throw std::invalid_argument(table + ": conflicting time declaration");
  }
  position->second.timed = timed;
}

void GraphBuilder::register_relation(const std::string& relation,
                                      const std::string& source,
                                      const std::string& target) {
  if (!tables_.count(source) || !tables_.count(target)) {
    throw std::invalid_argument(relation + ": missing source or target nodes");
  }
  if (relation.empty() || relation.rfind("rev:", 0) == 0 ||
      relations_.count(relation) ||
      relations_.count("rev:" + relation)) {
    throw std::invalid_argument("Duplicate or empty relation name");
  }
  auto& state = relations_[relation];
  state.source = source;
  state.target = target;
}

void GraphBuilder::add_nodes(const std::string& table,
                             const std::vector<std::string>& key_columns,
                             const ArrowBatch& batch) {
  const auto found = tables_.find(table);
  if (found == tables_.end()) {
    throw std::invalid_argument(table + ": table is not registered");
  }
  auto& state = found->second;
  auto& ids = state.ids;
  for (int64_t row = 0; row < batch.length(); ++row) {
    auto key = batch.key(key_columns, row);
    if (!key) {
      throw std::invalid_argument(table + ": identity key contains NULL");
    }
    if (!ids.emplace(*key, static_cast<int64_t>(ids.size())).second) {
      throw std::invalid_argument(table + ": duplicate identity key");
    }
    state.time.push_back(state.timed ? batch.time_us("__relx_time_us", row)
                                     : kStaticTime);
  }
}

void GraphBuilder::add_edges(
    const std::string& relation, const std::string& source,
    const std::string& target,
    const std::vector<std::string>& source_key_columns,
    const std::vector<std::string>& foreign_key_columns,
    const ArrowBatch& batch) {
  const auto source_it = tables_.find(source);
  const auto target_it = tables_.find(target);
  if (source_it == tables_.end() || target_it == tables_.end()) {
    throw std::invalid_argument(relation + ": missing source or target nodes");
  }
  const auto relation_it = relations_.find(relation);
  if (relation_it == relations_.end() || relation_it->second.source != source ||
      relation_it->second.target != target) {
    throw std::invalid_argument(relation + ": relation is not registered");
  }
  auto& state = relation_it->second;

  for (int64_t row = 0; row < batch.length(); ++row) {
    auto source_key = batch.key(source_key_columns, row);
    if (!source_key) {
      throw std::invalid_argument(source + ": identity key contains NULL");
    }
    const auto source_id = source_it->second.ids.find(*source_key);
    if (source_id == source_it->second.ids.end()) {
      throw std::invalid_argument(source + ": edge source is not in the node index");
    }
    auto target_key = batch.key(foreign_key_columns, row);
    if (!target_key) {
      ++state.null_keys;
      continue;
    }
    const auto target_id = target_it->second.ids.find(*target_key);
    if (target_id == target_it->second.ids.end()) {
      ++state.dangling_keys;
      continue;
    }
    const auto source_time = source_it->second.time[source_id->second];
    const auto target_time = target_it->second.time[target_id->second];
    state.edges.push_back({source_id->second, target_id->second,
                           std::max(source_time, target_time)});
  }
}

GraphIndex GraphBuilder::finish() {
  GraphIndex result;
  result.inclusive = inclusive_;
  for (auto& [table, state] : tables_) {
    result.node_counts.emplace(table, static_cast<int64_t>(state.ids.size()));
    result.node_ids.emplace(table, std::move(state.ids));
    result.node_time.emplace(table, std::move(state.time));
  }
  for (auto& [name, state] : relations_) {
    std::vector<EdgeRecord> reversed;
    reversed.reserve(state.edges.size());
    for (const auto& edge : state.edges) {
      reversed.push_back({edge.destination, edge.source, edge.time});
    }
    auto forward = make_csc(
        state.source, state.target, result.node_counts.at(state.target),
        std::move(state.edges), state.null_keys, state.dangling_keys);
    auto reverse = make_csc(state.target, state.source,
                            result.node_counts.at(state.source),
                            std::move(reversed), state.null_keys,
                            state.dangling_keys);
    result.relations.emplace(name, std::move(forward));
    result.relations.emplace("rev:" + name, std::move(reverse));
  }
  return result;
}

}  // namespace relx
