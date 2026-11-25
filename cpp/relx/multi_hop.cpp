#include "graph_index.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace relx {
namespace {

struct OwnedNode {
  int64_t seed;
  int64_t id;

  bool operator==(const OwnedNode& other) const {
    return seed == other.seed && id == other.id;
  }
};

struct OwnedNodeHash {
  size_t operator()(const OwnedNode& value) const {
    const auto a = std::hash<int64_t>{}(value.seed);
    const auto b = std::hash<int64_t>{}(value.id);
    return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
  }
};

}  // namespace

SampledBatch GraphIndex::sample_multi_hop(
    const std::string& seed_table, const std::vector<int64_t>& seed_ids,
    const std::vector<int64_t>& cutoffs_us,
    const std::vector<int64_t>& fanouts, uint64_t random_seed,
    int64_t workers) const {
  if (!node_counts.count(seed_table)) {
    throw std::invalid_argument("Unknown graph table: " + seed_table);
  }
  if (seed_ids.size() != cutoffs_us.size()) {
    throw std::invalid_argument("seed_ids and cutoffs_us must have equal length");
  }
  if (fanouts.empty() ||
      std::any_of(fanouts.begin(), fanouts.end(),
                  [](int64_t fanout) { return fanout < -1; }) ||
      workers < 1) {
    throw std::invalid_argument("fanouts must be nonempty and >= -1; workers must be positive");
  }
  for (const auto id : seed_ids) {
    if (id < 0 || id >= node_counts.at(seed_table)) {
      throw std::out_of_range("Seed node ID is out of range");
    }
  }

  SampledBatch result;
  std::unordered_map<std::string,
                     std::unordered_map<OwnedNode, int64_t, OwnedNodeHash>> local;
  const auto hops = fanouts.size();
  for (const auto& [table, _] : node_counts) {
    result.node_ids.try_emplace(table);
    result.node_seed_index.try_emplace(table);
    result.nodes_per_hop.emplace(table, std::vector<int64_t>(hops + 1, 0));
    local.try_emplace(table);
  }
  std::vector<std::string> relation_names;
  relation_names.reserve(relations.size());
  for (const auto& [name, _] : relations) {
    relation_names.push_back(name);
    result.edge_index.try_emplace(name);
    result.edges_per_hop.emplace(name, std::vector<int64_t>(hops, 0));
  }
  std::sort(relation_names.begin(), relation_names.end());

  auto& seeds = result.node_ids.at(seed_table);
  auto& owners = result.node_seed_index.at(seed_table);
  seeds = seed_ids;
  owners.reserve(seed_ids.size());
  for (size_t index = 0; index < seed_ids.size(); ++index) {
    owners.push_back(static_cast<int64_t>(index));
    local.at(seed_table).emplace(
        OwnedNode{static_cast<int64_t>(index), seed_ids[index]},
        static_cast<int64_t>(index));
  }
  result.nodes_per_hop.at(seed_table)[0] = static_cast<int64_t>(seed_ids.size());
  std::unordered_map<std::string, std::vector<int64_t>> frontier;
  auto& seed_frontier = frontier[seed_table];
  seed_frontier.reserve(seed_ids.size());
  for (size_t index = 0; index < seed_ids.size(); ++index) {
    seed_frontier.push_back(static_cast<int64_t>(index));
  }

  for (size_t hop = 0; hop < hops; ++hop) {
    std::unordered_map<std::string, std::vector<int64_t>> next_frontier;
    for (const auto& name : relation_names) {
      const auto& relation = relations.at(name);
      const auto found = frontier.find(relation.target);
      if (found == frontier.end() || found->second.empty()) continue;

      const auto& target_nodes = result.node_ids.at(relation.target);
      const auto& target_owners = result.node_seed_index.at(relation.target);
      std::vector<int64_t> destinations;
      std::vector<int64_t> cutoffs;
      destinations.reserve(found->second.size());
      cutoffs.reserve(found->second.size());
      for (const auto local_id : found->second) {
        destinations.push_back(target_nodes[static_cast<size_t>(local_id)]);
        cutoffs.push_back(
            cutoffs_us[static_cast<size_t>(
                target_owners[static_cast<size_t>(local_id)])]);
      }
      const auto sampled = sample_neighbors(
          name, destinations, cutoffs, fanouts[hop],
          random_seed ^ (0x9e3779b97f4a7c15ULL * (hop + 1)), workers);
      auto& source_nodes = result.node_ids.at(relation.source);
      auto& source_owners = result.node_seed_index.at(relation.source);
      auto& source_local = local.at(relation.source);
      auto& new_nodes = next_frontier[relation.source];
      auto& edges = result.edge_index.at(name);
      for (size_t index = 0; index < sampled.size(); ++index) {
        const auto target_local = found->second[index];
        const auto owner = target_owners[static_cast<size_t>(target_local)];
        for (const auto neighbor : sampled[index]) {
          const auto candidate = static_cast<int64_t>(source_nodes.size());
          const auto [entry, inserted] =
              source_local.emplace(OwnedNode{owner, neighbor}, candidate);
          if (inserted) {
            source_nodes.push_back(neighbor);
            source_owners.push_back(owner);
            new_nodes.push_back(candidate);
          }
          edges.source.push_back(entry->second);
          edges.target.push_back(target_local);
          ++result.edges_per_hop.at(name)[hop];
        }
      }
    }
    for (const auto& [table, indices] : next_frontier) {
      result.nodes_per_hop.at(table)[hop + 1] =
          static_cast<int64_t>(indices.size());
    }
    frontier = std::move(next_frontier);
  }
  return result;
}

}  // namespace relx
