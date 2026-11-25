#pragma once

#include "arrow_batch.h"

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace relx {

inline constexpr int64_t kStaticTime = std::numeric_limits<int64_t>::min();

struct EdgeRecord {
  int64_t source;
  int64_t destination;
  int64_t time;
};

struct RelationIndex {
  std::string source;
  std::string target;
  std::vector<int64_t> colptr;
  std::vector<int64_t> row;
  std::vector<int64_t> edge_time;
  int64_t null_keys = 0;
  int64_t dangling_keys = 0;
};

struct SampledEdges {
  std::vector<int64_t> source;
  std::vector<int64_t> target;
};

struct SampledBatch {
  std::unordered_map<std::string, std::vector<int64_t>> node_ids;
  std::unordered_map<std::string, std::vector<int64_t>> node_seed_index;
  std::unordered_map<std::string, SampledEdges> edge_index;
  std::unordered_map<std::string, std::vector<int64_t>> nodes_per_hop;
  std::unordered_map<std::string, std::vector<int64_t>> edges_per_hop;
};

struct GraphIndex {
  std::unordered_map<std::string, int64_t> node_counts;
  std::unordered_map<std::string, RelationIndex> relations;
  std::unordered_map<std::string, std::unordered_map<std::string, int64_t>>
      node_ids;
  std::unordered_map<std::string, std::vector<int64_t>> node_time;
  bool inclusive = true;

  std::vector<int64_t> lookup_nodes(
      const std::string& table, const std::vector<std::string>& key_columns,
      const ArrowBatch& batch) const;
  std::vector<int64_t> visible_neighbors(const std::string& relation,
                                         int64_t destination,
                                         int64_t cutoff_us) const;
  std::vector<std::vector<int64_t>> sample_neighbors(
      const std::string& relation, const std::vector<int64_t>& destinations,
      const std::vector<int64_t>& cutoffs_us, int64_t fanout,
      uint64_t random_seed, int64_t workers) const;
  SampledBatch sample_one_hop(
      const std::string& seed_table, const std::vector<int64_t>& seed_ids,
      const std::vector<int64_t>& cutoffs_us, int64_t fanout,
      uint64_t random_seed, int64_t workers) const;
  SampledBatch sample_multi_hop(
      const std::string& seed_table, const std::vector<int64_t>& seed_ids,
      const std::vector<int64_t>& cutoffs_us,
      const std::vector<int64_t>& fanouts, uint64_t random_seed,
      int64_t workers) const;
};

class GraphBuilder {
 public:
  explicit GraphBuilder(bool inclusive = true) : inclusive_(inclusive) {}

  void register_table(const std::string& table, bool timed = false);
  void register_relation(const std::string& relation,
                         const std::string& source,
                         const std::string& target);

  void add_nodes(const std::string& table,
                 const std::vector<std::string>& key_columns,
                 const ArrowBatch& batch);

  void add_edges(const std::string& relation, const std::string& source,
                 const std::string& target,
                 const std::vector<std::string>& source_key_columns,
                 const std::vector<std::string>& foreign_key_columns,
                 const ArrowBatch& batch);

  GraphIndex finish();

 private:
  struct TableState {
    std::unordered_map<std::string, int64_t> ids;
    std::vector<int64_t> time;
    bool timed = false;
  };

  struct RelationState {
    std::string source;
    std::string target;
    std::vector<EdgeRecord> edges;
    int64_t null_keys = 0;
    int64_t dangling_keys = 0;
  };

  std::unordered_map<std::string, TableState> tables_;
  std::unordered_map<std::string, RelationState> relations_;
  bool inclusive_ = true;
};

}  // namespace relx
