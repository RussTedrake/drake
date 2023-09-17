#include "drake/planning/trajectory_optimization/fast_path_planner.h"


#include "drake/solvers/solve.h"

namespace drake {
namespace planning {
namespace trajectory_optimization {

using geometry::optimization::ConvexSets;
using EdgesBetweenSubgraphs = FastPathPlanner::EdgesBetweenSubgraphs;
using Subgraph = FastPathPlanner::Subgraph;
using VertexId = FastPathPlanner::VertexId;

//const double kInf = std::numeric_limits<double>::infinity();

FastPathPlanner::FastPathPlanner(int num_positions)
    : num_positions_{num_positions} {}

FastPathPlanner::~FastPathPlanner() = default;

Subgraph::~Subgraph() = default;

Subgraph::Subgraph(const std::vector<FastPathPlanner::VertexId> vertex_ids,
                   int order, double h_min, double h_max, std::string name)
    : vertex_ids_{vertex_ids},
      order_{order},
      h_min_{h_min},
      h_max_{h_max},
      name_(std::move(name)) {}

EdgesBetweenSubgraphs::~EdgesBetweenSubgraphs() = default;

Subgraph& FastPathPlanner::AddRegions(
      const ConvexSets& regions,
      const std::vector<std::pair<int, int>>& edges_between_regions, int order,
      double h_min, double h_max, std::string name) {
  // Copy vertices (assigning them a VertexId).
  std::vector<VertexId> ids(regions.size());
  for (int i=0; i<ssize(regions); ++i) {
    ids[i] = VertexId::get_new_id();
    vertices_.emplace(ids[i], regions[i]);
  }

  // Add the edges.
  for (const auto& e : edges_between_regions) {
    edges_.emplace_back(std::make_pair(ids[e.first], ids[e.second]));
  }

  // Set the dirty bit.
  needs_preprocessing_ = true;

  // Create the Subgraph.
  Subgraph* subgraph = new Subgraph(ids, order, h_min, h_max, std::move(name));
  return *subgraphs_.emplace_back(subgraph);
}

Subgraph& FastPathPlanner::AddRegions(
    const ConvexSets& regions, int order, double h_min,
    double h_max, std::string name) {
  // TODO(russt): parallelize this.
  std::vector<std::pair<int, int>> edges_between_regions;
  for (size_t i = 0; i < regions.size(); ++i) {
    for (size_t j = i + 1; j < regions.size(); ++j) {
      if (regions[i]->IntersectsWith(*regions[j])) {
        // Regions are overlapping, add edge.
        edges_between_regions.emplace_back(i, j);
        edges_between_regions.emplace_back(j, i);
      }
    }
  }

  return AddRegions(regions, edges_between_regions, order, h_min, h_max,
                    std::move(name));
}

void FastPathPlanner::Preprocess() {
  needs_preprocessing_ = false;  
}

}  // namespace trajectory_optimization
}  // namespace planning
}  // namespace drake
