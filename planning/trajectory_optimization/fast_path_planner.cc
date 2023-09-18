#include "drake/planning/trajectory_optimization/fast_path_planner.h"


#include "drake/solvers/solve.h"
#include "drake/solvers/get_program_type.h"

namespace drake {
namespace planning {
namespace trajectory_optimization {

using geometry::optimization::ConvexSet;
using geometry::optimization::ConvexSets;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using solvers::VectorXDecisionVariable;

using EdgesBetweenSubgraphs = FastPathPlanner::EdgesBetweenSubgraphs;
using Subgraph = FastPathPlanner::Subgraph;
using VertexId = FastPathPlanner::VertexId;

const double kInf = std::numeric_limits<double>::infinity();

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
    edges_.emplace_back(Edge{ids[e.first], ids[e.second]});
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
  for (int i = 0; i < ssize(regions); ++i) {
    for (int j = i + 1; j < ssize(regions); ++j) {
      if (regions[i]->IntersectsWith(*regions[j])) {
        // Regions are overlapping, add edge.
        edges_between_regions.emplace_back(i, j);
      }
    }
  }

  return AddRegions(regions, edges_between_regions, order, h_min, h_max,
                    std::move(name));
}

void FastPathPlanner::Preprocess() {
  const int num_edges = edges_.size();
  // Compute line graph.
  edge_ids_by_vertex_.clear();
  for (int i=0; i<num_edges; ++i) {
    edge_ids_by_vertex_[edges_[i].u].emplace_back(i);
    edge_ids_by_vertex_[edges_[i].v].emplace_back(i);
  }
  for (const auto& [vertex_id, edge_ids] : edge_ids_by_vertex_) {
    for (int i = 0; i < ssize(edge_ids); ++i) {
      for (int j = i + 1; j < ssize(edge_ids); ++j) {
        line_graph_edges_.emplace_back(LineGraphEdge{i, j, kInf});
      }
    }
  }

  // Optimize points.
  solvers::MathematicalProgram prog;
  auto x = prog.NewContinuousVariables(num_positions_, num_edges);
  for (int i=0; i<num_edges; ++i) {
    vertices_.at(edges_[i].u)->AddPointInSetConstraints(&prog, x.col(i));
    vertices_.at(edges_[i].v)->AddPointInSetConstraints(&prog, x.col(i));
  }
  MatrixXd A(num_positions_, 2 * num_positions_);
  A.leftCols(num_positions_) =
      MatrixXd::Identity(num_positions_, num_positions_);
  A.rightCols(num_positions_) =
      -MatrixXd::Identity(num_positions_, num_positions_);
  const VectorXd b = VectorXd::Zero(num_positions_);
  VectorXDecisionVariable vars(2 * num_positions_);
  for (const auto& e : line_graph_edges_) {
    vars.head(num_positions_) = x.col(e.u);
    vars.tail(num_positions_) = x.col(e.v);
    prog.AddL2NormCostUsingConicConstraint(A, b, vars);
  }
  auto result = Solve(prog);
  DRAKE_DEMAND(result.is_success());
  points_ = result.GetSolution(x);

  // Assign weights.
  for (auto& e : line_graph_edges_) {
    e.weight = (points_.col(e.u) - points_.col(e.v)).norm();
  }

  needs_preprocessing_ = false;  
}

trajectories::CompositeTrajectory<double> FastPathPlanner::SolvePath(
    const Eigen::Ref<const Eigen::VectorXd>& q_start,
    const Eigen::Ref<const Eigen::VectorXd>& q_goal) {
  // Stabbing problem.
  // TODO(russt): we could parallelize this.
  std::vector<VertexId> start_vertices;
  std::vector<VertexId> goal_vertices;
  for (const auto& [vertex_id, region] : vertices_) {
    if (region.PointInSet(q_start)) {
      start_vertices.push_back(vertex_id);
    }
    if (region.PointInSet(q_goal)) {
      goal_vertices.push_back(vertex_id);
    }
  }

  // Plan with A*.

}

}  // namespace trajectory_optimization
}  // namespace planning
}  // namespace drake
