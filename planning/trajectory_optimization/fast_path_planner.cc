#include "drake/planning/trajectory_optimization/fast_path_planner.h"

#include <common_robotics_utilities/simple_astar_search.hpp>

#include "drake/solvers/solve.h"
#include "drake/solvers/get_program_type.h"

namespace drake {
namespace planning {
namespace trajectory_optimization {

namespace {
  using common_robotics_utilities::simple_astar_search::StateWithCost;

  template <typename T>
  class MyStateWithCost : public StateWithCost<T> {
   public:
    MyStateWithCost(const T& state, double cost)
        : StateWithCost<T>(state, cost) {}
  };
}  // namespace

using geometry::optimization::ConvexSet;
using geometry::optimization::ConvexSets;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using solvers::VectorXDecisionVariable;

using EdgesBetweenSubgraphs = FastPathPlanner::EdgesBetweenSubgraphs;
using Subgraph = FastPathPlanner::Subgraph;
using VertexId = FastPathPlanner::VertexId;

const double kInf = std::numeric_limits<double>::infinity();

struct FastPathPlanner::LineGraph {
public:
  std::vector<std::vector<MyStateWithCost<int>>> edges;
};


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

  // Build line graph data.
  edge_ids_by_vertex_.clear();
  for (int i=0; i<num_edges; ++i) {
    edge_ids_by_vertex_[edges_[i].u].emplace_back(i);
    edge_ids_by_vertex_[edges_[i].v].emplace_back(i);
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
  for (const auto& [vertex_id, edge_ids] : edge_ids_by_vertex_) {
    for (int i = 0; i < ssize(edge_ids); ++i) {
      for (int j = i + 1; j < ssize(edge_ids); ++j) {
        vars.head(num_positions_) = x.col(i);
        vars.tail(num_positions_) = x.col(j);
        prog.AddL2NormCostUsingConicConstraint(A, b, vars);
      }
    }
  }
  auto result = Solve(prog);
  DRAKE_DEMAND(result.is_success());
  points_ = result.GetSolution(x);

  // Store line graph with edge weights.
  line_graph_ = std::make_unique<LineGraph>();
  line_graph_->edges.resize(num_edges);
  for (const auto& [vertex_id, edge_ids] : edge_ids_by_vertex_) {
    for (int i = 0; i < ssize(edge_ids); ++i) {
      for (int j = i + 1; j < ssize(edge_ids); ++j) {
        double cost =
            (points_.col(edge_ids[i]) - points_.col(edge_ids[j])).norm();
        line_graph_->edges[edge_ids[i]].emplace_back(
            MyStateWithCost(edge_ids[j], cost));
        line_graph_->edges[edge_ids[j]].emplace_back(
            MyStateWithCost(edge_ids[i], cost));
      }
    }
  }

  needs_preprocessing_ = false;  
}

trajectories::CompositeTrajectory<double> FastPathPlanner::SolvePath(
    const Eigen::Ref<const Eigen::VectorXd>& q_start,
    const Eigen::Ref<const Eigen::VectorXd>& q_goal) {
  // TODO(russt): Support restricting the start and goal to subgraphs.

  // Stabbing problem.
  // TODO(russt): we could parallelize this.
  StatesWithCosts start_states;
  const int GOAL_ID = -1;
  for (const auto& [vertex_id, region] : vertices_) {
    if (region->PointInSet(q_start)) {
      for (const auto& edge_id : edge_ids_by_vertex_.at(vertex_id)) {
        start_states.emplace_back(
            StateWithCost(vertex_id, (q_start - points.col(edge_id)).norm()));
        };
    }
    if (region->PointInSet(q_goal)) {
      for (const auto& edge_id : edge_ids_by_vertex_.at(vertex_id)) {
        cost = (q_goal - points.col(edge_id)).norm();
        line_graph_->edges[edge_id].emplace_back(
            MyStateWithCost(GOAL_ID, cost));
    }
  }

  auto goal_check_fn = [](const int& edge_id) {
    return edge_id == GOAL_ID;
  };

  auto generate_valid_children_fn(const int& edge_id) {
    return line_graph_->edges[edge_id];
  };

  auto heuristic_fn = [](const int& edge_id) {
    return (points[edge_id] - q_goal).norma();
  };

  // Plan with A*.
  auto result =
      common_robotics_utilities::simple_astar_search::PerformAstarSearch(
          start_states, goal_checK_fn, generate_valid_children_fn,
          heuristic_fn);

  return trajectories::CompositeTrajectory<double>({});
}

}  // namespace trajectory_optimization
}  // namespace planning
}  // namespace drake
