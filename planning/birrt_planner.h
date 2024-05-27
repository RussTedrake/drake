#pragma once

#include <cstdint>
#include <vector>

#include "planning/default_state_types.h"
#include "planning/path_planning_result.h"
#include "planning/planning_space.h"

namespace drake {
namespace planning {
/// Bi-directional RRT planner.
template <typename StateType>
class BiRRTPlanner {
 public:
  /// Parameters to the BiRRT planner.
  // TODO(calderpg) Provide/document good defaults.
  struct Parameters {
    /// Probability that the sampled state comes from the target tree.
    double tree_sampling_bias{0.0};
    /// Probability that the active tree switches after each iteration.
    double p_switch_trees{0.0};
    /// Time limit for planning.
    double time_limit{0.0};
    /// Tolerance for connect checks between start and goal tree.
    double connection_tolerance{0.0};
    /// Seed for internal random number generator.
    uint64_t prng_seed{0};
    /// Should nearest neighbor checks be parallelized? To be performed in
    /// parallel both this parameter must be true, and the planning space must
    /// support parallel operations.
    bool parallelize_nearest_neighbor{false};
  };

  /// Plan a path from the provided start state to goal state.
  /// @param start Starting state.
  /// @param goal Ending state.
  /// @param parameters Parameters to planner.
  /// @param planning_space Planning space to use. @pre not null.
  /// @return First path found from start to goal, if found in time limit.
  static PathPlanningResult<StateType> Plan(
      const StateType& start, const StateType& goal,
      const Parameters& parameters, PlanningSpace<StateType>* planning_space);

  /// Plan a path from the provided start states to goal states.
  /// @param start Starting states.
  /// @param goal Ending states.
  /// @param parameters Parameters to planner.
  /// @param planning_space Planning space to use. @pre not null.
  /// @return First path found from *a* start to *a* goal, if found in time
  /// limit.
  static PathPlanningResult<StateType> Plan(
      const std::vector<StateType>& starts, const std::vector<StateType>& goals,
      const Parameters& parameters, PlanningSpace<StateType>* planning_space);

  // Delete all constructors of this static-only class.
  BiRRTPlanner(const BiRRTPlanner&) = delete;
};
}  // namespace planning
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_PLANNING_STATE_TYPES(
    class ::drake::planning::BiRRTPlanner)
