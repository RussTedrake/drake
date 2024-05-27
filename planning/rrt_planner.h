#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "planning/default_state_types.h"
#include "planning/path_planning_result.h"
#include "planning/planning_space.h"

namespace drake {
namespace planning {
/// Single-directional RRT planner.
template <typename StateType>
class RRTPlanner {
 public:
  /// Parameters to the RRT planner.
  // TODO(calderpg) Provide/document good defaults.
  struct Parameters {
    /// Probability that the sampled state comes from the goal state(s)/goal
    /// sampling function. Ignored when using an implicit goal check function.
    double goal_sampling_bias{0.0};
    /// Time limit for planning.
    double time_limit{0.0};
    /// Tolerance for checks against goal states.
    double goal_tolerance{0.0};
    /// Seed for internal random number generator.
    uint64_t prng_seed{0};
    /// Should nearest neighbor checks be parallelized? To be performed in
    /// parallel both this parameter must be true, and the planning space must
    /// support parallel operations.
    bool parallelize_nearest_neighbor{false};
  };

  /// Function type for sampling a possible goal state.
  using GoalSampleFunction = std::function<StateType(void)>;

  /// Function type for checking if the provided state meets goal conditions.
  using GoalCheckFunction = std::function<bool(const StateType&)>;

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

  /// Plan a path from the provided start state to sampled goal state(s).
  /// @param start Starting state.
  /// @param goal_sample_fn Goal sampling function.
  /// @param parameters Parameters to planner.
  /// @param planning_space Planning space to use. @pre not null.
  /// @return First path found from start to *a* goal, if found in time limit.
  static PathPlanningResult<StateType> PlanGoalSampling(
      const StateType& start, const GoalSampleFunction& goal_sample_fn,
      const Parameters& parameters, PlanningSpace<StateType>* planning_space);

  /// Plan a path from the provided start states to sampled goal state(s).
  /// @param starts Starting states.
  /// @param goal_sample_fn Goal sampling function.
  /// @param parameters Parameters to planner.
  /// @param planning_space Planning space to use. @pre not null.
  /// @return First path found from *a* start to *a* goal, if found in time
  /// limit.
  static PathPlanningResult<StateType> PlanGoalSampling(
      const std::vector<StateType>& starts,
      const GoalSampleFunction& goal_sample_fn, const Parameters& parameters,
      PlanningSpace<StateType>* planning_space);

  /// Plan a path from the provided start state to the goal implicitly defined
  /// by the goal check function.
  /// @param start Starting state.
  /// @param goal_check_fn Goal check function.
  /// @param parameters Parameters to planner.
  /// @param planning_space Planning space to use. @pre not null.
  /// @return First path found from start to *a* state meeting the goal check,
  /// if found in time limit.
  static PathPlanningResult<StateType> PlanGoalCheck(
      const StateType& start, const GoalCheckFunction& goal_check_fn,
      const Parameters& parameters, PlanningSpace<StateType>* planning_space);

  /// Plan a path from the provided start states to the goal implicitly defined
  /// by the goal check function.
  /// @param start Starting state.
  /// @param goal_check_fn Goal check function.
  /// @param parameters Parameters to planner.
  /// @param planning_space Planning space to use. @pre not null.
  /// @return First path found from *a* start to *a* state meeting the goal
  /// check, if found in time limit.
  static PathPlanningResult<StateType> PlanGoalCheck(
      const std::vector<StateType>& starts,
      const GoalCheckFunction& goal_check_fn, const Parameters& parameters,
      PlanningSpace<StateType>* planning_space);

  // Delete all constructors of this static-only class.
  RRTPlanner(const RRTPlanner&) = delete;
};
}  // namespace planning
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_PLANNING_STATE_TYPES(
    class ::drake::planning::RRTPlanner)
