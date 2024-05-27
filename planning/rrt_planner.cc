#include "planning/rrt_planner.h"

#include <map>
#include <string>

#include <common_robotics_utilities/print.hpp>
#include <common_robotics_utilities/simple_knearest_neighbors.hpp>
#include <common_robotics_utilities/simple_rrt_planner.hpp>

#include "drake/common/drake_throw.h"
#include "drake/common/text_logging.h"

namespace drake {
namespace planning {
using common_robotics_utilities::simple_rrt_planner::ForwardPropagation;
using common_robotics_utilities::simple_rrt_planner::
    MakeLinearRRTNearestNeighborsFunction;
using common_robotics_utilities::simple_rrt_planner::
    MakeRRTTimeoutTerminationFunction;
using common_robotics_utilities::simple_rrt_planner::
    MakeStateAndGoalsSamplingFunction;
using common_robotics_utilities::simple_rrt_planner::
    RRTCheckGoalReachedFunction;
using common_robotics_utilities::simple_rrt_planner::
    RRTForwardPropagationFunction;
using common_robotics_utilities::simple_rrt_planner::RRTPlanSinglePath;
using common_robotics_utilities::simple_rrt_planner::SamplingFunction;
using common_robotics_utilities::simple_rrt_planner::SimpleRRTPlannerTree;

template <typename StateType>
PathPlanningResult<StateType> RRTPlanner<StateType>::Plan(
    const StateType& start, const StateType& goal, const Parameters& parameters,
    PlanningSpace<StateType>* planning_space) {
  return Plan(std::vector<StateType>{start}, std::vector<StateType>{goal},
              parameters, planning_space);
}

template <typename StateType>
PathPlanningResult<StateType> RRTPlanner<StateType>::PlanGoalSampling(
    const StateType& start, const GoalSampleFunction& goal_sample_fn,
    const Parameters& parameters, PlanningSpace<StateType>* planning_space) {
  return PlanGoalSampling(std::vector<StateType>{start}, goal_sample_fn,
                          parameters, planning_space);
}

template <typename StateType>
PathPlanningResult<StateType> RRTPlanner<StateType>::PlanGoalCheck(
    const StateType& start, const GoalCheckFunction& goal_check_fn,
    const Parameters& parameters, PlanningSpace<StateType>* planning_space) {
  return PlanGoalCheck(std::vector<StateType>{start}, goal_check_fn, parameters,
                       planning_space);
}

template <typename StateType>
PathPlanningResult<StateType> RRTPlanner<StateType>::Plan(
    const std::vector<StateType>& starts, const std::vector<StateType>& goals,
    const Parameters& parameters, PlanningSpace<StateType>* planning_space) {
  DRAKE_THROW_UNLESS(parameters.goal_sampling_bias > 0.0);
  DRAKE_THROW_UNLESS(parameters.time_limit > 0.0);
  DRAKE_THROW_UNLESS(parameters.goal_tolerance > 0.0);
  DRAKE_THROW_UNLESS(planning_space != nullptr);

  const auto& [valid_starts, valid_goals, status] =
      planning_space->ExtractValidStartsAndGoals(starts, goals);
  if (!status.is_success()) {
    return PathPlanningResult<StateType>(status);
  }

  const SamplingFunction<StateType> state_sampling_fn = [&]() {
    return planning_space->SampleState();
  };

  // Statistics for edge propagation function.
  std::map<std::string, double> propagation_statistics;

  const RRTForwardPropagationFunction<StateType, StateType>
      forward_propagation_fn =
          [&](const StateType& nearest, const StateType& sample) {
            const std::vector<StateType> propagated_states =
                planning_space->PropagateForwards(nearest, sample,
                                                  &propagation_statistics);

            ForwardPropagation<StateType> forward_propagation;
            forward_propagation.reserve(propagated_states.size());
            int64_t relative_parent_index = -1;
            for (const auto& propagated_state : propagated_states) {
              forward_propagation.emplace_back(propagated_state,
                                               relative_parent_index);
              relative_parent_index++;
            }
            return forward_propagation;
          };

  const RRTCheckGoalReachedFunction<StateType> goal_check_fn =
      [&](const StateType& state) {
        for (const auto& goal : goals) {
          if (planning_space->StateDistanceForwards(state, goal) <=
              parameters.goal_tolerance) {
            return true;
          }
        }
        return false;
      };

  SimpleRRTPlannerTree<StateType> tree(starts.size());
  for (const auto& start : starts) {
    tree.AddNode(start);
  }

  std::mt19937_64 prng(parameters.prng_seed);
  std::uniform_real_distribution<double> uniform_unit_dist(0.0, 1.0);
  const common_robotics_utilities::utility::UniformUnitRealFunction
      uniform_unit_real_fn = [&]() {
        return uniform_unit_dist(prng);
      };

  drake::log()->debug("Calling RRTPlanSinglePath() with timeout {}s...",
                      parameters.time_limit);
  const auto result = RRTPlanSinglePath(
      tree,
      MakeStateAndGoalsSamplingFunction(state_sampling_fn, goals,
                                        parameters.goal_sampling_bias,
                                        uniform_unit_real_fn),
      MakeLinearRRTNearestNeighborsFunction<
          StateType, SimpleRRTPlannerTree<StateType>, StateType>(
          [&](const StateType& tree_state, const StateType& sampled) {
            return planning_space->NearestNeighborDistanceForwards(tree_state,
                                                                   sampled);
          },
          parameters.parallelize_nearest_neighbor),
      forward_propagation_fn, {}, goal_check_fn, {},
      MakeRRTTimeoutTerminationFunction(parameters.time_limit));

  auto combined_statistics = result.Statistics();
  combined_statistics.merge(propagation_statistics);
  drake::log()->debug(
      "RRT statistics {}",
      common_robotics_utilities::print::Print(combined_statistics));

  if (result.Path().empty()) {
    drake::log()->warn("RRT failed to plan a path");
    return PathPlanningResult<StateType>(PathPlanningStatus::kTimeout);
  } else {
    const double path_length = planning_space->CalcPathLength(result.Path());
    drake::log()->debug("RRT found path of length {} with {} states",
                        path_length, result.Path().size());
    return PathPlanningResult<StateType>(result.Path(), path_length);
  }
}

template <typename StateType>
PathPlanningResult<StateType> RRTPlanner<StateType>::PlanGoalSampling(
    const std::vector<StateType>& starts,
    const GoalSampleFunction& goal_sample_fn, const Parameters& parameters,
    PlanningSpace<StateType>* planning_space) {
  DRAKE_THROW_UNLESS(parameters.goal_sampling_bias > 0.0);
  DRAKE_THROW_UNLESS(parameters.time_limit > 0.0);
  DRAKE_THROW_UNLESS(parameters.goal_tolerance > 0.0);
  DRAKE_THROW_UNLESS(goal_sample_fn != nullptr);
  DRAKE_THROW_UNLESS(planning_space != nullptr);

  const auto& [valid_starts, status] =
      planning_space->ExtractValidStarts(starts);
  if (!status.is_success()) {
    return PathPlanningResult<StateType>(status);
  }

  const SamplingFunction<StateType> state_sampling_fn = [&]() {
    return planning_space->SampleState();
  };

  // Storage for sampled goal states.
  std::vector<StateType> sampled_goal_states;

  const SamplingFunction<StateType> goal_sampling_fn = [&]() {
    const StateType goal_sample = planning_space->SampleState();
    sampled_goal_states.emplace_back(goal_sample);
    return goal_sample;
  };

  // Statistics for edge propagation function.
  std::map<std::string, double> propagation_statistics;

  const RRTForwardPropagationFunction<StateType, StateType>
      forward_propagation_fn =
          [&](const StateType& nearest, const StateType& sample) {
            const std::vector<StateType> propagated_states =
                planning_space->PropagateForwards(nearest, sample,
                                                  &propagation_statistics);

            ForwardPropagation<StateType> forward_propagation;
            forward_propagation.reserve(propagated_states.size());
            int64_t relative_parent_index = -1;
            for (const auto& propagated_state : propagated_states) {
              forward_propagation.emplace_back(propagated_state,
                                               relative_parent_index);
              relative_parent_index++;
            }
            return forward_propagation;
          };

  const RRTCheckGoalReachedFunction<StateType> goal_check_fn =
      [&](const StateType& state) {
        for (const auto& goal : sampled_goal_states) {
          if (planning_space->StateDistanceForwards(state, goal) <=
              parameters.goal_tolerance) {
            return true;
          }
        }
        return false;
      };

  SimpleRRTPlannerTree<StateType> tree(starts.size());
  for (const auto& start : starts) {
    tree.AddNode(start);
  }

  std::mt19937_64 prng(parameters.prng_seed);
  std::uniform_real_distribution<double> uniform_unit_dist(0.0, 1.0);
  const common_robotics_utilities::utility::UniformUnitRealFunction
      uniform_unit_real_fn = [&]() {
        return uniform_unit_dist(prng);
      };

  drake::log()->debug("Calling RRTPlanSinglePath() with timeout {}s...",
                      parameters.time_limit);
  const auto result = RRTPlanSinglePath(
      tree, SamplingFunction<StateType>([&]() {
        if (uniform_unit_real_fn() > parameters.goal_sampling_bias) {
          return state_sampling_fn();
        } else {
          return goal_sampling_fn();
        }
      }),
      MakeLinearRRTNearestNeighborsFunction<
          StateType, SimpleRRTPlannerTree<StateType>, StateType>(
          [&](const StateType& tree_state, const StateType& sampled) {
            return planning_space->NearestNeighborDistanceForwards(tree_state,
                                                                   sampled);
          },
          parameters.parallelize_nearest_neighbor),
      forward_propagation_fn, {}, goal_check_fn, {},
      MakeRRTTimeoutTerminationFunction(parameters.time_limit));

  auto combined_statistics = result.Statistics();
  combined_statistics.merge(propagation_statistics);
  drake::log()->debug(
      "RRT statistics {}",
      common_robotics_utilities::print::Print(combined_statistics));

  if (result.Path().empty()) {
    drake::log()->warn("RRT failed to plan a path");
    return PathPlanningResult<StateType>(PathPlanningStatus::kTimeout);
  } else {
    const double path_length = planning_space->CalcPathLength(result.Path());
    drake::log()->debug("RRT found path of length {} with {} states",
                        path_length, result.Path().size());
    return PathPlanningResult<StateType>(result.Path(), path_length);
  }
}

template <typename StateType>
PathPlanningResult<StateType> RRTPlanner<StateType>::PlanGoalCheck(
    const std::vector<StateType>& starts,
    const GoalCheckFunction& goal_check_fn, const Parameters& parameters,
    PlanningSpace<StateType>* planning_space) {
  DRAKE_THROW_UNLESS(parameters.time_limit > 0.0);
  DRAKE_THROW_UNLESS(goal_check_fn != nullptr);
  DRAKE_THROW_UNLESS(planning_space != nullptr);

  const auto& [valid_starts, status] =
      planning_space->ExtractValidStarts(starts);
  if (!status.is_success()) {
    return PathPlanningResult<StateType>(status);
  }

  const SamplingFunction<StateType> state_sampling_fn = [&]() {
    return planning_space->SampleState();
  };

  // Statistics for edge propagation function.
  std::map<std::string, double> propagation_statistics;

  const RRTForwardPropagationFunction<StateType, StateType>
      forward_propagation_fn =
          [&](const StateType& nearest, const StateType& sample) {
            const std::vector<StateType> propagated_states =
                planning_space->PropagateForwards(nearest, sample,
                                                  &propagation_statistics);

            ForwardPropagation<StateType> forward_propagation;
            forward_propagation.reserve(propagated_states.size());
            int64_t relative_parent_index = -1;
            for (const auto& propagated_state : propagated_states) {
              forward_propagation.emplace_back(propagated_state,
                                               relative_parent_index);
              relative_parent_index++;
            }
            return forward_propagation;
          };

  SimpleRRTPlannerTree<StateType> tree(starts.size());
  for (const auto& start : starts) {
    tree.AddNode(start);
  }

  std::mt19937_64 prng(parameters.prng_seed);
  std::uniform_real_distribution<double> uniform_unit_dist(0.0, 1.0);
  const common_robotics_utilities::utility::UniformUnitRealFunction
      uniform_unit_real_fn = [&]() {
        return uniform_unit_dist(prng);
      };

  drake::log()->debug("Calling RRTPlanSinglePath() with timeout {}s...",
                      parameters.time_limit);
  const auto result = RRTPlanSinglePath(
      tree, state_sampling_fn,
      MakeLinearRRTNearestNeighborsFunction<
          StateType, SimpleRRTPlannerTree<StateType>, StateType>(
          [&](const StateType& tree_state, const StateType& sampled) {
            return planning_space->NearestNeighborDistanceForwards(tree_state,
                                                                   sampled);
          },
          parameters.parallelize_nearest_neighbor),
      forward_propagation_fn, {}, goal_check_fn, {},
      MakeRRTTimeoutTerminationFunction(parameters.time_limit));

  auto combined_statistics = result.Statistics();
  combined_statistics.merge(propagation_statistics);
  drake::log()->debug(
      "RRT statistics {}",
      common_robotics_utilities::print::Print(combined_statistics));

  if (result.Path().empty()) {
    drake::log()->warn("RRT failed to plan a path");
    return PathPlanningResult<StateType>(PathPlanningStatus::kTimeout);
  } else {
    const double path_length = planning_space->CalcPathLength(result.Path());
    drake::log()->debug("RRT found path of length {} with {} states",
                        path_length, result.Path().size());
    return PathPlanningResult<StateType>(result.Path(), path_length);
  }
}

}  // namespace planning
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_PLANNING_STATE_TYPES(
    class ::drake::planning::RRTPlanner)
