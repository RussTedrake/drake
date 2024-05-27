#include "planning/birrt_planner.h"

#include <map>
#include <string>

#include <common_robotics_utilities/print.hpp>
#include <common_robotics_utilities/simple_knearest_neighbors.hpp>
#include <common_robotics_utilities/simple_rrt_planner.hpp>

#include "drake/common/drake_throw.h"
#include "drake/common/text_logging.h"

namespace drake {
namespace planning {
using common_robotics_utilities::simple_knearest_neighbors::
    GetKNearestNeighbors;
using common_robotics_utilities::simple_rrt_planner::BiRRTActiveTreeType;
using common_robotics_utilities::simple_rrt_planner::
    BiRRTNearestNeighborFunction;
using common_robotics_utilities::simple_rrt_planner::BiRRTPlanSinglePath;
using common_robotics_utilities::simple_rrt_planner::BiRRTPropagationFunction;
using common_robotics_utilities::simple_rrt_planner::
    BiRRTSelectActiveTreeFunction;
using common_robotics_utilities::simple_rrt_planner::
    BiRRTSelectSampleTypeFunction;
using common_robotics_utilities::simple_rrt_planner::
    BiRRTStatesConnectedFunction;
using common_robotics_utilities::simple_rrt_planner::BiRRTTreeSamplingFunction;
using common_robotics_utilities::simple_rrt_planner::ForwardPropagation;
using common_robotics_utilities::simple_rrt_planner::
    MakeBiRRTTimeoutTerminationFunction;
using common_robotics_utilities::simple_rrt_planner::
    MakeUniformRandomBiRRTSelectActiveTreeFunction;
using common_robotics_utilities::simple_rrt_planner::
    MakeUniformRandomBiRRTSelectSampleTypeFunction;
using common_robotics_utilities::simple_rrt_planner::
    MakeUniformRandomBiRRTTreeSamplingFunction;
using common_robotics_utilities::simple_rrt_planner::SamplingFunction;
using common_robotics_utilities::simple_rrt_planner::SimpleRRTPlannerTree;

namespace {
template <typename StateType>
using TreeNodeDistanceFunction = std::function<double(
    const typename SimpleRRTPlannerTree<StateType>::NodeType&,
    const StateType&)>;

template <typename StateType>
int64_t GetRRTNearestNeighbor(
    const SimpleRRTPlannerTree<StateType>& tree, const StateType& sample,
    const TreeNodeDistanceFunction<StateType>& distance_fn, bool use_parallel) {
  const auto neighbors = GetKNearestNeighbors(tree.GetNodesImmutable(), sample,
                                              distance_fn, 1, use_parallel);
  if (neighbors.size() > 0) {
    const auto& nearest_neighbor = neighbors.at(0);
    return nearest_neighbor.Index();
  } else {
    throw std::runtime_error("NN check produced no neighbors");
  }
}
}  // namespace

template <typename StateType>
PathPlanningResult<StateType> BiRRTPlanner<StateType>::Plan(
    const StateType& start, const StateType& goal, const Parameters& parameters,
    PlanningSpace<StateType>* planning_space) {
  return Plan(std::vector<StateType>{start}, std::vector<StateType>{goal},
              parameters, planning_space);
}

template <typename StateType>
PathPlanningResult<StateType> BiRRTPlanner<StateType>::Plan(
    const std::vector<StateType>& starts, const std::vector<StateType>& goals,
    const Parameters& parameters, PlanningSpace<StateType>* planning_space) {
  DRAKE_THROW_UNLESS(parameters.tree_sampling_bias > 0.0);
  DRAKE_THROW_UNLESS(parameters.p_switch_trees > 0.0);
  DRAKE_THROW_UNLESS(parameters.time_limit > 0.0);
  DRAKE_THROW_UNLESS(parameters.connection_tolerance >= 0.0);
  DRAKE_THROW_UNLESS(planning_space != nullptr);

  const auto& [valid_starts, valid_goals, status] =
      planning_space->ExtractValidStartsAndGoals(starts, goals);
  if (!status.is_success()) {
    return PathPlanningResult<StateType>(status);
  }

  // Build helper functions. For operations performed on/from the start tree, we
  // use the "forwards" methods provided by the planning space; for operations
  // on/from the goal tree, we use the "backwards" methods instead.
  drake::log()->debug("Building BiRRT components & helpers...");

  // Sampling function.
  const SamplingFunction<StateType> sampling_fn = [&]() {
    return planning_space->SampleState();
  };

  // Nearest-neighbor function.
  const BiRRTNearestNeighborFunction<StateType> nearest_neighbor_fn =
      [&](const SimpleRRTPlannerTree<StateType>& tree, const StateType& sample,
          const BiRRTActiveTreeType active_tree_type) {
        switch (active_tree_type) {
          case BiRRTActiveTreeType::START_TREE:
            return GetRRTNearestNeighbor<StateType>(
                tree, sample,
                [&](const typename SimpleRRTPlannerTree<StateType>::NodeType&
                        from,
                    const StateType& to) {
                  return planning_space->NearestNeighborDistanceForwards(
                      from.GetValueImmutable(), to);
                },
                parameters.parallelize_nearest_neighbor);
          case BiRRTActiveTreeType::GOAL_TREE:
            return GetRRTNearestNeighbor<StateType>(
                tree, sample,
                [&](const typename SimpleRRTPlannerTree<StateType>::NodeType&
                        from,
                    const StateType& to) {
                  return planning_space->NearestNeighborDistanceBackwards(
                      from.GetValueImmutable(), to);
                },
                parameters.parallelize_nearest_neighbor);
        }
        DRAKE_UNREACHABLE();
      };

  // Statistics for edge propagation function.
  std::map<std::string, double> propagation_statistics;

  // Edge propagation function.
  const BiRRTPropagationFunction<StateType> propagation_fn =
      [&](const StateType& nearest, const StateType& sampled,
          const BiRRTActiveTreeType active_tree_type) {
        std::vector<StateType> propagated_states;

        switch (active_tree_type) {
          case BiRRTActiveTreeType::START_TREE:
            propagated_states = planning_space->PropagateForwards(
                nearest, sampled, &propagation_statistics);
            break;
          case BiRRTActiveTreeType::GOAL_TREE:
            propagated_states = planning_space->PropagateBackwards(
                nearest, sampled, &propagation_statistics);
            break;
        }

        ForwardPropagation<StateType> forward_propagation;
        forward_propagation.reserve(propagated_states.size());
        int64_t relative_parent_index = -1;
        for (const auto& propagated_config : propagated_states) {
          forward_propagation.emplace_back(propagated_config,
                                           relative_parent_index);
          relative_parent_index++;
        }
        return forward_propagation;
      };

  // State-state connection check function.
  const BiRRTStatesConnectedFunction<StateType> states_connected_fn =
      [&](const StateType& from, const StateType& to,
          const BiRRTActiveTreeType active_tree_type) {
        double distance = 0.0;
        switch (active_tree_type) {
          case BiRRTActiveTreeType::START_TREE:
            distance = planning_space->StateDistanceForwards(from, to);
            break;
          case BiRRTActiveTreeType::GOAL_TREE:
            distance = planning_space->StateDistanceBackwards(from, to);
            break;
        }
        return distance <= parameters.connection_tolerance;
      };

  // Assemble starts & goals.
  SimpleRRTPlannerTree<StateType> start_tree(valid_starts.size());
  for (const StateType& start : valid_starts) {
    start_tree.AddNode(start);
  }
  SimpleRRTPlannerTree<StateType> goal_tree(valid_goals.size());
  for (const StateType& goal : valid_goals) {
    goal_tree.AddNode(goal);
  }

  // TODO(calderpg) This could just use the planning_space->Draw() method.
  std::mt19937_64 prng(parameters.prng_seed);
  std::uniform_real_distribution<double> uniform_unit_dist(0.0, 1.0);
  const common_robotics_utilities::utility::UniformUnitRealFunction
      uniform_unit_real_fn = [&]() {
        return uniform_unit_dist(prng);
      };

  const BiRRTSelectSampleTypeFunction<StateType> select_sample_type_fn =
      MakeUniformRandomBiRRTSelectSampleTypeFunction<StateType>(
          uniform_unit_real_fn, parameters.tree_sampling_bias);

  const BiRRTTreeSamplingFunction<StateType> tree_sampling_fn =
      MakeUniformRandomBiRRTTreeSamplingFunction<StateType>(
          uniform_unit_real_fn);

  const BiRRTSelectActiveTreeFunction<StateType> select_active_tree_fn =
      MakeUniformRandomBiRRTSelectActiveTreeFunction<StateType>(
          uniform_unit_real_fn, parameters.p_switch_trees);

  // Call the planner
  drake::log()->debug("Calling BiRRTPlanSinglePath() with timeout {}s...",
                      parameters.time_limit);

  const auto result = BiRRTPlanSinglePath(
      start_tree, goal_tree, select_sample_type_fn, sampling_fn,
      tree_sampling_fn, nearest_neighbor_fn, propagation_fn, {},
      states_connected_fn, {}, select_active_tree_fn,
      MakeBiRRTTimeoutTerminationFunction(parameters.time_limit));

  auto combined_statistics = result.Statistics();
  combined_statistics.merge(propagation_statistics);
  drake::log()->debug(
      "BiRRT statistics {}",
      common_robotics_utilities::print::Print(combined_statistics));

  if (result.Path().empty()) {
    drake::log()->warn("BiRRT failed to plan a path");
    return PathPlanningResult<StateType>(PathPlanningStatus::kTimeout);
  } else {
    const double path_length = planning_space->CalcPathLength(result.Path());
    drake::log()->debug("BiRRT found path of length {} with {} states",
                        path_length, result.Path().size());
    return PathPlanningResult<StateType>(result.Path(), path_length);
  }
}

}  // namespace planning
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_PLANNING_STATE_TYPES(
    class ::drake::planning::BiRRTPlanner)
