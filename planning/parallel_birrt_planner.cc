#include "planning/parallel_birrt_planner.h"

#include <atomic>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include <common_robotics_utilities/print.hpp>
#include <common_robotics_utilities/simple_knearest_neighbors.hpp>
#include <common_robotics_utilities/simple_rrt_planner.hpp>

#include "drake/common/drake_throw.h"
#include "drake/common/text_logging.h"
#include "drake/planning/parallel_rrt_planner_tree.h"

namespace drake {
namespace planning {
using common_robotics_utilities::simple_rrt_planner::BiRRTActiveTreeType;
using common_robotics_utilities::simple_rrt_planner::
    BiRRTGoalBridgeCallbackFunction;
using common_robotics_utilities::simple_rrt_planner::
    BiRRTNearestNeighborFunction;
using common_robotics_utilities::simple_rrt_planner::BiRRTPlanMultiPath;
using common_robotics_utilities::simple_rrt_planner::BiRRTPropagationFunction;
using common_robotics_utilities::simple_rrt_planner::
    BiRRTSelectActiveTreeFunction;
using common_robotics_utilities::simple_rrt_planner::
    BiRRTSelectSampleTypeFunction;
using common_robotics_utilities::simple_rrt_planner::
    BiRRTStatesConnectedFunction;
using common_robotics_utilities::simple_rrt_planner::
    BiRRTTerminationCheckFunction;
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
using common_robotics_utilities::simple_rrt_planner::
    MultipleSolutionPlanningResults;
using common_robotics_utilities::simple_rrt_planner::SamplingFunction;
using common_robotics_utilities::simple_rrt_planner::SimpleRRTPlannerTree;
using common_robotics_utilities::simple_rrt_planner::
    SingleSolutionPlanningResults;
using common_robotics_utilities::simple_rrt_planner::TimeoutCheckHelper;

namespace {
/// Parallel worker that handles per-worker shared state (e.g. collision
/// checking context, sampler, constraints) and runs the per-worker BiRRT
/// planner.
template <typename StateType>
class ParallelBiRRTWorker {
 public:
  ParallelBiRRTWorker(int32_t worker_num,
                      std::unique_ptr<PlanningSpace<StateType>> planning_space,
                      std::atomic<bool>* const solution_found,
                      int64_t prng_seed)
      : worker_num_(worker_num),
        planning_space_(std::move(planning_space)),
        solution_found_(solution_found),
        prng_(prng_seed),
        uniform_unit_distribution_(0.0, 1.0) {
    DRAKE_THROW_UNLESS(planning_space_ != nullptr);
    DRAKE_THROW_UNLESS(solution_found_ != nullptr);
    // Reseed our planning space.
    this->mutable_planning_space().random_source().ReseedGenerators(prng_seed);
  }

  const SingleSolutionPlanningResults<StateType>& solution() const {
    return solution_;
  }

  void Plan(
      const typename ParallelBiRRTPlanner<StateType>::Parameters& parameters,
      internal::ParallelRRTPlannerTree<StateType>* const start_tree,
      internal::ParallelRRTPlannerTree<StateType>* const goal_tree) {
    DRAKE_THROW_UNLESS(start_tree != nullptr);
    DRAKE_THROW_UNLESS(goal_tree != nullptr);

    // Build helper functions. For operations performed on/from the start tree,
    // we use the "forwards" methods provided by the planning space; for
    // operations on/from the goal tree, we use the "backwards" methods instead.
    drake::log()->info("[Worker {}] building helpers...", worker_num());

    // Sampling function.
    const SamplingFunction<StateType> sampling_fn = [&]() {
      return mutable_planning_space().SampleState();
    };

    // Nearest-neighbor function.
    const BiRRTNearestNeighborFunction<
        StateType, internal::ParallelRRTPlannerTree<StateType>>
        nearest_neighbor_fn =
            [&](const internal::ParallelRRTPlannerTree<StateType>& tree,
                const StateType& sample,
                const BiRRTActiveTreeType active_tree_type) {
              switch (active_tree_type) {
                case BiRRTActiveTreeType::START_TREE:
                  return internal::GetParallelRRTNearestNeighbor<StateType>(
                      tree, sample,
                      [&](const StateType& from, const StateType& to) {
                        return planning_space().NearestNeighborDistanceForwards(
                            from, to);
                      },
                      parameters.parallelize_nearest_neighbor);
                case BiRRTActiveTreeType::GOAL_TREE:
                  return internal::GetParallelRRTNearestNeighbor<StateType>(
                      tree, sample,
                      [&](const StateType& from, const StateType& to) {
                        return planning_space()
                            .NearestNeighborDistanceBackwards(from, to);
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
              propagated_states = mutable_planning_space().PropagateForwards(
                  nearest, sampled, &propagation_statistics);
              break;
            case BiRRTActiveTreeType::GOAL_TREE:
              propagated_states = mutable_planning_space().PropagateBackwards(
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
              distance = planning_space().StateDistanceForwards(from, to);
              break;
            case BiRRTActiveTreeType::GOAL_TREE:
              distance = planning_space().StateDistanceBackwards(from, to);
              break;
          }
          return distance <= parameters.connection_tolerance;
        };

    // Define our own solution-found callback to check when the first path is
    // found.
    const BiRRTGoalBridgeCallbackFunction<
        StateType, internal::ParallelRRTPlannerTree<StateType>>
        solution_found_fn = [&](internal::ParallelRRTPlannerTree<StateType>&,
                                int64_t,
                                internal::ParallelRRTPlannerTree<StateType>&,
                                int64_t, BiRRTActiveTreeType) {
          solution_found_->store(true);
        };

    // Define our own timeout-based termination function.
    TimeoutCheckHelper timeout_check_helper(parameters.time_limit);

    const BiRRTTerminationCheckFunction termination_check_fn = [&](int64_t,
                                                                   int64_t) {
      return (solution_found_->load() || timeout_check_helper.CheckOrStart());
    };

    // TODO(calderpg) This could just use the planning_space->Draw() method.
    const common_robotics_utilities::utility::UniformUnitRealFunction
        uniform_unit_real_fn = [&]() {
          return GetUniformUnitReal();
        };

    const BiRRTSelectSampleTypeFunction<
        StateType, internal::ParallelRRTPlannerTree<StateType>>
        select_sample_type_fn = MakeUniformRandomBiRRTSelectSampleTypeFunction<
            StateType, internal::ParallelRRTPlannerTree<StateType>>(
            uniform_unit_real_fn, parameters.tree_sampling_bias);

    const BiRRTTreeSamplingFunction<StateType,
                                    internal::ParallelRRTPlannerTree<StateType>>
        tree_sampling_fn = MakeUniformRandomBiRRTTreeSamplingFunction<
            StateType, internal::ParallelRRTPlannerTree<StateType>>(
            uniform_unit_real_fn);

    const BiRRTSelectActiveTreeFunction<
        StateType, internal::ParallelRRTPlannerTree<StateType>>
        select_active_tree_fn = MakeUniformRandomBiRRTSelectActiveTreeFunction<
            StateType, internal::ParallelRRTPlannerTree<StateType>>(
            uniform_unit_real_fn, parameters.p_switch_trees);

    drake::log()->info("[Worker {}] Starting BiRRT planner...", worker_num());
    // Note: we call BiRRTPlanMultiPath rather than BiRRTPlanSinglePath to
    // avoid having two layers of solution-found checks.
    const MultipleSolutionPlanningResults<StateType> birrt_result =
        BiRRTPlanMultiPath(*start_tree, *goal_tree, select_sample_type_fn,
                           sampling_fn, tree_sampling_fn, nearest_neighbor_fn,
                           propagation_fn, {}, states_connected_fn,
                           solution_found_fn, select_active_tree_fn,
                           termination_check_fn);

    drake::log()->info("[Worker {}] Collecting result...", worker_num());
    if (birrt_result.Paths().size() > 0) {
      // Note: a given worker will only ever produce a single path, as it will
      // stop planning after the first solution is found.
      solution_ = SingleSolutionPlanningResults<StateType>(
          birrt_result.Paths().at(0), birrt_result.Statistics());
    } else {
      solution_ =
          SingleSolutionPlanningResults<StateType>(birrt_result.Statistics());
    }
  }

 private:
  int32_t worker_num() const { return worker_num_; }

  const PlanningSpace<StateType>& planning_space() const {
    return *planning_space_;
  }

  PlanningSpace<StateType>& mutable_planning_space() {
    return *planning_space_;
  }

  double GetUniformUnitReal() { return uniform_unit_distribution_(prng_); }

  const int32_t worker_num_;
  std::unique_ptr<PlanningSpace<StateType>> planning_space_;

  std::atomic<bool>* const solution_found_;

  std::mt19937_64 prng_;
  std::uniform_real_distribution<double> uniform_unit_distribution_;

  SingleSolutionPlanningResults<StateType> solution_;
};

template <typename StateType>
PathPlanningResult<StateType> GetPathPlanningResult(
    const std::vector<SingleSolutionPlanningResults<StateType>>& solutions,
    const PlanningSpace<StateType>& planning_space) {
  // Merge statistics.
  std::map<std::string, double> merged_statistics;
  for (const auto& solution : solutions) {
    for (const auto& [key, value] : solution.Statistics()) {
      merged_statistics[key] += value;
    }
  }

  drake::log()->info(
      "ParallelBiRRT statistics {}",
      common_robotics_utilities::print::Print(merged_statistics));

  // Return the best solution.
  int32_t best_solution_index = -1;
  double best_solution_length = std::numeric_limits<double>::infinity();

  for (size_t solution_num = 0; solution_num < solutions.size();
       ++solution_num) {
    const auto& solution = solutions.at(solution_num);
    if (solution.Path().size() > 0) {
      const double solution_length =
          planning_space.CalcPathLength(solution.Path());
      if (solution_length < best_solution_length) {
        best_solution_index = static_cast<int32_t>(solution_num);
        best_solution_length = solution_length;
      }
    }
  }

  if (best_solution_index < 0) {
    drake::log()->warn("ParallelBiRRT failed to plan a path");
    return PathPlanningResult<StateType>(PathPlanningStatus::kTimeout);
  } else {
    const auto& best_solution_path = solutions.at(best_solution_index).Path();
    drake::log()->info("ParallelBiRRT found path of length {} with {} states",
                       best_solution_length, best_solution_path.size());
    return PathPlanningResult<StateType>(best_solution_path,
                                         best_solution_length);
  }
}

}  // namespace

template <typename StateType>
PathPlanningResult<StateType> ParallelBiRRTPlanner<StateType>::Plan(
    const StateType& start, const StateType& goal, const Parameters& parameters,
    const PlanningSpace<StateType>& planning_space) {
  return Plan(std::vector<StateType>{start}, std::vector<StateType>{goal},
              parameters, planning_space);
}

template <typename StateType>
PathPlanningResult<StateType> ParallelBiRRTPlanner<StateType>::Plan(
    const std::vector<StateType>& starts, const std::vector<StateType>& goals,
    const Parameters& parameters,
    const PlanningSpace<StateType>& planning_space) {
  DRAKE_THROW_UNLESS(parameters.tree_sampling_bias > 0.0);
  DRAKE_THROW_UNLESS(parameters.p_switch_trees > 0.0);
  DRAKE_THROW_UNLESS(parameters.time_limit > 0.0);
  DRAKE_THROW_UNLESS(parameters.connection_tolerance >= 0.0);
  DRAKE_THROW_UNLESS(parameters.num_workers > 0);
  DRAKE_THROW_UNLESS(parameters.initial_tree_capacity >= 0);

  const auto& [valid_starts, valid_goals, status] =
      planning_space.ExtractValidStartsAndGoals(starts, goals);
  if (!status.is_success()) {
    return PathPlanningResult<StateType>(status);
  }

  // Assemble starts & goals.
  internal::ParallelRRTPlannerTree<StateType> start_tree(
      parameters.initial_tree_capacity);
  for (const StateType& start : valid_starts) {
    start_tree.AddNode(start);
  }
  internal::ParallelRRTPlannerTree<StateType> goal_tree(
      parameters.initial_tree_capacity);
  for (const StateType& goal : valid_goals) {
    goal_tree.AddNode(goal);
  }

  // Assemble workers.
  drake::log()->info("Building {} ParallelBiRRT workers...",
                     parameters.num_workers);
  std::atomic<bool> solution_found(false);

  std::mt19937_64 seed_dist(parameters.prng_seed);
  std::vector<ParallelBiRRTWorker<StateType>> workers;
  for (int32_t worker_num = 0; worker_num < parameters.num_workers;
       ++worker_num) {
    const int64_t worker_prng_seed = seed_dist();
    workers.emplace_back(worker_num, planning_space.Clone(), &solution_found,
                         worker_prng_seed);
  }

  // Start planners.
  drake::log()->info("Dispatching ParallelBiRRT planner threads...");
  std::vector<std::thread> worker_threads;
  for (int32_t worker_num = 0; worker_num < parameters.num_workers;
       ++worker_num) {
    ParallelBiRRTWorker<StateType>& worker = workers.at(worker_num);
    const auto worker_thread_fn = [&]() {
      worker.Plan(parameters, &start_tree, &goal_tree);
    };
    worker_threads.emplace_back(worker_thread_fn);
  }

  // Wait for planners to finish.
  drake::log()->info(
      "Waiting for ParallelBiRRT planner threads to complete...");
  for (auto& worker_thread : worker_threads) {
    worker_thread.join();
  }

  // Collect solutions.
  std::vector<SingleSolutionPlanningResults<StateType>> solutions;
  for (const ParallelBiRRTWorker<StateType>& worker : workers) {
    solutions.push_back(worker.solution());
  }

  return GetPathPlanningResult(solutions, planning_space);
}

}  // namespace planning
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_PLANNING_STATE_TYPES(
    class ::drake::planning::ParallelBiRRTPlanner)
