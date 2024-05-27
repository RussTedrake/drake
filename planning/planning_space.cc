#include "planning/planning_space.h"

#include "drake/common/drake_throw.h"
#include "drake/common/text_logging.h"

namespace drake {
namespace planning {

template <typename StateType>
PlanningSpace<StateType>::~PlanningSpace() = default;

template <typename StateType>
double PlanningSpace<StateType>::CalcPathLength(
    const std::vector<StateType>& path) const {
  double path_length = 0.0;
  for (size_t index = 1; index < path.size(); ++index) {
    path_length += StateDistanceForwards(path.at(index - 1), path.at(index));
  }
  return path_length;
}

template <typename StateType>
ValidStarts<StateType> PlanningSpace<StateType>::ExtractValidStarts(
    const std::vector<StateType>& starts) const {
  std::vector<StateType> valid_starts;
  for (size_t start_index = 0; start_index < starts.size(); ++start_index) {
    const StateType& start = starts.at(start_index);
    if (CheckStateValidity(start)) {
      valid_starts.push_back(start);
    } else {
      drake::log()->warn("Start {}/{} is invalid", start_index + 1,
                         starts.size());
    }
  }

  return ValidStarts<StateType>(valid_starts);
}

template <typename StateType>
ValidStartsAndGoals<StateType>
PlanningSpace<StateType>::ExtractValidStartsAndGoals(
    const std::vector<StateType>& starts,
    const std::vector<StateType>& goals) const {
  std::vector<StateType> valid_starts;
  for (size_t start_index = 0; start_index < starts.size(); ++start_index) {
    const StateType& start = starts.at(start_index);
    if (CheckStateValidity(start)) {
      valid_starts.push_back(start);
    } else {
      drake::log()->warn("Start {}/{} is invalid", start_index + 1,
                         starts.size());
    }
  }

  std::vector<StateType> valid_goals;
  for (size_t goal_index = 0; goal_index < goals.size(); ++goal_index) {
    const StateType& goal = goals.at(goal_index);
    if (CheckStateValidity(goal)) {
      valid_goals.push_back(goal);
    } else {
      drake::log()->warn("Goal {}/{} is invalid", goal_index + 1, goals.size());
    }
  }

  return ValidStartsAndGoals<StateType>(valid_starts, valid_goals);
}

template <typename StateType>
bool PlanningSpace<StateType>::CheckPathValidity(
    const std::vector<StateType>& path) const {
  if (path.size() > 1) {
    for (size_t index = 1; index < path.size(); ++index) {
      const StateType& previous = path.at(index - 1);
      const StateType& current = path.at(index);
      if (!CheckEdgeValidity(previous, current)) {
        drake::log()->warn("Edge from waypoint {} to waypoint {} invalid",
                           index - 1, index);
        return false;
      }
    }
    return true;
  } else if (path.size() == 1) {
    return CheckStateValidity(path.at(0));
  } else {
    throw std::runtime_error("Cannot check zero-waypoint paths for validity");
  }
}

template <typename StateType>
std::optional<StateType> PlanningSpace<StateType>::MaybeSampleValidState(
    const int max_attempts) {
  DRAKE_THROW_UNLESS(max_attempts > 0);
  for (int attempt = 0; attempt < max_attempts; ++attempt) {
    StateType sample = SampleState();
    if (CheckStateValidity(sample)) {
      return sample;
    }
  }
  return std::nullopt;
}

template <typename StateType>
StateType PlanningSpace<StateType>::SampleValidState(const int max_attempts) {
  std::optional<StateType> maybe_valid_sample =
      MaybeSampleValidState(max_attempts);
  if (maybe_valid_sample) {
    return maybe_valid_sample.value();
  } else {
    throw std::runtime_error(fmt::format(
        "Failed to sample valid state in {} attempts", max_attempts));
  }
}

template <typename StateType>
PlanningSpace<StateType>::PlanningSpace(const PlanningSpace<StateType>&) =
    default;

template <typename StateType>
PlanningSpace<StateType>::PlanningSpace(const uint64_t seed,
                                        const bool supports_parallel,
                                        const bool is_symmetric)
    : random_source_(seed),
      supports_parallel_(supports_parallel),
      is_symmetric_(is_symmetric) {}

}  // namespace planning
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_PLANNING_STATE_TYPES(
    class ::drake::planning::PlanningSpace)
