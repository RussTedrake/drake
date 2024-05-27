#include "planning/valid_starts_and_goals.h"

#include <utility>

#include "drake/common/text_logging.h"

namespace drake {
namespace planning {

template <typename StateType>
ValidStartsAndGoals<StateType>::ValidStartsAndGoals(
    std::vector<StateType> valid_starts, std::vector<StateType> valid_goals)
    : valid_starts_(std::move(valid_starts)),
      valid_goals_(std::move(valid_goals)) {
  SetStatus();
}

template <typename StateType>
ValidStartsAndGoals<StateType>::ValidStartsAndGoals()
    : ValidStartsAndGoals<StateType>({}, {}) {}

template <typename StateType>
void ValidStartsAndGoals<StateType>::SetStatus() {
  auto status = PathPlanningStatusSet::Success();
  if (valid_starts().empty()) {
    status.set_flag(PathPlanningStatus::kNoValidStart);
  }
  if (valid_goals().empty()) {
    status.set_flag(PathPlanningStatus::kNoValidGoal);
  }
  status_ = status;
}

}  // namespace planning
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_PLANNING_STATE_TYPES(
    class ::drake::planning::ValidStartsAndGoals)
