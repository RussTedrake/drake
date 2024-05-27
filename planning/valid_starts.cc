#include "planning/valid_starts.h"

#include <utility>

namespace drake {
namespace planning {

template <typename StateType>
ValidStarts<StateType>::ValidStarts(std::vector<StateType> valid_starts)
    : valid_starts_(std::move(valid_starts)) {
  SetStatus();
}

template <typename StateType>
ValidStarts<StateType>::ValidStarts()
    : ValidStarts<StateType>(std::vector<StateType>{}) {}

template <typename StateType>
void ValidStarts<StateType>::SetStatus() {
  auto status = PathPlanningStatusSet::Success();
  if (valid_starts().empty()) {
    status.set_flag(PathPlanningStatus::kNoValidStart);
  }
  status_ = status;
}

}  // namespace planning
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_PLANNING_STATE_TYPES(
    class ::drake::planning::ValidStarts)
