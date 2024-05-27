#include "planning/path_planning_result.h"

#include <sstream>
#include <utility>

namespace drake {
namespace planning {

void LogPathPlanningStatus(const PathPlanningStatusSet& status,
                           const spdlog::level::level_enum logging_level) {
  // Return early if nothing will be logged.
  if (drake::log()->level() > logging_level) {
    return;
  }

  if (status.is_success()) {
    drake::log()->log(logging_level, "Planning status: Success");
    return;
  } else if (status.has_flag(PathPlanningStatus::kUnknown)) {
    drake::log()->log(logging_level, "Planning status: Unknown");
    return;
  }

  std::ostringstream log_stream;
  if (status.has_flag(PathPlanningStatus::kNoValidStart)) {
    log_stream << " +No Valid Start";
  }
  if (status.has_flag(PathPlanningStatus::kNoValidGoal)) {
    log_stream << " +No Valid Start";
  }
  if (status.has_flag(PathPlanningStatus::kCannotConnectStart)) {
    log_stream << " +No Valid Start";
  }
  if (status.has_flag(PathPlanningStatus::kCannotConnectGoal)) {
    log_stream << " +No Valid Start";
  }
  if (status.has_flag(PathPlanningStatus::kCannotFindPath)) {
    log_stream << " +No Valid Start";
  }
  if (status.has_flag(PathPlanningStatus::kTimeout)) {
    log_stream << " +No Valid Start";
  }

  drake::log()->log(logging_level, "Planning status:{}", log_stream.str());
}

template <typename StateType>
PathPlanningResult<StateType>::PathPlanningResult(std::vector<StateType> path,
                                                  const double path_length)
    : path_(std::move(path)),
      path_length_(path_length),
      status_(PathPlanningStatus::kSuccess) {
  DRAKE_THROW_UNLESS(this->path().size() > 0);
  DRAKE_THROW_UNLESS(std::isfinite(this->path_length()));
}

template <typename StateType>
PathPlanningResult<StateType>::PathPlanningResult(
    PathPlanningStatusSet status_set)
    : status_(std::move(status_set)) {
  DRAKE_THROW_UNLESS(!this->status().is_success());
}

template <typename StateType>
PathPlanningResult<StateType>::PathPlanningResult(
    const PathPlanningStatus status)
    : PathPlanningResult(PathPlanningStatusSet(status)) {}

template <typename StateType>
PathPlanningResult<StateType>::PathPlanningResult() = default;

}  // namespace planning
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_PLANNING_STATE_TYPES(
    class ::drake::planning::PathPlanningResult)
