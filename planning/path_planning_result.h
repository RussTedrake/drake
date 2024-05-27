#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include "planning/default_state_types.h"

#include "drake/common/drake_copyable.h"
#include "drake/common/text_logging.h"

namespace drake {
namespace planning {
/// Status flags for planning.
enum class PathPlanningStatus : uint8_t {
  kSuccess = 0x00,
  kNoValidStart = 0x01,
  kNoValidGoal = 0x02,
  kCannotConnectStart = 0x04,
  kCannotConnectGoal = 0x08,
  kCannotFindPath = 0x10,
  kTimeout = 0x20,
  kUnknown = 0x40
};

/// Like a std::set<PathPlanningStatus> but much more compact.
struct PathPlanningStatusSet {
  uint8_t bits{static_cast<uint8_t>(PathPlanningStatus::kUnknown)};

  static PathPlanningStatusSet Success() {
    return PathPlanningStatusSet(PathPlanningStatus::kSuccess);
  }

  static PathPlanningStatusSet Unknown() {
    return PathPlanningStatusSet(PathPlanningStatus::kUnknown);
  }

  PathPlanningStatusSet() = default;

  explicit PathPlanningStatusSet(PathPlanningStatus status)
      : bits(static_cast<uint8_t>(status)) {}

  void set_flag(PathPlanningStatus flag) { bits |= static_cast<uint8_t>(flag); }

  bool has_flag(PathPlanningStatus flag) const {
    return (bits & static_cast<uint8_t>(flag)) != 0;
  }

  bool is_success() const {
    return bits == static_cast<uint8_t>(PathPlanningStatus::kSuccess);
  }
};

/// Logs the provided planning status at the specified log level.
void LogPathPlanningStatus(
    const PathPlanningStatusSet& status,
    spdlog::level::level_enum logging_level = spdlog::level::info);

/// Holds the results from planning {path, length, status} where path is the
/// planned sequence of configurations, length is the length of the planned
/// path, and status is the status of the plan. If a solution cannot be found,
/// path is empty, length is infinity, and status is non-zero. Use
/// PathPlanningStatus to check the meaning of the returned status/error.
template <typename StateType>
class PathPlanningResult {
 public:
  /// Provides all copy/move/assign operations.
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(PathPlanningResult);

  PathPlanningResult(std::vector<StateType> path, double path_length);

  /// Constructs a PathPlanningResult with the specified status set.
  /// @param status_set Planning status set. @pre status_set is not success.
  explicit PathPlanningResult(PathPlanningStatusSet status_set);

  /// Constructs a PathPlanningResult with the specified status.
  /// @param status Planning status. @pre status is not success.
  explicit PathPlanningResult(PathPlanningStatus status);

  PathPlanningResult();

  const std::vector<StateType>& path() const { return path_; }

  double path_length() const { return path_length_; }

  const PathPlanningStatusSet& status() const { return status_; }

  bool has_solution() const { return status().is_success(); }

 private:
  std::vector<StateType> path_;
  double path_length_ = std::numeric_limits<double>::infinity();
  PathPlanningStatusSet status_ = PathPlanningStatusSet::Unknown();
};

}  // namespace planning
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_PLANNING_STATE_TYPES(
    class ::drake::planning::PathPlanningResult)
