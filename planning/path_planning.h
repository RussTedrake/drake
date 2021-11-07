#pragma once

#include <Eigen/Dense>

#include "drake/common/drake_throw.h"

namespace drake {
namespace planning {

/// Status flags for planning.
enum class PlanningStatusFlags : uint8_t {
  Success = 0x00,
  NoValidStart = 0x01,
  NoValidGoal = 0x02,
  CannotConnectStart = 0x04,
  CannotConnectGoal = 0x08,
  CannotFindPath = 0x10,
  Timeout = 0x20
};

/// Hold the results from planning {path, length, status} where path is the
/// planned sequence of configurations, length is the length of the planned
/// path, and status is the status of the plan. If a solution cannot be found,
/// path is empty, length is infinity, and status is non-zero. Use
/// PlanningStatusFlags to check the meaning of the returned status/error.
class PlanningResult {
 public:
  PlanningResult(const std::vector<Eigen::VectorXd>& path, double path_length,
                 uint8_t status = 0x00)
      : path_(path),
        path_length_(path_length),
        planning_status_(planning_status) {}

  explicit PlanningResult(uint8_t status)
      : path_length_(std::numeric_limits<double>::infinity()),
        planning_status_(status) {
    DRAKE_THROW_UNLESS(planning_status_ != PlanningStatusFlags::Success);
  }

  const std::vector<Eigen::VectorXd>& Path() const { return path_; }

  double PathLength() const { return path_length_; }

  uint8_t Status() const { return planning_status_; }

  bool HasSolution() const { return (Status() == 0x00); }

 private:
  std::vector<Eigen::VectorXd> path_;
  double path_length_ = std::numeric_limits<double>::infinity();
  uint8_t planning_status_ = 0x00;
};

}  // namespace planning
}  // namespace drake
