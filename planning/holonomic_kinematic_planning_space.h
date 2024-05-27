#pragma once

#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "planning/joint_limits.h"
#include "planning/symmetric_planning_space.h"
#include <Eigen/Geometry>
#include <fmt/format.h>

#include "drake/common/copyable_unique_ptr.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/drake_throw.h"
#include "drake/planning/collision_checker.h"

namespace drake {
namespace planning {
/// Implementation of a "holonomic" kinematic planning space. Broadly, this
/// covers all non-constrained planning uses to date, i.e. anywhere the
/// distance, interpolation, and validity provided by a single collision checker
/// is sufficient. Notably, it *does not* necessarily cover planning using a
/// mobile base.
/// Note: two member functions, SampleState() and MotionCost(), may be further
/// overridden by derived classes.
class HolonomicKinematicPlanningSpace
    : public SymmetricPlanningSpace<Eigen::VectorXd> {
 public:
  // The copy constructor is protected for use in implementing Clone().
  // Does not allow copy, move, or assignment.
  HolonomicKinematicPlanningSpace(HolonomicKinematicPlanningSpace&&) = delete;
  HolonomicKinematicPlanningSpace& operator=(
      const HolonomicKinematicPlanningSpace&) = delete;
  HolonomicKinematicPlanningSpace& operator=(
      HolonomicKinematicPlanningSpace&&) = delete;

  /// Constructor.
  /// @param collision_checker Collision checker to use.
  /// @param joint_limits Joint limits to use for sampling. @pre size of
  /// position limits must match the configuration size of collision_checker.
  /// @param propagation_step_size Step size for propagation functions.
  /// @param seed Seed for per-thread random source.
  HolonomicKinematicPlanningSpace(
      std::unique_ptr<drake::planning::CollisionChecker> collision_checker,
      const JointLimits& joint_limits, double propagation_step_size,
      uint64_t seed);

  ~HolonomicKinematicPlanningSpace() override;

  // Implement SymmetricPlanningSpace API.

  std::unique_ptr<PlanningSpace<Eigen::VectorXd>> Clone() const override;

  bool CheckStateValidity(const Eigen::VectorXd& state) const final;

  bool CheckEdgeValidity(const Eigen::VectorXd& from,
                         const Eigen::VectorXd& to) const final;

  Eigen::VectorXd SampleState() override;

  double StateDistance(const Eigen::VectorXd& from,
                       const Eigen::VectorXd& to) const final;

  Eigen::VectorXd Interpolate(const Eigen::VectorXd& from,
                              const Eigen::VectorXd& to,
                              double ratio) const final;

  std::vector<Eigen::VectorXd> Propagate(
      const Eigen::VectorXd& from, const Eigen::VectorXd& to,
      std::map<std::string, double>* propagation_statistics) final;

  double MotionCost(const Eigen::VectorXd& from,
                    const Eigen::VectorXd& to) const override;

  // Class-specific getters and setters.

  const drake::planning::CollisionChecker& collision_checker() const {
    return *collision_checker_;
  }

  drake::planning::CollisionChecker& mutable_collision_checker() {
    return *collision_checker_;
  }

  const JointLimits& joint_limits() const { return joint_limits_; }

  /// Sets new joint limits for sampling.
  /// @param joint_limits New joint limits. @pre size of position limits must
  /// match the configuration size of the collision checker.
  void SetJointLimits(const JointLimits& joint_limits) {
    DRAKE_THROW_UNLESS(joint_limits.num_positions() ==
                       collision_checker().GetZeroConfiguration().size());
    joint_limits_ = joint_limits;
  }

  double propagation_step_size() const { return propagation_step_size_; }

  /// Sets new propagation step size.
  /// @param propagation_step_size New propagation step size.
  /// @pre > 0.0 and finite.
  void SetPropagationStepSize(double propagation_step_size) {
    DRAKE_THROW_UNLESS(std::isfinite(propagation_step_size));
    DRAKE_THROW_UNLESS(propagation_step_size > 0.0);
    propagation_step_size_ = propagation_step_size;
  }

 protected:
  // Copy constructor for use in Clone().
  HolonomicKinematicPlanningSpace(
      const HolonomicKinematicPlanningSpace& other) = default;

 private:
  drake::copyable_unique_ptr<drake::planning::CollisionChecker>
      collision_checker_;
  JointLimits joint_limits_;
  double propagation_step_size_ = 0.0;
};

// Separate implementation for constrained case, see if this makes sense?
}  // namespace planning
}  // namespace drake
