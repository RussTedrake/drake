#include "planning/holonomic_kinematic_planning_space.h"

#include <utility>

#include <common_robotics_utilities/math.hpp>

namespace drake {
namespace planning {
HolonomicKinematicPlanningSpace::HolonomicKinematicPlanningSpace(
    std::unique_ptr<drake::planning::CollisionChecker> collision_checker,
    const JointLimits& joint_limits, const double propagation_step_size,
    const uint64_t seed)
    : SymmetricPlanningSpace<Eigen::VectorXd>(seed, true),
      collision_checker_(std::move(collision_checker)) {
  DRAKE_THROW_UNLESS(collision_checker_ != nullptr);
  SetJointLimits(joint_limits);
  SetPropagationStepSize(propagation_step_size);
}

HolonomicKinematicPlanningSpace::~HolonomicKinematicPlanningSpace() = default;

std::unique_ptr<PlanningSpace<Eigen::VectorXd>>
HolonomicKinematicPlanningSpace::Clone() const {
  return std::unique_ptr<HolonomicKinematicPlanningSpace>(
      new HolonomicKinematicPlanningSpace(*this));
}

bool HolonomicKinematicPlanningSpace::CheckStateValidity(
    const Eigen::VectorXd& state) const {
  // TODO(calderpg) Incorporate joint limits check into state validity.
  return collision_checker().CheckConfigCollisionFree(state);
}

bool HolonomicKinematicPlanningSpace::CheckEdgeValidity(
    const Eigen::VectorXd& from, const Eigen::VectorXd& to) const {
  // TODO(calderpg) Incorporate joint limits check into edge validity.
  return collision_checker().CheckEdgeCollisionFree(from, to);
}

Eigen::VectorXd HolonomicKinematicPlanningSpace::SampleState() {
  const JointLimits& limits = joint_limits();
  Eigen::VectorXd sample = Eigen::VectorXd::Zero(limits.num_positions());
  for (int index = 0; index < sample.size(); ++index) {
    const double lower = limits.position_lower()(index);
    const double upper = limits.position_upper()(index);
    const double ratio = random_source().DrawUniformUnitReal();
    sample(index) =
        common_robotics_utilities::math::Interpolate(lower, upper, ratio);
  }
  return sample;
}

double HolonomicKinematicPlanningSpace::StateDistance(
    const Eigen::VectorXd& from, const Eigen::VectorXd& to) const {
  return collision_checker().ComputeConfigurationDistance(from, to);
}

Eigen::VectorXd HolonomicKinematicPlanningSpace::Interpolate(
    const Eigen::VectorXd& from, const Eigen::VectorXd& to,
    const double ratio) const {
  return collision_checker().InterpolateBetweenConfigurations(from, to, ratio);
}

std::vector<Eigen::VectorXd> HolonomicKinematicPlanningSpace::Propagate(
    const Eigen::VectorXd& from, const Eigen::VectorXd& to,
    std::map<std::string, double>* propagation_statistics) {
  DRAKE_THROW_UNLESS(propagation_statistics != nullptr);

  // Ensure that propagation_statistics contains the right keys. If additional
  // tracking statistics are added below, they must be added here as well. Note
  // that try_emplace only adds a new element if it does not already exist.
  propagation_statistics->try_emplace("edges_considered", 0.0);
  propagation_statistics->try_emplace("valid_edges", 0.0);
  propagation_statistics->try_emplace("edges_in_collision", 0.0);
  propagation_statistics->try_emplace("complete_propagation_successful", 0.0);

  std::vector<Eigen::VectorXd> propagated_states;

  // Compute a maximum number of steps to take.
  const double total_distance = StateDistance(from, to);
  const int total_steps =
      static_cast<int>(std::ceil(total_distance / propagation_step_size()));

  Eigen::VectorXd current = from;
  for (int step = 1; step <= total_steps; ++step) {
    (*propagation_statistics)["edges_considered"] += 1.0;
    const double ratio =
        static_cast<double>(step) / static_cast<double>(total_steps);
    const Eigen::VectorXd intermediate =
        collision_checker().InterpolateBetweenConfigurations(from, to, ratio);
    if (CheckEdgeValidity(current, intermediate)) {
      (*propagation_statistics)["valid_edges"] += 1.0;
      if (step == total_steps) {
        (*propagation_statistics)["complete_propagation_successful"] += 1.0;
      }
      propagated_states.emplace_back(intermediate);
      current = intermediate;
    } else {
      (*propagation_statistics)["edges_in_collision"] += 1.0;
      break;
    }
  }

  return propagated_states;
}

double HolonomicKinematicPlanningSpace::MotionCost(
    const Eigen::VectorXd& from, const Eigen::VectorXd& to) const {
  return StateDistance(from, to);
}

}  // namespace planning
}  // namespace drake
