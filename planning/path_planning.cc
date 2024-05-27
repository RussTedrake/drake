#include "planning/path_planning.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/path_processing.hpp>
#include <common_robotics_utilities/print.hpp>

#include "drake/common/drake_throw.h"
#include "drake/common/text_logging.h"
#include "drake/multibody/inverse_kinematics/angle_between_vectors_constraint.h"
#include "drake/multibody/inverse_kinematics/gaze_target_constraint.h"
#include "drake/multibody/inverse_kinematics/orientation_constraint.h"
#include "drake/multibody/inverse_kinematics/point_to_point_distance_constraint.h"
#include "drake/multibody/inverse_kinematics/position_constraint.h"
#include "drake/multibody/inverse_kinematics/unit_quaternion_constraint.h"
#include "drake/planning/collision_checker.h"
#include "drake/solvers/snopt_solver.h"
#include "drake/solvers/solve.h"
#include "planning/joint_limits.h"
#include "planning/mbp_constraint_types.h"

namespace drake {
namespace planning {
using drake::multibody::AngleBetweenVectorsConstraint;
using drake::multibody::GazeTargetConstraint;
using drake::multibody::OrientationConstraint;
using drake::multibody::PointToPointDistanceConstraint;
using drake::multibody::PositionConstraint;
using drake::multibody::UnitQuaternionConstraint;
using drake::planning::CollisionChecker;
using drake::planning::CollisionCheckerContext;
using drake::planning::ConfigurationDistanceFunction;

namespace {
template<typename ConstraintType>
bool IsConstraintType(const drake::solvers::Constraint& constraint) {
  return (dynamic_cast<const ConstraintType*>(&constraint) != nullptr);
}
}  // namespace

bool CheckConstraintsSatisfied(
    const std::vector<std::shared_ptr<drake::solvers::Constraint>>& constraints,
    const Eigen::VectorXd& q, const double tolerance,
    const spdlog::level::level_enum logging_level) {
  std::function<void(const std::string&)> log_function = nullptr;
  if (drake::log()->level() <= logging_level) {
    log_function = [logging_level] (const std::string& msg) {
        drake::log()->log(logging_level, "CheckConstraintsSatisfied: {}", msg);
    };
  }

  bool constraints_met = true;
  for (const auto& constraint : constraints) {
    if (!CheckConstraintSatisfied(*constraint, q, tolerance, log_function)) {
      constraints_met = false;
      if (log_function == nullptr) {
        break;
      }
    }
  }

  if (constraints_met) {
    if (log_function != nullptr) {
      log_function("all constraints met");
    }
    return true;
  } else {
    return false;
  }
}

bool CheckConstraintSatisfied(
    const drake::solvers::Constraint& constraint, const Eigen::VectorXd& q,
    const double tolerance,
    const std::function<void(const std::string&)>& log_function) {
  if (constraint.CheckSatisfied(q, tolerance)) {
    return true;
  } else {
    if (log_function != nullptr) {
      if (IsConstraintType<CollisionAvoidanceConstraint>(constraint)) {
        log_function("CollisionAvoidanceConstraint not met");
      } else if (IsConstraintType<RelativePoseConstraint>(constraint)) {
        log_function("RelativePoseConstraint not met");
      } else if (IsConstraintType<PositionConstraint>(constraint)) {
        log_function("PositionConstraint not met");
      } else if (IsConstraintType<GazeTargetConstraint>(constraint)) {
        log_function("GazeTargetConstraint not met");
      } else if (IsConstraintType<AngleBetweenVectorsConstraint>(constraint)) {
        log_function("AngleBetweenVectorsConstraint not met");
      } else if (IsConstraintType<OrientationConstraint>(constraint)) {
        log_function("OrientationConstraint not met");
      } else if (IsConstraintType<PointToPointDistanceConstraint>(constraint)) {
        log_function("PointToPointDistanceConstraint not met");
      } else if (IsConstraintType<UnitQuaternionConstraint>(constraint)) {
        log_function("UnitQuaternionConstraint not met");
      } else {
        log_function("unidentified constraint not met");
      }
    }
    return false;
  }
}

ConstrainedEdgeCheckResult CheckEdgeCollisionFreeAndSatisfiesConstraints(
    const Eigen::VectorXd& start, const Eigen::VectorXd& end,
    const CollisionChecker& collision_checker,
    const std::vector<std::shared_ptr<drake::solvers::Constraint>>&
        constraints,
    const double tolerance, CollisionCheckerContext* model_context) {
  const double distance =
      collision_checker.ComputeConfigurationDistance(start, end);
  const double step_size = collision_checker.edge_step_size();
  const int32_t num_steps =
      static_cast<int32_t>(std::max(1.0, std::ceil(distance / step_size)));
  for (int32_t step = 0; step <= num_steps; step++) {
    const double ratio =
        static_cast<double>(step) / static_cast<double>(num_steps);
    const Eigen::VectorXd qinterp =
        collision_checker.InterpolateBetweenConfigurations(start, end, ratio);
    const bool collision_free = (model_context != nullptr)
        ? collision_checker.CheckContextConfigCollisionFree(
            model_context, qinterp)
        : collision_checker.CheckConfigCollisionFree(qinterp);
    if (!collision_free) {
      return ConstrainedEdgeCheckResult::EdgeInCollision;
    } else {
      if (!CheckConstraintsSatisfied(constraints, qinterp, tolerance)) {
        return ConstrainedEdgeCheckResult::EdgeViolatesConstraints;
      }
    }
  }
  return ConstrainedEdgeCheckResult::EdgeValid;
}

std::vector<Eigen::VectorXd> PropagateConstrained(
    const Eigen::VectorXd& start, const Eigen::VectorXd& target,
    const CollisionChecker& collision_checker,
    const std::vector<std::shared_ptr<drake::solvers::Constraint>>&
        constraints,
    const JointLimits& joint_limits, const double tolerance,
    const double projection_step_size,
    const double minimum_propagation_progress,
    std::map<std::string, double>* propagation_statistics,
    CollisionCheckerContext* model_context,
    const std::string& log_message_prefix) {
  DRAKE_THROW_UNLESS(tolerance >= 0.0);
  DRAKE_THROW_UNLESS(projection_step_size > 0.0);
  DRAKE_THROW_UNLESS(minimum_propagation_progress >= 0.0);
  DRAKE_THROW_UNLESS(projection_step_size > minimum_propagation_progress);
  DRAKE_THROW_UNLESS(propagation_statistics != nullptr);

  // Ensure that propagation_statistics contains the right keys. If additional
  // tracking statistics are added below, they must be added here as well. Note
  // that try_emplace only adds a new element if it does not already exist.
  propagation_statistics->try_emplace("targets_considered", 0.0);
  propagation_statistics->try_emplace("projected_targets", 0.0);
  propagation_statistics->try_emplace("projected_target_in_collision", 0.0);
  propagation_statistics->try_emplace(
      "projected_target_insufficient_progress", 0.0);
  propagation_statistics->try_emplace("target_failed_to_project", 0.0);
  propagation_statistics->try_emplace("edges_considered", 0.0);
  propagation_statistics->try_emplace("valid_edges", 0.0);
  propagation_statistics->try_emplace("edges_in_collision", 0.0);
  propagation_statistics->try_emplace("edges_violated_constraints", 0.0);
  propagation_statistics->try_emplace("complete_propagation_successful", 0.0);

  const std::string real_log_message_prefix =
      (log_message_prefix.size() > 0) ? "[" + log_message_prefix + "] " : "";

  std::vector<Eigen::VectorXd> propagated_configs;

  // Compute a maximum number of steps to take.
  const double total_distance =
      collision_checker.ComputeConfigurationDistance(start, target);
  const int32_t total_steps =
      static_cast<int32_t>(
          std::ceil(total_distance / projection_step_size));
  Eigen::VectorXd current = start;
  int32_t steps = 0;
  bool complete_propagation_successful = false;
  while (steps < total_steps) {
    // Compute the next intermediate target state.
    Eigen::VectorXd current_target = target;
    const double target_distance =
        collision_checker.ComputeConfigurationDistance(current, current_target);

    if (std::abs(target_distance) <= std::numeric_limits<double>::epsilon()) {
      // If we've reached the target state, stop.
      complete_propagation_successful = true;
      break;
    } else if (target_distance > projection_step_size) {
      // If we have more than one stop left, interpolate a target state.
      const double step_fraction = projection_step_size / target_distance;
      const Eigen::VectorXd interpolated_target =
          collision_checker.InterpolateBetweenConfigurations(
              current, target, step_fraction);
      current_target = interpolated_target;
    }

    // Try projecting target to meet constraints.
    (*propagation_statistics)["targets_considered"] += 1.0;
    const auto projected =
        ProjectToConstraints(collision_checker.plant(), joint_limits,
                             constraints, tolerance, current_target);

    if (!projected) {
      drake::log()->trace(
          "{}Constrained propagation stopped because current_target failed to "
          "project to meet constraints",
          real_log_message_prefix);
      (*propagation_statistics)["target_failed_to_project"] += 1.0;
      break;
    }

    (*propagation_statistics)["projected_targets"] += 1.0;
    current_target = projected.value();

    const bool projection_collision_free = (model_context != nullptr)
        ? collision_checker.CheckContextConfigCollisionFree(
            model_context, current_target)
        : collision_checker.CheckConfigCollisionFree(current_target);

    if (!projection_collision_free) {
      drake::log()->trace(
          "{}Constrained propagation stopped due to projected target in "
          "collision (pre edge check)",
          real_log_message_prefix);
      (*propagation_statistics)["projected_target_in_collision"] += 1.0;
      break;
    }

    const double propagation_progress =
        collision_checker.ComputeConfigurationDistance(
            current, projected.value());

    if (propagation_progress < minimum_propagation_progress) {
      drake::log()->trace(
          "{}Constrained propagation stopped due to projected target not "
          "making enough forward progress (pre edge check)",
          real_log_message_prefix);
      (*propagation_statistics)["projected_target_stalled"] += 1.0;
      break;
    }

    (*propagation_statistics)["edges_considered"] += 1.0;

    const ConstrainedEdgeCheckResult edge_check_result =
        CheckEdgeCollisionFreeAndSatisfiesConstraints(
            current, current_target, collision_checker, constraints, tolerance,
            model_context);

    switch (edge_check_result) {
      case ConstrainedEdgeCheckResult::EdgeValid: {
        propagated_configs.push_back(current_target);
        current = current_target;
        steps++;
        // If this is the last step, record that it was successful.
        if (steps == total_steps) {
          complete_propagation_successful = true;
        }
        (*propagation_statistics)["valid_edges"] += 1.0;
        break;
      }
      case ConstrainedEdgeCheckResult::EdgeInCollision: {
        drake::log()->trace(
            "{}Constrained propagation stopped due to edge in collision",
            real_log_message_prefix);
        (*propagation_statistics)["edges_in_collision"] += 1.0;
        break;
      }
      case ConstrainedEdgeCheckResult::EdgeViolatesConstraints: {
        drake::log()->trace(
            "{}Constrained propagation stopped due to edge violating "
            "constraints",
            real_log_message_prefix);
        (*propagation_statistics)["edges_violated_constraints"] += 1.0;
        break;
      }
    }

    if (edge_check_result != ConstrainedEdgeCheckResult::EdgeValid) {
      break;
    }
  }

  if (complete_propagation_successful) {
    (*propagation_statistics)["complete_propagation_successful"] += 1.0;
  }
  return propagated_configs;
}

std::optional<Eigen::VectorXd> ProjectToConstraints(
    const drake::multibody::MultibodyPlant<double>& plant,
    const JointLimits& joint_limits,
    const std::vector<std::shared_ptr<drake::solvers::Constraint>>&
        constraints,
    const double tolerance, const Eigen::VectorXd& q) {
  drake::solvers::MathematicalProgram projection;
  // Make joint limits constraint + quaternion constraint if needed.
  auto q_var = projection.NewContinuousVariables(
      joint_limits.num_positions(), "q");
  projection.AddBoundingBoxConstraint(
      joint_limits.position_lower(), joint_limits.position_upper(), q_var);
  drake::multibody::AddUnitQuaternionConstraintOnPlant(
      plant, q_var, &projection);
  // Add a cost to the difference between q and projection.
  projection.AddQuadraticCost((q - q_var).squaredNorm());
  // Add the user-provided constraints.
  for (const auto& constraint : constraints) {
    projection.AddConstraint(constraint, q_var);
  }
  // Set solver options.
  // TODO(calderpg) Should these tolerance values be the same as tolerance
  // parameter?
  projection.SetSolverOption(drake::solvers::SnoptSolver::id(),
                             "Major optimality tolerance", 1e-5);
  projection.SetSolverOption(drake::solvers::SnoptSolver::id(),
                             "Major feasibility tolerance", 1e-5);
  // Solve projection.
  const auto result = drake::solvers::Solve(projection, q);
  const Eigen::VectorXd& q_projected = result.GetSolution();
  if (result.is_success() ||
      CheckConstraintsSatisfied(constraints, q_projected, tolerance)) {
    drake::log()->trace("Projected {} to {}", q, q_projected);
    return std::optional<Eigen::VectorXd>(q_projected);
  } else {
    drake::log()->trace("Failed to project");
    return std::nullopt;
  }
}

Eigen::VectorXd SampleConstrained(
    const drake::multibody::MultibodyPlant<double>& plant,
    const JointLimits& joint_limits,
    const std::vector<std::shared_ptr<drake::solvers::Constraint>>&
        constraints,
    const double tolerance, const int32_t max_sample_iterations,
    const std::function<Eigen::VectorXd(void)>& sampling_fn) {
  DRAKE_THROW_UNLESS(tolerance >= 0.0);
  DRAKE_THROW_UNLESS(max_sample_iterations > 0);
  DRAKE_THROW_UNLESS(sampling_fn != nullptr);

  int32_t iterations = 0;
  while (iterations < max_sample_iterations) {
    iterations++;
    const Eigen::VectorXd q_sample = sampling_fn();
    drake::log()->trace(
        "SampleConstrained iteration {} trying to project {} to constraints",
        iterations, common_robotics_utilities::print::Print(q_sample));
    const auto projection = ProjectToConstraints(
        plant, joint_limits, constraints, tolerance, q_sample);
    if (projection) {
      const Eigen::VectorXd& q_projected = projection.value();
      drake::log()->trace(
          "SampleConstrained iteration {} projected {} to constraints {}",
          iterations, common_robotics_utilities::print::Print(q_sample),
          common_robotics_utilities::print::Print(q_projected));
      return q_projected;
    } else {
      drake::log()->trace(
        "SampleConstrained iteration {} failed to project {} to constraints",
        iterations, common_robotics_utilities::print::Print(q_sample));
    }
  }
  throw std::runtime_error(fmt::format(
      "Failed to sample and project in max {} iterations",
      max_sample_iterations));
}

bool CheckPathCollisionFreeAndSatisfiesConstraints(
    const std::vector<Eigen::VectorXd>& path,
    const CollisionChecker& collision_checker,
    const std::vector<std::shared_ptr<drake::solvers::Constraint>>&
        constraints,
    const double tolerance) {
  if (path.size() >= 2) {
    for (size_t idx = 1; idx < path.size(); idx++) {
      const Eigen::VectorXd& prev = path.at(idx - 1);
      const Eigen::VectorXd& current = path.at(idx);
      if (CheckEdgeCollisionFreeAndSatisfiesConstraints(
              prev, current, collision_checker, constraints, tolerance)
          != ConstrainedEdgeCheckResult::EdgeValid) {
        drake::log()->info(
            "Edge from {} [{}] to {} [{}] in collision or violates constraints",
            prev.transpose(), idx - 1, current.transpose(), idx);
        return false;
      }
    }
    return true;
  } else if (path.size() == 1) {
    return
        (collision_checker.CheckConfigCollisionFree(path.at(0))
         && CheckConstraintsSatisfied(constraints, path.at(0), tolerance));
  } else {
    throw std::runtime_error("Cannot check empty path");
  }
}
}  // namespace planning
}  // namespace drake 
