#pragma once

#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/simple_astar_search.hpp>

#include "drake/planning/collision_checker.h"
#include "drake/solvers/solve.h"
#include "planning/joint_limits.h"
#include "planning/valid_starts_and_goals.h"

// TODO(calderpg) Move/remove functions here made surplus by the new
// PlanningSpace API. Find a renamed home for constraint checking helpers.
namespace drake {
namespace planning {
/// Constrained sampling & projection.

/// Project the provided configuration to meet constraints.
/// @param plant MultibodyPlant model.
/// @param joint_limits Joint limits to apply to constraint projection.
/// @param constraints User-provided constraints.
/// @param tolerance Tolerance to use checking constraints.
/// @param q Configuration to project.
/// @return Projected configuration or nullopt if projection unsuccessful.
std::optional<Eigen::VectorXd> ProjectToConstraints(
    const drake::multibody::MultibodyPlant<double>& plant,
    const JointLimits& joint_limits,
    const std::vector<std::shared_ptr<drake::solvers::Constraint>>&
        constraints,
    double tolerance, const Eigen::VectorXd& q);

/// Sample a configuration that satisfies constraints.
/// @param plant MultibodyPlant model.
/// @param joint_limits Joint limits to apply to constraint projection.
/// @param constraints User-provided constraints.
/// @param tolerance Tolerance to use checking constraints.
/// @param max_sample_iterations Max number of sampling attempts.
/// @param sampling_fn Configuration sampling function.
/// @return Sampled configuration or throws if max sampling attempts exceeded.
Eigen::VectorXd SampleConstrained(
    const drake::multibody::MultibodyPlant<double>& plant,
    const JointLimits& joint_limits,
    const std::vector<std::shared_ptr<drake::solvers::Constraint>>&
        constraints,
    double tolerance, int32_t max_sample_iterations,
    const std::function<Eigen::VectorXd(void)>& sampling_fn);

/// Checks if the provided configuration satisfies constraints.
/// @param constraints User-provided constraints.
/// @param q Configuration to check.
/// @param tolerance Tolerance to use checking constraints.
/// @return If the provided sample satisfies constraints.
bool CheckConstraintsSatisfied(
    const std::vector<std::shared_ptr<drake::solvers::Constraint>>& constraints,
    const Eigen::VectorXd& q, double tolerance,
    spdlog::level::level_enum logging_level = spdlog::level::debug);

/// Checks if the provided configuration satisfies constraint.
/// @param constraint User-provided constraint.
/// @param q Configuration to check.
/// @param tolerance Tolerance to use checking constraints.
/// @param log_function (Optional) Logging function.
/// @return If the provided sample satisfies constraint.
bool CheckConstraintSatisfied(
    const drake::solvers::Constraint& constraint, const Eigen::VectorXd& q,
    double tolerance,
    const std::function<void(const std::string&)>& log_function = {});

/// Different outcomes for checking if an edge is collision-free and meets
/// constraints. Note that an edge may be *both* in collision and violate
/// constraints, but this will only be reported as EdgeInCollision.
enum class ConstrainedEdgeCheckResult : uint8_t {
  EdgeValid = 0,
  EdgeInCollision = 1,
  EdgeViolatesConstraints = 2
};

/// Checks that the provided edge is collision-free and meets constraints.
/// @param collision_checker Collision checker.
/// @param start Start configuration.
/// @param end End configuration.
/// @param constraints User-provided constraints.
/// @param tolerance Tolerance for constraint checking.
/// @param model_context Optional explict collision checking context to use.
ConstrainedEdgeCheckResult CheckEdgeCollisionFreeAndSatisfiesConstraints(
    const Eigen::VectorXd& start, const Eigen::VectorXd& end,
    const drake::planning::CollisionChecker& collision_checker,
    const std::vector<std::shared_ptr<drake::solvers::Constraint>>& constraints,
    double tolerance,
    drake::planning::CollisionCheckerContext* model_context = nullptr);

/// TODO(calderpg) Consider additional documentation/explanation of propagation
/// strategy before wider public release.
/// Performs kinematic collision-free forward propagation subject to constrains.
/// Serves as the core of constrained BiRRT forward propagation function, in
/// which successive intermediate states between `start` and `target` are
/// projected to meet constraints, and the edge between the previous
/// intermediate state and next intermediate state is checked to be
/// collision-free and satisfying constraints. Terminates when either `target`
/// is reached, an intermediate state fails to project to meet constraints, or
/// an edge between previous intermediate state and next intermediate state is
/// not collision-free or constraint-satisfying.
/// @param start Start configuration. @pre satisfies constraints.
/// @param target Target configuration. Does not need to satisfy constraints.
/// @param collision_checker Collision checker.
/// @param constraints User-provided constraints.
/// @param joint_limits Joint limits to apply during constraint projection.
/// @param tolerance Tolerance for constraint checking.
/// @param projection_step_size Nominal step size between `start` and `target`
/// to control how many intermediate states are projected to meet constraints.
/// @param minimum_propagation_progress Minimum progress for propagation. If a
/// propagated state is less than this distance from the previous intermediate
/// state, propagation terminates.
/// @param propagation_statistics Statistics about propagation.
/// @param model_context Optional explict collision checking context to use.
/// @param log_message_prefix Optional prefix to log messages. Used by parallel
/// planner so that log messages from separate worker threads may be
/// distinguished.
/// @return Vector of propagated collision-free and constraint-satisfying
/// intermediate states.
std::vector<Eigen::VectorXd> PropagateConstrained(
    const Eigen::VectorXd& start, const Eigen::VectorXd& target,
    const drake::planning::CollisionChecker& collision_checker,
    const std::vector<std::shared_ptr<drake::solvers::Constraint>>& constraints,
    const JointLimits& joint_limits, double tolerance,
    double projection_step_size, double minimum_propagation_progress,
    std::map<std::string, double>* propagation_statistics,
    drake::planning::CollisionCheckerContext* model_context = nullptr,
    const std::string& log_message_prefix = "");

/// Checks that the provided path is collision-free and meets constraints.
/// @param path Path to smooth.
/// @param collision_checker Collision checker.
/// @param constraints User-provided constraints.
/// @param tolerance Tolerance for constraint checking.
bool CheckPathCollisionFreeAndSatisfiesConstraints(
    const std::vector<Eigen::VectorXd>& path,
    const drake::planning::CollisionChecker& collision_checker,
    const std::vector<std::shared_ptr<drake::solvers::Constraint>>&
        constraints,
    double tolerance);
}  // namespace planning
}  // namespace drake
