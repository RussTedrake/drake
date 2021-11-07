#pragma once

#include <Eigen/Dense>

#include "drake/planning/path_planning.h"

namespace drake {
namespace planning {

/// Parameters to bidirectional RRT planner.
struct BiRRTPlannerParameters {
  /// Probability that the next "sample" is drawn from the target tree instead
  /// of from the sampler.
  double tree_sampling_bias{0.0};

  /// Probability that after each extend/connect operation, the planner should
  /// switch active/target trees.
  double p_switch_trees{0.0};

  /// Planning time limit, in seconds.
  double time_limit{0.0};

  /// Distance threshold to check if two states are the same (i.e. if the goal
  /// has been reached).
  double goal_tolerance{0.0};

  /// Size of a single extend step.
  double rrt_step_size{0.0};

  /// Select whether to use extend or connect in the RRT.
  bool use_connect{false};
};

/// Unconstrained kinematic planning using BiRRT planner.

/// Plans a path using a bidirectional RRT from start to goal.
/// @param start Start configuration.
/// @param goal Goal configuration.
/// @param plant A multibody::MultibodyPlant, which must be connected (in a
///              systems::Diagram) to a geometry::SceneGraph.
/// @param sampler A StateSampler that produces query points.
/// @param ignored_bodies Bodies to ignore in collision checks.
/// @param parameters Parameters to the planner.
/// @return PlanningResult result of planning.
PlanningResult PlanBiRRTPath(
    const Eigen::VectorXd& start, const Eigen::VectorXd& goal,
    const multibody::MultibodyPlant& plant,
    const StateSampler& sampler,
    const std::unordered_set<multibody::BodyIndex>& ignored_bodies,
    const BiRRTPlannerParameters& parameters);

}  // namespace planning
}  // namespace drake
