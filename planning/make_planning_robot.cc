#include "planning/make_planning_robot.h"

#include <limits>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "drake/planning/collision_checker.h"

using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;
using drake::planning::ConfigurationDistanceFunction;

namespace drake {
namespace planning {
ConfigurationDistanceFunction MakeWeightedConfigurationDistanceFunction(
    const drake::multibody::MultibodyPlant<double>& plant,
    const std::vector<drake::multibody::ModelInstanceIndex>&
        robot_model_instances,
    const std::map<std::string, double>& named_joint_distance_weights,
    const double default_joint_distance_weight) {
  DRAKE_THROW_UNLESS(default_joint_distance_weight >= 0.0);
  Eigen::VectorXd joint_distance_weights =
      Eigen::VectorXd::Ones(plant.num_positions()) *
      default_joint_distance_weight;

  // Collect all possible joints that are part of the robot model.
  std::vector<drake::multibody::JointIndex> joints;
  for (const auto& robot_model_instance : robot_model_instances) {
    auto instance_joints = plant.GetJointIndices(robot_model_instance);
    joints.insert(joints.end(), instance_joints.begin(), instance_joints.end());
  }

  // Go through the model and set joint distance weights accordingly.
  for (const drake::multibody::JointIndex& idx : joints) {
    const drake::multibody::Joint<double>& joint = plant.get_joint(idx);
    if (joint.num_positions() > 0) {
      DRAKE_THROW_UNLESS(joint.num_positions() == 1);
      // TODO(calderpg) Find a solution that incorporates model instances.
      // Note: this ignores model instances, so two joints with the same name
      // but in different model instances will receive the same weight. This
      // also assumes that joint names are usefully unique; this is not enforced
      // by drake::multibody::Joint, but is reasonably true for any sane model.
      const auto found_itr = named_joint_distance_weights.find(joint.name());
      if (found_itr != named_joint_distance_weights.end()) {
        const double joint_distance_weight = found_itr->second;
        DRAKE_THROW_UNLESS(joint_distance_weight >= 0.0);
        joint_distance_weights(joint.position_start()) = joint_distance_weight;
        drake::log()->debug(
            "Set joint {} [{}] distance weight to non-default {}", idx,
            joint.name(), joint_distance_weight);
      }
    }
  }

  const ConfigurationDistanceFunction cspace_distance_fn =
      [joint_distance_weights](const Eigen::VectorXd& q1,
                               const Eigen::VectorXd& q2) {
        return (q1 - q2).cwiseProduct(joint_distance_weights).norm();
      };

  return cspace_distance_fn;
}
}  // namespace planning
}  // namespace drake
