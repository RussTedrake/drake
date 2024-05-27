#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "drake/planning/collision_checker.h"
namespace drake {
namespace planning {
/// TODO(calderpg) When this is restructured as part of //planning->Drake,
/// expose the creation of `joint_distance_weights` for testing, especially in
/// cases with multiple robot model instances.
/// Makes a weighted configuration distance function of the form
/// distance = (q1 - q2).cwiseProduct(joint_distance_weights).norm().
drake::planning::ConfigurationDistanceFunction
MakeWeightedConfigurationDistanceFunction(
    const drake::multibody::MultibodyPlant<double>& plant,
    const std::vector<drake::multibody::ModelInstanceIndex>&
        robot_model_instances,
    const std::map<std::string, double>& named_joint_distance_weights,
    double default_joint_distance_weight = 1.0);
}  // namespace planning
}  // namespace drake
