#pragma once

#include <chrono>
#include <functional>
#include <unordered_map>
#include <vector>

#include <Eigen/Geometry>
#include <common_robotics_utilities/voxel_grid.hpp>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "planning/sphere_robot_model_collision_checker.h"
#include <voxelized_geometry_tools/collision_map.hpp>
#include <voxelized_geometry_tools/tagged_object_collision_map.hpp>

#include "drake/common/drake_throw.h"
#include "drake/common/text_logging.h"
#include "drake/multibody/plant/multibody_plant.h"

namespace drake {
namespace planning {
/// Self-filter implementation for CollisionMap environments.
/// Self-filter marks voxels belonging to the robot as empty so that they do not
/// produce false collisions in a voxelized environment used for collision
/// checking.
/// @param collision_checker Sphere-model collision checker that provides the
/// sphere model of robot geometry and performs forward kinematics.
/// @param q Current configuration of the robot.
/// @param inflation_ratio Ratio to imflate the spheres of the collision model
/// to use in the self-filter.
/// @param collision_map Current environment.
void SelfFilter(const SphereRobotModelCollisionChecker& collision_checker,
                const Eigen::VectorXd& q, double inflation_ratio,
                drake::multibody::BodyIndex grid_body_index,
                voxelized_geometry_tools::CollisionMap* const collision_map);

/// Self-filter implementation for TaggedObjectCollisionMap environments.
/// Self-filter marks voxels belonging to the robot as empty so that they do not
/// produce false collisions in a voxelized environment used for collision
/// checking.
/// @param collision_checker Sphere-model collision checker that provides the
/// sphere model of robot geometry and performs forward kinematics.
/// @param q Current configuration of the robot.
/// @param inflation_ratio Ratio to imflate the spheres of the collision model
/// to use in the self-filter.
/// @param collision_map Current environment.
void SelfFilter(
    const SphereRobotModelCollisionChecker& collision_checker,
    const Eigen::VectorXd& q, double inflation_ratio,
    drake::multibody::BodyIndex grid_body_index,
    voxelized_geometry_tools::TaggedObjectCollisionMap* const collision_map);

/// Self-filter implementation for generic voxelized environments.
/// Self-filter marks voxels belonging to the robot as empty so that they do not
/// produce false collisions in a voxelized environment used for collision
/// checking.
/// @param collision_checker Sphere-model collision checker that provides the
/// sphere model of robot geometry and performs forward kinematics.
/// @param q Current configuration of the robot.
/// @param inflation_ratio Ratio to imflate the spheres of the collision model
/// to use in the self-filter.
/// @param environment Voxelized environment.
/// @param set_cell_empty_fn Function to set the provided cell of the
/// environment to "empty".
template <typename T, typename BackingStore = std::vector<T>>
void SelfFilter(
    const SphereRobotModelCollisionChecker& collision_checker,
    const Eigen::VectorXd& q, double inflation_ratio,
    drake::multibody::BodyIndex grid_body_index,
    common_robotics_utilities::voxel_grid::VoxelGridBase<T, BackingStore>* const
        environment,
    const std::function<void(T&)>& set_cell_empty_fn) {
  const auto start_time = std::chrono::steady_clock::now();
  DRAKE_THROW_UNLESS(inflation_ratio >= 1.0);
  DRAKE_THROW_UNLESS(environment != nullptr);
  DRAKE_THROW_UNLESS(environment->IsInitialized());
  DRAKE_THROW_UNLESS(environment->HasUniformCellSize());

  // Add check padding equal to the cell center->cell corner distance.
  const double cell_size = environment->GetCellSizes().x();
  const double check_padding = cell_size * 0.5 * std::sqrt(3.0);

  // Get the self-filter spheres.
  const std::vector<Eigen::Isometry3d> X_WB_set =
      collision_checker.ComputeBodyPoses(q);
  const std::vector<
      std::unordered_map<drake::geometry::GeometryId, SphereSpecification>>
      spheres_in_world_frame =
          collision_checker.ComputeSphereLocationsInWorldFrame(X_WB_set);

  // Get the pose of the grid body in world.
  const Eigen::Isometry3d X_BW = X_WB_set.at(grid_body_index).inverse();

  // Transform the self-filter spheres into grid body frame.
  std::vector<SphereSpecification> filter_spheres;
  for (const auto& body_spheres : spheres_in_world_frame) {
    for (const auto& [sphere_id, sphere] : body_spheres) {
      drake::unused(sphere_id);
      const Eigen::Vector4d& p_WSo = sphere.Origin();
      const Eigen::Vector4d p_BSo = X_BW * p_WSo;
      const double radius = sphere.Radius();
      const double filter_radius = radius * inflation_ratio;
      const SphereSpecification filter_sphere(p_BSo, filter_radius);
      filter_spheres.push_back(filter_sphere);
    }
  }

  // Loop through environment
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int64_t data_index = 0; data_index < environment->GetTotalCells();
       data_index++) {
    const common_robotics_utilities::voxel_grid::GridIndex index =
        environment->DataIndexToGridIndex(data_index);
    const Eigen::Vector4d p_BCo = environment->GridIndexToLocation(index);
    for (const SphereSpecification& filter_sphere : filter_spheres) {
      const Eigen::Vector4d p_BSo = filter_sphere.Origin();
      const double squared_distance = (p_BSo - p_BCo).squaredNorm();
      const double squared_threshold =
          std::pow(filter_sphere.Radius() + check_padding, 2.0);
      if (squared_distance <= squared_threshold) {
        set_cell_empty_fn(environment->GetDataIndexMutable(data_index));
        break;
      }
    }
  }

  const auto end_time = std::chrono::steady_clock::now();
  drake::log()->info(
      "Self-filter took {} seconds",
      std::chrono::duration<double>(end_time - start_time).count());
}
}  // namespace planning
}  // namespace drake
