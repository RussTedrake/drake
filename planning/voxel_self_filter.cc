#include "planning/voxel_self_filter.h"

#include <vector>

#include "planning/sphere_robot_model_collision_checker.h"
#include <Eigen/Geometry>
#include <common_robotics_utilities/voxel_grid.hpp>

namespace drake {
namespace planning {
void SelfFilter(const SphereRobotModelCollisionChecker& collision_checker,
                const Eigen::VectorXd& q, const double inflation_ratio,
                const drake::multibody::BodyIndex grid_body_index,
                voxelized_geometry_tools::CollisionMap* const collision_map) {
  DRAKE_THROW_UNLESS(collision_map != nullptr);
  const std::function<void(voxelized_geometry_tools::CollisionCell&)>
      set_empty_fn = [](voxelized_geometry_tools::CollisionCell& cell) {
        cell.Occupancy() = 0.0f;
        cell.Component() = 0u;
      };
  return SelfFilter(collision_checker, q, inflation_ratio, grid_body_index,
                    collision_map, set_empty_fn);
}

void SelfFilter(
    const SphereRobotModelCollisionChecker& collision_checker,
    const Eigen::VectorXd& q, const double inflation_ratio,
    drake::multibody::BodyIndex grid_body_index,
    voxelized_geometry_tools::TaggedObjectCollisionMap* const collision_map) {
  DRAKE_THROW_UNLESS(collision_map != nullptr);
  const std::function<void(
      voxelized_geometry_tools::TaggedObjectCollisionCell&)>
      set_empty_fn =
          [](voxelized_geometry_tools::TaggedObjectCollisionCell& cell) {
            cell.Occupancy() = 0.0f;
            cell.ObjectId() = 0u;
            cell.Component() = 0u;
            cell.SpatialSegment() = 0u;
          };
  return SelfFilter(collision_checker, q, inflation_ratio, grid_body_index,
                    collision_map, set_empty_fn);
}
}  // namespace planning
}  // namespace drake
