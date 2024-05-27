#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "planning/robot_diagram.h"
#include "planning/sphere_robot_model_collision_checker.h"
#include <Eigen/Geometry>

#include "drake/geometry/scene_graph.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/planning/collision_checker.h"
#include "drake/planning/collision_checker_params.h"

namespace drake {
namespace planning {

/// Sphere-model robot collision checker using MbP/SG to model environment
/// geometry.
class MbpEnvironmentCollisionChecker final
    : public SphereRobotModelCollisionChecker {
 public:
  /** @name     Does not allow copy, move, or assignment. */
  /** @{ */
  // N.B. The copy constructor is protected for use in implementing Clone().
  void operator=(const MbpEnvironmentCollisionChecker&) = delete;
  /** @} */

  /// Construct a collision checker using MbP/SG to model environment geometry.
  explicit MbpEnvironmentCollisionChecker(
      drake::planning::CollisionCheckerParams params);

  /// Query the (distance, gradient) of the provided point from obstacles.
  /// @param context Context of the MbP model.
  /// @param query_object Query object for `context`.
  /// @param p_WQ Query position in world frame W.
  /// @param query_radius Gradients do not need to be computed for queries
  /// with distance > query_radius. This parameter is needed because the
  /// default implementation calls ComputePointSignedDistanceAndGradient, and
  /// only needing to check within a bound can improve performance.
  /// @param X_WB_set Poses X_WB for all bodies in the model. Unused.
  /// @param X_WB_inverse_set Poses X_BW for all bodies in the model. Unused.
  /// @return pair<signed distance, gradient> where signed distance is positive
  /// if @param p_WQ is outside of objects, and negative if it is inside. The
  /// gradient is ∂d/∂p.
  PointSignedDistanceAndGradientResult
  ComputePointToEnvironmentSignedDistanceAndGradient(
      const drake::systems::Context<double>& context,
      const drake::geometry::QueryObject<double>& query_object,
      const Eigen::Vector4d& p_WQ, double query_radius,
      const std::vector<Eigen::Isometry3d>& X_WB_set,
      const std::vector<Eigen::Isometry3d>& X_WB_inverse_set) const override;

 protected:
  /// To support Clone(), allow copying (but not move nor assign).
  explicit MbpEnvironmentCollisionChecker(
      const MbpEnvironmentCollisionChecker&);

 private:
  std::unique_ptr<drake::planning::CollisionChecker> DoClone() const override;

  std::optional<drake::geometry::GeometryId> AddEnvironmentCollisionShapeToBody(
      const std::string& group_name,
      const drake::multibody::Body<double>& bodyA,
      const drake::geometry::Shape& shape,
      const drake::math::RigidTransform<double>& X_AG) override;

  void RemoveAllAddedEnvironment(
      const std::vector<drake::planning::CollisionChecker::AddedShape>& shapes)
      override;
};
}  // namespace planning
}  // namespace drake
