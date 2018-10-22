#pragma once

#include <memory>

#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/multibody_tree/multibody_plant/multibody_plant.h"
#include "drake/systems/framework/diagram.h"

namespace drake {
namespace examples {
namespace manipulation_station {

/// Constructs a System that connects via message-passing to the real
/// manipulation station.
/// @{
///
/// @system{ ManipulationStationHardwareInterface,
///   @input_port{iiwa_position}
///   @input_port{iiwa_feedforward_torque}
///   @input_port{wsg_position}
///   @input_porT{wsg_force_limit},
///   @output_port{iiwa_position_commanded}
///   @output_port{iiwa_position_measured}
///   @output_port{iiwa_velocity_estimated}
///   @output_port{iiwa_torque_commanded}
///   @output_port{iiwa_torque_measured}
///   @output_port{iiwa_torque_external} }
///
/// The `geometry_pose` output is available if a
/// SceneGraph is passed in to the constructor; it passes the measured /
/// estimated pose of the arm/hand (only) to the SceneGraph.
///
/// @ingroup manipulation_station_systems
/// @}
///
class ManipulationStationHardwareInterface : public systems::Diagram<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ManipulationStationHardwareInterface)

  /// @param lcm A pointer to the LCM subsystem to use, which must
  ///   remain valid for the lifetime of this object. If null, a
  ///   drake::lcm::DrakeLcm object is allocated and maintained internally,
  ///   but you should generally provide an LCM interface yourself, since
  ///   there should normally be just one of these typically-heavyweight
  ///   objects per program.
  explicit ManipulationStationHardwareInterface(
      lcm::DrakeLcmInterface* lcm = nullptr);

  /// @param scene_graph Registers the known geometries with the @p
  ///   scene_graph, and creates a `geometry_pose` output port.  Only the
  ///   station table (without cupboard), the IIWA arm, and the Schunk gripper
  ///   are registered by default.
  //  void RegisterGeometry(geometry::SceneGraph* scene_graph);

  /// TODO(russt): Consider adding AddCupboard(), but this time return the
  /// SourceId for the SceneGraph, because a user-supplied estimator will be
  /// necessary to publish the pose.

  /// For parity with ManipulationStation, we maintain a MultibodyPlant of
  /// the IIWA arm, with the lumped-mass equivalent spatial inertia of the
  /// Schunk WSG gripper.
  // TODO(russt): Actually add the equivalent mass of the WSG.
  const multibody::multibody_plant::MultibodyPlant<double>&
  get_controller_plant() const {
    return *owned_controller_plant_;
  }

 private:
  std::unique_ptr<multibody::multibody_plant::MultibodyPlant<double>>
      owned_controller_plant_;
};

}  // namespace manipulation_station
}  // namespace examples
}  // namespace drake
