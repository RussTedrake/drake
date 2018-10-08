#pragma once

#include <memory>

#include "drake/lcm/drake_lcm.h"
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
///   @input_port{iiwa_feedforward_torque},
///   @output_port{iiwa_position_commanded}
///   @output_port{iiwa_position_measured}
///   @output_port{iiwa_velocity_estimated}
///   @output_port{iiwa_torque_commanded}
///   @output_port{iiwa_torque_measured}
///   @output_port{iiwa_torque_external} }
///
/// @ingroup manipulation_station_systems
/// @}
///
/// @param lcm A pointer to the LCM subsystem to use, which must
/// remain valid for the lifetime of this object. If null, a
/// drake::lcm::DrakeLcm object is allocated and maintained internally, but
/// You should generally provide an LCM interface yourself, since there
/// should normally be just one of these typically-heavyweight objects per
/// program.
std::unique_ptr<systems::Diagram<double>>
MakeManipulationStationHardwareInterface(lcm::DrakeLcmInterface* lcm = nullptr);

}  // namespace manipulation_station
}  // namespace examples
}  // namespace drake

