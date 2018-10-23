/** @file
 Provides a set of functions to facilitate visualization of the sensor outputs.
 */

#pragma once

#include "drake/lcm/drake_lcm_interface.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/output_port.h"
#include "drake/systems/lcm/lcm_publisher_system.h"

namespace drake {
namespace systems {
namespace sensors {

/// Note: Currently only supports one camera.  See #9767.
/// Sets the system to publish periodically at 10Hz.  Use
/// `set_publish_period` on the returned system to change this.
systems::lcm::LcmPublisherSystem* ConnectRgbdCameraToDrakeVisualizer(
    DiagramBuilder<double>* builder,
    const systems::OutputPort<double>& color_image_port,
    const OutputPort<double>& depth_image_port,
    const OutputPort<double>& label_image_port,
    drake::lcm::DrakeLcmInterface* lcm = nullptr);

// TODO(russt/SeanCurtis-TRI): Also add a version:
//   ConnectDrakeVisualizer(DiagramBuilder*, RgbdCamera&, lcm)
// (maybe dev::RgbdCamera, too?) that dispatches to the method above.

}  // namespace sensors
}  // namespace systems
}  // namespace drake
