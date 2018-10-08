#include "drake/examples/manipulation_station/manipulation_station_hardware_interface.h"

#include "drake/examples/kuka_iiwa_arm/iiwa_lcm.h"
#include "drake/lcmt_iiwa_command.hpp"
#include "drake/lcmt_iiwa_status.hpp"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"

namespace drake {
namespace examples {
namespace manipulation_station {

// TODO(russt): Set publishing defaults.

std::unique_ptr<systems::Diagram<double>>
MakeManipulationStationHardwareInterface(lcm::DrakeLcmInterface* lcm_optional) {
  systems::DiagramBuilder<double> builder;

  // Publish IIWA command.
  auto iiwa_command_sender = builder
      .AddSystem<examples::kuka_iiwa_arm::IiwaCommandSender>();
  auto iiwa_command_publisher = builder.AddSystem
      (systems::lcm::LcmPublisherSystem::Make<drake::lcmt_iiwa_command>
           ("IIWA_COMMAND", lcm_optional));
  builder.ExportInput(iiwa_command_sender->get_position_input_port(),
                      "iiwa_position");
  builder.ExportInput(iiwa_command_sender->get_torque_input_port(),
                      "iiwa_feedforward_torque");
  builder.Connect(iiwa_command_sender->get_output_port(0),
                  iiwa_command_publisher->get_input_port());

  // The iiwa_command_publisher may have created the LCM object, or may just
  // be passing back a reference.  In either case, use the same reference for
  // all of the LCM systems constructed/added here.
  lcm::DrakeLcmInterface& lcm = iiwa_command_publisher->lcm();

  // Receive IIWA status and populate the output ports.
  auto iiwa_status_receiver = builder
      .AddSystem<examples::kuka_iiwa_arm::IiwaStatusReceiver>();
  auto iiwa_status_subscriber = builder.AddSystem
      (systems::lcm::LcmSubscriberSystem::Make<drake::lcmt_iiwa_status>
           ("IIWA_STATUS", &lcm));
  builder.ExportOutput
      (iiwa_status_receiver->get_position_commanded_output_port(),
       "iiwa_position_commanded");
  builder.ExportOutput
      (iiwa_status_receiver->get_position_measured_output_port(),
       "iiwa_position_measured");
  builder.ExportOutput
      (iiwa_status_receiver->get_velocity_estimated_output_port(),
       "iiwa_velocity_estimated");
  builder.ExportOutput
      (iiwa_status_receiver->get_torque_commanded_output_port(),
       "iiwa_torque_commanded");
  builder.ExportOutput
      (iiwa_status_receiver->get_torque_measured_output_port(),
       "iiwa_torque_measured");
  builder.ExportOutput
      (iiwa_status_receiver->get_torque_external_output_port(),
       "iiwa_torque_external");
  builder.Connect(iiwa_status_subscriber->get_output_port(),
                  iiwa_status_receiver->get_input_port(0));

  auto diagram = builder.Build();
  diagram->set_name("manipulation_station");
  return std::move(diagram);
}

}  // namespace manipulation_station
}  // namespace examples
}  // namespace drake

