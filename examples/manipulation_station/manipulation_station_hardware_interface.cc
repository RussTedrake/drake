#include "drake/examples/manipulation_station/manipulation_station_hardware_interface.h"

#include "drake/common/find_resource.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_lcm.h"
#include "drake/lcmt_iiwa_command.hpp"
#include "drake/lcmt_iiwa_status.hpp"
#include "drake/multibody/multibody_tree/parsing/multibody_plant_sdf_parser.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/systems/primitives/pass_through.h"

namespace drake {
namespace examples {
namespace manipulation_station {

using Eigen::Isometry3d;
using Eigen::Vector3d;
using multibody::multibody_plant::MultibodyPlant;
using multibody::parsing::AddModelFromSdfFile;

// TODO(russt): Set publishing defaults.

ManipulationStationHardwareInterface::ManipulationStationHardwareInterface(
    lcm::DrakeLcmInterface* lcm_optional)
    : owned_controller_plant_(std::make_unique<MultibodyPlant<double>>()) {
  systems::DiagramBuilder<double> builder;

  // Publish IIWA command.
  auto iiwa_command_sender =
      builder.AddSystem<examples::kuka_iiwa_arm::IiwaCommandSender>();
  auto iiwa_command_publisher = builder.AddSystem(
      systems::lcm::LcmPublisherSystem::Make<drake::lcmt_iiwa_command>(
          "IIWA_COMMAND", lcm_optional));
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
  auto iiwa_status_receiver =
      builder.AddSystem<examples::kuka_iiwa_arm::IiwaStatusReceiver>();
  auto iiwa_status_subscriber = builder.AddSystem(
      systems::lcm::LcmSubscriberSystem::Make<drake::lcmt_iiwa_status>(
          "IIWA_STATUS", &lcm));
  builder.ExportOutput(
      iiwa_status_receiver->get_position_commanded_output_port(),
      "iiwa_position_commanded");
  builder.ExportOutput(
      iiwa_status_receiver->get_position_measured_output_port(),
      "iiwa_position_measured");
  builder.ExportOutput(
      iiwa_status_receiver->get_velocity_estimated_output_port(),
      "iiwa_velocity_estimated");
  builder.ExportOutput(iiwa_status_receiver->get_torque_commanded_output_port(),
                       "iiwa_torque_commanded");
  builder.ExportOutput(iiwa_status_receiver->get_torque_measured_output_port(),
                       "iiwa_torque_measured");
  builder.ExportOutput(iiwa_status_receiver->get_torque_external_output_port(),
                       "iiwa_torque_external");
  builder.Connect(iiwa_status_subscriber->get_output_port(),
                  iiwa_status_receiver->get_input_port(0));

  // TODO(russt): Actually publish WSG command.  These are only placeholders.
  auto wsg_position = builder.AddSystem<systems::PassThrough>(1);
  builder.ExportInput(wsg_position->get_input_port(), "wsg_position");
  auto wsg_force_limit = builder.AddSystem<systems::PassThrough>(1);
  builder.ExportInput(wsg_force_limit->get_input_port(), "wsg_force_limit");

  builder.BuildInto(this);
  this->set_name("manipulation_station");

  // Build the controller's version of the plant, which only contains the
  // IIWA and the equivalent inertia of the gripper.
  const std::string iiwa_sdf_path = FindResourceOrThrow(
      "drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf");
  //      "drake/external/models_robotlocomotion/iiwa7/iiwa7_no_collision.sdf");
  const auto controller_iiwa_model =
      AddModelFromSdfFile(iiwa_sdf_path, "iiwa", owned_controller_plant_.get());
  owned_controller_plant_->WeldFrames(owned_controller_plant_->world_frame(),
                                      owned_controller_plant_->GetFrameByName(
                                          "iiwa_link_0", controller_iiwa_model),
                                      Isometry3d::Identity());
  owned_controller_plant_
      ->template AddForceElement<multibody::UniformGravityFieldElement>(
          -9.81 * Vector3d::UnitZ());
  owned_controller_plant_->Finalize();
}

}  // namespace manipulation_station
}  // namespace examples
}  // namespace drake
