#include <limits>

#include <gflags/gflags.h>

#include "drake/common/eigen_types.h"
#include "drake/common/find_resource.h"
#include "drake/common/is_approx_equal_abstol.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_lcm.h"
#include "drake/examples/manipulation_station/manipulation_station.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/lcmt_iiwa_command.hpp"
#include "drake/lcmt_iiwa_status.hpp"
#include "drake/multibody/multibody_tree/parsing/multibody_plant_sdf_parser.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/systems/primitives/demultiplexer.h"
#include "drake/systems/primitives/matrix_gain.h"

namespace drake {
namespace examples {
namespace manipulation_station {
namespace {

// Runs a simulation of the manipulation station plant as a stand-alone
// simulation which mocks the network inputs and outputs of the real robot
// station.

using Eigen::VectorXd;

DEFINE_double(target_realtime_rate, 1.0,
              "Playback speed.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");
DEFINE_double(duration, std::numeric_limits<double>::infinity(),
              "Simulation duration.");

int do_main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  systems::DiagramBuilder<double> builder;

  // Create the "manipulation station".
  auto station = builder.AddSystem<ManipulationStation>();
  station->AddCupboard();
  // TODO(russt): Load sdf objects specified at the command line.  Requires
  // #9747.
  auto object = multibody::parsing::AddModelFromSdfFile(
      FindResourceOrThrow(
          "drake/external/models_robotlocomotion/ycb_objects/apple.sdf"),
      "apple", &station->get_mutable_multibody_plant(),
      &station->get_mutable_scene_graph());
  station->Finalize();

  geometry::ConnectDrakeVisualizer(&builder, station->get_mutable_scene_graph(),
                                   station->GetOutputPort("pose_bundle"));

  lcm::DrakeLcm lcm;
  lcm.StartReceiveThread();

  // TODO(russt): IiwaCommandReceiver should output positions, not
  // state.  (We are adding delay twice in this current implementation).
  auto iiwa_command_subscriber = builder.AddSystem(
      systems::lcm::LcmSubscriberSystem::Make<drake::lcmt_iiwa_command>(
          "IIWA_COMMAND", &lcm));
  auto iiwa_command = builder.AddSystem<kuka_iiwa_arm::IiwaCommandReceiver>();
  builder.Connect(iiwa_command_subscriber->get_output_port(),
                  iiwa_command->get_input_port(0));

  // Pull the positions out of the state.
  auto demux = builder.AddSystem<systems::Demultiplexer>(14, 7);
  builder.Connect(iiwa_command->get_commanded_state_output_port(),
                  demux->get_input_port(0));
  builder.Connect(demux->get_output_port(0),
                  station->GetInputPort("iiwa_position"));
  builder.Connect(iiwa_command->get_commanded_torque_output_port(),
                  station->GetInputPort("iiwa_feedforward_torque"));

  auto iiwa_status = builder.AddSystem<kuka_iiwa_arm::IiwaStatusSender>();
  // TODO(russt): Ok seriously, the IiwaStatusSender input port wants size
  // 14, but only uses the first 7.
  auto zero_padding =
      builder.AddSystem<systems::MatrixGain>(Eigen::MatrixXd::Identity(14, 7));
  builder.Connect(station->GetOutputPort("iiwa_position_commanded"),
                  zero_padding->get_input_port());
  builder.Connect(zero_padding->get_output_port(),
                  iiwa_status->get_command_input_port());
  builder.Connect(station->GetOutputPort("iiwa_state_estimated"),
                  iiwa_status->get_state_input_port());
  builder.Connect(station->GetOutputPort("iiwa_torque_commanded"),
                  iiwa_status->get_commanded_torque_input_port());
  builder.Connect(station->GetOutputPort("iiwa_torque_measured"),
                  iiwa_status->get_measured_torque_input_port());
  builder.Connect(station->GetOutputPort("iiwa_torque_external"),
                  iiwa_status->get_external_torque_input_port());
  auto iiwa_status_publisher = builder.AddSystem(
      systems::lcm::LcmPublisherSystem::Make<drake::lcmt_iiwa_status>(
          "IIWA_STATUS", &lcm));
  builder.Connect(iiwa_status->get_output_port(0),
                  iiwa_status_publisher->get_input_port());

  auto diagram = builder.Build();

  systems::Simulator<double> simulator(*diagram);
  auto& context = simulator.get_mutable_context();
  auto& station_context =
      diagram->GetMutableSubsystemContext(*station, &context);

  // Set initial conditions for the IIWA:
  VectorXd q0(7);
  q0 << 0, 0.6, 0, -1.75, 0, 1.0, 0;
  iiwa_command->set_initial_position(
      &diagram->GetMutableSubsystemContext(*iiwa_command, &context), q0);
  station->SetIiwaPosition(q0, &station_context);
  const VectorXd qdot0 = VectorXd::Zero(7);
  station->SetIiwaVelocity(qdot0, &station_context);

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = Eigen::Vector3d(.6, 0, 0);
  station->get_mutable_multibody_plant().tree().SetFreeBodyPoseOrThrow(
      station->get_mutable_multibody_plant().GetBodyByName("base_link_apple",
                                                           object),
      pose, &station->GetMutableSubsystemContext(
                station->get_mutable_multibody_plant(), &station_context));

  // Nominal WSG position is open.
  station_context.FixInputPort(
      station->GetInputPort("wsg_position").get_index(), Vector1d(0.05));
  // Force limit at 40N.
  station_context.FixInputPort(
      station->GetInputPort("wsg_force_limit").get_index(), Vector1d(40.0));

  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.StepTo(FLAGS_duration);

  return 0;
}

}  // namespace
}  // namespace manipulation_station
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  return drake::examples::manipulation_station::do_main(argc, argv);
}
