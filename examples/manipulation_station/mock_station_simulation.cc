#include <limits>

#include <gflags/gflags.h>

#include "drake/common/eigen_types.h"
#include "drake/common/find_resource.h"
#include "drake/common/is_approx_equal_abstol.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_lcm.h"
#include "drake/examples/manipulation_station/manipulation_station.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/multibody/multibody_tree/parsing/multibody_plant_sdf_parser.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/demultiplexer.h"

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

  // TODO(russt): IiwaCommandReceiver should output positions, not
  // state.  (We are adding delay twice in this current implementation).
  auto iiwa_command = builder.AddSystem<kuka_iiwa_arm::IiwaCommandReceiver>();
  // Pull the positions out of the state.
  auto demux = builder.AddSystem<systems::Demultiplexer>(14, 7);
  builder.Connect(iiwa_command->get_commanded_state_output_port(),
                  demux->get_input_port(0));
  builder.Connect(demux->get_output_port(0),
                  station->GetInputPort("iiwa_position"));
  builder.Connect(iiwa_command->get_commanded_torque_output_port(),
                  station->GetInputPort("iiwa_feedforward_torque"));

  auto iiwa_status = builder.AddSystem<kuka_iiwa_arm::IiwaStatusSender>();
  builder.Connect(station->GetOutputPort("iiwa_position_commanded"),
      iiwa_status->get_command_input_port());
  builder.Connect(station->GetOutputPort("iiwa_state_estimated"),
      iiwa_status->get_state_input_port());
  builder.Connect(station->GetOutputPort("iiwa_torque_commanded"),
      iiwa_status->get_commanded_torque_input_port());
  builder.Connect(station->GetOutputPort("iiwa_torque_measured"),
      iiwa_status->get_measured_torque_input_port());
  builder.Connect(station->GetOutputPort("iiwa_torque_external"),
      iiwa_status->get_external_torque_input_port());

  auto diagram = builder.Build();

  systems::Simulator<double> simulator(*diagram);
  auto& context = diagram->GetMutableSubsystemContext(
      *station, &simulator.get_mutable_context());

  // Set initial conditions for the IIWA:
  VectorXd q0(7);
  q0 << 0, 0.6, 0, -1.75, 0, 1.0, 0;
  station->SetIiwaPosition(q0, &context);
  const VectorXd qdot0 = VectorXd::Zero(7);
  station->SetIiwaVelocity(qdot0, &context);

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = Eigen::Vector3d(.6, 0, 0);
  station->get_mutable_multibody_plant().tree().SetFreeBodyPoseOrThrow(
      station->get_mutable_multibody_plant().GetBodyByName("base_link_apple",
                                                           object),
      pose, &station->GetMutableSubsystemContext(
                station->get_mutable_multibody_plant(), &context));

  // Nominal WSG position is open.
  context.FixInputPort(station->GetInputPort("wsg_position").get_index(),
                       Vector1d(0.05));
  // Force limit at 40N.
  context.FixInputPort(station->GetInputPort("wsg_force_limit").get_index(),
                       Vector1d(40.0));

  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();
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
