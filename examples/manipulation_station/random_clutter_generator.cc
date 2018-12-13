#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/multibody_tree/multibody_plant/multibody_plant.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace examples {
namespace manipulation_station {
namespace {

// Simple version of the "random clutter generation" used in the manipulation
// station (and beyond).  This sets up a MultibodyPlant (only) with the only
// dynamic objects being the clutter (no robot).


int do_main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // What I want to have:
  // for i=1:N
  //   sim = SimulationFactory();
  //   for j=1:M
  //      plant->SetRandomState()
  //      diagram->Publish()
  // or perhaps a single call to MonteCarlo?

  systems::DiagramBuilder<double> builder;

  auto plant = builder.AddSystem<multibody::multibody_plant::MultibodyPlant
      <double>>(0.001);
  auto scene_graph = builder.AddSystem<geometry::SceneGraph<double>>();
  multibody::Parser parser(plant, scene_graph);

  auto bin = parser.AddModelFromFile(
      FindResourceOrThrow(
          "drake/examples/kuka_iiwa_arm/models/objects/open_top_box.urdf"),
      "bin");
  plant->WeldFrames(plant->world_frame(), plant->GetFrameByName("bin", bin),
      Eigen::Isometry3d::Identity());

  parser.AddModelFromFile(
      FindResourceOrThrow(
          "drake/examples/manipulation_station/models/061_foam_brick.sdf"),
      "object");
  plant->template AddForceElement<multibody::UniformGravityFieldElement>(
      -9.81 * Eigen::Vector3d::UnitZ());
  plant->Finalize();

  builder.Connect(plant->get_geometry_poses_output_port(),
      scene_graph->get_source_pose_port(plant->get_source_id().value()));

  geometry::ConnectDrakeVisualizer(&builder, *scene_graph);

  auto diagram = builder.Build();
  auto context = diagram->CreateDefaultContext();

  systems::Simulator<double> simulator(*diagram);
  simulator.Initialize(); // To send load message.

  diagram->Publish(*context);

  return 0;
}

}  // namespace
}  // namespace manipulation_station
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  return drake::examples::manipulation_station::do_main(argc, argv);
}
