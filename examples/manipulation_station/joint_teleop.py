"""
Runs the manipulation_station example with a simple tcl/tk joint slider ui for
directly tele-operating the joints.
"""
import Tkinter as tk
import numpy as np

from pydrake.common import FindResourceOrThrow
from pydrake.examples.manipulation_station import ManipulationStation
from pydrake.geometry import ConnectDrakeVisualizer
from pydrake.manipulation.simple_ui import JointSliders, SchunkWsgButtons
from pydrake.multibody.multibody_tree.parsing import AddModelFromSdfFile
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.util.eigen_geometry import Isometry3


builder = DiagramBuilder()

station = builder.AddSystem(ManipulationStation())
station.AddCupboard()
object = AddModelFromSdfFile(FindResourceOrThrow(
    "drake/external/models_robotlocomotion/ycb_objects/apple.sdf"),
                           "object",
                           station.get_mutable_multibody_plant(),
 station.get_mutable_scene_graph() )
station.Finalize()

teleop = builder.AddSystem(JointSliders(station.get_controller_plant()))
builder.Connect(teleop.get_output_port(0), station.GetInputPort(
    "iiwa_position"))

wsg_buttons = builder.AddSystem(SchunkWsgButtons(teleop.window))
builder.Connect(wsg_buttons.GetOutputPort("position"), station.GetInputPort(
    "wsg_position"))
builder.Connect(wsg_buttons.GetOutputPort("force_limit"),
                station.GetInputPort("wsg_force_limit"))

ConnectDrakeVisualizer(builder, station.get_mutable_scene_graph(),
                       station.GetOutputPort("pose_bundle"))

diagram = builder.Build()
simulator = Simulator(diagram)

context = diagram.GetMutableSubsystemContext(station,
                                             simulator.get_mutable_context())

q0 = [0, 0.6, 0, -1.75, 0, 1.0, 0]
station.SetIiwaPosition(q0, context)
station.SetIiwaVelocity(np.zeros(7), context)
station.SetWsgState(0.05, 0, context)
teleop.set(q0)
X_WObject = Isometry3.Identity()
X_WObject.set_translation([.6, 0, 0])
station.get_mutable_multibody_plant().tree().SetFreeBodyPoseOrThrow(
    station.get_mutable_multibody_plant().GetBodyByName("base_link_apple",
                                                       object), X_WObject,
    station.GetMutableSubsystemContext(station.get_mutable_multibody_plant(),
        context))

context.FixInputPort(station.GetInputPort(
    "iiwa_feedforward_torque").get_index(), np.zeros(7))

simulator.set_target_realtime_rate(1.0)
simulator.StepTo(np.inf)
