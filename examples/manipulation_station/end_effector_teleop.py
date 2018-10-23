
import argparse

import Tkinter as tk
import numpy as np

from pydrake.examples.manipulation_station import ManipulationStation
from pydrake.geometry import ConnectDrakeVisualizer
from pydrake.multibody.multibody_tree.multibody_plant import MultibodyPlant
from pydrake.manipulation.simple_ui import SchunkWsgButtons
from pydrake.manipulation.planner import (
    DifferentialInverseKinematicsParameters, DoDifferentialInverseKinematics)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import ( AbstractValue, BasicVector,
                                        DiagramBuilder,
                                        LeafSystem,
                                        PortDataType )
from pydrake.util.eigen_geometry import Isometry3, AngleAxis


class EndEffectorTeleop(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self._DeclareAbstractOutputPort("X_WE",
              lambda: AbstractValue.Make(Isometry3.Identity()),
              self._DoCalcOutput)

        self._DeclarePeriodicPublish(0.1, 0.0)

        self.window = tk.Tk()
        self.window.title("End-Effector TeleOp")

        self.roll  = tk.Scale(self.window, from_=-2 * np.pi, to=2 * np.pi,
                              resolution=-1,
                              label="roll",
                              length=800,
                              orient=tk.HORIZONTAL)
        self.roll.pack()
        self.pitch = tk.Scale(self.window, from_=-2 * np.pi, to=2 * np.pi,
                              resolution=-1,
                              label="pitch",
                              length=800,
                              orient=tk.HORIZONTAL)
        self.pitch.pack()
        self.yaw   = tk.Scale(self.window, from_=-2 * np.pi, to=2 * np.pi,
                              resolution=-1,
                              label="yaw",
                              length=800,
                              orient=tk.HORIZONTAL)
        self.yaw.pack()
        self.x     = tk.Scale(self.window, from_=0.1, to=0.8,
                              resolution=-1,
                              label="x",
                              length=800,
                              orient=tk.HORIZONTAL)
        self.x.pack()
        self.y     = tk.Scale(self.window, from_=-0.3, to=0.3,
                              resolution=-1,
                              label="y",
                              length=800,
                              orient=tk.HORIZONTAL)
        self.y.pack()
        self.z     = tk.Scale(self.window, from_=0, to=0.8,
                              resolution=-1,
                              label="z",
                              length=800,
                              orient=tk.HORIZONTAL)
        self.z.pack()

    # @param pose is an Isometry3
    def SetPose(self, pose):
        tf = RigidTransform(pose)
        self.SetRPY(RollPitchYaw(tf.rotation()))
        self.SetXYZ(pose.translation())

    # @param rpy is a 3 element vector of roll, pitch, yaw
    def SetRPY(self, rpy):
        self.roll.set(rpy.roll_angle())
        self.pitch.set(rpy.pitch_angle())
        self.yaw.set(rpy.yaw_angle())

    def SetXYZ(self, xyz):
        self.x.set(xyz[0])
        self.y.set(xyz[1])
        self.z.set(xyz[2])

    def _DoPublish(self, context, event):
        self.window.update_idletasks()
        self.window.update()

    def _DoCalcOutput(self, context, output):
        output.get_mutable_value().set_rotation(RollPitchYaw(self.roll.get(),
                                                             self.pitch.get(),
                                                             self.yaw.get()).
                                                ToRotationMatrix().matrix())
        output.get_mutable_value().set_translation([self.x.get(), self.y.get(),
                                                    self.z.get()])

class DifferentialIK(LeafSystem):
    # @param robot is a reference to a MultibodyPlant.
    # @param frame_E is a multibody::Frame on the robot.
    # @param params is a DifferentialIKParams.
    def __init__(self, robot, frame_E, parameters, time_step):
        LeafSystem.__init__(self)
        self.robot = robot
        self.frame_E = frame_E
        self.parameters = parameters
        self.time_step = time_step
        self.context = robot.CreateDefaultContext()

        # Desired pose of frame E in world frame.
        self._DeclareAbstractInputPort("X_WE_desired")
        self._DeclareInputPort("robot_state", PortDataType.kVectorValued,
                               robot.num_positions()+robot.num_velocities())

        # Output is desired joint velocities.
        self._DeclareVectorOutputPort("desired_velocity", BasicVector(
            robot.num_velocities()), self._DoCalcVelocityOutput)

        # Alternatively, provide the output as desired positions.
        self._DeclareVectorOutputPort("desired_position", BasicVector(
            robot.num_positions()), self._DoCalcPositionOutput)

        # Pose error
        self._DeclareVectorOutputPort("pose_error", BasicVector(6),
                                      self._DoCalcPoseError)

    def ForwardKinematics(self, q):
        x = self.robot.tree().get_mutable_multibody_state_vector(
            self.context)
        x[:robot.num_positions()] = q
        return self.robot.tree().EvalBodyPoseInWorld(
            self.context, self.frame_E.body())

    def DoDifferentialIK(self, context):
        X_WE_desired = self.EvalAbstractInput(context, 0).get_value()
        robot_state = self.EvalVectorInput(context, 1).get_value()
        if (self.context.get_num_discrete_state_groups() > 0):
            self.context.get_mutable_discrete_state_vector().SetFromVector(
                robot_state)
        else:
            self.context.get_mutable_continuous_state_vector().SetFromVector(
                robot_state)
        result = DoDifferentialInverseKinematics(self.robot,
                                                self.context,
                                                X_WE_desired, self.frame_E,
                                                self.parameters)
        if (result.status != result.status.kSolutionFound):
            return np.zeros(self.robot.num_velocities())
        return result.joint_velocities

    def _DoCalcVelocityOutput(self, context, output):
        output.SetFromVector(self.DoDifferentialIK(context))
        assert(false)

    def _DoCalcPositionOutput(self, context, output):
        q = self.EvalVectorInput(context, 1).get_value()[:self.robot.num_positions()]
        # TODO(russt): Should be effectively a PD controller.
        output.SetFromVector(q + self.time_step*self.DoDifferentialIK(
            context))

    def _DoCalcPoseError(self, context, output):
        X_WE_desired = self.EvalAbstractInput(context, 0).get_value()
        q = self.EvalVectorInput(context, 1).get_value()[:self.robot.num_positions()]
        pose = self.ForwardKinematics(q)

        output.get_mutable_value()[-3:] = X_WE_desired.translation() - pose.translation()

        rot_err = AngleAxis(X_WE_desired.rotation() * pose.rotation().transpose())
        output.get_mutable_value()[:3] = rot_err.axis() * rot_err.angle();


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target_realtime_rate", type=float, default=1.0,
    help="Desired rate relative to real time.  See documentation for "
         "Simulator::set_target_realtime_rate() for details.")
parser.add_argument(
    "--duration", type=float, default=np.inf,
    help="Desired duration of the simulation in seconds.")
parser.add_argument(
    "--hardware", action='store_true',
    help="Use the ManipulationStationHardwareInterface instead of an "
         "in-process simulation.")
args = parser.parse_args()

builder = DiagramBuilder()

time_step = 0.002
station = builder.AddSystem(ManipulationStation(time_step))
station.AddCupboard()
station.Finalize()
q0 = [0, 0.6, 0, -1.75, 0, 1.0, 0]

robot = station.get_controller_plant()
params = DifferentialInverseKinematicsParameters(robot.num_positions(),
                                                 robot.num_velocities())

params.set_timestep(time_step)
#params.set_end_effector_velocity_gain([0.2,0.2,0.2,1,1,1])
params.set_nominal_joint_position(q0)
params.set_unconstrained_degrees_of_freedom_velocity_limit(0.6)

differential_ik = builder.AddSystem(DifferentialIK(
    robot, robot.GetFrameByName("iiwa_link_7"), params, time_step))
builder.Connect(station.GetOutputPort("iiwa_state_estimated"),
                differential_ik.get_input_port(1))

builder.Connect(differential_ik.GetOutputPort("desired_position"),
 station.GetInputPort("iiwa_position"))

teleop = builder.AddSystem(EndEffectorTeleop())
builder.Connect(teleop.get_output_port(0),
                differential_ik.get_input_port(0))

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

station.SetIiwaPosition(q0, context)
station.SetIiwaVelocity(np.zeros(7), context)
teleop.SetPose(differential_ik.ForwardKinematics(q0))

context.FixInputPort(station.GetInputPort(
    "iiwa_feedforward_torque").get_index(), np.zeros(7))

simulator.set_target_realtime_rate(args.target_realtime_rate)
simulator.StepTo(args.duration)
