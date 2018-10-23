#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"

#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/examples/manipulation_station/manipulation_station.h"
#include "drake/examples/manipulation_station/manipulation_station_hardware_interface.h"  // noqa

using std::make_unique;
using std::unique_ptr;
using std::vector;

namespace drake {
namespace pydrake {

PYBIND11_MODULE(manipulation_station, m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::systems;
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::examples::manipulation_station;

  m.doc() = "Bindings for the Manipulation Station example.";
  constexpr auto& doc = pydrake_doc.drake.examples.manipulation_station;

  py::module::import("pydrake.systems.framework");

  // ManipulationStation currently only supports double.
  using T = double;

  py::class_<ManipulationStation<T>, Diagram<T>>(m, "ManipulationStation")
      .def(py::init<double>(), py::arg("time_step") = 0.002,
           doc.ManipulationStation.ctor.doc_3)
      .def("AddCupboard", &ManipulationStation<T>::AddCupboard,
           doc.ManipulationStation.AddCupboard.doc)
      .def("Finalize", &ManipulationStation<T>::Finalize,
           doc.ManipulationStation.Finalize.doc)
      .def("get_mutable_multibody_plant",
           &ManipulationStation<T>::get_mutable_multibody_plant,
           py_reference_internal,
           doc.ManipulationStation.get_mutable_multibody_plant.doc)
      .def("get_mutable_scene_graph",
           &ManipulationStation<T>::get_mutable_scene_graph,
           py_reference_internal,
           doc.ManipulationStation.get_mutable_scene_graph.doc)
      .def("get_controller_plant",
           &ManipulationStation<T>::get_controller_plant, py_reference_internal,
           doc.ManipulationStation.get_controller_plant.doc)
      .def("GetIiwaPosition", &ManipulationStation<T>::GetIiwaPosition,
           doc.ManipulationStation.GetIiwaPosition.doc)
      .def("SetIiwaPosition", &ManipulationStation<T>::SetIiwaPosition,
           doc.ManipulationStation.SetIiwaPosition.doc)
      .def("GetIiwaVelocity", &ManipulationStation<T>::GetIiwaVelocity,
           doc.ManipulationStation.GetIiwaVelocity.doc)
      .def("SetIiwaVelocity", &ManipulationStation<T>::SetIiwaVelocity,
           doc.ManipulationStation.SetIiwaVelocity.doc)
      .def("SetWsgState", &ManipulationStation<T>::SetWsgState,
           doc.ManipulationStation.SetWsgState.doc);

  py::class_<ManipulationStationHardwareInterface, Diagram<double>>(
      m, "ManipulationStationHardwareInterface")
      .def(py::init<lcm::DrakeLcmInterface*>(), py::arg("lcm") = nullptr,
           doc.ManipulationStationHardwareInterface.ctor.doc_3)
      .def("get_controller_plant",
           &ManipulationStationHardwareInterface::get_controller_plant,
           py_reference_internal, doc.ManipulationStationHardwareInterface
                                      .get_controller_plant.doc);
}

}  // namespace pydrake
}  // namespace drake
