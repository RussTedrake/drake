#include "pybind11/eigen.h"
#include "pybind11/functional.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <common_robotics_utilities/openmp_helpers.hpp>
#include <common_robotics_utilities/path_processing.hpp>
#include <common_robotics_utilities/simple_astar_search.hpp>
#include <common_robotics_utilities/simple_graph.hpp>
#include <common_robotics_utilities/simple_prm_planner.hpp>
#include <common_robotics_utilities/simple_rrt_planner.hpp>

#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/common/random.h"

namespace drake {
namespace pydrake {
namespace internal {

void DefinePlanningCommonRoboticsUtilities(py::module m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  // using namespace drake::planning;
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace common_robotics_utilities;
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace common_robotics_utilities::simple_astar_search;
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace common_robotics_utilities::simple_graph;
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace common_robotics_utilities::simple_prm_planner;
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace common_robotics_utilities::simple_rrt_planner;
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace common_robotics_utilities::openmp_helpers;

  // using RNG = RandomGenerator;
  using T = Eigen::VectorXd;

  py::module::import("pydrake.common");

  // OMP helpers
  m.def("IsOmpEnabledInBuild", &IsOmpEnabledInBuild, "");
  m.def("GetNumOmpThreads", &GetNumOmpThreads, "");
  m.def("IsOmpInParallel", &IsOmpInParallel, "");

  {
    using Class = ChangeOmpNumThreadsWrapper;
    py::class_<Class>(m, "ChangeOmpNumThreadsWrapper", "")
        .def(py::init<const int64_t>(), py::arg("num_threads"), "");
  }

  {
    using Class = SimpleRRTPlannerState<T>;
    py::class_<Class>(m, "SimpleRRTPlannerState", "")
        .def(py::init<const T&>(), py::arg("value"), "")
        .def(py::init<const T&, const int64_t>(), py::arg("value"),
            py::arg("parent_index"), "")
        .def(py::init<const T&, const int64_t, const std::vector<int64_t>&>(),
            py::arg("value"), py::arg("parent_index"), py::arg("child_indices"),
            "")
        .def("GetValueImmutable", &Class::GetValueImmutable, "")
        .def("GetValueMutable", &Class::GetValueMutable, "")
        .def("GetParentIndex", &Class::GetParentIndex, "")
        .def("SetParentIndex", &Class::SetParentIndex, py::arg("parent_index"),
            "")
        .def("GetChildIndices", &Class::GetChildIndices, "")
        .def("ClearChildIndicies", &Class::ClearChildIndicies, "")
        .def("AddChildIndex", &Class::AddChildIndex, py::arg("child_index"), "")
        .def("RemoveChildIndex", &Class::RemoveChildIndex,
            py::arg("child_index"), "");
  }

  {
    using Class = PropagatedState<T>;
    py::class_<Class>(m, "PropagatedState", "")
        .def(py::init<const T&, const int64_t>(), py::arg("state"),
            py::arg("relative_parent_index"), "")
        .def("State", &Class::State, "")
        .def("MutableState", &Class::MutableState, "")
        .def("RelativeParentIndex", &Class::RelativeParentIndex, "")
        .def("SetRelativeParentIndex", &Class::SetRelativeParentIndex,
            py::arg("relative_parent_index"), "");
  }

  {
    using Class = SimpleRRTPlannerTree<T>;
    py::class_<Class>(m, "SimpleRRTPlannerTree", "")
        .def(py::init<>(), "")
        .def(py::init<const int64_t>(), py::arg("anticipated_size"), "")
        .def(py::init<const SimpleRRTPlannerStateVector<T>&>(),
            py::arg("nodes"), "")
        .def("Size", &Class::Size, "")
        .def("Empty", &Class::Empty, "")
        .def("Clear", &Class::Clear, "")
        .def("GetNodeImmutable", &Class::GetNodeImmutable, py::arg("index"), "")
        .def("GetNodeMutable", &Class::GetNodeMutable, py::arg("index"), "")
        .def("AddNode", &Class::AddNode, py::arg("value"), "")
        .def("AddNodeAndConnect", &Class::AddNodeAndConnect, py::arg("value"),
            py::arg("parent_index"), "")
        .def("GetNodesImmutable", &Class::GetNodesImmutable, "")
        .def("GetNodesMutable", &Class::GetNodesMutable, "")
        .def_static(
            "CheckNodeLinkage", &Class::CheckNodeLinkage, py::arg("nodes"), "");
  }

  {
    using Class = MultipleSolutionPlanningResults<T>;
    py::class_<Class>(m, "MultipleSolutionPlanningResults", "")
        .def("Paths", &Class::Paths, "")
        .def("Statistics", &Class::Statistics, "");
  }

  {
    using Class = SingleSolutionPlanningResults<T>;
    py::class_<Class>(m, "SingleSolutionPlanningResults", "")
        .def("Path", &Class::Path, "")
        .def("Statistics", &Class::Statistics, "");
  }

  m.def("MakeKinematicLinearRRTNearestNeighborsFunction",
      &MakeKinematicLinearRRTNearestNeighborsFunction<T>,
      py::arg("distance_fn"), py::arg("use_parallel") = true, "");

  m.def("MakeKinematicLinearBiRRTNearestNeighborsFunction",
      &MakeKinematicLinearBiRRTNearestNeighborsFunction<T>,
      py::arg("distance_fn"), py::arg("use_parallel") = true, "");

  m.def("MakeKinematicBiRRTExtendPropagationFunction",
      &MakeKinematicBiRRTExtendPropagationFunction<T>, py::arg("distance_fn"),
      py::arg("state_interpolation_fn"), py::arg("edge_validity_check_fn"),
      py::arg("step_size"), "");

  m.def("MakeKinematicBiRRTConnectPropagationFunction",
      &MakeKinematicBiRRTConnectPropagationFunction<T>, py::arg("distance_fn"),
      py::arg("state_interpolation_fn"), py::arg("edge_validity_check_fn"),
      py::arg("step_size"), "");

  m.def("MakeRRTTimeoutTerminationFunction", &MakeRRTTimeoutTerminationFunction,
      py::arg("planning_timeout"), "");

  m.def("MakeBiRRTTimeoutTerminationFunction",
      &MakeBiRRTTimeoutTerminationFunction, py::arg("planning_timeout"), "");

  m.def("RRTPlanMultiPath", &RRTPlanMultiPath<T>, py::arg("tree"),
      py::arg("sampling_fn"), py::arg("nearest_neighbor_fn"),
      py::arg("forward_propagation_fn"), py::arg("state_added_callback_fn"),
      py::arg("check_goal_reached_fn"), py::arg("goal_reached_callback_fn"),
      py::arg("termination_check_fn"), "");

  m.def("RRTPlanSinglePath", &RRTPlanSinglePath<T>, py::arg("tree"),
      py::arg("sampling_fn"), py::arg("nearest_neighbor_fn"),
      py::arg("forward_propagation_fn"), py::arg("state_added_callback_fn"),
      py::arg("check_goal_reached_fn"), py::arg("goal_reached_callback_fn"),
      py::arg("termination_check_fn"), "");

  m.def("BiRRTPlanMultiPath", &BiRRTPlanMultiPath<T>, py::arg("start_tree"),
      py::arg("goal_tree"), py::arg("select_sample_type_fn"),
      py::arg("state_sampling_fn"), py::arg("tree_sampling_fn"),
      py::arg("nearest_neighbor_fn"), py::arg("propagation_fn"),
      py::arg("state_added_callback_fn"), py::arg("states_connected_fn"),
      py::arg("goal_bridge_callback_fn"), py::arg("tree_sampling_bias"),
      py::arg("termination_check_fn"), "");

  m.def("BiRRTPlanSinglePath", &BiRRTPlanSinglePath<T>, py::arg("start_tree"),
      py::arg("goal_tree"), py::arg("select_sample_type_fn"),
      py::arg("state_sampling_fn"), py::arg("tree_sampling_fn"),
      py::arg("nearest_neighbor_fn"), py::arg("propagation_fn"),
      py::arg("state_added_callback_fn"), py::arg("states_connected_fn"),
      py::arg("goal_bridge_callback_fn"), py::arg("tree_sampling_bias"),
      py::arg("termination_check_fn"), "");

  // Path Processing
  //  m.def("ResamplePath", &path_processing::ResamplePath<T>, py::arg("path"),
  //      py::arg("resampled_state_distance"), py::arg("state_distance_fn"),
  //      py::arg("state_interpolation_fn"), "");

  m.def("ShortcutSmoothPath", &path_processing::ShortcutSmoothPath<T>,
      py::arg("path"), py::arg("max_iterations"),
      py::arg("max_failed_iterations"), py::arg("max_backtracking_steps"),
      py::arg("max_shortcut_fraction"), py::arg("resample_shortcuts_interval"),
      py::arg("check_for_marginal_shortcuts"),
      py::arg("edge_validity_check_fn"), py::arg("state_distance_fn"),
      py::arg("state_interpolation_fn"), py::arg("uniform_unit_real_fn"), "");

  // Simple Astar Search
  {
    using Class = AstarResult<T>;
    py::class_<Class>(m, "AstarResult", "")
        .def(py::init<>(), "")
        .def(py::init<const std::vector<T>&, const double>(), py::arg("path"),
            py::arg("path_cost"), "")
        .def("Path", &Class::Path, "")
        .def("PathCost", &Class::PathCost, "");
  }

  // Simple Graph
  {
    using Class = GraphEdge;
    py::class_<Class>(m, "GraphEdge", "")
        .def(py::init<>(), "")
        .def(py::init<const int64_t, int64_t, double>(), py::arg("from_index"),
            py::arg("to_index"), py::arg("weight"), "")
        .def(py::init<const int64_t, int64_t, double, uint64_t>(),
            py::arg("from_index"), py::arg("to_index"), py::arg("weight"),
            py::arg("scratchpad"), "")
        .def("GetFromIndex", &Class::GetFromIndex, "")
        .def("GetToIndex", &Class::GetToIndex, "")
        .def("GetWeight", &Class::GetWeight, "")
        .def("GetScratchpad", &Class::GetScratchpad, "")
        .def(
            "SetFromIndex", &Class::SetFromIndex, py::arg("new_from_index"), "")
        .def("SetToIndex", &Class::SetToIndex, py::arg("new_to_index"), "")
        .def("SetWeight", &Class::SetWeight, py::arg("new_weight"), "")
        .def("SetScratchpad", &Class::SetScratchpad, py::arg("new_scratchpad"),
            "")
        .def("__str__", &Class::Print)
        .def(py::pickle(
            [](const Class& self) {
              return std::make_tuple(self.GetFromIndex(), self.GetToIndex(),
                  self.GetWeight(), self.GetScratchpad());
            },
            [](std::tuple<int64_t, int64_t, double, uint64_t> args) {
              return Class(std::get<0>(args), std::get<1>(args),
                  std::get<2>(args), std::get<3>(args));
            }));
  }
  {
    using Class = GraphNode<T>;
    py::class_<Class>(m, "GraphNode", "")
        .def(py::init<>(), "")
        .def(py::init<const T&>(), py::arg("value"), "")
        .def(py::init<const T&, std::vector<GraphEdge>&,
                 std::vector<GraphEdge>&>(),
            py::arg("value"), py::arg("new_in_edges"), py::arg("new_out_edges"),
            "")
        .def("GetValueImmutable", &Class::GetValueImmutable, "")
        .def("GetValueMutable", &Class::GetValueMutable, "")
        .def("AddInEdge", &Class::AddInEdge, py::arg("new_in_edge"), "")
        .def("AddOutEdge", &Class::AddOutEdge, py::arg("new_out_edge"), "")
        .def("AddEdgePair", &Class::AddEdgePair, py::arg("new_in_edge"),
            py::arg("new_out_edge"), "")
        .def("GetInEdgesImmutable", &Class::GetInEdgesImmutable, "")
        .def("GetInEdgesMutable", &Class::GetInEdgesMutable, "")
        .def("GetOutEdgesImmutable", &Class::GetOutEdgesImmutable, "")
        .def("GetOutEdgesMutable", &Class::GetOutEdgesMutable, "")
        .def("SetInEdges", &Class::SetInEdges, py::arg("new_in_edges"), "")
        .def("SetOutEdges", &Class::SetOutEdges, py::arg("new_out_edges"), "")
        .def("__str__", &Class::Print)
        .def(py::pickle(
            [](const Class& self) {
              return std::make_tuple(self.GetValueImmutable(),
                  self.GetInEdgesImmutable(), self.GetOutEdgesImmutable());
            },
            [](std::tuple<T, std::vector<GraphEdge>, std::vector<GraphEdge>>
                    args) {
              return Class(
                  std::get<0>(args), std::get<1>(args), std::get<2>(args));
            }));
  }

  {
    using Class = Graph<T>;
    py::class_<Class>(m, "Graph", "")
        .def(py::init<>(), "")
        .def(py::init<const size_t>(), py::arg("expected_size"), "")
        .def(py::init<const std::vector<GraphNode<T>,
                 Eigen::aligned_allocator<GraphNode<T>>>&>(),
            py::arg("nodes"), "")
        .def("MakePrunedCopy", &Class::MakePrunedCopy,
            py::arg("nodes_to_prune"), py::arg("use_parallel"), "")
        .def("Size", &Class::Size, "")
        .def("IndexInRange", &Class::IndexInRange, py::arg("index"), "")
        .def("CheckGraphLinkage",
            overload_cast_explicit<bool>(&Class::CheckGraphLinkage), "")
        .def("GetNodesImmutable", &Class::GetNodesImmutable, "")
        .def("GetNodesMutable", &Class::GetNodesMutable, "")
        .def("GetNodeImmutable", &Class::GetNodeImmutable, py::arg("index"), "")
        .def("AddNode", py::overload_cast<const GraphNode<T>&>(&Class::AddNode),
            py::arg("new_node"), "")
        .def("AddNode", py::overload_cast<const T&>(&Class::AddNode),
            py::arg("new_value"), "")
        .def("AddEdgeBetweenNodes", &Class::AddEdgeBetweenNodes,
            py::arg("from_index"), py::arg("to_index"), py::arg("edge_weight"),
            "")
        .def("__str__", &Class::Print)
        .def(py::pickle(
            [](const Class& self) { return self.GetNodesImmutable(); },
            [](std::vector<GraphNode<T>, Eigen::aligned_allocator<GraphNode<T>>>
                    node) { return Class(node); }));
  }

  // PRM
  {
    using Class = NNDistanceDirection;
    py::class_<Class>(m, "NNDistanceDirection", "").def(py::init<>(), "");
  }

  m.def("AddNodeToRoadmap", &AddNodeToRoadmap<T, Graph<T>>, py::arg("state"),
      py::arg("nn_distance_direction"), py::arg("roadmap"),
      py::arg("distance_fn"), py::arg("edge_validity_check_fn"), py::arg("K"),
      py::arg("max_node_index_for_knn"), py::arg("use_parallel") = true,
      py::arg("connection_is_symmetric") = true,
      py::arg("add_duplicate_states") = false, "");

  m.def("GrowRoadMap", &GrowRoadMap<T, Graph<T>>, py::arg("roadmap"),
      py::arg("sampling_fn"), py::arg("distance_fn"),
      py::arg("state_validity_check_fn"), py::arg("edge_validity_check_fn"),
      py::arg("termination_check_fn"), py::arg("K"),
      py::arg("use_parallel") = true, py::arg("connection_is_symmetric") = true,
      py::arg("add_duplicate_states") = false, "");

  m.def("BuildRoadMap", &BuildRoadMap<T, Graph<T>>, py::arg("roadmap_size"),
      py::arg("sampling_fn"), py::arg("distance_fn"),
      py::arg("state_validity_check_fn"), py::arg("edge_validity_check_fn"),
      py::arg("K"), py::arg("max_valid_sample_tries"),
      py::arg("initial_states"), py::arg("use_parallel") = true,
      py::arg("connection_is_symmetric") = true,
      py::arg("add_duplicate_states") = false, "");

  m.def("UpdateRoadMapEdges", &UpdateRoadMapEdges<T, Graph<T>>,
      py::arg("roadmap"), py::arg("edge_validity_check_fn"),
      py::arg("distance_fn"), py::arg("use_parallel") = true, "");

  m.def("ExtractSolution", &ExtractSolution<T, std::vector<T>, Graph<T>>,
      py::arg("roadmap"), py::arg("astar_index_solution"), "");

  m.def("LazyQueryPathAndAddNodes",
      &LazyQueryPathAndAddNodes<T, std::vector<T>, Graph<T>>, py::arg("starts"),
      py::arg("goals"), py::arg("roadmap"), py::arg("distance_fn"),
      py::arg("edge_validity_check_fn"), py::arg("K"),
      py::arg("use_parallel") = true, py::arg("connection_is_symmetric") = true,
      py::arg("add_duplicate_states") = false,
      py::arg("limit_astar_pqueue_duplicates") = true, "");

  m.def("QueryPathAndAddNodes",
      &QueryPathAndAddNodes<T, std::vector<T>, Graph<T>>, py::arg("starts"),
      py::arg("goals"), py::arg("roadmap"), py::arg("distance_fn"),
      py::arg("edge_validity_check_fn"), py::arg("K"),
      py::arg("use_parallel") = true, py::arg("connection_is_symmetric") = true,
      py::arg("add_duplicate_states") = false,
      py::arg("limit_astar_pqueue_duplicates") = true, "");

  m.def("LazyQueryPath", &LazyQueryPath<T, std::vector<T>, Graph<T>>,
      py::arg("starts"), py::arg("goals"), py::arg("roadmap"),
      py::arg("distance_fn"), py::arg("edge_validity_check_fn"), py::arg("K"),
      py::arg("use_parallel") = true, py::arg("connection_is_symmetric") = true,
      py::arg("add_duplicate_states") = false,
      py::arg("limit_astar_pqueue_duplicates") = true,
      py::arg("use_roadmap_overlay") = true, "");

  m.def("QueryPath", &QueryPath<T, std::vector<T>, Graph<T>>, py::arg("starts"),
      py::arg("goals"), py::arg("roadmap"), py::arg("distance_fn"),
      py::arg("edge_validity_check_fn"), py::arg("K"),
      py::arg("use_parallel") = true, py::arg("connection_is_symmetric") = true,
      py::arg("add_duplicate_states") = false,
      py::arg("limit_astar_pqueue_duplicates") = true,
      py::arg("use_roadmap_overlay") = true, "");
}

}  // namespace internal
}  // namespace pydrake
}  // namespace drake
