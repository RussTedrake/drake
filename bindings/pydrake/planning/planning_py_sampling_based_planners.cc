#include "pybind11/eigen.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <common_robotics_utilities/openmp_helpers.hpp>
#include <common_robotics_utilities/simple_graph.hpp>
#include <common_robotics_utilities/simple_prm_planner.hpp>
#include <common_robotics_utilities/utility.hpp>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/bindings/pydrake/symbolic_types_pybind.h"
#include "drake/common/timer.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/planning/collision_checker.h"
#include "drake/planning/holonomic_kinematic_planning_space.h"
#include "drake/planning/joint_limits.h"
#include "drake/planning/path_planning_result.h"
#include "drake/planning/sampling_based_planners.h"
#include "drake/planning/voxelized_environment_builder.h"

namespace drake {
namespace pydrake {
namespace internal {

void DefinePlanningSamplingBasedPlanners(py::module m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::planning;
  // constexpr auto& doc = pydrake_doc.drake.planning;

  using T = Eigen::VectorXd;
  using Roadmap = common_robotics_utilities::simple_graph::Graph<T>;

  m.def("BuildCollisionMap", &BuildCollisionMap, py::arg("plant"),
      py::arg("plant_context"),
      py::arg("geometries_to_ignore") =
          std::unordered_set<geometry::GeometryId>(),
      py::arg("parent_body_name") = "world",
      py::arg("X_BG") = Eigen::Isometry3d::Identity(),
      py::arg("grid_size") = Eigen::Vector3d(2, 2, 2),
      py::arg("grid_resolution") = 0.5,
      py::arg("override_parent_body_index") = std::nullopt, "");

  // Planning Space
  {
    using Class = PlanningSpace<T>;
    py::class_<Class> cls(m, "PlanningSpace", "");
    // cls.def(py::init<>(), "")
    cls.def("Clone", &Class::Clone, "")
        .def("NearestNeighborDistanceForwards",
            &Class::NearestNeighborDistanceForwards, py::arg("from"),
            py::arg("to"), "")
        .def("NearestNeighborDistanceBackwards",
            &Class::NearestNeighborDistanceBackwards, py::arg("from"),
            py::arg("to"), "")
        .def("StateDistanceForwards", &Class::StateDistanceForwards,
            py::arg("from"), py::arg("to"), "")
        .def("StateDistanceBackwards", &Class::StateDistanceBackwards,
            py::arg("from"), py::arg("to"), "")
        .def("CalcPathLength", &Class::CalcPathLength, py::arg("path"), "")
        .def("InterpolateForwards", &Class::InterpolateForwards,
            py::arg("from"), py::arg("to"), py::arg("ratio"), "")
        .def("InterpolateBackwards", &Class::InterpolateBackwards,
            py::arg("from"), py::arg("to"), py::arg("ratio"), "")
        .def("PropagateForwards", &Class::PropagateForwards, py::arg("from"),
            py::arg("to"), py::arg("propagation_statistics"), "")
        .def("PropagateBackwards", &Class::PropagateBackwards, py::arg("from"),
            py::arg("to"), py::arg("propagation_statistics"), "")
        .def("MotionCostForwards", &Class::MotionCostForwards, py::arg("from"),
            py::arg("to"), "")
        .def("MotionCostBackwards", &Class::MotionCostBackwards,
            py::arg("from"), py::arg("to"), "")
        .def("CheckStateValidity", &Class::CheckStateValidity, py::arg("state"),
            "")
        .def("ExtractValidStarts", &Class::ExtractValidStarts,
            py::arg("starts"), "")
        .def("ExtractValidStartsAndGoals", &Class::ExtractValidStartsAndGoals,
            py::arg("starts"), py::arg("goals"), "")
        .def("CheckEdgeValidity", &Class::CheckEdgeValidity, py::arg("from"),
            py::arg("to"), "")
        .def("SampleState", &Class::SampleState, "")
        .def("MaybeSampleValidState", &Class::MaybeSampleValidState,
            py::arg("max_attempts"), "")
        .def("SampleValidState", &Class::SampleValidState,
            py::arg("max_attempts"), "")
        .def("random_source", &Class::random_source, "")
        .def("supports_parallel", &Class::supports_parallel, "")
        .def("is_symmetric", &Class::is_symmetric, "");
  }

  // SymmetricPlanningSpace

  {
    using Class = SymmetricPlanningSpace<T>;
    py::class_<Class, PlanningSpace<T>> cls(m, "SymmetricPlanningSpace", "");
    cls.def("NearestNeighborDistanceForwards",
           &Class::NearestNeighborDistanceForwards, py::arg("from"),
           py::arg("to"), "")
        .def("NearestNeighborDistanceBackwards",
            &Class::NearestNeighborDistanceBackwards, py::arg("from"),
            py::arg("to"), "")
        .def("StateDistanceForwards", &Class::StateDistanceForwards,
            py::arg("from"), py::arg("to"), "")
        .def("StateDistanceBackwards", &Class::StateDistanceBackwards,
            py::arg("from"), py::arg("to"), "")
        .def("InterpolateForwards", &Class::InterpolateForwards,
            py::arg("from"), py::arg("to"), py::arg("ratio"), "")
        .def("InterpolateBackwards", &Class::InterpolateBackwards,
            py::arg("from"), py::arg("to"), py::arg("ratio"), "")
        .def("PropagateForwards", &Class::PropagateForwards, py::arg("from"),
            py::arg("to"), py::arg("propagation_statistics"), "")
        .def("PropagateBackwards", &Class::PropagateBackwards, py::arg("from"),
            py::arg("to"), py::arg("propagation_statistics"), "")
        .def("MotionCostForwards", &Class::MotionCostForwards, py::arg("from"),
            py::arg("to"), "")
        .def("MotionCostBackwards", &Class::MotionCostBackwards,
            py::arg("from"), py::arg("to"), "")
        .def("NearestNeighborDistance", &Class::NearestNeighborDistance,
            py::arg("from"), py::arg("to"), "")
        .def("StateDistance", &Class::StateDistance, py::arg("from"),
            py::arg("to"), "")
        .def("Interpolate", &Class::Interpolate, py::arg("from"), py::arg("to"),
            py::arg("ratio"), "")
        .def("Propagate", &Class::Propagate, py::arg("from"), py::arg("to"),
            py::arg("propagation_statistics"), "")
        .def("MotionCost", &Class::MotionCost, py::arg("from"), py::arg("to"),
            "");
  }
  // HolonomicKinematicPlanningSpace
  {
    using Class = HolonomicKinematicPlanningSpace;
    py::class_<Class, SymmetricPlanningSpace<Eigen::VectorXd>> cls(
        m, "HolonomicKinematicPlanningSpace", "");
    cls.def(py::init<std::unique_ptr<CollisionChecker>, const JointLimits&,
                double, uint64_t>(),
           py::arg("collision_checker"), py::arg("joint_limits"),
           py::arg("propagation_step_size"), py::arg("seed"), "")
        .def("Clone", &Class::Clone, "")
        .def("CheckStateValidity", &Class::CheckStateValidity, py::arg("state"),
            "")
        .def("CheckEdgeValidity", &Class::CheckEdgeValidity, py::arg("from"),
            py::arg("to"), "")
        .def("SampleState", &Class::SampleState, "")
        .def("SateDistance", &Class::StateDistance, py::arg("from"),
            py::arg("to"), "")
        .def("Interpolate", &Class::Interpolate, py::arg("from"), py::arg("to"),
            py::arg("ratio"), "")
        .def("Propagate", &Class::Propagate, py::arg("from"), py::arg("to"),
            py::arg("propagation_statistics"), "")
        .def("MotionCost", &Class::MotionCost, py::arg("from"), py::arg("to"),
            "")
        .def("collision_checker", &Class::collision_checker, "")
        .def("mutable_collision_checker", &Class::mutable_collision_checker, "")
        .def("joint_limits", &Class::joint_limits, "")
        .def("SetJointLimits", &Class::SetJointLimits, py::arg("joint_limits"),
            "")
        .def("propagation_step_size", &Class::propagation_step_size, "")
        .def("SetPropagationStepSize", &Class::SetPropagationStepSize,
            py::arg("propagation_step_size"), "");
  }

  // PRM
  {
    using Class = PRMPlanner<T>;
    py::class_<Class::CreationParameters> cls(
        m, "PRMPlannerCreationParameters", "");
    cls.def(py::init<>(), "")
        .def_readwrite(
            "roadmap_size", &Class::CreationParameters::roadmap_size, "")
        .def_readwrite(
            "num_neighbors", &Class::CreationParameters::num_neighbors, "")
        .def_readwrite("max_valid_sample_tries",
            &Class::CreationParameters::max_valid_sample_tries, "")
        .def_readwrite(
            "parallelize", &Class::CreationParameters::parallelize, "")
        .def("__repr__", [](const Class::CreationParameters& self) {
          return py::str(
              "PRMPlannerCreationParameters("
              "roadmap_size={}, "
              "num_neighbors={}, "
              "max_valid_sample_tries={}, "
              "parallelize={})")
              .format(self.roadmap_size, self.num_neighbors,
                  self.max_valid_sample_tries, self.parallelize);
        });
  }

  {
    using Class = PRMPlanner<T>;
    py::class_<Class::QueryParameters> cls(m, "PRMPlannerQueryParameters", "");
    cls.def(py::init<>(), "")
        .def_readwrite(
            "num_neighbors", &Class::QueryParameters::num_neighbors, "")
        .def_readwrite("parallelize", &Class::QueryParameters::parallelize, "")
        .def("__repr__", [](const Class::QueryParameters& self) {
          return py::str(
              "PRMPlannerQueryParameters("
              "num_neighbors={}, "
              "parallelize={})")
              .format(self.num_neighbors, self.parallelize);
        });
  }

  using namespace common_robotics_utilities::openmp_helpers;
  m.def(
      "BuildPRMsplinedBiRRT",
      [](const std::vector<std::pair<T, T>>& spline_pairs,
          const PRMPlanner<T>::CreationParameters& prm_parameters,
          const BiRRTPlanner<T>::Parameters& birrt_parameters,
          PlanningSpace<T>* planning_space, const int32_t prm_threads,
          const int32_t birrt_threads) {
        SteadyTimer timer;

        const std::function<double(const T&, const T&)> state_distance_fn =
            [&](const T& from, const T& to) {
              return planning_space->StateDistanceForwards(from, to);
            };
        const std::function<bool(const T&, const T&)> edge_validity_check_fn =
            [&](const T& from, const T& to) {
              return planning_space->CheckEdgeValidity(from, to);
            };

        ChangeOmpNumThreadsWrapper{birrt_threads};
        timer.Start();

        double max_birrt_runtime = 0;
        double total_birrt_runtime = 0;
        // Plan between each pair of nodes with BiRRT
        // TODO(wrangelvid) this could be parallelized
        std::vector<T> spline_nodes;
        for (const auto& [from, to] : spline_pairs) {
          const double birrt_start_time = timer.Tick();
          auto result =
              BiRRTPlanner<T>::Plan(from, to, birrt_parameters, planning_space);
          double birrt_runtime = timer.Tick() - birrt_start_time;

          if (birrt_runtime > max_birrt_runtime) {
            max_birrt_runtime = birrt_runtime;
          }
          total_birrt_runtime += birrt_runtime;

          if (result.has_solution()) {
            for (const auto& node : result.path()) {
              spline_nodes.push_back(node);
            }
          }
        }

        // Generate a start roadmap
        ChangeOmpNumThreadsWrapper{prm_threads};
        const double prm_start_time = timer.Tick();
        auto roadmap = PRMPlanner<T>::BuildRoadmap(
            prm_parameters, spline_nodes, planning_space);
        const double prm_runtime = timer.Tick() - prm_start_time;

        return std::tuple<Roadmap, double, double, double>(
            roadmap, prm_runtime, total_birrt_runtime, max_birrt_runtime);
      },
      py::arg("post_spline_pairs"), py::arg("prm_parameters"),
      py::arg("birrt_parameters"), py::arg("planning_space"),
      py::arg("prm_threads"), py::arg("birrt_threads"), "");

  {
    using Class = PRMPlanner<T>;
    py::class_<Class> cls(m, "PRMPlanner", "");
    cls.def_static("BuildRoadmap", &Class::BuildRoadmap, py::arg("parameters"),
        py::arg("initial_states"), py::arg("planning_space"), "");
    cls.def_static(
        "TimedBuildRoadmap",
        [](const Class::CreationParameters& parameters,
            const std::vector<T>& initial_states,
            PlanningSpace<T>* planning_space) {
          SteadyTimer timer;
          timer.Start();
          auto result =
              Class::BuildRoadmap(parameters, initial_states, planning_space);
          const double run_time = timer.Tick();
          return std::make_pair(result, run_time);
        },
        py::arg("parameters"), py::arg("initial_states"),
        py::arg("planning_space"), "");

    cls.def_static("GrowRoadmap", &Class::GrowRoadmap, py::arg("roadmap"),
        py::arg("parameters"), py::arg("planning_space"), "");
    cls.def_static(
        "TimedGrowRoadmap",
        [](Roadmap* roadmap, const Class::CreationParameters& parameters,
            PlanningSpace<T>* planning_space) {
          SteadyTimer timer;
          timer.Start();
          auto result = Class::GrowRoadmap(roadmap, parameters, planning_space);
          const double run_time = timer.Tick();
          return std::make_pair(result, run_time);
        },
        py::arg("roadmap"), py::arg("parameters"), py::arg("planning_space"),
        "");

    cls.def_static(
        "TimedAddNodesToRoadmap",
        [](std::vector<T>& nodes, Roadmap* roadmap,
            const Class::CreationParameters& parameters,
            PlanningSpace<T>* planning_space) {
          SteadyTimer timer;
          // Since only valid states are sampled, state validity check is a
          // no-op.
          const std::function<double(const T&, const T&)> state_distance_fn =
              [&](const T& from, const T& to) {
                return planning_space->StateDistanceForwards(from, to);
              };
          const std::function<bool(const T&, const T&)> edge_validity_check_fn =
              [&](const T& from, const T& to) {
                return planning_space->CheckEdgeValidity(from, to);
              };
          const bool parallelize_prm =
              parameters.parallelize && planning_space->supports_parallel();

          timer.Start();
          for (const auto& node : nodes) {
            common_robotics_utilities::simple_prm_planner::AddNodeToRoadmap<T,
                Roadmap>(node,
                common_robotics_utilities::simple_prm_planner::
                    NNDistanceDirection::ROADMAP_TO_NEW_STATE,
                *roadmap, state_distance_fn, edge_validity_check_fn,
                parameters.num_neighbors, roadmap->Size(), parallelize_prm,
                planning_space->is_symmetric(), false);
          }
          const double run_time = timer.Tick();
          return std::make_pair(roadmap, run_time);
        },
        py::arg("nodes"), py::arg("roadmap"), py::arg("parameters"),
        py::arg("planning_space"), "");

    cls.def_static("UpdateRoadmap", &Class::UpdateRoadmap,
        py::arg("planning_space"), py::arg("roadmap"), py::arg("parallelize"),
        "");

    cls.def_static("SaveRoadmapToFile", &Class::SaveRoadmapToFile,
        py::arg("roadmap"), py::arg("filename"), "");

    cls.def_static("LoadRoadmapFromFile", &Class::LoadRoadmapFromFile,
        py::arg("filename"), "");

    cls.def_static("Plan",
        overload_cast_explicit<PathPlanningResult<T>, const T&, const T&,
            const Class::QueryParameters&, const PlanningSpace<T>&,
            const Roadmap&>(&Class::Plan),
        py::arg("start"), py::arg("goal"), py::arg("parameters"),
        py::arg("planning_space"), py::arg("roadmap"), "");

    cls.def_static(
        "TimedPlan",
        [](const T& start, const T& goal,
            const Class::QueryParameters& parameters,
            const PlanningSpace<T>& planning_space, const Roadmap& roadmap) {
          SteadyTimer timer;
          timer.Start();
          auto result =
              Class::Plan(start, goal, parameters, planning_space, roadmap);
          const double run_time = timer.Tick();
          return std::make_pair(result, run_time);
        },
        py::arg("start"), py::arg("goal"), py::arg("parameters"),
        py::arg("planning_space"), py::arg("roadmap"), "");

    cls.def_static("PlanAddingNodes",
        overload_cast_explicit<PathPlanningResult<T>, const T&, const T&,
            const Class::QueryParameters&, const PlanningSpace<T>&, Roadmap*>(
            &Class::PlanAddingNodes),
        py::arg("start"), py::arg("goal"), py::arg("parameters"),
        py::arg("planning_space"), py::arg("roadmap"), "");

    cls.def_static(
        "TimedPlanAddingNodes",
        [](const T& start, const T& goal,
            const Class::QueryParameters& parameters,
            const PlanningSpace<T>& planning_space, Roadmap* roadmap) {
          SteadyTimer timer;
          timer.Start();
          auto result = Class::PlanAddingNodes(
              start, goal, parameters, planning_space, roadmap);
          const double run_time = timer.Tick();
          return std::make_pair(result, run_time);
        },
        py::arg("start"), py::arg("goal"), py::arg("parameters"),
        py::arg("planning_space"), py::arg("roadmap"), "");

    cls.def_static("PlanLazy",
        overload_cast_explicit<PathPlanningResult<T>, const T&, const T&,
            const Class::QueryParameters&, const PlanningSpace<T>&,
            const Roadmap&>(&Class::PlanLazy),
        py::arg("start"), py::arg("goal"), py::arg("parameters"),
        py::arg("planning_space"), py::arg("roadmap"), "");

    cls.def_static(
        "TimedPlanLazy",
        [](const T& start, const T& goal,
            const Class::QueryParameters& parameters,
            const PlanningSpace<T>& planning_space, const Roadmap& roadmap) {
          SteadyTimer timer;
          timer.Start();
          auto result =
              Class::PlanLazy(start, goal, parameters, planning_space, roadmap);
          const double run_time = timer.Tick();
          return std::make_pair(result, run_time);
        },
        py::arg("start"), py::arg("goal"), py::arg("parameters"),
        py::arg("planning_space"), py::arg("roadmap"), "");

    cls.def_static("PlanLazyAddingNodes",
        overload_cast_explicit<PathPlanningResult<T>, const T&, const T&,
            const Class::QueryParameters&, const PlanningSpace<T>&, Roadmap*>(
            &Class::PlanLazyAddingNodes),
        py::arg("start"), py::arg("goal"), py::arg("parameters"),
        py::arg("planning_space"), py::arg("roadmap"), "");

    cls.def_static(
        "TimedPlanLazyAddingNodes",
        [](const T& start, const T& goal,
            const Class::QueryParameters& parameters,
            const PlanningSpace<T>& planning_space, Roadmap* roadmap) {
          SteadyTimer timer;
          timer.Start();
          auto result = Class::PlanLazyAddingNodes(
              start, goal, parameters, planning_space, roadmap);
          const double run_time = timer.Tick();
          return std::make_pair(result, run_time);
        },
        py::arg("start"), py::arg("goal"), py::arg("parameters"),
        py::arg("planning_space"), py::arg("roadmap"), "");
  }

  // BiRRT
  {
    using Class = BiRRTPlanner<T>;
    py::class_<Class::Parameters> cls(m, "BiRRTPlannerParameters", "");
    cls.def(py::init<>(), "")
        .def_readwrite(
            "tree_sampling_bias", &Class::Parameters::tree_sampling_bias, "")
        .def_readwrite("p_switch_trees", &Class::Parameters::p_switch_trees, "")
        .def_readwrite("time_limit", &Class::Parameters::time_limit, "")
        .def_readwrite("connection_tolerance",
            &Class::Parameters::connection_tolerance, "")
        .def_readwrite("prng_seed", &Class::Parameters::prng_seed, "")
        .def_readwrite("parallelize_nearest_neighbor",
            &Class::Parameters::parallelize_nearest_neighbor, "")
        .def("__repr__", [](const Class::Parameters& self) {
          return py::str(
              "ParallelBiRRTPlannerParameters("
              "tree_sampling_bias={}, "
              "p_switch_trees={}, "
              "time_limit={}, "
              "connection_tolerance={}, "
              "prng_seed={}, "
              "parallelize_nearest_neighbor={})")
              .format(self.tree_sampling_bias, self.p_switch_trees,
                  self.time_limit, self.connection_tolerance, self.prng_seed,
                  self.parallelize_nearest_neighbor);
        });
  }

  {
    using Class = BiRRTPlanner<T>;
    py::class_<Class> cls(m, "BiRRTPlanner", "");
    cls.def_static("Plan",
           overload_cast_explicit<PathPlanningResult<T>, const T&, const T&,
               const Class::Parameters&, PlanningSpace<T>*>(&Class::Plan),
           py::arg("start"), py::arg("goal"), py::arg("parameters"),
           py::arg("planning_space"), "")
        .def_static(
            "TimedPlan",
            [](const T& start, const T& goal,
                const Class::Parameters& parameters,
                PlanningSpace<T>* planning_space) {
              SteadyTimer timer;
              timer.Start();
              auto result =
                  Class::Plan(start, goal, parameters, planning_space);
              const double run_time = timer.Tick();
              return std::make_pair(result, run_time);
            },
            py::arg("start"), py::arg("goal"), py::arg("parameters"),
            py::arg("planning_space"), "");
  }

  // Path Processor
  {
    using Class = PathProcessor<T>;
    py::class_<Class::Parameters> cls(m, "PathProcessorParameters", "");
    cls.def(py::init<>(), "")
        .def_readwrite("max_smoothing_shortcut_fraction",
            &Class::Parameters::max_smoothing_shortcut_fraction, "")
        .def_readwrite("resampled_state_interval",
            &Class::Parameters::resampled_state_interval, "")
        .def_readwrite("prng_seed", &Class::Parameters::prng_seed, "")
        .def_readwrite("max_smoothing_iterations",
            &Class::Parameters::max_smoothing_iterations, "")
        .def_readwrite("max_failed_smoothing_iterations",
            &Class::Parameters::max_failed_smoothing_iterations, "")
        .def_readwrite("max_backtracking_steps",
            &Class::Parameters::max_backtracking_steps, "")
        .def_readwrite("use_shortcut_smoothing",
            &Class::Parameters::use_shortcut_smoothing, "")
        .def_readwrite("resample_before_smoothing",
            &Class::Parameters::resample_before_smoothing, "")
        .def_readwrite("resample_before_smoothing",
            &Class::Parameters::resample_before_smoothing, "")
        .def_readwrite(
            "resample_shortcuts", &Class::Parameters::resample_shortcuts, "")
        .def_readwrite("resample_after_smoothing",
            &Class::Parameters::resample_after_smoothing, "")
        .def_readwrite("check_for_marginal_shortcuts",
            &Class::Parameters::check_for_marginal_shortcuts, "")
        .def_readwrite(
            "safety_check_path", &Class::Parameters::safety_check_path, "")
        .def("__repr__", [](const Class::Parameters& self) {
          return py::str(
              "PathProcessingParameters("
              "max_smoothing_shortcut_fraction={}, "
              "resampled_state_interval={}, "
              "prng_seed={}, "
              "max_smoothing_iterations={}, "
              "max_failed_smoothing_iterations={}, "
              "max_backtracking_steps={}, "
              "use_shortcut_smoothing={}, "
              "resample_before_smoothing={}, "
              "resample_shortcuts={}, "
              "resample_after_smoothing={}, "
              "check_for_marginal_shortcuts={}, "
              "safety_check_path={})")
              .format(self.max_smoothing_shortcut_fraction,
                  self.resampled_state_interval, self.prng_seed,
                  self.max_smoothing_iterations,
                  self.max_failed_smoothing_iterations,
                  self.max_backtracking_steps, self.use_shortcut_smoothing,
                  self.resample_before_smoothing, self.resample_shortcuts,
                  self.resample_after_smoothing,
                  self.check_for_marginal_shortcuts, self.safety_check_path);
        });
  }

  {
    using Class = PathProcessor<T>;
    py::class_<Class> cls(m, "PathProcessor", "");
    cls.def_static("ProcessPath", &Class::ProcessPath, py::arg("path"),
        py::arg("parameters"), py::arg("planning_space"), "");
    cls.def_static(
        "TimedProcessPath",
        [](const std::vector<T>& path, const Class::Parameters& parameters,
            const PlanningSpace<T>& planning_space) {
          SteadyTimer timer;
          timer.Start();
          auto result = Class::ProcessPath(path, parameters, planning_space);
          const double run_time = timer.Tick();
          return std::make_pair(result, run_time);
        },
        py::arg("path"), py::arg("parameters"), py::arg("planning_space"), "");
  }

  // Path Planning Result
  {
    using Class = PathPlanningStatusSet;
    py::class_<Class> cls(m, "PathPlanningStatusSet", "");
    cls.def(py::init<>(), "").def("is_success", &Class::is_success, "");
  }

  {
    using Class = PathPlanningResult<T>;
    py::class_<Class> cls(m, "PathPlanningResult", "");
    cls.def(py::init<std::vector<T>, double>(), py::arg("path"),
           py::arg("path_length"), "")
        .def("path", &Class::path, "")
        .def("path_length", &Class::path_length, "")
        .def("status", &Class::status, "")
        .def("has_solution", &Class::has_solution, "");
  }

  // Joint Limits
  {
    using Class = JointLimits;
    py::class_<Class> cls(m, "JointLimits");
    cls.def(py::init<const drake::multibody::MultibodyPlant<double>&, bool,
                bool, bool>(),
           py::arg("plant"), py::arg("require_finite_positions") = false,
           py::arg("require_finite_velocities") = false,
           py::arg("require_finite_accelerations") = false, "")
        .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&,
                 const Eigen::VectorXd&, const Eigen::VectorXd&,
                 const Eigen::VectorXd&, const Eigen::VectorXd&, bool, bool,
                 bool>(),
            py::arg("position_lower"), py::arg("position_upper"),
            py::arg("velocity_lower"), py::arg("velocity_upper"),
            py::arg("acceleration_lower"), py::arg("acceleration_upper"),
            py::arg("require_finite_positions") = false,
            py::arg("require_finite_velocities") = false,
            py::arg("require_finite_accelerations") = false, "");
  }
}

}  // namespace internal
}  // namespace pydrake
}  // namespace drake
