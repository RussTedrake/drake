#include "planning/prm_planner.h"

#include <Eigen/Geometry>
#include <common_robotics_utilities/simple_graph_search.hpp>
#include <common_robotics_utilities/simple_prm_planner.hpp>
#include <common_robotics_utilities/zlib_helpers.hpp>

#include "drake/common/drake_throw.h"
#include "drake/common/text_logging.h"

namespace drake {
namespace planning {
// Don't add duplicate states to a roadmap.
const bool kAddDuplicateStates = false;
// Limit duplicate states added to the priority queue in graph search.
const bool kLimitPQueueDuplicates = true;
// Use roadmap overlays, rather than copying the roadmap.
const bool kUseRoadmapOverlay = true;

using common_robotics_utilities::serialization::Deserialized;
using common_robotics_utilities::serialization::DeserializeIsometry3d;
using common_robotics_utilities::serialization::DeserializeMapLike;
using common_robotics_utilities::serialization::DeserializeString;
using common_robotics_utilities::serialization::DeserializeVector2d;
using common_robotics_utilities::serialization::DeserializeVector3d;
using common_robotics_utilities::serialization::DeserializeVectorXd;
using common_robotics_utilities::serialization::MakeDeserialized;
using common_robotics_utilities::serialization::SerializeIsometry3d;
using common_robotics_utilities::serialization::SerializeMapLike;
using common_robotics_utilities::serialization::SerializeString;
using common_robotics_utilities::serialization::SerializeVector2d;
using common_robotics_utilities::serialization::SerializeVector3d;
using common_robotics_utilities::serialization::SerializeVectorXd;
using common_robotics_utilities::simple_astar_search::AstarResult;
using common_robotics_utilities::simple_graph::NonOwningGraphOverlay;
using common_robotics_utilities::simple_graph_search::PerformLazyAstarSearch;
using common_robotics_utilities::simple_prm_planner::AddNodeToRoadmap;
using common_robotics_utilities::simple_prm_planner::BuildRoadMap;
using common_robotics_utilities::simple_prm_planner::ExtractSolution;
using common_robotics_utilities::simple_prm_planner::GrowRoadMap;
using common_robotics_utilities::simple_prm_planner::LazyQueryPath;
using common_robotics_utilities::simple_prm_planner::LazyQueryPathAndAddNodes;
using common_robotics_utilities::simple_prm_planner::NNDistanceDirection;
using common_robotics_utilities::simple_prm_planner::QueryPath;
using common_robotics_utilities::simple_prm_planner::QueryPathAndAddNodes;
using common_robotics_utilities::simple_prm_planner::UpdateRoadMapEdges;
using common_robotics_utilities::zlib_helpers::CompressAndWriteToFile;
using common_robotics_utilities::zlib_helpers::LoadFromFileAndDecompress;

namespace {
// Signature for templated state serialization. Signature matches serialization
// methods in common_robotics_utilities.
template <typename StateType>
uint64_t SerializeStateType(const StateType& state,
                            // NOLINTNEXTLINE(runtime/references)
                            std::vector<uint8_t>& buffer);

// Concrete implementations of state serialization for supported state types.
template <>
uint64_t SerializeStateType(const Eigen::Vector2d& state,
                            // NOLINTNEXTLINE(runtime/references)
                            std::vector<uint8_t>& buffer) {
  return SerializeVector2d(state, buffer);
}

template <>
uint64_t SerializeStateType(const Eigen::Vector3d& state,
                            // NOLINTNEXTLINE(runtime/references)
                            std::vector<uint8_t>& buffer) {
  return SerializeVector3d(state, buffer);
}

template <>
uint64_t SerializeStateType(const drake::math::RigidTransformd& state,
                            // NOLINTNEXTLINE(runtime/references)
                            std::vector<uint8_t>& buffer) {
  return SerializeIsometry3d(state.GetAsIsometry3(), buffer);
}

template <>
uint64_t SerializeStateType(const Eigen::VectorXd& state,
                            // NOLINTNEXTLINE(runtime/references)
                            std::vector<uint8_t>& buffer) {
  return SerializeVectorXd(state, buffer);
}

template <>
uint64_t SerializeStateType(const ControlPlanningState<Eigen::Vector2d>& state,
                            // NOLINTNEXTLINE(runtime/references)
                            std::vector<uint8_t>& buffer) {
  uint64_t bytes_written = SerializeVector2d(state.state(), buffer);
  bytes_written += SerializeVectorXd(state.control(), buffer);
  return bytes_written;
}

template <>
uint64_t SerializeStateType(const ControlPlanningState<Eigen::Vector3d>& state,
                            // NOLINTNEXTLINE(runtime/references)
                            std::vector<uint8_t>& buffer) {
  uint64_t bytes_written = SerializeVector3d(state.state(), buffer);
  bytes_written += SerializeVectorXd(state.control(), buffer);
  return bytes_written;
}

template <>
uint64_t SerializeStateType(
    const ControlPlanningState<drake::math::RigidTransformd>& state,
    // NOLINTNEXTLINE(runtime/references)
    std::vector<uint8_t>& buffer) {
  uint64_t bytes_written =
      SerializeIsometry3d(state.state().GetAsIsometry3(), buffer);
  bytes_written += SerializeVectorXd(state.control(), buffer);
  return bytes_written;
}

template <>
uint64_t SerializeStateType(const ControlPlanningState<Eigen::VectorXd>& state,
                            // NOLINTNEXTLINE(runtime/references)
                            std::vector<uint8_t>& buffer) {
  uint64_t bytes_written = SerializeVectorXd(state.state(), buffer);
  bytes_written += SerializeVectorXd(state.control(), buffer);
  return bytes_written;
}

// Signature for templated state deserialization. Signature matches
// deserialization methods in common_robotics_utilities.
template <typename StateType>
Deserialized<StateType> DeserializeStateType(const std::vector<uint8_t>& buffer,
                                             uint64_t starting_offset);

// Concrete implementations of state deserialization for supported state types.
template <>
Deserialized<Eigen::Vector2d> DeserializeStateType(
    const std::vector<uint8_t>& buffer, uint64_t starting_offset) {
  return DeserializeVector2d(buffer, starting_offset);
}

template <>
Deserialized<Eigen::Vector3d> DeserializeStateType(
    const std::vector<uint8_t>& buffer, uint64_t starting_offset) {
  return DeserializeVector3d(buffer, starting_offset);
}

template <>
Deserialized<drake::math::RigidTransformd> DeserializeStateType(
    const std::vector<uint8_t>& buffer, uint64_t starting_offset) {
  auto deserialized_isometry3d = DeserializeIsometry3d(buffer, starting_offset);
  return MakeDeserialized(
      drake::math::RigidTransformd(deserialized_isometry3d.Value()),
      deserialized_isometry3d.BytesRead());
}

template <>
Deserialized<Eigen::VectorXd> DeserializeStateType(
    const std::vector<uint8_t>& buffer, uint64_t starting_offset) {
  return DeserializeVectorXd(buffer, starting_offset);
}

template <>
Deserialized<ControlPlanningState<Eigen::Vector2d>> DeserializeStateType(
    const std::vector<uint8_t>& buffer, uint64_t starting_offset) {
  uint64_t current_offset = starting_offset;
  auto deserialized_state = DeserializeVector2d(buffer, current_offset);
  current_offset += deserialized_state.BytesRead();
  auto deserialized_control = DeserializeVectorXd(buffer, current_offset);
  current_offset += deserialized_control.BytesRead();
  const uint64_t bytes_read = current_offset - starting_offset;
  return MakeDeserialized(ControlPlanningState(deserialized_state.Value(),
                                               deserialized_control.Value()),
                          bytes_read);
}

template <>
Deserialized<ControlPlanningState<Eigen::Vector3d>> DeserializeStateType(
    const std::vector<uint8_t>& buffer, uint64_t starting_offset) {
  uint64_t current_offset = starting_offset;
  auto deserialized_state = DeserializeVector3d(buffer, current_offset);
  current_offset += deserialized_state.BytesRead();
  auto deserialized_control = DeserializeVectorXd(buffer, current_offset);
  current_offset += deserialized_control.BytesRead();
  const uint64_t bytes_read = current_offset - starting_offset;
  return MakeDeserialized(ControlPlanningState(deserialized_state.Value(),
                                               deserialized_control.Value()),
                          bytes_read);
}

template <>
Deserialized<ControlPlanningState<drake::math::RigidTransformd>>
DeserializeStateType(const std::vector<uint8_t>& buffer,
                     uint64_t starting_offset) {
  uint64_t current_offset = starting_offset;
  auto deserialized_state = DeserializeIsometry3d(buffer, current_offset);
  current_offset += deserialized_state.BytesRead();
  auto deserialized_control = DeserializeVectorXd(buffer, current_offset);
  current_offset += deserialized_control.BytesRead();
  const uint64_t bytes_read = current_offset - starting_offset;
  return MakeDeserialized(ControlPlanningState(drake::math::RigidTransformd(
                                                   deserialized_state.Value()),
                                               deserialized_control.Value()),
                          bytes_read);
}

template <>
Deserialized<ControlPlanningState<Eigen::VectorXd>> DeserializeStateType(
    const std::vector<uint8_t>& buffer, uint64_t starting_offset) {
  uint64_t current_offset = starting_offset;
  auto deserialized_state = DeserializeVectorXd(buffer, current_offset);
  current_offset += deserialized_state.BytesRead();
  auto deserialized_control = DeserializeVectorXd(buffer, current_offset);
  current_offset += deserialized_control.BytesRead();
  const uint64_t bytes_read = current_offset - starting_offset;
  return MakeDeserialized(ControlPlanningState(deserialized_state.Value(),
                                               deserialized_control.Value()),
                          bytes_read);
}

// Helper to construct a PathPlanningResult from an AstarResult.
template <typename StateType>
PathPlanningResult<StateType> MakePathPlanningResult(
    const AstarResult<StateType>& planning_result) {
  if (planning_result.Path().size() > 0) {
    return PathPlanningResult<StateType>(planning_result.Path(),
                                         planning_result.PathCost());
  } else {
    return PathPlanningResult<StateType>(PathPlanningStatus::kCannotFindPath);
  }
}

}  // namespace

template <typename StateType>
typename PRMPlanner<StateType>::Roadmap PRMPlanner<StateType>::BuildRoadmap(
    const CreationParameters& parameters,
    const std::vector<StateType>& initial_states,
    PlanningSpace<StateType>* planning_space) {
  DRAKE_THROW_UNLESS(parameters.roadmap_size > 0);
  DRAKE_THROW_UNLESS(parameters.num_neighbors > 0);
  DRAKE_THROW_UNLESS(parameters.max_valid_sample_tries > 0);
  DRAKE_THROW_UNLESS(planning_space != nullptr);

  // Only sample valid states.
  const std::function<StateType(void)> state_sampling_fn = [&]() {
    return planning_space->SampleValidState(parameters.max_valid_sample_tries);
  };
  // Since only valid states are sampled, state validity check is a no-op.
  const std::function<bool(const StateType&)> state_validity_check_fn =
      [](const StateType&) {
        return true;
      };
  const std::function<double(const StateType&, const StateType&)>
      state_distance_fn = [&](const StateType& from, const StateType& to) {
        return planning_space->StateDistanceForwards(from, to);
      };
  const std::function<bool(const StateType&, const StateType&)>
      edge_validity_check_fn = [&](const StateType& from, const StateType& to) {
        return planning_space->CheckEdgeValidity(from, to);
      };

  const bool parallelize_prm =
      parameters.parallelize && planning_space->supports_parallel();

  return BuildRoadMap<StateType, Roadmap>(
      parameters.roadmap_size, state_sampling_fn, state_distance_fn,
      state_validity_check_fn, edge_validity_check_fn, parameters.num_neighbors,
      1 /* the state sampling function already incorporates multiple tries */,
      initial_states, parallelize_prm, planning_space->is_symmetric(),
      true /* allow duplicate states to be added for performance */);
}

template <typename StateType>
typename PRMPlanner<StateType>::Roadmap PRMPlanner<StateType>::GrowRoadmap(
    Roadmap* roadmap, const CreationParameters& parameters,
    PlanningSpace<StateType>* planning_space) {
  DRAKE_THROW_UNLESS(parameters.roadmap_size > 0);
  DRAKE_THROW_UNLESS(parameters.num_neighbors > 0);
  DRAKE_THROW_UNLESS(parameters.max_valid_sample_tries > 0);
  DRAKE_THROW_UNLESS(planning_space != nullptr);

  // Only sample valid states.
  const std::function<StateType(void)> state_sampling_fn = [&]() {
    return planning_space->SampleValidState(parameters.max_valid_sample_tries);
  };
  // Since only valid states are sampled, state validity check is a no-op.
  const std::function<bool(const StateType&)> state_validity_check_fn =
      [](const StateType&) {
        return true;
      };
  const std::function<double(const StateType&, const StateType&)>
      state_distance_fn = [&](const StateType& from, const StateType& to) {
        return planning_space->StateDistanceForwards(from, to);
      };
  const std::function<bool(const StateType&, const StateType&)>
      edge_validity_check_fn = [&](const StateType& from, const StateType& to) {
        return planning_space->CheckEdgeValidity(from, to);
      };
  const std::function<bool(const int64_t)> termination_check_fn =
      [&](const int64_t current_roadmap_size) {
        return current_roadmap_size >=
               static_cast<int64_t>(parameters.roadmap_size);
      };

  const bool parallelize_prm =
      parameters.parallelize && planning_space->supports_parallel();

  const std::map<std::string, double> roadmap_growth_statistics =
      GrowRoadMap<StateType, Roadmap>(
          *roadmap, state_sampling_fn, state_distance_fn,
          state_validity_check_fn, edge_validity_check_fn, termination_check_fn,
          parameters.num_neighbors, parallelize_prm,
          planning_space->is_symmetric(), kAddDuplicateStates);
  drake::log()->debug(
      "GrowRoadmap statistics:\n{}",
      common_robotics_utilities::print::Print(roadmap_growth_statistics));

  return *roadmap;
}

template <typename StateType>
uint64_t PRMPlanner<StateType>::SerializeRoadmap(
    const Roadmap& roadmap,
    // NOLINTNEXTLINE(runtime/references)
    std::vector<uint8_t>& buffer) {
  return Roadmap::Serialize(roadmap, buffer, SerializeStateType<StateType>);
}

template <typename StateType>
typename PRMPlanner<StateType>::DeserializedRoadmap
PRMPlanner<StateType>::DeserializeRoadmap(const std::vector<uint8_t>& buffer,
                                          const uint64_t starting_offset) {
  return Roadmap::Deserialize(buffer, starting_offset,
                              DeserializeStateType<StateType>);
}

template <typename StateType>
uint64_t PRMPlanner<StateType>::SerializeNamedRoadmaps(
    const NamedRoadmaps& named_roadmaps,
    // NOLINTNEXTLINE(runtime/references)
    std::vector<uint8_t>& buffer) {
  return SerializeMapLike<std::string, Roadmap>(
      named_roadmaps, buffer, SerializeString<char>, SerializeRoadmap);
}

template <typename StateType>
typename PRMPlanner<StateType>::DeserializedNamedRoadmaps
PRMPlanner<StateType>::DeserializeNamedRoadmaps(
    const std::vector<uint8_t>& buffer, const uint64_t starting_offset) {
  return DeserializeMapLike<std::string, Roadmap>(
      buffer, starting_offset, DeserializeString<char>, DeserializeRoadmap);
}

template <typename StateType>
void PRMPlanner<StateType>::SaveRoadmapToFile(const Roadmap& roadmap,
                                              const std::string& filename) {
  std::vector<uint8_t> buffer;
  SerializeRoadmap(roadmap, buffer);
  CompressAndWriteToFile(buffer, filename);
}

template <typename StateType>
typename PRMPlanner<StateType>::Roadmap
PRMPlanner<StateType>::LoadRoadmapFromFile(const std::string& filename) {
  const std::vector<uint8_t> decompressed_serialized_roadmap =
      LoadFromFileAndDecompress(filename);
  const uint64_t starting_offset = 0;
  return DeserializeRoadmap(decompressed_serialized_roadmap, starting_offset)
      .Value();
}

template <typename StateType>
void PRMPlanner<StateType>::SaveNamedRoadmapsToFile(
    const NamedRoadmaps& named_roadmaps, const std::string& filename) {
  std::vector<uint8_t> buffer;
  SerializeNamedRoadmaps(named_roadmaps, buffer);
  CompressAndWriteToFile(buffer, filename);
}

template <typename StateType>
typename PRMPlanner<StateType>::NamedRoadmaps
PRMPlanner<StateType>::LoadNamedRoadmapsFromFile(const std::string& filename) {
  const std::vector<uint8_t> decompressed_serialized_roadmaps =
      LoadFromFileAndDecompress(filename);
  const uint64_t starting_offset = 0;
  return DeserializeNamedRoadmaps(decompressed_serialized_roadmaps,
                                  starting_offset)
      .Value();
}

template <typename StateType>
void PRMPlanner<StateType>::UpdateRoadmap(
    const PlanningSpace<StateType>& planning_space, Roadmap* roadmap,
    const bool parallelize) {
  DRAKE_THROW_UNLESS(roadmap != nullptr);

  const bool parallelize_prm =
      parallelize && planning_space.supports_parallel();

  const std::function<double(const StateType&, const StateType&)>
      state_distance_fn = [&](const StateType& from, const StateType& to) {
        return planning_space.StateDistanceForwards(from, to);
      };
  const std::function<bool(const StateType&, const StateType&)>
      edge_validity_check_fn = [&](const StateType& from, const StateType& to) {
        return planning_space.CheckEdgeValidity(from, to);
      };

  UpdateRoadMapEdges(*roadmap, edge_validity_check_fn, state_distance_fn,
                     parallelize_prm);
}

template <typename StateType>
PathPlanningResult<StateType> PRMPlanner<StateType>::Plan(
    const StateType& start, const StateType& goal,
    const QueryParameters& parameters,
    const PlanningSpace<StateType>& planning_space, const Roadmap& roadmap) {
  return Plan(std::vector<StateType>{start}, std::vector<StateType>{goal},
              parameters, planning_space, roadmap);
}

template <typename StateType>
PathPlanningResult<StateType> PRMPlanner<StateType>::PlanAddingNodes(
    const StateType& start, const StateType& goal,
    const QueryParameters& parameters,
    const PlanningSpace<StateType>& planning_space, Roadmap* roadmap) {
  return PlanAddingNodes(std::vector<StateType>{start},
                         std::vector<StateType>{goal}, parameters,
                         planning_space, roadmap);
}

template <typename StateType>
PathPlanningResult<StateType> PRMPlanner<StateType>::PlanLazy(
    const StateType& start, const StateType& goal,
    const QueryParameters& parameters,
    const PlanningSpace<StateType>& planning_space, const Roadmap& roadmap) {
  return PlanLazy(std::vector<StateType>{start}, std::vector<StateType>{goal},
                  parameters, planning_space, roadmap);
}

template <typename StateType>
PathPlanningResult<StateType> PRMPlanner<StateType>::PlanLazyAddingNodes(
    const StateType& start, const StateType& goal,
    const QueryParameters& parameters,
    const PlanningSpace<StateType>& planning_space, Roadmap* roadmap) {
  return PlanLazyAddingNodes(std::vector<StateType>{start},
                             std::vector<StateType>{goal}, parameters,
                             planning_space, roadmap);
}

template <typename StateType>
PathPlanningResult<StateType> PRMPlanner<StateType>::PlanEdgeValidity(
    const StateType& start, const StateType& goal,
    const QueryParameters& parameters,
    const PlanningSpace<StateType>& planning_space, const Roadmap& roadmap,
    const std::vector<int32_t>& edge_validity_map,
    const StateOverrideFunction& state_override_fn) {
  return PlanEdgeValidity(
      std::vector<StateType>{start}, std::vector<StateType>{goal}, parameters,
      planning_space, roadmap, edge_validity_map, state_override_fn);
}

template <typename StateType>
PathPlanningResult<StateType> PRMPlanner<StateType>::Plan(
    const std::vector<StateType>& starts, const std::vector<StateType>& goals,
    const QueryParameters& parameters,
    const PlanningSpace<StateType>& planning_space, const Roadmap& roadmap) {
  DRAKE_THROW_UNLESS(parameters.num_neighbors >= 0);

  const auto& [valid_starts, valid_goals, status] =
      planning_space.ExtractValidStartsAndGoals(starts, goals);
  if (!status.is_success()) {
    return PathPlanningResult<StateType>(status);
  }

  const bool parallelize_prm =
      parameters.parallelize && planning_space.supports_parallel();

  const std::function<double(const StateType&, const StateType&)>
      state_distance_fn = [&](const StateType& from, const StateType& to) {
        return planning_space.StateDistanceForwards(from, to);
      };
  const std::function<bool(const StateType&, const StateType&)>
      edge_validity_check_fn = [&](const StateType& from, const StateType& to) {
        return planning_space.CheckEdgeValidity(from, to);
      };

  return MakePathPlanningResult<StateType>(QueryPath(
      valid_starts, valid_goals, roadmap, state_distance_fn,
      edge_validity_check_fn, parameters.num_neighbors, parallelize_prm,
      planning_space.is_symmetric(), kAddDuplicateStates,
      kLimitPQueueDuplicates, kUseRoadmapOverlay));
}

template <typename StateType>
PathPlanningResult<StateType> PRMPlanner<StateType>::PlanAddingNodes(
    const std::vector<StateType>& starts, const std::vector<StateType>& goals,
    const QueryParameters& parameters,
    const PlanningSpace<StateType>& planning_space, Roadmap* roadmap) {
  DRAKE_THROW_UNLESS(parameters.num_neighbors >= 0);
  DRAKE_THROW_UNLESS(roadmap != nullptr);

  const auto& [valid_starts, valid_goals, status] =
      planning_space.ExtractValidStartsAndGoals(starts, goals);
  if (!status.is_success()) {
    return PathPlanningResult<StateType>(status);
  }

  const bool parallelize_prm =
      parameters.parallelize && planning_space.supports_parallel();

  const std::function<double(const StateType&, const StateType&)>
      state_distance_fn = [&](const StateType& from, const StateType& to) {
        return planning_space.StateDistanceForwards(from, to);
      };
  const std::function<bool(const StateType&, const StateType&)>
      edge_validity_check_fn = [&](const StateType& from, const StateType& to) {
        return planning_space.CheckEdgeValidity(from, to);
      };

  return MakePathPlanningResult<StateType>(QueryPathAndAddNodes(
      valid_starts, valid_goals, *roadmap, state_distance_fn,
      edge_validity_check_fn, parameters.num_neighbors, parallelize_prm,
      planning_space.is_symmetric(), kAddDuplicateStates,
      kLimitPQueueDuplicates));
}

template <typename StateType>
PathPlanningResult<StateType> PRMPlanner<StateType>::PlanLazy(
    const std::vector<StateType>& starts, const std::vector<StateType>& goals,
    const QueryParameters& parameters,
    const PlanningSpace<StateType>& planning_space, const Roadmap& roadmap) {
  DRAKE_THROW_UNLESS(parameters.num_neighbors >= 0);

  const auto& [valid_starts, valid_goals, status] =
      planning_space.ExtractValidStartsAndGoals(starts, goals);
  if (!status.is_success()) {
    return PathPlanningResult<StateType>(status);
  }

  const bool parallelize_prm =
      parameters.parallelize && planning_space.supports_parallel();

  const std::function<double(const StateType&, const StateType&)>
      state_distance_fn = [&](const StateType& from, const StateType& to) {
        return planning_space.StateDistanceForwards(from, to);
      };
  const std::function<bool(const StateType&, const StateType&)>
      edge_validity_check_fn = [&](const StateType& from, const StateType& to) {
        return planning_space.CheckEdgeValidity(from, to);
      };

  return MakePathPlanningResult<StateType>(LazyQueryPath(
      valid_starts, valid_goals, roadmap, state_distance_fn,
      edge_validity_check_fn, parameters.num_neighbors, parallelize_prm,
      planning_space.is_symmetric(), kAddDuplicateStates,
      kLimitPQueueDuplicates, kUseRoadmapOverlay));
}

template <typename StateType>
PathPlanningResult<StateType> PRMPlanner<StateType>::PlanLazyAddingNodes(
    const std::vector<StateType>& starts, const std::vector<StateType>& goals,
    const QueryParameters& parameters,
    const PlanningSpace<StateType>& planning_space, Roadmap* roadmap) {
  DRAKE_THROW_UNLESS(parameters.num_neighbors >= 0);
  DRAKE_THROW_UNLESS(roadmap != nullptr);

  const auto& [valid_starts, valid_goals, status] =
      planning_space.ExtractValidStartsAndGoals(starts, goals);
  if (!status.is_success()) {
    return PathPlanningResult<StateType>(status);
  }

  const bool parallelize_prm =
      parameters.parallelize && planning_space.supports_parallel();

  const std::function<double(const StateType&, const StateType&)>
      state_distance_fn = [&](const StateType& from, const StateType& to) {
        return planning_space.StateDistanceForwards(from, to);
      };
  const std::function<bool(const StateType&, const StateType&)>
      edge_validity_check_fn = [&](const StateType& from, const StateType& to) {
        return planning_space.CheckEdgeValidity(from, to);
      };

  return MakePathPlanningResult<StateType>(LazyQueryPathAndAddNodes(
      valid_starts, valid_goals, *roadmap, state_distance_fn,
      edge_validity_check_fn, parameters.num_neighbors, parallelize_prm,
      planning_space.is_symmetric(), kAddDuplicateStates,
      kLimitPQueueDuplicates));
}

template <typename StateType>
PathPlanningResult<StateType> PRMPlanner<StateType>::PlanEdgeValidity(
    const std::vector<StateType>& starts, const std::vector<StateType>& goals,
    const QueryParameters& parameters,
    const PlanningSpace<StateType>& planning_space, const Roadmap& roadmap,
    const std::vector<int32_t>& edge_validity_map,
    const StateOverrideFunction& state_override_fn) {
  DRAKE_THROW_UNLESS(parameters.num_neighbors >= 0);
  DRAKE_THROW_UNLESS(state_override_fn != nullptr);

  const auto& [valid_starts, valid_goals, status] =
      planning_space.ExtractValidStartsAndGoals(starts, goals);
  if (!status.is_success()) {
    return PathPlanningResult<StateType>(status);
  }

  const bool parallelize_prm =
      parameters.parallelize && planning_space.supports_parallel();

  using OverlaidRoadmap = NonOwningGraphOverlay<StateType, Roadmap>;
  OverlaidRoadmap overlaid_roadmap(roadmap);

  // Distance and edge validity functions for connecting start and goal states.
  const std::function<double(const StateType&, const StateType&)>
      state_distance_fn = [&](const StateType& from, const StateType& to) {
        const StateType override_from = state_override_fn(from);
        const StateType override_to = state_override_fn(to);
        return planning_space.StateDistanceForwards(override_from, override_to);
      };
  const std::function<bool(const StateType&, const StateType&)>
      edge_validity_check_fn = [&](const StateType& from, const StateType& to) {
        const StateType override_from = state_override_fn(from);
        const StateType override_to = state_override_fn(to);
        return planning_space.CheckEdgeValidity(override_from, override_to);
      };

  // Edge validity check for roadmap edges.
  const std::function<bool(const OverlaidRoadmap&,
                           const typename OverlaidRoadmap::EdgeType&)>
      roadmap_edge_validity_check_fn =
          [&](const OverlaidRoadmap&,
              const typename OverlaidRoadmap::EdgeType& edge) {
            const uint64_t identifier = edge.GetScratchpad();
            if (identifier > 0) {
              // All edge identifiers are >= 1.
              const int32_t validity = edge_validity_map.at(identifier - 1);
              // Edges with validity = 1 are valid
              // Edges with validity = 2 are unknown
              // Edges with validity = 0 are invalid
              return (validity == 1);
            } else {
              // The only edges that don't have identifiers are the ones to the
              // start and goal nodes that have just been added. We know that
              // these edges are collision-free.
              return true;
            }
          };

  // Distance function for edges in the roadmap.
  const std::function<double(const OverlaidRoadmap&,
                             const typename OverlaidRoadmap::EdgeType&)>
      edge_distance_fn = [&](const OverlaidRoadmap&,
                             const typename OverlaidRoadmap::EdgeType& edge) {
        return edge.GetWeight();
      };

  // Heuristic function for nodes in the roadmap.
  const std::function<double(const OverlaidRoadmap&, int64_t, int64_t)>
      heuristic_fn = [&](const OverlaidRoadmap& roadmap_graph,
                         const int64_t from_index, const int64_t to_index) {
        return planning_space.StateDistanceForwards(
            roadmap_graph.GetNodeImmutable(from_index).GetValueImmutable(),
            roadmap_graph.GetNodeImmutable(to_index).GetValueImmutable());
      };

  // Add start states to the roadmap.
  const int64_t pre_starts_size = overlaid_roadmap.Size();
  std::vector<int64_t> start_node_indices;
  for (const StateType& start : valid_starts) {
    const int64_t node_index = AddNodeToRoadmap(
        start, NNDistanceDirection::NEW_STATE_TO_ROADMAP, overlaid_roadmap,
        state_distance_fn, edge_validity_check_fn, parameters.num_neighbors,
        pre_starts_size, parallelize_prm, planning_space.is_symmetric(),
        kAddDuplicateStates);
    start_node_indices.emplace_back(node_index);
  }

  // Add goal states to the roadmap.
  const int64_t pre_goals_size = overlaid_roadmap.Size();
  std::vector<int64_t> goal_node_indices;
  for (const StateType& goal : valid_goals) {
    const int64_t node_index = AddNodeToRoadmap(
        goal, NNDistanceDirection::ROADMAP_TO_NEW_STATE, overlaid_roadmap,
        state_distance_fn, edge_validity_check_fn, parameters.num_neighbors,
        pre_goals_size, parallelize_prm, planning_space.is_symmetric(),
        kAddDuplicateStates);
    goal_node_indices.emplace_back(node_index);
  }

  // Call graph A* to find path.
  const AstarResult<int64_t> astar_result = PerformLazyAstarSearch(
      overlaid_roadmap, start_node_indices, goal_node_indices,
      roadmap_edge_validity_check_fn, edge_distance_fn, heuristic_fn,
      kLimitPQueueDuplicates);

  // Extract the solution path found by A*.
  const auto raw_result = ExtractSolution<StateType, std::vector<StateType>>(
      overlaid_roadmap, astar_result);

  if (raw_result.Path().size() > 0) {
    std::vector<StateType> override_solution_path;
    override_solution_path.reserve(raw_result.Path().size());
    for (const StateType& raw_path_state : raw_result.Path()) {
      override_solution_path.emplace_back(state_override_fn(raw_path_state));
    }
    return PathPlanningResult<StateType>(override_solution_path,
                                         raw_result.PathCost());
  } else {
    return PathPlanningResult<StateType>(PathPlanningStatus::kCannotFindPath);
  }
}

}  // namespace planning
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_PLANNING_STATE_TYPES(
    class ::drake::planning::PRMPlanner)
