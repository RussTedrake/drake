#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "planning/default_state_types.h"
#include "planning/path_planning_result.h"
#include "planning/planning_space.h"
#include <common_robotics_utilities/serialization.hpp>
#include <common_robotics_utilities/simple_graph.hpp>

namespace drake {
namespace planning {
/// PRM planner.
template <typename StateType>
class PRMPlanner {
 public:
  /// Parameters for roadmap creation.
  // TODO(calderpg) Provide/document good defaults.
  struct CreationParameters {
    /// Size of the roadmap. @pre > 0.
    int roadmap_size{0};
    /// Number of neighbors to consider when creating the roadmap. @pre > 0.
    int num_neighbors{0};
    /// Maximum number of tries to sample a valid state. @pre > 0.
    int max_valid_sample_tries{0};
    /// Should roadmap creation be parallelized? To be performed in parallel
    /// both this parameter must be true, and the planning space must support
    /// parallel operations.
    bool parallelize{false};
  };

  /// Parameters for roadmap planning queries.
  // TODO(calderpg) Provide/document good defaults.
  struct QueryParameters {
    /// Number of neighbors to consider when connecting the start and goal
    /// states to the roadmap. @pre > 0.
    int num_neighbors{0};
    /// Should query operations be parallelized? To be performed in parallel
    /// both this parameter must be true, and the planning space must support
    /// parallel operations.
    bool parallelize{false};
  };

  using Roadmap = common_robotics_utilities::simple_graph::Graph<StateType>;
  using RoadmapEdge = typename Roadmap::EdgeType;
  using NamedRoadmaps = std::map<std::string, Roadmap>;
  using DeserializedRoadmap =
      common_robotics_utilities::serialization::Deserialized<Roadmap>;
  using DeserializedNamedRoadmaps =
      common_robotics_utilities::serialization::Deserialized<NamedRoadmaps>;

  using StateOverrideFunction = std::function<StateType(const StateType&)>;

  /// Create a roadmap using the "build roadmap" strategy that first samples all
  /// roadmap states, then connects them together. This approach is much faster
  /// than the "grow roadmap" strategy, but *must only* be used in spaces that
  /// do not sample duplicate states. If duplicate states are sampled, all the
  /// duplicate states will be added to the roadmap.
  /// @param parameters Parameters for roadmap creation.
  /// @param planning_space Planning space to use. @pre non null.
  /// @return Roadmap.
  static Roadmap BuildRoadmap(const CreationParameters& parameters,
                              const std::vector<StateType>& initial_states,
                              PlanningSpace<StateType>* planning_space);

  /// Create a roadmap using the "grow roadmap" strategy that iteratively adds
  /// states to the roadmap. This is slower than the "build roadmap" strategy,
  /// but is more generally applicable.
  /// @param roadmap Roadmap to grow. @pre non null.
  /// @param parameters Parameters for roadmap creation.
  /// @param planning_space Planning space to use. @pre non null.
  /// @return Roadmap.
  static Roadmap GrowRoadmap(Roadmap* roadmap,
                             const CreationParameters& parameters,
                             PlanningSpace<StateType>* planning_space);

  /// Serialize the provided roadmap into the provided buffer. API to match
  /// common_robotics_utilities.
  /// @param roadmap Roadmap to serialize.
  /// @param buffer Buffer to serialize into.
  /// @return Number of bytes written into buffer.
  static uint64_t SerializeRoadmap(const Roadmap& roadmap,
                                   // NOLINTNEXTLINE(runtime/references)
                                   std::vector<uint8_t>& buffer);

  /// Deserialize a roadmap from the provided buffer.
  /// @param buffer Buffer to deserialize from.
  /// @param starting_offset Starting offset in buffer.
  /// @return Deserialized roadmap and number of bytes read from buffer.
  static DeserializedRoadmap DeserializeRoadmap(
      const std::vector<uint8_t>& buffer, uint64_t starting_offset);

  /// Serialize the provided named roadmaps into the provided buffer. API to
  /// match common_robotics_utilities.
  /// @param named_roadmaps Named roadmaps to serialize.
  /// @param buffer Buffer to serialize into.
  /// @return Number of bytes written into buffer.
  static uint64_t SerializeNamedRoadmaps(const NamedRoadmaps& named_roadmaps,
                                         // NOLINTNEXTLINE(runtime/references)
                                         std::vector<uint8_t>& buffer);

  /// Deserialize named roadmaps from the provided buffer.
  /// @param buffer Buffer to deserialize from.
  /// @param starting_offset Starting offset in buffer.
  /// @return Deserialized named roadmaps and number of bytes read from buffer.
  static DeserializedNamedRoadmaps DeserializeNamedRoadmaps(
      const std::vector<uint8_t>& buffer, uint64_t starting_offset);

  /// Save the provided roadmap to the specified file.
  /// @param roadmap Roadmap to save.
  /// @param filename Filename to write to.
  static void SaveRoadmapToFile(const Roadmap& roadmap,
                                const std::string& filename);

  /// Load a roadmap from the specified file.
  /// @param filename File to read from.
  /// @return Loaded roadmap.
  static Roadmap LoadRoadmapFromFile(const std::string& filename);

  /// Save the provided named roadmaps to the specified file.
  /// @param roadmap Named roadmaps to save.
  /// @param filename Filename to write to.
  static void SaveNamedRoadmapsToFile(const NamedRoadmaps& named_roadmaps,
                                      const std::string& filename);

  /// Load named roadmaps from the specified file.
  /// @param filename File to read from.
  /// @return Loaded named roadmaps.
  static NamedRoadmaps LoadNamedRoadmapsFromFile(const std::string& filename);

  /// Update the validity and edge weights of edges in the provided roadmap.
  /// Invalid edges are set to infinite weight.
  /// @param planning_space Planning space to use.
  /// @param roadmap Roadmap to update. @pre non null.
  /// @param parallelize Should update be parallelized? To be performed in
  /// parallel both this parameter must be true, and the planning space must
  /// support parallel operations.
  static void UpdateRoadmap(const PlanningSpace<StateType>& planning_space,
                            Roadmap* roadmap, bool parallelize = true);

  /// Plans a path through the provided roadmap from the provided start state to
  /// the provided goal state.
  /// @param start Starting state.
  /// @param goal Ending state.
  /// @param parameters Parameters to planner.
  /// @param planning_space Planning space to use.
  /// @param roadmap Roadmap to use.
  /// @return Shortest path between start and goal, if one exists.
  static PathPlanningResult<StateType> Plan(
      const StateType& start, const StateType& goal,
      const QueryParameters& parameters,
      const PlanningSpace<StateType>& planning_space, const Roadmap& roadmap);

  /// Plans a path through the provided roadmap from the provided start states
  /// to the provided goal states.
  /// @param starts Starting states.
  /// @param goals Ending states.
  /// @param parameters Parameters to planner.
  /// @param planning_space Planning space to use.
  /// @param roadmap Roadmap to use.
  /// @return Shortest path between *a* start and *a* goal, if one exists.
  static PathPlanningResult<StateType> Plan(
      const std::vector<StateType>& starts, const std::vector<StateType>& goals,
      const QueryParameters& parameters,
      const PlanningSpace<StateType>& planning_space, const Roadmap& roadmap);

  /// Plans a path through the provided roadmap from the provided start state to
  /// the provided goal state. Start and goal states are added to the roadmap.
  /// @param start Starting state.
  /// @param goal Ending state.
  /// @param parameters Parameters to planner.
  /// @param planning_space Planning space to use.
  /// @param roadmap Roadmap to use. @pre non null.
  /// @return Shortest path between start and goal, if one exists.
  static PathPlanningResult<StateType> PlanAddingNodes(
      const StateType& start, const StateType& goal,
      const QueryParameters& parameters,
      const PlanningSpace<StateType>& planning_space, Roadmap* roadmap);

  /// Plans a path through the provided roadmap from the provided start states
  /// to the provided goal states. Start and goal states are added to the
  /// roadmap.
  /// @param starts Starting states.
  /// @param goals Ending states.
  /// @param parameters Parameters to planner.
  /// @param planning_space Planning space to use.
  /// @param roadmap Roadmap to use. @pre non null.
  /// @return Shortest path between *a* start and *a* goal, if one exists.
  static PathPlanningResult<StateType> PlanAddingNodes(
      const std::vector<StateType>& starts, const std::vector<StateType>& goals,
      const QueryParameters& parameters,
      const PlanningSpace<StateType>& planning_space, Roadmap* roadmap);

  /// Plans a path through the provided roadmap from the provided start state to
  /// the provided goal state. Lazy planning means that edge feasiblity is
  /// queried as-needed by the roadmap search, and the edges of the provided
  /// roadmap are not assumed to be feasible.
  /// @param start Starting state.
  /// @param goal Ending state.
  /// @param parameters Parameters to planner.
  /// @param planning_space Planning space to use.
  /// @param roadmap Roadmap to use.
  /// @return Shortest path between start and goal, if one exists.
  static PathPlanningResult<StateType> PlanLazy(
      const StateType& start, const StateType& goal,
      const QueryParameters& parameters,
      const PlanningSpace<StateType>& planning_space, const Roadmap& roadmap);

  /// Plans a path through the provided roadmap from the provided start states
  /// to the provided goal states. Lazy planning means that edge feasiblity is
  /// queried as-needed by the roadmap search, and the edges of the provided
  /// roadmap are not assumed to be feasible.
  /// @param starts Starting states.
  /// @param goals Ending states.
  /// @param parameters Parameters to planner.
  /// @param planning_space Planning space to use.
  /// @param roadmap Roadmap to use.
  /// @return Shortest path between *a* start and *a* goal, if one exists.
  static PathPlanningResult<StateType> PlanLazy(
      const std::vector<StateType>& starts, const std::vector<StateType>& goals,
      const QueryParameters& parameters,
      const PlanningSpace<StateType>& planning_space, const Roadmap& roadmap);

  /// Plans a path through the provided roadmap from the provided start state to
  /// the provided goal state. Start and goal states are added to the roadmap.
  /// Lazy planning means that edge feasiblity is queried as-needed by the
  /// roadmap search, and the edges of the provided roadmap are not assumed to
  /// be feasible.
  /// @param start Starting state.
  /// @param goal Ending state.
  /// @param parameters Parameters to planner.
  /// @param planning_space Planning space to use.
  /// @param roadmap Roadmap to use. @pre non null.
  /// @return Shortest path between start and goal, if one exists.
  static PathPlanningResult<StateType> PlanLazyAddingNodes(
      const StateType& start, const StateType& goal,
      const QueryParameters& parameters,
      const PlanningSpace<StateType>& planning_space, Roadmap* roadmap);

  /// Plans a path through the provided roadmap from the provided start states
  /// to the provided goal states. Start and goal states are added to the
  /// roadmap. Lazy planning means that edge feasiblity is queried as-needed by
  /// the graph search, and the edges of the provided roadmap are not assumed to
  /// be feasible.
  /// @param starts Starting states.
  /// @param goals Ending states.
  /// @param parameters Parameters to planner.
  /// @param planning_space Planning space to use.
  /// @param roadmap Roadmap to use. @pre non null.
  /// @return Shortest path between *a* start and *a* goal, if one exists.
  static PathPlanningResult<StateType> PlanLazyAddingNodes(
      const std::vector<StateType>& starts, const std::vector<StateType>& goals,
      const QueryParameters& parameters,
      const PlanningSpace<StateType>& planning_space, Roadmap* roadmap);

  /// (Advanced)
  /// Plans a path through the provided roadmap from the provided start state to
  /// the provided goal state, using the provided edge validity map to identify
  /// which edges in the roadmap are valid. Note that no "add node" version may
  /// be provided, as the edge validity mechanism does not support modifications
  /// to the roadmap.
  /// @param start Starting state.
  /// @param goal Ending state.
  /// @param parameters Parameters to planner.
  /// @param planning_space Planning space to use.
  /// @param roadmap Roadmap to use.
  /// @param edge_validity_map Vector of edge feasibility (feasible = 1,
  /// colliding = 0, unknown = 2) for each distinct edge in the roadmap.
  /// Note int32_t is used as edge_validity_map generally is processed by GPU.
  /// @param state_override_fn Function to override states stored in roadmap.
  /// By default, the override function does nothing.
  /// @return Shortest path between start and goal, if one exists.
  static PathPlanningResult<StateType> PlanEdgeValidity(
      const StateType& start, const StateType& goal,
      const QueryParameters& parameters,
      const PlanningSpace<StateType>& planning_space, const Roadmap& roadmap,
      const std::vector<int32_t>& edge_validity_map,
      const StateOverrideFunction& state_override_fn =
          [](const StateType& state) {
            return state;
          });

  /// (Advanced)
  /// Plans a path through the provided roadmap from the provided start states
  /// to the provided goal states, using the provided edge validity map to
  /// identify which edges in the roadmap are valid. Note that no "add node"
  /// version may be provided, as the edge validity mechanism does not support
  /// modifications to the roadmap.
  /// @param starts Starting states.
  /// @param goals Ending states.
  /// @param parameters Parameters to planner.
  /// @param planning_space Planning space to use.
  /// @param roadmap Roadmap to use.
  /// @param edge_validity_map Vector of edge feasibility (feasible = 1,
  /// colliding = 0, unknown = 2) for each distinct edge in the roadmap.
  /// Note int32_t is used as edge_validity_map generally is processed by GPU.
  /// @param state_override_fn Function to override states stored in roadmap.
  /// By default, the override function does nothing.
  /// @return Shortest path between *a* start and *a* goal, if one exists.
  static PathPlanningResult<StateType> PlanEdgeValidity(
      const std::vector<StateType>& starts, const std::vector<StateType>& goals,
      const QueryParameters& parameters,
      const PlanningSpace<StateType>& planning_space, const Roadmap& roadmap,
      const std::vector<int32_t>& edge_validity_map,
      const StateOverrideFunction& state_override_fn =
          [](const StateType& state) {
            return state;
          });

  // Delete all constructors of this static-only class.
  PRMPlanner(const PRMPlanner&) = delete;
};
}  // namespace planning
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_PLANNING_STATE_TYPES(
    class ::drake::planning::PRMPlanner)
