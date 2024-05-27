#pragma once

#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "planning/prm_planner.h"
#include "planning/sphere_robot_model_collision_checker.h"
#include <Eigen/Geometry>
#include <common_robotics_utilities/serialization.hpp>

#include "drake/common/drake_throw.h"

// TODO(calderpg) Support more state types than Eigen::VectorXd.
namespace drake {
namespace planning {

std::vector<float> FlattenConfigurationSphereLocations(
    const std::vector<std::unordered_map<
        drake::geometry::GeometryId, SphereSpecification>>& sphere_locations,
    const std::unordered_set<drake::multibody::BodyIndex>& bodies_to_copy);

/// Assigns unique identifiers to each edge in the provided `roadmap`, where
/// "symmetric" means edge 1->2 receives the same identifier as edge 2->1.
/// @return the number of unique identifiers assigned, these are guaranteed to
/// be contiguous from 1 to the number of distinct edges so that they can be
/// used as an index value. Note that identifier 0 is not used, as this is the
/// default value of the per-edge scratchpad space.
int64_t AssignSymmetricEdgeIdentifiers(
    PRMPlanner<Eigen::VectorXd>::Roadmap* roadmap);

/// Wrapper class for a roadmap, packed sphere model of edges, and packed edge
/// info describing the edges. Note that the contained roadmap and packed edges
/// can only be accessed via const reference to prevent modification. If the
/// roadmap needs to be modified, the only safe answer is to construct a new
/// RoadmapWithPackedEdgeSpheres around the modified roadmap.
class RoadmapWithPackedEdgeSpheres {
 public:
  // Default constructor to build an empty container.
  RoadmapWithPackedEdgeSpheres() {}

  /// This constructor is expensive!
  /// Builds packed edge sphere and edge info using the provided roadmap
  /// `roadmap` and collision checker `collision_checker`. Requires a sphere
  /// model collision checker to access robot sphere model.
  /// For ignored bodies, the edge spheres corresponding to the ignored bodies
  /// are not stored.
  /// Edge sphere pruning reduces the number of spheres stored per edge by
  /// coalescing multiple spheres together within the provided resolution. This
  /// is not enabled by default.
  RoadmapWithPackedEdgeSpheres(
      const PRMPlanner<Eigen::VectorXd>::Roadmap& roadmap,
      const SphereRobotModelCollisionChecker& collision_checker,
      const std::unordered_set<drake::multibody::BodyIndex>& ignored_bodies,
      const std::optional<double>& edge_pruning_resolution = {});

  /// Serialize provided `roadmap_with_edges` into the provided buffer `buffer`.
  /// @return Number of bytes written to `buffer`.
  /// Uses mutable reference and uint64_t to match external API.
  static uint64_t Serialize(
      const RoadmapWithPackedEdgeSpheres& roadmap_with_edges,
      // NOLINTNEXTLINE(runtime/references)
      std::vector<uint8_t>& buffer);

  /// Deserialize from the provided buffer `buffer`, starting from
  /// `starting_offset`.
  /// @return pair<loaded, bytes_read>, where loaded is the loaded
  /// RoadmapWithPackedEdgeSpheres and bytes_read is the number of bytes read
  /// from `buffer`.
  /// Uses uint64_t to match external API.
  static common_robotics_utilities::serialization ::Deserialized<
      RoadmapWithPackedEdgeSpheres>
  Deserialize(const std::vector<uint8_t>& buffer, uint64_t starting_offset);

  /// Save provided `roadmap_with_edges` to the provided file path `filename`.
  static void SaveToFile(const RoadmapWithPackedEdgeSpheres& roadmap_with_edges,
                         const std::string& filename);

  /// Load from the provided file path `filename`.
  static RoadmapWithPackedEdgeSpheres LoadFromFile(const std::string& filename);

  /// Serialize ourself into the provided buffer `buffer`.
  /// @return Number of bytes written to `buffer`.
  /// Uses mutable reference and uint64_t to match external API.
  // NOLINTNEXTLINE(runtime/references)
  uint64_t SerializeSelf(std::vector<uint8_t>& buffer) const;

  /// Deserialize from the provided buffer `buffer`, starting from
  /// `starting_offset`.
  /// @return Number of bytes read from `buffer`.
  /// Uses uint64_t to match external API.
  uint64_t DeserializeSelf(const std::vector<uint8_t>& buffer,
                           uint64_t starting_offset);

  /// Get const reference to contained roadmap.
  const PRMPlanner<Eigen::VectorXd>::Roadmap& roadmap() const {
    return roadmap_;
  }

  /// Get const reference to packed spheres.
  const std::vector<float>& packed_spheres() const { return packed_spheres_; }

  /// Get const reference to packed edge info.
  const std::vector<int32_t>& packed_edge_info() const {
    return packed_edge_info_;
  }

  /// Get number of distinct edges.
  int64_t NumDistinctEdges() const { return packed_edge_info_.size() / 2; }

 private:
  PRMPlanner<Eigen::VectorXd>::Roadmap roadmap_;
  std::vector<float> packed_spheres_;
  std::vector<int32_t> packed_edge_info_;
};

/// Serializes a map of <name, packed roadmap> into a binary buffer.
/// @param named_roadmaps Roadmaps to serialize.
/// @param buffer Buffer to serialize into. Mutable reference to match API in
/// common_robotics_utilities external.
/// @return Number of bytes written. uint64_t to match external API.
uint64_t SerializeNamedPackedRoadmaps(
    const std::map<std::string, RoadmapWithPackedEdgeSpheres>&
        named_packed_roadmaps,
    // NOLINTNEXTLINE(runtime/references)
    std::vector<uint8_t>& buffer);

/// Deserializes a map of <name, packed roadmap> from binary.
/// @param buffer Buffer to deserialze from.
/// @param starting_offset Starting offset into the buffer. uint64_t to match
/// API in common_robotics_utilities external.
/// @return Deserialized<map<name, roadmap>>
common_robotics_utilities::serialization ::Deserialized<
    std::map<std::string, RoadmapWithPackedEdgeSpheres>>
DeserializeNamedPackedRoadmaps(const std::vector<uint8_t>& buffer,
                               uint64_t starting_offset);

/// Serializes and saves a map of <name, packed roadmap> into a file.
/// @param named_packed_roadmaps Roadmaps to serialize.
/// @param filename Path of file to save roadmap.
void SaveNamedPackedRoadmapsToFile(
    const std::map<std::string, RoadmapWithPackedEdgeSpheres>&
        named_packed_roadmaps,
    const std::string& filename);

/// Loads a map of <name, packed roadmap> from a file and deserializes it.
/// @param filename Path of file to load roadmaps.
/// @return Loaded roadmaps.
std::map<std::string, RoadmapWithPackedEdgeSpheres>
LoadNamedPackedRoadmapsFromFile(const std::string& filename);
}  // namespace planning
}  // namespace drake
