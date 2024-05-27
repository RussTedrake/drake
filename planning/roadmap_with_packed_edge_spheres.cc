#include "planning/roadmap_with_packed_edge_spheres.h"

#include <algorithm>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "planning/sampling_based_planners.h"
#include "planning/sphere_robot_model_collision_checker.h"
#include <Eigen/Geometry>
#include <common_robotics_utilities/serialization.hpp>
#include <common_robotics_utilities/voxel_grid.hpp>
#include <common_robotics_utilities/zlib_helpers.hpp>

#include "drake/common/drake_throw.h"
#include "drake/common/text_logging.h"

namespace drake {
namespace planning {
using RoadmapEdge = typename PRMPlanner<Eigen::VectorXd>::RoadmapEdge;
using drake::geometry::GeometryId;
using drake::multibody::BodyIndex;

namespace {
common_robotics_utilities::voxel_grid::GridIndex ComputeVirtualGridIndex(
    const Eigen::Vector4d& p_WSo, double virtual_grid_resolution) {
  const int64_t x_index =
      static_cast<int64_t>(p_WSo(0) / virtual_grid_resolution);
  const int64_t y_index =
      static_cast<int64_t>(p_WSo(1) / virtual_grid_resolution);
  const int64_t z_index =
      static_cast<int64_t>(p_WSo(2) / virtual_grid_resolution);
  return common_robotics_utilities::voxel_grid::GridIndex(x_index, y_index,
                                                          z_index);
}

SphereSpecification CoalesceBinSpheres(
    const std::vector<SphereSpecification>& bin_spheres,
    double edge_pruning_resolution) {
  DRAKE_THROW_UNLESS(bin_spheres.size() > 0);
  if (bin_spheres.size() > 1) {
    auto compare_fn = [](const SphereSpecification& a,
                         const SphereSpecification& b) {
      return (a.Radius() < b.Radius());
    };
    const auto largest_sphere =
        std::max_element(bin_spheres.begin(), bin_spheres.end(), compare_fn);
    DRAKE_THROW_UNLESS(largest_sphere != bin_spheres.end());
    return SphereSpecification(
        largest_sphere->Origin(),
        largest_sphere->Radius() + (edge_pruning_resolution * std::sqrt(3.0)));
  } else {
    return bin_spheres.at(0);
  }
}

std::vector<SphereSpecification> PruneEdgeSpheres(
    const std::vector<SphereSpecification>& complete_edge_spheres,
    double edge_pruning_resolution) {
  std::unordered_map<common_robotics_utilities::voxel_grid::GridIndex,
                     std::vector<SphereSpecification>>
      sphere_binning_map;
  for (const auto& sphere : complete_edge_spheres) {
    const auto virtual_grid_index =
        ComputeVirtualGridIndex(sphere.Origin(), edge_pruning_resolution);
    sphere_binning_map[virtual_grid_index].push_back(sphere);
  }
  std::vector<SphereSpecification> pruned_spheres;
  for (const auto& bin : sphere_binning_map) {
    const auto bin_sphere =
        CoalesceBinSpheres(bin.second, edge_pruning_resolution);
    pruned_spheres.push_back(bin_sphere);
  }
  drake::log()->debug(
      "Coalesced {} edge spheres into {} bins and {} pruned edge spheres",
      complete_edge_spheres.size(), sphere_binning_map.size(),
      pruned_spheres.size());
  return pruned_spheres;
}

std::vector<SphereSpecification> GenerateEdgeSpheres(
    const PRMPlanner<Eigen::VectorXd>::Roadmap& roadmap,
    const RoadmapEdge& edge,
    const SphereRobotModelCollisionChecker& collision_checker,
    const std::unordered_set<BodyIndex>& ignored_bodies,
    const std::optional<double>& edge_pruning_resolution) {
  const int64_t lower_node_index =
      std::min(edge.GetFromIndex(), edge.GetToIndex());
  const int64_t upper_node_index =
      std::max(edge.GetFromIndex(), edge.GetToIndex());
  const auto& lower_node = roadmap.GetNodeImmutable(lower_node_index);
  const auto& upper_node = roadmap.GetNodeImmutable(upper_node_index);
  const Eigen::VectorXd& lower_q = lower_node.GetValueImmutable();
  const Eigen::VectorXd& upper_q = upper_node.GetValueImmutable();
  std::vector<SphereSpecification> edge_spheres;
  const double distance =
      collision_checker.ComputeConfigurationDistance(lower_q, upper_q);
  const double step_size = collision_checker.edge_step_size();
  const int32_t num_steps =
      static_cast<int32_t>(std::max(1.0, std::ceil(distance / step_size)));
  for (int32_t step = 0; step <= num_steps; step++) {
    const double ratio =
        static_cast<double>(step) / static_cast<double>(num_steps);
    const Eigen::VectorXd qinterp =
        collision_checker.InterpolateBetweenConfigurations(lower_q, upper_q,
                                                           ratio);
    const std::vector<Eigen::Isometry3d> X_WB_set =
        collision_checker.ComputeBodyPoses(qinterp);
    const std::vector<std::unordered_map<GeometryId, SphereSpecification>>
        spheres_in_world_frame =
            collision_checker.ComputeSphereLocationsInWorldFrame(X_WB_set);
    // Skip body 0, which is world
    for (BodyIndex body_index(1);
         body_index < static_cast<int32_t>(spheres_in_world_frame.size());
         body_index++) {
      // We filter out spheres that belong to links that have allowed collision
      // with the world/environment (body 0) and that belong to ignored bodies.
      if (!collision_checker.IsCollisionFilteredBetween(BodyIndex(0),
                                                        body_index) &&
          ignored_bodies.count(body_index) == 0) {
        const auto& body_spheres = spheres_in_world_frame.at(body_index);
        for (const auto& [sphere_id, sphere] : body_spheres) {
          drake::unused(sphere_id);
          edge_spheres.emplace_back(sphere);
        }
      }
    }
  }
  if (edge_pruning_resolution) {
    return PruneEdgeSpheres(edge_spheres, edge_pruning_resolution.value());
  } else {
    return edge_spheres;
  }
}

// Generates an identifier for edges such that edge 1->2 and edge 2->1 receive
// the same identifier.
uint64_t MakeKeyFromEdge(const RoadmapEdge& edge) {
  const uint64_t lower_half =
      static_cast<uint64_t>(std::min(edge.GetFromIndex(), edge.GetToIndex()));
  const uint64_t upper_half =
      static_cast<uint64_t>(std::max(edge.GetFromIndex(), edge.GetToIndex()));
  DRAKE_THROW_UNLESS(lower_half <
                     static_cast<size_t>(std::numeric_limits<uint32_t>::max()));
  DRAKE_THROW_UNLESS(upper_half <
                     static_cast<size_t>(std::numeric_limits<uint32_t>::max()));
  const uint64_t combined =
      static_cast<uint64_t>(upper_half << 32) |
      static_cast<uint64_t>(lower_half & 0x00000000ffffffff);
  return combined;
}
}  // namespace

std::vector<float> FlattenConfigurationSphereLocations(
    const std::vector<std::unordered_map<GeometryId, SphereSpecification>>&
        sphere_locations,
    const std::unordered_set<BodyIndex>& bodies_to_copy) {
  std::vector<float> packed_spheres;
  for (BodyIndex idx(0); idx < sphere_locations.size(); idx++) {
    const auto& link_spheres = sphere_locations.at(idx);
    if (bodies_to_copy.count(idx) > 0) {
      for (const auto& [sphere_id, sphere] : link_spheres) {
        drake::unused(sphere_id);
        const Eigen::Vector4d& p_WSo = sphere.Origin();
        const double radius = sphere.Radius();
        packed_spheres.push_back(static_cast<float>(p_WSo(0)));
        packed_spheres.push_back(static_cast<float>(p_WSo(1)));
        packed_spheres.push_back(static_cast<float>(p_WSo(2)));
        packed_spheres.push_back(static_cast<float>(radius));
      }
    }
  }
  return packed_spheres;
}

int64_t AssignSymmetricEdgeIdentifiers(
    PRMPlanner<Eigen::VectorXd>::Roadmap* roadmap) {
  DRAKE_THROW_UNLESS(roadmap != nullptr);
  std::unordered_map<uint64_t, uint64_t> tracking_map;
  for (int64_t idx = 0; idx < static_cast<int64_t>(roadmap->Size()); idx++) {
    auto& node = roadmap->GetNodeMutable(idx);
    auto& in_edges = node.GetInEdgesMutable();
    for (auto& in_edge : in_edges) {
      const uint64_t key = MakeKeyFromEdge(in_edge);
      auto found_itr = tracking_map.find(key);
      if (found_itr != tracking_map.end()) {
        // If we've already encountered this edge before, we already have an id.
        in_edge.SetScratchpad(found_itr->second);
      } else {
        // If we haven't seen it before, we assign a new id to size + 1.
        const uint64_t new_id = tracking_map.size() + 1;
        tracking_map[key] = new_id;
        in_edge.SetScratchpad(new_id);
      }
    }
    auto& out_edges = node.GetOutEdgesMutable();
    for (auto& out_edge : out_edges) {
      const uint64_t key = MakeKeyFromEdge(out_edge);
      auto found_itr = tracking_map.find(key);
      if (found_itr != tracking_map.end()) {
        // If we've already encountered this edge before, we already have an id.
        out_edge.SetScratchpad(found_itr->second);
      } else {
        // If we haven't seen it before, we assign a new id to size + 1.
        const uint64_t new_id = tracking_map.size() + 1;
        tracking_map[key] = new_id;
        out_edge.SetScratchpad(new_id);
      }
    }
  }
  const int64_t num_distinct_edges = static_cast<int64_t>(tracking_map.size());
  drake::log()->info("{} distinct edges identified in roadmap with {} nodes",
                     num_distinct_edges, roadmap->Size());
  return num_distinct_edges;
}

RoadmapWithPackedEdgeSpheres::RoadmapWithPackedEdgeSpheres(
    const PRMPlanner<Eigen::VectorXd>::Roadmap& roadmap,
    const SphereRobotModelCollisionChecker& collision_checker,
    const std::unordered_set<BodyIndex>& ignored_bodies,
    const std::optional<double>& edge_pruning_resolution)
    : roadmap_(roadmap) {
  if (edge_pruning_resolution) {
    DRAKE_THROW_UNLESS(edge_pruning_resolution.value() > 0.0);
  }
  // Assign edge identifiers to the roadmap and get number of unique edges.
  const int64_t num_distinct_edges = AssignSymmetricEdgeIdentifiers(&roadmap_);
  std::vector<std::vector<SphereSpecification>> per_edge_spheres(
      num_distinct_edges);
  // Compute edge spheres
  for (int64_t idx = 0; idx < static_cast<int64_t>(roadmap_.Size()); idx++) {
    const auto& node = roadmap_.GetNodeImmutable(idx);
    const auto& in_edges = node.GetInEdgesImmutable();
    for (auto& in_edge : in_edges) {
      const uint64_t edge_identifier = in_edge.GetScratchpad();
      const uint64_t edge_num = edge_identifier - 1;
      // If the per-edge spheres aren't empty, we've already processed it.
      if (per_edge_spheres.at(edge_num).empty()) {
        per_edge_spheres.at(edge_num) =
            GenerateEdgeSpheres(roadmap_, in_edge, collision_checker,
                                ignored_bodies, edge_pruning_resolution);
      }
    }
    const auto& out_edges = node.GetOutEdgesImmutable();
    for (auto& out_edge : out_edges) {
      const uint64_t edge_identifier = out_edge.GetScratchpad();
      const uint64_t edge_num = edge_identifier - 1;
      // If the per-edge spheres aren't empty, we've already processed it.
      if (per_edge_spheres.at(edge_num).empty()) {
        per_edge_spheres.at(edge_num) =
            GenerateEdgeSpheres(roadmap_, out_edge, collision_checker,
                                ignored_bodies, edge_pruning_resolution);
      }
    }
  }
  // Pack spheres and edge info
  for (const auto& edge_spheres : per_edge_spheres) {
    // Set the edge info
    DRAKE_THROW_UNLESS(
        packed_spheres_.size() <
        static_cast<size_t>(std::numeric_limits<int32_t>::max()));
    const int32_t starting_offset =
        static_cast<int32_t>(packed_spheres_.size());
    const int32_t num_edge_spheres = static_cast<int32_t>(edge_spheres.size());
    packed_edge_info_.push_back(starting_offset);
    packed_edge_info_.push_back(num_edge_spheres);
    // Pack the edge spheres
    for (const auto& sphere : edge_spheres) {
      const Eigen::Vector4d& p_WSo = sphere.Origin();
      const double radius = sphere.Radius();
      packed_spheres_.push_back(static_cast<float>(p_WSo(0)));
      packed_spheres_.push_back(static_cast<float>(p_WSo(1)));
      packed_spheres_.push_back(static_cast<float>(p_WSo(2)));
      packed_spheres_.push_back(static_cast<float>(radius));
    }
  }
  DRAKE_THROW_UNLESS(packed_edge_info_.size() ==
                     static_cast<size_t>(num_distinct_edges * 2));
  if (edge_pruning_resolution) {
    drake::log()->info("Packed roadmap into {} spheres (with pruning {})",
                       packed_spheres_.size() / 4,
                       edge_pruning_resolution.value());
  } else {
    drake::log()->info("Packed roadmap into {} spheres (without pruning)",
                       packed_spheres_.size() / 4);
  }
  drake::log()->info("Packed spheres size {} bytes, packed edge info {} bytes",
                     packed_spheres_.size() * sizeof(float),
                     packed_edge_info_.size() * sizeof(int32_t));
}

uint64_t RoadmapWithPackedEdgeSpheres::Serialize(
    const RoadmapWithPackedEdgeSpheres& roadmap_with_edges,
    // NOLINTNEXTLINE(runtime/references)
    std::vector<uint8_t>& buffer) {
  return roadmap_with_edges.SerializeSelf(buffer);
}

common_robotics_utilities::serialization ::Deserialized<
    RoadmapWithPackedEdgeSpheres>
RoadmapWithPackedEdgeSpheres::Deserialize(const std::vector<uint8_t>& buffer,
                                          const uint64_t starting_offset) {
  RoadmapWithPackedEdgeSpheres roadmap_with_edges;
  const uint64_t bytes_read =
      roadmap_with_edges.DeserializeSelf(buffer, starting_offset);
  return common_robotics_utilities::serialization::MakeDeserialized(
      roadmap_with_edges, bytes_read);
}

void RoadmapWithPackedEdgeSpheres::SaveToFile(
    const RoadmapWithPackedEdgeSpheres& roadmap_with_edges,
    const std::string& filename) {
  std::vector<uint8_t> buffer;
  Serialize(roadmap_with_edges, buffer);
  common_robotics_utilities::zlib_helpers::CompressAndWriteToFile(buffer,
                                                                  filename);
}

RoadmapWithPackedEdgeSpheres RoadmapWithPackedEdgeSpheres::LoadFromFile(
    const std::string& filename) {
  const std::vector<uint8_t> decompressed_serialized_roadmap_with_edges =
      common_robotics_utilities::zlib_helpers::LoadFromFileAndDecompress(
          filename);
  const uint64_t starting_offset = 0;
  return Deserialize(decompressed_serialized_roadmap_with_edges,
                     starting_offset)
      .Value();
}

uint64_t RoadmapWithPackedEdgeSpheres::SerializeSelf(
    // NOLINTNEXTLINE(runtime/references)
    std::vector<uint8_t>& buffer) const {
  const uint64_t start_buffer_size = buffer.size();
  PRMPlanner<Eigen::VectorXd>::SerializeRoadmap(roadmap(), buffer);
  common_robotics_utilities::serialization ::SerializeMemcpyableVectorLike<
      float>(packed_spheres(), buffer);
  common_robotics_utilities::serialization ::SerializeMemcpyableVectorLike<
      int32_t>(packed_edge_info(), buffer);
  const uint64_t end_buffer_size = buffer.size();
  const uint64_t bytes_written = end_buffer_size - start_buffer_size;
  drake::log()->info("Serialized roadmap + edges with {} nodes into {} bytes",
                     roadmap().Size(), bytes_written);
  return bytes_written;
}

uint64_t RoadmapWithPackedEdgeSpheres::DeserializeSelf(
    const std::vector<uint8_t>& buffer, const uint64_t starting_offset) {
  uint64_t current_position = starting_offset;
  const auto deserialized_roadmap =
      PRMPlanner<Eigen::VectorXd>::DeserializeRoadmap(buffer, current_position);
  roadmap_ = deserialized_roadmap.Value();
  current_position += deserialized_roadmap.BytesRead();
  const auto packed_spheres_deserialized =
      common_robotics_utilities::serialization ::
          DeserializeMemcpyableVectorLike<float>(buffer, current_position);
  packed_spheres_ = packed_spheres_deserialized.Value();
  current_position += packed_spheres_deserialized.BytesRead();
  const auto packed_edge_info_deserialized =
      common_robotics_utilities::serialization ::
          DeserializeMemcpyableVectorLike<int32_t>(buffer, current_position);
  packed_edge_info_ = packed_edge_info_deserialized.Value();
  current_position += packed_edge_info_deserialized.BytesRead();
  const uint64_t bytes_read = current_position - starting_offset;
  drake::log()->info("Deserialized roadmap + edges with {} nodes from {} bytes",
                     roadmap().Size(), bytes_read);
  return bytes_read;
}

uint64_t SerializeNamedPackedRoadmaps(
    const std::map<std::string, RoadmapWithPackedEdgeSpheres>&
        named_packed_roadmaps,
    // NOLINTNEXTLINE(runtime/references)
    std::vector<uint8_t>& buffer) {
  return common_robotics_utilities::serialization ::SerializeMapLike<
      std::string, RoadmapWithPackedEdgeSpheres>(
      named_packed_roadmaps, buffer,
      common_robotics_utilities::serialization::SerializeString<char>,
      RoadmapWithPackedEdgeSpheres::Serialize);
}

common_robotics_utilities::serialization ::Deserialized<
    std::map<std::string, RoadmapWithPackedEdgeSpheres>>
DeserializeNamedPackedRoadmaps(const std::vector<uint8_t>& buffer,
                               uint64_t starting_offset) {
  return common_robotics_utilities::serialization ::DeserializeMapLike<
      std::string, RoadmapWithPackedEdgeSpheres>(
      buffer, starting_offset,
      common_robotics_utilities::serialization::DeserializeString<char>,
      RoadmapWithPackedEdgeSpheres::Deserialize);
}

void SaveNamedPackedRoadmapsToFile(
    const std::map<std::string, RoadmapWithPackedEdgeSpheres>&
        named_packed_roadmaps,
    const std::string& filename) {
  std::vector<uint8_t> buffer;
  SerializeNamedPackedRoadmaps(named_packed_roadmaps, buffer);
  std::ofstream output_file(filename, std::ios::out | std::ios::binary);
  output_file.write(reinterpret_cast<const char*>(buffer.data()),
                    static_cast<std::streamsize>(buffer.size()));
  output_file.close();
}

std::map<std::string, RoadmapWithPackedEdgeSpheres>
LoadNamedPackedRoadmapsFromFile(const std::string& filename) {
  std::ifstream input_file(filename,
                           std::ios::binary | std::ios::in | std::ios::ate);
  if (!input_file.is_open()) {
    throw std::runtime_error("Failed to open file [" + filename + "]");
  }
  std::streamsize size = input_file.tellg();
  input_file.seekg(0, std::ios::beg);
  std::vector<uint8_t> file_buffer(static_cast<size_t>(size));
  if (!(input_file.read(reinterpret_cast<char*>(file_buffer.data()), size))) {
    throw std::runtime_error("Failed to read entire contents of file");
  }
  const uint64_t starting_offset = 0;
  return DeserializeNamedPackedRoadmaps(file_buffer, starting_offset).Value();
}
}  // namespace planning
}  // namespace drake
