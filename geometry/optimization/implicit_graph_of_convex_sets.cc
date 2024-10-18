#include "drake/geometry/optimization/implicit_graph_of_convex_sets.h"

#include <queue>
#include <unordered_set>

namespace drake {
namespace geometry {
namespace optimization {

using Edge = GraphOfConvexSets::Edge;
using EdgeId = GraphOfConvexSets::EdgeId;
using Vertex = GraphOfConvexSets::Vertex;
using VertexId = GraphOfConvexSets::VertexId;

ImplicitGraphOfConvexSets::ImplicitGraphOfConvexSets(const ConvexSet& source,
                                                     std::string name) {
  source_ = AddVertex(source, name);
}

ImplicitGraphOfConvexSets::~ImplicitGraphOfConvexSets() = default;

std::vector<Edge*> ImplicitGraphOfConvexSets::Successors(Vertex* v) {
  if (successor_cache_.find(v->id()) != successor_cache_.end()) {
    return successor_cache_[v->id()];
  }
  auto result = DoSuccessors(v);
  successor_cache_[v->id()] = result;
  return result;
}

const GraphOfConvexSets& ImplicitGraphOfConvexSets::BuildExplicitGcs(
    int max_vertices) {
  std::queue<Vertex*> queue;
  std::unordered_set<VertexId> visited;
  queue.push(source_);
  while (!queue.empty() && expanded_gcs_.num_vertices() < max_vertices) {
    Vertex* u = queue.front();
    queue.pop();
    visited.insert(u->id());
    for (Edge* e : Successors(u)) {
      Vertex* v = &e->v();
      if (!visited.contains(v->id())) {
        queue.push(v);
      }
    }
  }
  if (expanded_gcs_.num_vertices() == max_vertices) {
    log()->warn(
        "ImplicitGraphOfConvexSets::BuildExplicitGCS() reached the max number "
        "of vertices. The graph is not fully expanded.");
  }
  return expanded_gcs_;
}

Vertex* ImplicitGraphOfConvexSets::AddVertex(const ConvexSet& set,
                                             std::string name) {
  return expanded_gcs_.AddVertex(set, name);
}

Edge* ImplicitGraphOfConvexSets::AddEdge(Vertex* u, Vertex* v,
                                         std::string name) {
  return expanded_gcs_.AddEdge(u, v, name);
}

}  // namespace optimization
}  // namespace geometry
}  // namespace drake
