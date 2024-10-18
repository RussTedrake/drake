#pragma once

#include <map>
#include <string>
#include <vector>

#include "drake/geometry/optimization/graph_of_convex_sets.h"

namespace drake {
namespace geometry {
namespace optimization {

/** A base class...

@experimental

@ingroup geometry_optimization
*/
class ImplicitGraphOfConvexSets {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ImplicitGraphOfConvexSets);

  virtual ~ImplicitGraphOfConvexSets();

  /** Returns the outgoing edges from `v`, which defines the "successors" of
  `v` in the common notation of implicit graph search. */
  std::vector<GraphOfConvexSets::Edge*> Successors(
      GraphOfConvexSets::Vertex* v);

  /** Returns a mutable pointer to the unique source vertex. */
  GraphOfConvexSets::Vertex* source() const { return source_; }

  /** For finite graphs, this makes repeated calls to Successors() until all
  vertices have been added to the graph, and returns a reference to the internal
  GCS which now contains the entire graph. To protect against infinite loops for
  infinite graphs, we set a maximum number of vertices; this can be set to
  infinity if you are confident your graph is finite. */
  const GraphOfConvexSets& BuildExplicitGcs(int max_vertices = 1000);

 protected:
  /** Constructs the implicit GCS. Calls AddVertex() to define the unique source
  vertex. Use source() to get a mutable pointer to this vertex, if you need
  to add further costs and constraints. */
  ImplicitGraphOfConvexSets(const ConvexSet& source,
                            std::string name = "source");

  /** DoSuccessors implementations should call AddVertex() and AddEdge() if
  needed in order for this class to maintain ownership of the underlying vertex
  and edge objects. This method will only be called for if `v`'s successors have
  *not* already been added to the graph. */
  virtual std::vector<GraphOfConvexSets::Edge*> DoSuccessors(
      GraphOfConvexSets::Vertex* v) = 0;

  /** Registers a new vertex with the graph. This must be called before
  DoSuccessors returns any edges which point to this vertex. */
  GraphOfConvexSets::Vertex* AddVertex(const ConvexSet& set,
                                       std::string name = "");

  /** Registers a new edge with the graph. This must be called before
  DoSuccessors returns this edge. */
  GraphOfConvexSets::Edge* AddEdge(GraphOfConvexSets::Vertex* u,
                                   GraphOfConvexSets::Vertex* v,
                                   std::string name = "");

 private:
  GraphOfConvexSets::Vertex* source_{nullptr};

  // The expanded GCS maintains ownership of all vertices and edges that have
  // been expanded over the lifetime of this object.
  GraphOfConvexSets expanded_gcs_;

  std::map<GraphOfConvexSets::VertexId, std::vector<GraphOfConvexSets::Edge*>>
      successor_cache_{};
};

}  // namespace optimization
}  // namespace geometry
}  // namespace drake
