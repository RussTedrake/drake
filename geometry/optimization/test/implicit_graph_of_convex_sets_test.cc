#include "drake/geometry/optimization/implicit_graph_of_convex_sets.h"

#include <gtest/gtest.h>

#include "drake/geometry/optimization/point.h"

namespace drake {
namespace geometry {
namespace optimization {

using Edge = GraphOfConvexSets::Edge;
using EdgeId = GraphOfConvexSets::EdgeId;
using Vertex = GraphOfConvexSets::Vertex;
using VertexId = GraphOfConvexSets::VertexId;

using Eigen::Vector2d;

/* A graph with one edge definitely on the optimal path, and one definitely off
it.
┌──────┐         ┌──────┐
│source├──e_on──►│target│
└───┬──┘         └──────┘
    │e_off
    │
┌───▼──┐
│ sink │
└──────┘
*/
class ThreePointsGcs : public ImplicitGraphOfConvexSets {
 public:
  ThreePointsGcs() : ImplicitGraphOfConvexSets(Point(Vector2d(3., 5.))) {}

 protected:
  std::vector<Edge*> DoSuccessors(Vertex* v) override {
    if (v == source()) {
      Vertex* target = AddVertex(Point(Vector2d(-2., 4.)), "target");
      Vertex* sink = AddVertex(Point(Vector2d(5., -2.3)), "sink");
      Edge* e_on = AddEdge(v, target);
      Edge* e_off = AddEdge(v, sink);
      return {e_on, e_off};
    }
    return {};
  }
};

GTEST_TEST(ImplicitGraphOfConvexSetsTest, Basic) {
  ThreePointsGcs implicit_gcs;
  const GraphOfConvexSets& explicit_gcs = implicit_gcs.BuildExplicitGcs();
  EXPECT_EQ(explicit_gcs.num_vertices(), 3);
  EXPECT_EQ(explicit_gcs.num_edges(), 2);
}

}  // namespace optimization
}  // namespace geometry
}  // namespace drake
