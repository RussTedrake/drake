#pragma once

#include "drake/common/trajectories/bezier_curve.h"
#include "drake/common/trajectories/composite_trajectory.h"
#include "drake/geometry/optimization/convex_set.h"

namespace drake {
namespace planning {
namespace trajectory_optimization {

/**
Fast Path Planner (aka FPP) is a motion planning framework inspired by
GcsTrajectoryOptimization. Rather than solving the full GCS problem,
considering the discrete and continuous variables simultaneously, FPP
alternates between solving the discrete and continuous problems to find a local
minima.

"Fast Path Planning Through Large Collections of Safe Boxes" by Tobia Marcucci,
Parth Nobel, Russ Tedrake, and Stephen Boyd.
https://web.stanford.edu/~boyd/papers/pdf/fpp.pdf

While the original paper considered regions defined only as axis-aligned
bounding boxes, this implementation supports more general convex regions, and
support for subgraphs.

FPP is (typically) faster than GcsTrajectoryOptimization, and by virtue of
using alternations, it can handle constraints on the derivatives (e.g.
acceleration constraints) which GcsTrajectoryOptimization cannot handle.
However, FPP is more limited on the class of convex constraints that it can
support.

The API for FPP should mirror the API for GcsTrajectoryOptimization to the
extent possible.
*/
class FastPathPlanner final {
 public:
  using VertexId = Identifier<class VertexTag>;
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FastPathPlanner);

  /** Constructs the motion planning problem.
  @param num_positions is the dimension of the configuration space. */
  explicit FastPathPlanner(int num_positions);

  ~FastPathPlanner();

  /** A Subgraph is a subset of the larger graph. It is defined by a set of
  regions and edges between them based on intersection. From an API standpoint,
  a Subgraph is useful to define a multi-modal motion planning problem. Further,
  it allows different constraints and objects to be added to different
  subgraphs. */
  class Subgraph final {
   public:
    DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(Subgraph);

    ~Subgraph();

    /** Returns the name of the subgraph. */
    const std::string& name() const { return name_; }

    /** Returns the order of the Bézier trajectory within the region. */
    int order() const { return order_; }

    /** Returns the number of vertices in the subgraph. */
    int size() const { return vertex_ids_.size(); }

    // TODO(russt): Implement additional supported costs and constraints from GcsTrajectoryOptimization and the FPP paper.

    /** Adds multiple L2Norm Costs on the upper bound of the path length.
    We upper bound the trajectory length by the sum of the distances between
    control points. For Bézier curves, this is equivalent to the sum
    of the L2Norm of the derivative control points of the curve divided by the
    order.

    @param weight is the relative weight of the cost.
    */
    void AddPathLengthCost(double weight = 1.0);

  private:
    /* Constructs a new subgraph and copies the regions. */
    Subgraph(const std::vector<FastPathPlanner::VertexId> vertex_ids,
             int order, double h_min, double h_max, std::string name);
    
    std::vector<FastPathPlanner::VertexId> vertex_ids_{};
    int order_;
    double h_min_;
    double h_max_;
    std::string name_{};

    friend class FastPathPlanner;
  };

  /** EdgesBetweenSubgraphs are defined as the connecting edges between two
  given subgraphs. These edges are a subset of the many other edges in the
  larger graph. From an API standpoint, EdgesBetweenSubgraphs enable transitions
  between Subgraphs, which can enable transitions between modes. Further, it
  allows different constraints to be added in the transition between subgraphs.
  Note that the EdgesBetweenSubgraphs can't be separated from the actual edges
  in the GraphOfConvexSets framework, thus mixing it with other instances of
  GCSTrajetoryOptimization is not supported.
  */
  class EdgesBetweenSubgraphs final {
   public:
    DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(EdgesBetweenSubgraphs);

    ~EdgesBetweenSubgraphs();

    // TODO(russt): Implement AddVelocityBounds() or decide that we shouldn't.

   private:
    EdgesBetweenSubgraphs(const Subgraph& from_subgraph,
                          const Subgraph& to_subgraph,
                          const geometry::optimization::ConvexSet* subspace,
                          FastPathPlanner* traj_opt);
  };

  /** Returns the number of position variables. */
  int num_positions() const { return num_positions_; }

  // TODO(russt): Implement GetGraphvizString().

  /** Creates a Subgraph with the given regions and indices.
  @param regions represent the valid set a control point can be in. We retain a
  copy of the regions since other functions may access them.
  @param edges_between_regions is a list of pairs of indices into the regions
  vector. For each pair representing an edge between two regions, an edge is
  added within the subgraph. Note that the edges are directed so (i,j) will only
  add an edge from region i to region j.
  @param order is the order of the Bézier curve.
  @param h_max is the maximum duration to spend in a region (seconds). Some
  solvers struggle numerically with large values.
  @param h_min is the minimum duration to spend in a region (seconds) if that
  region is visited on the optimal path. Some cost and constraints are only
  convex for h > 0. For example the perspective quadratic cost of the path
  energy ||ṙ(s)||² / h becomes non-convex for h = 0. Otherwise h_min can be set
  to 0.
  @param name is the name of the subgraph. A default name will be provided.
  */
  Subgraph& AddRegions(
      const geometry::optimization::ConvexSets& regions,
      const std::vector<std::pair<int, int>>& edges_between_regions, int order,
      double h_min = 0, double h_max = 20, std::string name = "");

  /** Creates a Subgraph with the given regions.
  This function will compute the edges between the regions based on the set
  intersections.
  @param regions represent the valid set a control point can be in. We retain a
  copy of the regions since other functions may access them.
  @param order is the order of the Bézier curve.
  @param h_min is the minimum duration to spend in a region (seconds) if that
  region is visited on the optimal path. Some cost and constraints are only
  convex for h > 0. For example the perspective quadratic cost of the path
  energy ||ṙ(s)||² / h becomes non-convex for h = 0. Otherwise h_min can be set
  to 0.
  @param h_max is the maximum duration to spend in a region (seconds). Some
  solvers struggle numerically with large values.
  @param name is the name of the subgraph. A default name will be provided.
  */
  Subgraph& AddRegions(const geometry::optimization::ConvexSets& regions,
                       int order, double h_min = 0, double h_max = 20,
                       std::string name = "");

  /** Connects two subgraphs with directed edges.
  @param from_subgraph is the subgraph to connect from. Must have been created
  from a call to AddRegions() on this object, not some other optimization
  program.
  @param to_subgraph is the subgraph to connect to. Must have been created from
  a call to AddRegions() on this object, not some other optimization program.
  @param subspace is the subspace that the connecting control points must be in.
  Subspace is optional. Only edges that connect through the subspace will be
  added, and the subspace is added as a constraint on the connecting control
  points. Subspaces of type point or HPolyhedron are supported since other sets
  require constraints that are not yet supported by the GraphOfConvexSets::Edge
  constraint, e.g., set containment of a HyperEllipsoid is formulated via
  LorentzCone constraints. Workaround: Create a subgraph of zero order with the
  subspace as the region and connect it between the two subgraphs. This works
  because GraphOfConvexSet::Vertex , supports arbitrary instances of ConvexSets.
  */
  EdgesBetweenSubgraphs& AddEdges(
      const Subgraph& from_subgraph, const Subgraph& to_subgraph,
      const geometry::optimization::ConvexSet* subspace = nullptr);

  // TODO(russt): Add global cost/constraint methods AddTimeCost, etc, to
  // mirror GcsTrajectoryOptimization.

  /** Performs the _offline_ preprocessing steps described in the paper. This
  only needs to be called when the graph changes, and can then support multiple
  planning queries. */
  void Preprocess();

  /** Formulates and solves the mixed-integer convex formulation of the
  shortest path problem on the whole graph. @see
  `geometry::optimization::GraphOfConvexSets::SolveShortestPath()` for further
  details.

  @param source specifies the source subgraph. Must have been created from a
  call to AddRegions() on this object, not some other optimization program. If
  the source is a subgraph with more than one region, an empty set will be
  added and optimizer will choose the best region to start in. To start in a
  particular point, consider adding a subgraph of order zero with a single
  region of type Point.
  @param target specifies the target subgraph. Must have been created from a
  call to AddRegions() on this object, not some other optimization program. If
  the target is a subgraph with more than one region, an empty set will be
  added and optimizer will choose the best region to end in. To end in a
  particular point, consider adding a subgraph of order zero with a single
  region of type Point.
  */
  trajectories::CompositeTrajectory<double>
  SolvePath(
      const Eigen::Ref<const Eigen::VectorXd>& p_start,
      const Eigen::Ref<const Eigen::VectorXd>& p_goal);

 private:
  const int num_positions_;
  bool needs_preprocessing_{true};

  std::map<VertexId, copyable_unique_ptr<geometry::optimization::ConvexSet>>
      vertices_{};
  std::vector<std::pair<VertexId, VertexId>> edges_{};
  std::map<VertexId, Eigen::VectorXd> points_{};
  std::vector<double> edge_weights_{};

  std::vector<std::unique_ptr<Subgraph>> subgraphs_{};
  std::vector<std::unique_ptr<EdgesBetweenSubgraphs>> subgraph_edges_{};
};

}  // namespace trajectory_optimization
}  // namespace planning
}  // namespace drake
