#include "drake/geometry/optimization/iris.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "drake/geometry/optimization/convex_set.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/ibex_solver.h"
#include "drake/solvers/snopt_solver.h"

namespace drake {
namespace geometry {
namespace optimization {

using Eigen::MatrixXd;
using Eigen::Ref;
using Eigen::VectorXd;
using symbolic::Expression;

HPolyhedron Iris(const ConvexSets& obstacles, const Ref<const VectorXd>& sample,
                 const HPolyhedron& domain, const IrisOptions& options) {
  const int dim = sample.size();
  const int N = obstacles.size();
  DRAKE_DEMAND(domain.ambient_dimension() == dim);
  for (int i = 0; i < N; ++i) {
    DRAKE_DEMAND(obstacles[i]->ambient_dimension() == dim);
  }
  DRAKE_DEMAND(domain.IsBounded());
  const double kEpsilonEllipsoid = 1e-2;
  Hyperellipsoid E = Hyperellipsoid::MakeHypersphere(kEpsilonEllipsoid, sample);
  HPolyhedron P = domain;

  // On each iteration, we will build the collision-free polytope represented as
  // {x | A * x <= b}.  Here we pre-allocate matrices of the maximum size.
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(
      domain.A().rows() + N, dim);
  VectorXd b(domain.A().rows() + N);
  A.topRows(domain.A().rows()) = domain.A();
  b.head(domain.A().rows()) = domain.b();
  // Use pairs {scale, index}, so that I can back out the indices after a sort.
  std::vector<std::pair<double, int>> scaling(N);
  MatrixXd closest_points(dim, N);

  double best_volume = E.Volume();
  int iteration = 0;
  MatrixXd tangent_matrix;

  while (true) {
    // Find separating hyperplanes
    for (int i = 0; i < N; ++i) {
      const auto touch = E.MinimumUniformScalingToTouch(*obstacles[i]);
      scaling[i].first = touch.first;
      scaling[i].second = i;
      closest_points.col(i) = touch.second;
    }
    std::sort(scaling.begin(), scaling.end());

    int num_constraints = domain.A().rows();
    tangent_matrix = 2.0 * E.A().transpose() * E.A();
    for (int i = 0; i < N; ++i) {
      const VectorXd point = closest_points.col(scaling[i].second);
      // Only add a constraint if this point is still within the set that has
      // been constructed so far on this iteration.
      if (((A.topRows(num_constraints) * point).array() <=
           b.head(num_constraints).array())
              .all()) {
        // Add the tangent to the (scaled) ellipsoid at this point as a
        // constraint.
        A.row(num_constraints) =
            (tangent_matrix * (point - E.center())).normalized();
        b[num_constraints] = A.row(num_constraints) * point;
        num_constraints++;
      }
    }

    if (options.require_sample_point_is_contained &&
        ((A.topRows(num_constraints) * sample).array() >=
         b.head(num_constraints).array())
            .any()) {
      break;
    }
    P = HPolyhedron(A.topRows(num_constraints), b.head(num_constraints));

    iteration++;
    if (iteration >= options.iteration_limit) {
      break;
    }

    E = P.MaximumVolumeInscribedEllipsoid();
    const double volume = E.Volume();
    if (volume - best_volume <= options.termination_threshold) {
      break;
    }
    best_volume = volume;
  }

  return P;
}

namespace {
// Constructs a ConvexSet for each supported Shape and adds it to the set.
class IrisConvexSetMaker final : public ShapeReifier {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(IrisConvexSetMaker)

  IrisConvexSetMaker(const QueryObject<double>& query,
                     std::optional<FrameId> reference_frame)
      : query_{query}, reference_frame_{reference_frame} {};

  void set_reference_frame(const FrameId& reference_frame) {
    DRAKE_DEMAND(reference_frame.is_valid());
    *reference_frame_ = reference_frame;
  }

  void set_geometry_id(const GeometryId& geom_id) { geom_id_ = geom_id; }

  using ShapeReifier::ImplementGeometry;

  void ImplementGeometry(const Sphere&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set = std::make_unique<Hyperellipsoid>(query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const HalfSpace&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set = std::make_unique<HPolyhedron>(query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const Box&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    // Note: We choose HPolyhedron over VPolytope here, but the IRIS paper
    // discusses a significant performance improvement using a "least-distance
    // programming" instance from CVXGEN that exploited the VPolytope
    // representation.  So we may wish to revisit this.
    set = std::make_unique<HPolyhedron>(query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const Ellipsoid&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set = std::make_unique<Hyperellipsoid>(query_, geom_id_, reference_frame_);
  }

 private:
  const QueryObject<double>& query_{};
  std::optional<FrameId> reference_frame_{};
  GeometryId geom_id_{};
};

}  // namespace

ConvexSets MakeIrisObstacles(const QueryObject<double>& query_object,
                             std::optional<FrameId> reference_frame) {
  const SceneGraphInspector<double>& inspector = query_object.inspector();
  const GeometrySet all_ids(inspector.GetAllGeometryIds());
  const std::unordered_set<GeometryId> geom_ids =
      inspector.GetGeometryIds(all_ids, Role::kProximity);
  ConvexSets sets(geom_ids.size());

  IrisConvexSetMaker maker(query_object, reference_frame);
  int count = 0;
  for (GeometryId geom_id : geom_ids) {
    maker.set_geometry_id(geom_id);
    inspector.GetShape(geom_id).Reify(&maker, &sets[count++]);
  }
  return sets;
}

namespace {

// Solves the optimization
// min_q (q-d)*CᵀC(q-d)
// s.t. setA on bodyA and setB on bodyB are in collision in q.
//      Aq ≤ b.
// where C, d are the center and matrix from the hyperellipsoid E.
// Returns true iff a collision is found.
// Sets `closest` to an optimizing solution q*, if a solution is found.
bool FindClosestCollision(const multibody::MultibodyPlant<Expression>& plant,
                          const multibody::Body<Expression>& bodyA,
                          const multibody::Body<Expression>& bodyB,
                          const ConvexSet& setA, const ConvexSet& setB,
                          const Hyperellipsoid& E,
                          const Eigen::Ref<const Eigen::MatrixXd>& A,
                          const Eigen::Ref<const Eigen::VectorXd>& b,
                          const solvers::SolverBase& solver,
                          systems::Context<Expression>* context,
                          VectorXd* closest) {
  solvers::MathematicalProgram prog;
  auto q = prog.NewContinuousVariables(plant.num_positions(), "q");

  prog.AddLinearConstraint(
      A, VectorXd::Constant(b.size(), -std::numeric_limits<double>::infinity()),
      b, q);
  prog.AddQuadraticErrorCost(E.A().transpose() * E.A(), E.center(), q);

  auto p_AA = prog.NewContinuousVariables<3>("p_AA");
  auto p_BB = prog.NewContinuousVariables<3>("p_BB");
  setA.AddPointInSetConstraints(&prog, p_AA);
  setB.AddPointInSetConstraints(&prog, p_BB);

  plant.SetPositions(context, q);
  const math::RigidTransform<Expression> X_WA =
      plant.EvalBodyPoseInWorld(*context, bodyA);
  const math::RigidTransform<Expression> X_WB =
      plant.EvalBodyPoseInWorld(*context, bodyB);
  prog.AddConstraint(X_WA * p_AA.cast<Expression>() ==
                     X_WB * p_BB.cast<Expression>());

  solvers::MathematicalProgramResult result = solver.Solve(prog);
  if (result.is_success()) {
    *closest = result.GetSolution(q);
    return true;
  }
  return false;
}

}  // namespace

HPolyhedron IrisInConfigurationSpace(
    const multibody::MultibodyPlant<Expression>& plant,
    const QueryObject<double>& query_object,
    const Eigen::Ref<const Eigen::VectorXd>& sample,
    const IrisOptions& options) {
  const int nq = plant.num_positions();
  DRAKE_DEMAND(sample.size() == nq);

  // Note: We require finite joint limits to define the bounding box for the
  // IRIS algorithm.
  DRAKE_DEMAND(plant.GetPositionLowerLimits().array().isFinite().all());
  DRAKE_DEMAND(plant.GetPositionUpperLimits().array().isFinite().all());
  HPolyhedron P = HPolyhedron::MakeBox(plant.GetPositionLowerLimits(),
                                       plant.GetPositionUpperLimits());
  DRAKE_DEMAND(P.A().rows() == 2 * nq);

  const double kEpsilonEllipsoid = 1e-2;
  Hyperellipsoid E = Hyperellipsoid::MakeHypersphere(kEpsilonEllipsoid, sample);

  const SceneGraphInspector<double>& inspector = query_object.inspector();

  // Make all of the convex sets.  TODO(russt): Consider factoring this out so
  // that I don't have to redo the work for every new region.
  IrisConvexSetMaker maker(query_object, inspector.world_frame_id());
  std::unordered_map<GeometryId, copyable_unique_ptr<ConvexSet>> sets{};
  std::unordered_map<GeometryId, const multibody::Body<Expression>&> bodies{};
  const std::unordered_set<GeometryId> geom_ids = inspector.GetGeometryIds(
      GeometrySet(inspector.GetAllGeometryIds()), Role::kProximity);
  copyable_unique_ptr<ConvexSet> temp_set;
  for (GeometryId geom_id : geom_ids) {
    // Make all sets in the local geometry frame.
    FrameId frame_id = inspector.GetFrameId(geom_id);
    maker.set_reference_frame(frame_id);
    maker.set_geometry_id(geom_id);
    inspector.GetShape(geom_id).Reify(&maker, &temp_set);
    sets.emplace(geom_id, std::move(temp_set));
    bodies.emplace(geom_id, *plant.GetBodyFromFrameId(frame_id));
  }

  auto pairs = inspector.GetCollisionCandidates();
  const int N = static_cast<int>(pairs.size());

  // On each iteration, we will build the collision-free polytope represented as
  // {x | A * x <= b}.  Here we pre-allocate matrices of the maximum size.
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(
      P.A().rows() + N, nq);
  VectorXd b(P.A().rows() + N);
  A.topRows(P.A().rows()) = P.A();
  b.head(P.A().rows()) = P.b();

  double best_volume = E.Volume();
  int iteration = 0;
  MatrixXd tangent_matrix;

  // solvers::IbexSolver solver;
  solvers::IbexSolver solver;
  DRAKE_DEMAND(solver.is_available() && solver.is_enabled());
  auto context = plant.CreateDefaultContext();
  VectorXd closest(nq);

  while (true) {
    tangent_matrix = 2.0 * E.A().transpose() * E.A();
    int num_constraints = 2 * nq;  // Start with just the joint limits.
    // Find separating hyperplanes
    for (const auto [geomA, geomB] : pairs) {
      while (true) {
        const bool collision = FindClosestCollision(
            plant, bodies.at(geomA), bodies.at(geomB), *sets.at(geomA),
            *sets.at(geomB), E, A.topRows(num_constraints),
            b.head(num_constraints), solver, context.get(), &closest);
        std::cout << "geomA=" << inspector.GetName(geomA)
                  << ", geomB=" << inspector.GetName(geomB);
        if (collision) {
          std::cout << ", closest=" << closest.transpose() << std::endl;
          // Add the tangent to the (scaled) ellipsoid at this point as a
          // constraint.
          A.row(num_constraints) =
              (tangent_matrix * (closest - E.center())).normalized();
          b[num_constraints] = A.row(num_constraints) * closest -
                               options.configuration_space_margin;
          std::cout << "  "
                    << A.row(num_constraints) *
                           symbolic::MakeVectorVariable(nq, "q")
                    << " ≤ " << b[num_constraints] << std::endl;
          num_constraints++;
        } else {
          std::cout << ".  No collisions." << std::endl;
          break;
        }
      }
    }

    if (options.require_sample_point_is_contained &&
        ((A.topRows(num_constraints) * sample).array() >=
         b.head(num_constraints).array())
            .any()) {
      break;
    }
    P = HPolyhedron(A.topRows(num_constraints), b.head(num_constraints));

    iteration++;
    if (iteration >= options.iteration_limit) {
      break;
    }

    E = P.MaximumVolumeInscribedEllipsoid();
    const double volume = E.Volume();
    if (volume - best_volume <= options.termination_threshold) {
      break;
    }
    best_volume = volume;
  }
  return P;
}

}  // namespace optimization
}  // namespace geometry
}  // namespace drake
