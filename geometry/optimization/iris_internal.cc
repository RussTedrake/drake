#include "drake/geometry/optimization/iris_internal.h"

#include <limits>

#include "drake/solvers/snopt_solver.h"

namespace drake {
namespace geometry {
namespace optimization {
namespace internal {

using Eigen::MatrixXd;
using Eigen::VectorXd;

ClosestCollisionProgram::ClosestCollisionProgram(
    std::shared_ptr<solvers::Constraint> min_distance_constraint,
    const Hyperellipsoid& E, const Eigen::Ref<const Eigen::MatrixXd>& A,
    const Eigen::Ref<const Eigen::VectorXd>& b) {
  q_ = prog_.NewContinuousVariables(A.cols(), "q");

  P_constraint_ = prog_.AddLinearConstraint(
      A,
      Eigen::VectorXd::Constant(b.size(),
                                -std::numeric_limits<double>::infinity()),
      b, q_);

  E_cost_ = prog_.AddQuadraticCost(MatrixXd::Identity(A.cols(), A.cols()),
                                   VectorXd::Zero(A.cols()), q_, true);
  UpdateEllipsoid(E);

  prog_.AddConstraint(min_distance_constraint, q_);
}

void ClosestCollisionProgram::UpdateEllipsoid(const Hyperellipsoid& E) {
  Eigen::MatrixXd Q = E.A().transpose() * E.A();
  // Scale the objective so the eigenvalues are close to 1, using
  // scale*lambda_min = 1/scale*lambda_max.
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Q);
  Q *= 1.0 /
       std::sqrt(es.eigenvalues().maxCoeff() * es.eigenvalues().minCoeff());
  const VectorXd b = -2 * Q * E.center();
  const double c = E.center().dot(Q * E.center());
  E_cost_->evaluator()->UpdateCoefficients(Q, b, c, true);
}

void ClosestCollisionProgram::UpdatePolytope(
    const Eigen::Ref<const Eigen::MatrixXd>& A,
    const Eigen::Ref<const Eigen::VectorXd>& b) {
  P_constraint_->evaluator()->UpdateCoefficients(
      A, VectorXd::Constant(b.size(), -std::numeric_limits<double>::infinity()),
      b);
}

// Returns true iff a collision is found.
// Sets `closest` to an optimizing solution q*, if a solution is found.
bool ClosestCollisionProgram::Solve(
    const solvers::SolverInterface& solver,
    const Eigen::Ref<const Eigen::VectorXd>& q_guess, VectorXd* closest) {
  prog_.SetInitialGuess(q_, q_guess);
  solvers::MathematicalProgramResult result;
  solver.Solve(prog_, std::nullopt, std::nullopt, &result);
  if (result.is_success() ||
      // We declare success on info 43 because SceneGraph gives NaNs precisely
      // on the boundary of collision. In this application, this is actually
      // success.
      // TODO(russt): This should be removed pending resolution of #14789.
      result.get_solver_details<solvers::SnoptSolver>().info == 43) {
    *closest = result.GetSolution(q_);
    return true;
  }
  return false;
}
}  // namespace internal
}  // namespace optimization
}  // namespace geometry
}  // namespace drake
