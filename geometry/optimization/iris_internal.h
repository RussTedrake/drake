#pragma once
/*
 This file contains the internal functions used by iris.cc. The users shouldn't
 call these functions, but we expose them for unit tests.
 */

#include <memory>
#include <optional>

#include "drake/geometry/optimization/hyperellipsoid.h"
#include "drake/solvers/constraint.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solver_interface.h"

namespace drake {
namespace geometry {
namespace optimization {
namespace internal {

/* Defines a MathematicalProgram to solve the problem
 min_q (q-d) CᵀC (q-d)
 s.t. setA in frameA and setB in frameB are in collision in q.
      Aq ≤ b.
 where C, d are the matrix and center from the hyperellipsoid E.

 The class design supports repeated solutions of the (nearly) identical
 problem from different initial guesses.
 */
class ClosestCollisionProgram {
 public:
  ClosestCollisionProgram(
      std::shared_ptr<solvers::Constraint> min_distance_constraint,
      const Hyperellipsoid& E, const Eigen::Ref<const Eigen::MatrixXd>& A,
      const Eigen::Ref<const Eigen::VectorXd>& b);

  void UpdateEllipsoid(const Hyperellipsoid& E);

  void UpdatePolytope(const Eigen::Ref<const Eigen::MatrixXd>& A,
                      const Eigen::Ref<const Eigen::VectorXd>& b);

  // Returns true iff a collision is found.
  // Sets `closest` to an optimizing solution q*, if a solution is found.
  bool Solve(const solvers::SolverInterface& solver,
             const Eigen::Ref<const Eigen::VectorXd>& q_guess,
             Eigen::VectorXd* closest);

 private:
  solvers::MathematicalProgram prog_;
  solvers::VectorXDecisionVariable q_;
  std::optional<solvers::Binding<solvers::QuadraticCost>> E_cost_{};
  std::optional<solvers::Binding<solvers::LinearConstraint>> P_constraint_{};
};
}  // namespace internal
}  // namespace optimization
}  // namespace geometry
}  // namespace drake
