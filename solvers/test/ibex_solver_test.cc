#include "drake/solvers/ibex_solver.h"

#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/test/generic_trivial_constraints.h"
#include "drake/solvers/test/generic_trivial_costs.h"

namespace drake {
namespace solvers {
namespace {

using std::pow;

// Reproduced from
// https://github.com/ibex-team/ibex-lib/blob/master/benchs/optim/easy/ex3_1_3.bch
GTEST_TEST(IbexSolverTest, IbexEasyEx3_1_3) {
  MathematicalProgram prog;
  auto x = prog.NewContinuousVariables(6, "x");
  Vector6d lb, ub, x_d, Q_diag;
  lb << 0, 0, 1, 0, 1, 0;
  ub << 1e8, 1e8, 5, 6, 5, 10;
  prog.AddBoundingBoxConstraint(lb, ub, x);

  prog.AddCost(-25 * pow(x[0] - 2, 2) - pow(x[1] - 2, 2) - pow(x[2] - 1, 2) -
               pow(x[3] - 4, 2) - pow(x[4] - 1, 2) - pow(x[5] - 4, 2));

  prog.AddConstraint(pow(x[2] - 3, 2) + x[3] >= 4);
  prog.AddConstraint(pow(x[4] - 3, 2) + x[5] >= 4);
  prog.AddConstraint(x[0] - 3 * x[1] <= 2);
  prog.AddConstraint(-x[0] + x[1] <= 2);
  prog.AddConstraint(x[0] + x[1] <= 6);
  prog.AddConstraint(x[0] + x[1] >= 2);

  IbexSolver solver;
  if (solver.available()) {
    auto result = solver.Solve(prog);
    EXPECT_TRUE(result.is_success());
  }
}

}  // namespace

}  // namespace solvers
}  // namespace drake
