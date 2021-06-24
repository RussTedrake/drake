#include "drake/solvers/ibex_solver.h"

#include <algorithm>  // To suppress cpplint.
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "./ibex.h"

#include "drake/common/text_logging.h"
#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace solvers {

using Eigen::VectorXd;

bool IbexSolver::is_available() { return true; }

void IbexSolver::DoSolve(
    const MathematicalProgram& prog,
    const Eigen::VectorXd& initial_guess,
    const SolverOptions& merged_options,
    MathematicalProgramResult* result) const {
  if (!prog.GetVariableScaling().empty()) {
    static const logging::Warn log_once(
      "IbexSolver doesn't support the feature of variable scaling.");
  }

  unused(initial_guess);
  unused(merged_options);
  
  ibex::Variable ibex_vars(prog.num_vars(), "x");
  ibex::SystemFactory factory;
  ibex::IntervalVector bounds(prog.num_vars());
  for (const auto& b : prog.bounding_box_constraints()) {
    const auto& c = b.evaluator();
    const auto& lb = c->lower_bound();
    const auto& ub = c->upper_bound();

    for (int k = 0; k < static_cast<int>(b.GetNumElements()); ++k) {
      const size_t index =
          prog.FindDecisionVariableIndex(b.variables()[k]);
      bounds[index] &= ibex::Interval(lb[k], ub[k]);
    }
  }
  factory.add_var(ibex_vars, bounds);

  for (const auto&b : prog.linear_constraints()) {
    symbolic::Formula f{b.evaluator()->CheckSatisfied(b.variables())};

  }

  ibex::System sys(factory);
  ibex::DefaultOptimizer o(sys, 1e-07); // TODO: take this as a parameter.
  o.optimize(sys.box);
}

}  // namespace solvers
}  // namespace drake
