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
#include "drake/solvers/ibex_converter.h"
#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace solvers {

using Eigen::VectorXd;

using std::shared_ptr;
using std::vector;

using symbolic::Expression;
using symbolic::Formula;
using symbolic::Variable;

namespace {
// Custom deleter for ibex::ExprCtr. It deletes the internal
// ibex::ExprNode while keeping the ExprSymbols intact. Note that the
// ExprSymbols will be deleted separately in
// ~ContractorIbexPolytope().
struct ExprCtrDeleter {
  void operator()(const ibex::ExprCtr* const p) const {
    if (p) {
      ibex::cleanup(p->e, false);
      delete p;
    }
  }
};

// Extracts linear costs in @p prog and push them into @p costs vector.
void ExtractLinearCosts(const MathematicalProgram& prog,
                        vector<Expression>* const costs) {
  for (const Binding<LinearCost>& binding : prog.linear_costs()) {
    const VectorXDecisionVariable& x{binding.variables()};
    const shared_ptr<LinearCost>& cost{binding.evaluator()};
    VectorX<Expression> y;
    cost->Eval(x, &y);
    costs->push_back(y[0]);
  }
}

// Extracts quadratic costs in @p prog and push them into @p costs vector.
void ExtractQuadraticCosts(const MathematicalProgram& prog,
                           vector<Expression>* const costs) {
  for (const Binding<QuadraticCost>& binding : prog.quadratic_costs()) {
    const VectorXDecisionVariable& x{binding.variables()};
    const shared_ptr<QuadraticCost>& cost{binding.evaluator()};
    VectorX<Expression> y;
    cost->Eval(x, &y);
    costs->push_back(y[0]);
  }
}

// Adds @p f into @p factor using @p ibex_converter and @p expr_ctrs.
void AddFormula(
    const Formula& f, IbexConverter* const ibex_converter,
    ibex::SystemFactory* const factory,
    std::vector<std::unique_ptr<const ibex::ExprCtr, ExprCtrDeleter>>* const
        expr_ctrs) {
  std::unique_ptr<const ibex::ExprCtr, ExprCtrDeleter> expr_ctr{
      ibex_converter->Convert(f)};
  if (expr_ctr) {
    factory->add_ctr(*expr_ctr);
    expr_ctrs->push_back(std::move(expr_ctr));
  }
}

}  // namespace

bool IbexSolver::is_available() { return true; }

void IbexSolver::DoSolve(const MathematicalProgram& prog,
                         const Eigen::VectorXd& initial_guess,
                         const SolverOptions& merged_options,
                         MathematicalProgramResult* result) const {
  if (!prog.GetVariableScaling().empty()) {
    static const logging::Warn log_once(
        "IbexSolver doesn't support the feature of variable scaling.");
  }

  unused(initial_guess);
  unused(merged_options);

  // Creates a converter. Internally, it creates ibex variables.
  IbexConverter ibex_converter(prog.decision_variables());

  ibex::SystemFactory factory;

  // Prepares bounds for variables.
  ibex::IntervalVector bounds(prog.num_vars());
  for (const auto& b : prog.bounding_box_constraints()) {
    const auto& c = b.evaluator();
    const auto& lb = c->lower_bound();
    const auto& ub = c->upper_bound();

    for (int k = 0; k < static_cast<int>(b.GetNumElements()); ++k) {
      const size_t index = prog.FindDecisionVariableIndex(b.variables()[k]);
      bounds[index] &= ibex::Interval(lb[k], ub[k]);
      std::cerr << "Bound: " << b.variables()[k] << "[" << lb[k] << ", "
                << ub[k] << "]\n";
    }
  }

  // Adds ibex variables + bounds into the factory.
  factory.add_var(ibex_converter.variables(), bounds);

  // This expr_ctrs holds the translated ibex::ExprCtr* objects. After adding
  // the ibex::Ctrs to the factory, we should not destruct them at the end of
  // the loop where they are created because they are used in the system factory
  // and the system.
  std::vector<std::unique_ptr<const ibex::ExprCtr, ExprCtrDeleter>> expr_ctrs;

  // Add constraints.
  for (const auto& b : prog.linear_constraints()) {
    Formula f{b.evaluator()->CheckSatisfied(b.variables())};
    std::cerr << "Add constraint: " << f << "\n";
    if (symbolic::is_conjunction(f)) {
      for (const auto& f_i : symbolic::get_operands(f)) {
        AddFormula(f_i, &ibex_converter, &factory, &expr_ctrs);
      }
    } else {
      AddFormula(f, &ibex_converter, &factory, &expr_ctrs);
    }
    ibex_converter.set_need_to_delete_variables(true);
  }

  // Extract costs
  vector<Expression> costs;
  ExtractLinearCosts(prog, &costs);
  ExtractQuadraticCosts(prog, &costs);

  // Add costs into the factory.
  std::vector<std::unique_ptr<const ibex::ExprNode>> expr_nodes;
  for (const Expression& cost : costs) {
    std::cerr << "Add cost: " << cost << "\n";
    std::unique_ptr<const ibex::ExprNode> expr_node{
        ibex_converter.Convert(cost)};
    if (expr_node) {
      factory.add_goal(*expr_node);
      // We need to postpone the destruction of expr_ctr as it is
      // still used inside of system_factory.
      expr_nodes.push_back(std::move(expr_node));
    }
  }

  ibex::System sys(factory);
  std::cerr << "Solving...\n";

  const double rel_eps_f = ibex::Optimizer::default_rel_eps_f;
  const double abs_eps_f = ibex::Optimizer::default_abs_eps_f;
  const double eps_h = ibex::NormalizedSystem::default_eps_h;
  const bool rigor = false;
  const bool inHC4 = !(sys.nb_ctr > 0 && sys.nb_ctr < sys.f_ctrs.image_dim());
  const double random_seed = ibex::DefaultOptimizer::default_random_seed;
  const double eps_x = ibex::Optimizer::default_eps_x;
  const double timeout = 30.0; /* sec */
  const bool trace = false;

  std::cerr << sys << "\n";

  ibex::DefaultOptimizer o(sys, rel_eps_f, abs_eps_f, eps_h, rigor, inHC4,
                           random_seed, eps_x);
  o.trace = trace;
  o.timeout = timeout;
  o.optimize(sys.box);
  std::cerr << "Solving... Done.\n";
  o.report();

  // TODO: Need to set the result...
}
}  // namespace solvers
}  // namespace drake
