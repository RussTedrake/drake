#include "drake/systems/analysis/region_of_attraction.h"

#include <algorithm>

#include "drake/common/symbolic_trigonometric_polynomial.h"
#include "drake/math/continuous_lyapunov_equation.h"
#include "drake/math/matrix_util.h"
#include "drake/math/quadratic_form.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"
#include "drake/systems/primitives/linear_system.h"

namespace drake {
namespace systems {
namespace analysis {

using Eigen::MatrixXd;
using Eigen::VectorXd;

using math::IsPositiveDefinite;

using solvers::MathematicalProgram;
using solvers::Solve;

using symbolic::Environment;
using symbolic::Expression;
using symbolic::MakeVectorVariable;
using symbolic::Polynomial;
using symbolic::SinCos;
using symbolic::SinCosSubstitution;
using symbolic::Substitution;
using symbolic::Variable;
using symbolic::Variables;

namespace {

// Assumes V positive semi-definite at the origin.
// If the Hessian of Vdot is negative definite at the origin, then we use
// Vdot(x)=0, g(x)=0 => V(x)≥ρ (or x=0) via
//   maximize   ρ
//   subject to (V-ρ)*(x'*x)ᵈ - λ*Vdot + λ_g*g is SOS.
// If we cannot confirm negative definiteness, then we must ask instead for
// Vdot(x)≥0, g(x)=0 => V(x)≥ρ (or x=0).
Expression FixedLyapunovConvex(
    const solvers::VectorXIndeterminate& x, const VectorXd& x0,
    const Expression& V, const Expression& Vdot, const VectorX<Polynomial>& g,
    const solvers::VectorXIndeterminate& extra_vars =
        Vector0<Variable>{}) {
  // Check if the Hessian of Vdot is negative definite.
  Environment env;
  for (int i = 0; i < x0.size(); i++) {
    env.insert(x(i), x0(i));
  }
  for (int i = 0; i < extra_vars.size(); i++) {
    env.insert(extra_vars(i), 0.0);
  }
  solvers::VectorXIndeterminate y(x.size() + extra_vars.size());
  y << x, extra_vars;
  const Eigen::MatrixXd P =
      symbolic::Evaluate(symbolic::Jacobian(Vdot.Jacobian(y), y), env);
  const double tolerance = 1e-8;
  bool Vdot_is_locally_negative_definite;
  if (g.size() == 0) {
    Vdot_is_locally_negative_definite = IsPositiveDefinite(-P, tolerance);
  } else {
    const Eigen::MatrixXd J_g =
        symbolic::Evaluate(symbolic::Jacobian(g, y), env);
    Eigen::FullPivLU<MatrixXd> lu(J_g);
    MatrixXd N = lu.kernel();
    Vdot_is_locally_negative_definite =
        IsPositiveDefinite(-N.transpose() * P * N, tolerance);
  }

  Polynomial V_balanced, Vdot_balanced;
  // TODO(russt): Figure out balancing on the quotient ring.
  if (Vdot_is_locally_negative_definite && g.size() == 0) {
    // Then "balance" V and Vdot.
    const Eigen::MatrixXd S =
        symbolic::Evaluate(symbolic::Jacobian(V.Jacobian(x), x), env);
    const Eigen::MatrixXd T = math::BalanceQuadraticForms(S, -P);
    const VectorX<Expression> Tx = T * x;
    Substitution subs;
    for (int i = 0; i < static_cast<int>(x.size()); i++) {
      subs.emplace(x(i), Tx(i));
    }
    V_balanced = Polynomial(V.Substitute(subs));
    Vdot_balanced = Polynomial(Vdot.Substitute(subs));
  } else {
    V_balanced = Polynomial(V);
    Vdot_balanced = Polynomial(Vdot);
  }

  MathematicalProgram prog;
  prog.AddIndeterminates(y);

  const int V_degree = V_balanced.TotalDegree();
  const int Vdot_degree = Vdot_balanced.TotalDegree();

  // TODO(russt): Add this as an option once I have an example that needs it.
  // This is a reasonable guess: we want the multiplier to be able to compete
  // with terms in Vdot, and to be even (since it may be SOS below).
  const int lambda_degree = std::ceil(Vdot_degree / 2.0) * 2;
  const auto lambda = prog.NewFreePolynomial(Variables(y), lambda_degree);
  VectorX<Polynomial> lambda_g(g.size());
  for (int i = 0; i < g.size(); ++i) {
    // Take λ_g[i] * g[i] to have the same degree as λ * Vdot.
    const int lambda_gi_degree = std::max(
        lambda_degree + Vdot_degree - g[i].TotalDegree(), 1);
    lambda_g[i] = prog.NewFreePolynomial(Variables(y), lambda_gi_degree);
  }

  const auto rho = prog.NewContinuousVariables<1>("rho")[0];

  VectorX<Expression> x_bar = x - x0;
  // Want (V-rho)(x_bar'x_bar)^d and Lambda*Vdot to be the same degree.
  const int d = std::floor((lambda_degree + Vdot_degree - V_degree) / 2);
  prog.AddSosConstraint((V_balanced - rho) *
                            Polynomial(pow((x_bar.transpose() * x_bar)[0], d)) -
                        lambda * Vdot_balanced + lambda_g.dot(g));
  std::cout << Vdot_balanced << " >= 0 && " << g << " > 0 => "
            << (V_balanced - rho) *
                   Polynomial(pow((x_bar.transpose() * x_bar)[0], d))
            << " is SOS" << std::endl;

  // If Vdot is indefinite, then the linearization does not inform us about the
  // local stability.  Add "lambda(x) is SOS" to confirm this local stability.
  if (!Vdot_is_locally_negative_definite) {
    prog.AddSosConstraint(lambda);
  }

  prog.AddCost(-rho);

  solvers::SolverOptions options;
  options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  prog.SetSolverOptions(options);
  const auto result = Solve(prog);

  if (!result.is_success()) {
    std::cout << "solution result: " << result.get_solution_result()
              << std::endl;
  }

  // TODO(russt): If the solution is unbounded, then try for a proof of global
  // stability.
  DRAKE_THROW_UNLESS(result.is_success());

  std::cout << "rho = " << result.GetSolution(rho) << std::endl;
  DRAKE_THROW_UNLESS(result.GetSolution(rho) > 0.0);
  return V / result.GetSolution(rho);
}

}  // namespace

Expression RegionOfAttraction(const System<double>& system,
                              const Context<double>& context,
                              const RegionOfAttractionOptions& options) {
  system.ValidateContext(context);
  DRAKE_THROW_UNLESS(context.has_only_continuous_state());
  const int num_states = context.num_continuous_states();
  DRAKE_THROW_UNLESS(options.state_variables.size() == 0 ||
                     options.state_variables.size() == num_states);

  VectorXd x0 = context.get_continuous_state_vector().CopyToVector();

  // Check that x0 is a fixed point.
  VectorXd xdot0 =
      system.EvalTimeDerivatives(context).get_vector().CopyToVector();
  DRAKE_THROW_UNLESS(xdot0.template lpNorm<Eigen::Infinity>() <= 1e-14);

  const auto symbolic_system = system.ToSymbolic();
  const auto symbolic_context = symbolic_system->CreateDefaultContext();
  symbolic_context->SetTimeStateAndParametersFrom(context);
  symbolic_system->FixInputPortsFrom(system, context, symbolic_context.get());

  VectorX<Variable> x = options.state_variables.size() > 0
                            ? options.state_variables
                            : MakeVectorVariable(num_states, "x_roa");

  // Define the relative coordinates: x_bar = x - x0
  const VectorX<Variable> x_bar =
      MakeVectorVariable(num_states, "x_bar");

  Substitution subs_x_to_x_bar;
  Substitution subs_x_bar_to_x;
  subs_x_to_x_bar.reserve(num_states);
  subs_x_bar_to_x.reserve(num_states);
  for (int i = 0; i < num_states; ++i) {
    subs_x_to_x_bar.emplace(x(i), x0(i) + x_bar(i));
    subs_x_bar_to_x.emplace(x_bar(i), x(i) - x0(i));
  }

  // x_trig is x_bar with the sin/cos variables replaced by new variables for s
  // and c.
  const int num_trig_states = num_states + options.sin_cos_variables.size();
  VectorX<Variable> x_trig(num_trig_states);
  VectorXd x0_trig(num_trig_states);
  VectorX<Polynomial> ring(options.sin_cos_variables.size());
  SinCosSubstitution subs_x_bar_to_x_trig{};
  Substitution subs_x_trig_to_x_bar{};
  subs_x_bar_to_x_trig.reserve(options.sin_cos_variables.size());
  int index = 0;
  int ring_index = 0;
  for (int i = 0; i < num_states; ++i) {
    if (options.sin_cos_variables.find(x[i]) !=
        options.sin_cos_variables.end()) {
      Variable s("s" + x[i].get_name());
      Variable c("c" + x[i].get_name());
      subs_x_bar_to_x_trig.emplace(x_bar[i], SinCos(s, c));
      subs_x_trig_to_x_bar.emplace(s, sin(x_bar[i]));
      subs_x_trig_to_x_bar.emplace(c, cos(x_bar[i]));
      x0_trig[index] = 0;
      x_trig[index++] = s;
      x0_trig[index] = 1;
      x_trig[index++] = c;
      ring[ring_index++] = Polynomial(s * s + c * c - 1, Variables{s, c});
    } else {
      x0_trig[index] = 0;
      x_trig[index++] = x_bar[i];
    }
  }

  // TODO(russt): Support quaternions (unit norm constraint) and other system
  // constraints.

  Expression V;
  bool user_provided_lyapunov_candidate =
      !options.lyapunov_candidate.EqualTo(Expression::Zero());

  if (user_provided_lyapunov_candidate) {
    V = options.lyapunov_candidate;

    // V = V - V(0).
    Environment env;
    for (int i = 0; i < x.size(); i++) {
      env.insert(x(i), x0(i));
    }
    const double V0 = V.Evaluate(env);
    V -= V0;

    // Check that V has the right Variables.
    DRAKE_THROW_UNLESS(V.GetVariables().IsSubsetOf(Variables(x)));

    // First convert to relative coordinates.
    V = V.Substitute(subs_x_to_x_bar);

    // Then apply the sin/cos substitution.
    if (options.sin_cos_variables.size() > 0) {
      V = Substitute(V, subs_x_bar_to_x_trig);
    }

    DRAKE_THROW_UNLESS(V.is_polynomial());

    // Check that V is SOS.
    MathematicalProgram prog;
    prog.AddIndeterminates(x_trig);
    VectorX<Polynomial> lambda_ring(ring.size());
    for (int i = 0; i < ring.size(); ++i) {
      // Take λ_ring[i] * ring[i] to >= the degree as V.
      const int lambda_i_degree = std::max(
          Polynomial(V).TotalDegree() - ring[i].TotalDegree(), 2);
      lambda_ring[i] =
          prog.NewFreePolynomial(Variables(x_trig), lambda_i_degree);
    }
    prog.AddSosConstraint(Polynomial(V, Variables(x_trig)) +
                          lambda_ring.dot(ring));
    const auto result = Solve(prog);
    if (!result.is_success()) {
      throw std::runtime_error(fmt::format(
          "'Lyapunov candidate is SOS' check failed for V - V(0)= {}", V));
    }
  } else {
    // Solve a Lyapunov equation to find a candidate.
    const auto linearized_system =
        Linearize(system, context, InputPortSelection::kNoInput,
                  OutputPortSelection::kNoOutput);
    const Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(num_states, num_states);
    const Eigen::MatrixXd P =
        math::RealContinuousLyapunovEquation(linearized_system->A(), Q);
    if (options.sin_cos_variables.size() > 0) {
      VectorX<Expression> dx = x_bar;
      index = 0;
      for (int i = 0; i < num_states; ++i) {
        if (options.sin_cos_variables.find(x[i]) !=
            options.sin_cos_variables.end()) {
          // We want a trigonometric function f(x-x0) with the following
          // properties:
          // - f(x-x0)^2 is periodic in 2π,
          // - df/dx(x) = x at 0,
          // - adequate support from Expression and SinCosSubstitution.
          //
          // f(x) = 2*sin(x/2) would be ideal, but it currently does not have
          // adequate support from our symbolic pipeline. For now we'll just
          // use f(x) = sin(x), even though it is periodic in π instead of 2π.
          dx[i] = x_trig[index];  // sin(x-x0)
          index += 2;
        } else {
          index += 1;
        }
      }
      V = dx.dot(P * dx);
    } else {
      V = x_bar.dot(P * x_bar);
    }
  }

  // Evaluate the dynamics.
  symbolic_context->SetContinuousState(x.cast<Expression>());

  // TODO(russt): Could make these more efficient for second-order systems.

  if (options.use_implicit_dynamics) {
    const auto derivatives = symbolic_system->AllocateTimeDerivatives();
    const VectorX<Variable> xdot =
        MakeVectorVariable(derivatives->size(), "xdot");
    Expression Vdot{0};
    index = 0;
    for (int i = 0; i < x.size(); ++i) {
      if (options.sin_cos_variables.find(x[i]) !=
          options.sin_cos_variables.end()) {
        // Vdot += dVdsi * ci * xdoti - dVdci * si * xdoti
        Variable s{x_trig[index++]}, c{x_trig[index++]};
        Vdot += V.Differentiate(s) * c * xdot[i];
        Vdot -= V.Differentiate(c) * s * xdot[i];
      } else {
        // Vdot += dVdxi * xdoti;
        Vdot += V.Differentiate(x_trig[index]) * xdot[i];
        index += 1;
      }
    }
    derivatives->SetFromVector(xdot.cast<Expression>());
    VectorX<Expression> g(
        symbolic_system->implicit_time_derivatives_residual_size());
    symbolic_system->CalcImplicitTimeDerivativesResidual(*symbolic_context,
                                                         *derivatives, &g);
    g = Substitute(g, subs_x_to_x_bar);
    if (options.sin_cos_variables.size() > 0) {
      g = Substitute(g, subs_x_bar_to_x_trig);
    }

    index = ring.size();
    ring.conservativeResize(ring.size() + g.size());
    for (int i = 0; i < g.size(); ++i) {
      ring[index++] = Polynomial(g[i]);
    }
    V = FixedLyapunovConvex(x_trig, x0_trig, V, Vdot, ring, xdot);
  } else {
    VectorX<Expression> f =
        symbolic_system->EvalTimeDerivatives(*symbolic_context)
            .get_vector()
            .CopyToVector();
    f = Substitute(f, subs_x_to_x_bar);
    if (options.sin_cos_variables.size() > 0) {
      f = Substitute(f, subs_x_bar_to_x_trig);
    }

    Expression Vdot{0};
    index = 0;
    for (int i = 0; i < num_states; ++i) {
      if (options.sin_cos_variables.find(x[i]) !=
          options.sin_cos_variables.end()) {
        // Vdot += dVdsi * ci * fi - dVdci * si * fi
        Variable s{x_trig[index++]}, c{x_trig[index++]};
        Vdot += V.Differentiate(s) * c * f[i];
        Vdot -= V.Differentiate(c) * s * f[i];
      } else {
        // Vdot += dVdxi * fi;
        Vdot += V.Differentiate(x_trig[index]) * f[i];
        index += 1;
      }
    }

    V = FixedLyapunovConvex(x_trig, x0_trig, V, Vdot, ring);
  }

  // Put V back into global coordinates.
  if (options.sin_cos_variables.size() > 0) {
    V = V.Substitute(subs_x_trig_to_x_bar);
  }
  V = V.Substitute(subs_x_bar_to_x);
  return V;
}

}  // namespace analysis
}  // namespace systems
}  // namespace drake
