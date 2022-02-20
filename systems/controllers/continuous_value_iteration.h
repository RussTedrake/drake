#pragma once

#include "drake/systems/controllers/continuous_value_iteration_options.h"
#include "drake/systems/primitives/multilayer_perceptron.h"

namespace drake {
namespace systems {
namespace controllers {

/** Implements a neural fitted value iteration for *control-affine* systems
 with continuous-time dynamics.

 For a dynamical system with time derivatives of the form:
   ẋ = f(x,u) = f₁(x) + f₂(x)u
 we attempt to minimize the infinite-horizon cost
   ∫ eᵞᵗ ℓ(x,u)dt, where ℓ(x,u) = ℓ₁(x) + uᵀRu, and R=R^T ≻ 0,
 by estimating the optimal cost-to-go function, Ĵ(x), described by a
 MultilayerPerceptron.

 At each step of the algorithm, we use the greedy policy w.r.t. Ĵ(x):
   u*(x) = argminᵤ [ ℓ(x,u) + ∂J/∂x f(x,u) ],
 which has a readily-obtained unique solution. The desired value is computed at
 (some minibatch set of) sample states, xᵢ, as the Euler approximation with time
 step, h, to the continuous-time value update: Jᵈ(x) = hℓ(x,u*) + eᵞʰĴ(x +
 hf(x,u)). Finally, the network is updated by taking several optimization steps
 to minimize the nonlinear least-squares objective min ∑ᵢ [ Ĵ(xᵢ) − Jᵈ(xᵢ) ]².

 For additional details and worked examples, see
 http://underactuated.mit.edu/dp.html .

 @param plant is a System, which must have (only) continuous-time state
 dynamics.
 @param plant_context is a Context for `plant`, which is only used to support
 plant parameters.
 @param value is a MultilayerPerceptron representing the estimated
 cost-to-go function.
 @param state_cost_function implements ℓ₁(x) in the description above.
 @param R_diagonal specifies the positive diagonal quadratic running cost uᵀRu.
 @param state_samples is a nx-by-N matrix, where nx is the number of
 continuous states in the plant. These are the xᵢ in the description above.
 @param generator is a RandomGenerator; the algorithm is deterministic given the
 state of this generator.
 @param options is a ContinuousValueIterationOptions struct.
 @param[out] value_context is a Context for `value`.  It should be
 initialized using, e.g. SetRandomContext(), or with the results from a
 proceeding run of the algorithm.

 @pre plant must be control affine. We don't current check this condition, but
 simply assume it.
 */
void ContinuousValueIteration(
    const System<double>& plant, const Context<double>& plant_context,
    const MultilayerPerceptron<double>& value,
    const std::function<double(const Context<double>&)>& state_cost,
    const Eigen::Ref<const Eigen::VectorXd>& R_diagonal,
    const Eigen::Ref<const Eigen::MatrixXd>& state_samples,
    Context<double>* value_context, RandomGenerator* generator,
    const ContinuousValueIterationOptions& options =
        ContinuousValueIterationOptions());

}  // namespace controllers
}  // namespace systems
}  // namespace drake
