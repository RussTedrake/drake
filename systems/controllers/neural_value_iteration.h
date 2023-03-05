#pragma once

#include "drake/systems/controllers/neural_value_iteration_options.h"
#include "drake/systems/primitives/multilayer_perceptron.h"

namespace drake {
namespace systems {
namespace controllers {

/** Implements a neural fitted value iteration discrete-time systems.

 For a dynamical system with dynamics of the form:
    ẋ = f(x,u),
 approximated by x[n+1] = x[n] + h*f(x,u), we attempt to minimize the
 infinite-horizon cost
    ∑ₙ γⁿ ℓ(x,u),
 by estimating the optimal cost-to-go function using a function, Ĵ(x),
 described by a MultilayerPerceptron.

 Note that ℓ(x,u) is the finite-time one-step cost. Continuous-time costs should
 already be integrated over the time step, e.g. by multiplying by the time_step.

 For additional details and worked examples, see
 http://underactuated.mit.edu/dp.html .

 @param plant is a System, which must have (only) continuous-time state
 dynamics.
 @param plant_context is a Context for `plant`, which is only used to support
 plant parameters.
 @param value is a MultilayerPerceptron representing the estimated cost-to-go
 function.
 @param state_samples is an nx-by-M matrix, where nx is the number of
 continuous states in the plant.
 @param input_samples is an nu-by-N matrix, where nu is the size of
 plant.get_input_port(options.input_port_index).
 @param cost is a matrix of length MxN with values ℓ(state_samples,
 input_samples).
 @param generator is a RandomGenerator; the algorithm is deterministic given
 the state of this generator.
 @param options is a NeuralValueIterationOptions struct.
 @param[out] value_context is a Context for `value_function`.  It should be
 initialized using, e.g. SetRandomContext(), or with the results from a
 proceeding run of the algorithm.

 @pre plant must be control affine. We don't current check this condition, but
 simply assume it.
 */
void NeuralValueIteration(
    const System<double>& plant, const Context<double>& plant_context,
    const MultilayerPerceptron<double>& value_function,
    const Eigen::Ref<const Eigen::MatrixXd>& state_samples,
    const Eigen::Ref<const Eigen::MatrixXd>& input_samples,
    const Eigen::Ref<const Eigen::MatrixXd>& cost,
    Context<double>* value_context, RandomGenerator* generator,
    const NeuralValueIterationOptions& options =
        NeuralValueIterationOptions());

}  // namespace controllers
}  // namespace systems
}  // namespace drake
