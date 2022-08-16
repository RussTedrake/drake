#pragma once

#include <optional>

#include "drake/systems/framework/input_port.h"

namespace drake {
namespace systems {
namespace controllers {

struct ContinuousValueIterationOptions {
  /** Time step to use for any time integral approximations used in the
   algorithm. */
  double time_step{0.01};

  /** A value between (0,1] that discounts future rewards. */
  double discount_factor{1.0};

  /** For systems with multiple input ports, we must specify which input port is
   being used in the control design. */
  InputPortIndex input_port_index{0};

  /** The size of the "minibatch"; a subset of states that will be updated
   at each iteration of the algorithm. Set minibatch size to 0 to use all
   samples on every iteration of the algorithm. */
  int minibatch_size{32};

  /** The maximum number of epochs. */
  int max_epochs{1000};

  /** The number of optimization (e.g. gradient descent) steps taken during
   each epoch. */
  int optimization_steps_per_epoch{100};

  // TODO(russt): Take a SolverInterface and SolverOptions instead. */
  /** Learning rate used in by the optimization algorithm. */
  double learning_rate{1e-3};

  /** The target network weights are updated with the discrete-time moving
  average filter: `target_params = α * critic_params + (1-α) * target_params`,
  where α ∈ [0,1] is the smoothing factor. The default value corresponds
  to a time constant of approximately 1/α = 20 epochs. */
  double target_network_smoothing_factor{0.05};

  /** If callable, this method is invoked during each
   `epochs_per_visualization_callback`, in order to facilitate e.g. graphical
   inspection/debugging of the results.

   @note The call happens at the end of the iteration (after the value
   iteration update has run). */
  std::function<void(int epoch, double loss)> visualization_callback{nullptr};

  /** If `visualization_callback != nullptr`, then it will be called when `epoch
  % epochs_per_visualization_callback == 0`. */
  int epochs_per_visualization_callback{1};

  /** Lower limit on the inputs. If non-empty, then it must be the size of the
   * number of inputs of the plant. */
  Eigen::VectorXd input_lower_limit{};

  /** Upper limit on the inputs. If non-empty, then it must be the size of the
   * number of inputs of the plant. */
  Eigen::VectorXd input_upper_limit{};

  /** A matrix whose columns specify any states that are known to have zero
   * cost-to-go (e.g. a goal state). Specifying these can help to stabilize the
   * numerics of the algorithm; they are included in *every* minibatch. */
  Eigen::MatrixXd zero_value_states{};

  /** Specify number of parallel threads to use. The default value, 1, specifies
   that simulations should be executed in serial. To use the default concurrency
   available on your hardware (equivalent to
   std::thread::hardware_concurrency()), use max_threads=std::nullopt.
   Otherwise, max_threads must be >= 1. */
  std::optional<int> max_threads{1};

  /** If this string is non-empty, then ContinuousValueIteration will log
  intermediate data to the named project.  You must run e.g.
  ```
  clear && rm -f /tmp/python_rpc && mkfifo /tmp/python_rpc && bazel run //common/proto:call_python_client_cli
  ```
  In another window for the results to be published.  Make sure you have done
  `pip3 install wandb` and run `wandb login`. */
  std::string wandb_project{};
};

}  // namespace controllers
}  // namespace systems
}  // namespace drake
