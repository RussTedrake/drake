#define EIGEN_USE_BLAS
#include "drake/systems/controllers/continuous_value_iteration.h"

#include <algorithm>
#include <vector>

#define LIMIT_MALLOC false

#include "drake/common/text_logging.h"

#if LIMIT_MALLOC
#include "drake/common/test_utilities/limit_malloc.h"
#endif

namespace drake {
namespace systems {
namespace controllers {

using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::VectorXd;

namespace {

// Adapted from the implementation in torch.optim.
// TODO(russt): Move this to drake::solvers and implement
// SolverInterface.
// TODO(russt): Implement weight decay and/or ams_grad.
class Adam {
 public:
  // TODO(russt): This should be EigenPtr, but the MLP parameters currently
  // return a VectorBlock.
  explicit Adam(Eigen::VectorBlock<VectorXd> params) : params_{params} {
    exp_avgs_ = VectorXd::Zero(params_.size());
    exp_avgs_sqs_ = VectorXd::Zero(params_.size());
    denom_.resize(params_.size());
  }

  void set_learning_rate(double learning_rate) {
    DRAKE_DEMAND(learning_rate > 0.0);
    learning_rate_ = learning_rate;
  }

  void Step(const Eigen::Ref<const VectorXd>& dloss_dparams) {
    num_steps_ += 1;

    const double bias_correction1 = 1 - std::pow(beta1_, num_steps_);
    const double bias_correction2 = 1 - std::pow(beta2_, num_steps_);

    exp_avgs_ *= beta1_;
    exp_avgs_.noalias() += (1 - beta1_) * dloss_dparams;

    exp_avgs_sqs_ *= beta2_;
    exp_avgs_sqs_.noalias() +=
        (1 - beta2_) * (dloss_dparams.array().square().matrix());

    denom_.noalias() =
        ((exp_avgs_sqs_.array().sqrt() / std::sqrt(bias_correction2)) + eps_)
            .matrix();
    const double step_size = learning_rate_ / bias_correction1;
    params_.noalias() -=
        (step_size * exp_avgs_.array() / denom_.array()).matrix();
  }

 private:
  // Parameters:
  double learning_rate_{1e-3};
  double beta1_{0.9};
  double beta2_{0.999};
  double eps_{1e-8};

  // State:
  int num_steps_{0};
  Eigen::VectorBlock<VectorXd> params_;
  VectorXd exp_avgs_;
  VectorXd exp_avgs_sqs_;

  // Temporary variables (to avoid dynamic memory allocations):
  VectorXd denom_;
};

}  // namespace

void ContinuousValueIteration(
    const System<double>& plant, const Context<double>& model_plant_context,
    const MultilayerPerceptron<double>& value,
    const std::function<double(const Context<double>&)>& state_cost,
    const Eigen::Ref<const VectorXd>& R_diagonal,
    const Eigen::Ref<const MatrixXd>& state_samples,
    Context<double>* value_context, RandomGenerator* generator,
    const ContinuousValueIterationOptions& options) {
  const InputPort<double>& input_port =
      plant.get_input_port(options.input_port_index);
  const int num_inputs = input_port.size();
  const int num_states = plant.num_continuous_states();
  const int num_samples = state_samples.cols();

  plant.ValidateContext(model_plant_context);
  auto plant_context = model_plant_context.Clone();
  DRAKE_DEMAND(plant_context->has_only_continuous_state());
  DRAKE_DEMAND(value.get_input_port().size() == num_states);
  DRAKE_DEMAND(value.layers().back() == 1);
  DRAKE_DEMAND(R_diagonal.size() == num_inputs);
  DRAKE_DEMAND(state_samples.rows() == num_states);
  DRAKE_DEMAND(options.time_step > 0.0);
  DRAKE_DEMAND(options.discount_factor > 0.0 && options.discount_factor <= 1.0);
  DRAKE_DEMAND(options.minibatch_size >= 0);
  DRAKE_DEMAND(options.max_epochs > 0);
  DRAKE_DEMAND(options.optimization_steps_per_epoch > 0);
  DRAKE_DEMAND(options.learning_rate > 0);
  DRAKE_DEMAND(options.target_network_smoothing_factor >= 0 &&
               options.target_network_smoothing_factor <= 1.0);
  DRAKE_DEMAND(options.epochs_per_visualization_callback > 0);
  DRAKE_DEMAND(options.input_lower_limit.size() == 0 ||
               options.input_lower_limit.size() == num_inputs);
  DRAKE_DEMAND(options.input_upper_limit.size() == 0 ||
               options.input_upper_limit.size() == num_inputs);
  DRAKE_DEMAND(value_context != nullptr);
  value.ValidateContext(*value_context);

  auto target_context = value_context->Clone();

  // Precompute dynamics and cost.
  VectorXd cost_x(num_samples);
  MatrixXd R_diagonal_inv = (1.0 / R_diagonal.array()).matrix();
  MatrixXd dynamics_x(num_states, num_samples);
  std::vector<MatrixXd> dynamics_u(num_inputs);
  for (int ui = 0; ui < num_inputs; ++ui) {
    dynamics_u[ui].resize(num_states, num_samples);
  }
  VectorXd u = VectorXd::Zero(num_inputs);

  drake::log()->info("Precomputing dynamics at sample points...");
  // TODO(russt): Parallelize this loop.
  for (int si = 0; si < num_samples; ++si) {
    plant_context->SetContinuousState(state_samples.col(si));
    input_port.FixValue(plant_context.get(), u);
    cost_x[si] = state_cost(*plant_context);
    dynamics_x.col(si) =
        plant.EvalTimeDerivatives(*plant_context).CopyToVector();
    for (int ui = 0; ui < num_inputs; ++ui) {
      u[ui] = 1.0;
      input_port.FixValue(plant_context.get(), u);
      dynamics_u[ui].col(si) =
          plant.EvalTimeDerivatives(*plant_context).CopyToVector() -
          dynamics_x.col(si);
      u[ui] = 0.0;
    }
  }

  Eigen::VectorBlock<VectorXd> value_parameters =
      value.GetMutableParameters(value_context);
  Eigen::VectorBlock<VectorXd> target_parameters =
      value.GetMutableParameters(target_context.get());
  Adam optimizer(value_parameters);
  optimizer.set_learning_rate(options.learning_rate);

  RowVectorXd Jd(num_samples);

  const int N =
      options.minibatch_size > 0 ? options.minibatch_size : num_samples;
  RowVectorXd J(num_samples);
  RowVectorXd Jnext(num_samples);
  MatrixXd dJdX(num_states, num_samples);
  VectorXd dloss_dparams(value.num_parameters());
  MatrixXd next_state(num_states, num_samples);
  MatrixXd dynamics_ui(num_states, num_samples);
  RowVectorXd cost(num_samples);
  RowVectorXd ui_star(num_samples);
  double loss{0.0};

  Eigen::VectorXi indices =
      Eigen::VectorXi::LinSpaced(num_samples, 0, num_samples);
  Eigen::VectorXi batch(N);
  MatrixXd batch_state(num_states, N);
  RowVectorXd batch_Jd(N);

  std::uniform_int_distribution random_sample(0, num_samples - 1);

  // Note: During development, the loop below was tested will a limit malloc
  // guard to have zero allocations after one call to Backpropagation.
#if LIMIT_MALLOC
  value.BatchOutput(*target_context, state_samples, &J, &dJdX);
  value.BackpropagationMeanSquaredError(*value_context, batch_state, batch_Jd,
                                        &dloss_dparams);
#endif

  // TODO(russt): Consider a softer moving average for the target network,
  // instead of computing Jd from the current critic.  E.g. target_params =
  // alpha*target_params + (1-alpha)*critic_params.

  drake::log()->info("Running value iteration...");
  for (int epoch = 0; epoch < options.max_epochs; ++epoch) {
#if LIMIT_MALLOC
    test::LimitMalloc guard({.max_num_allocations = 0});
#endif
    value.BatchOutput(*target_context, state_samples, &J, &dJdX);
    next_state.noalias() = state_samples + options.time_step * dynamics_x;
    cost = cost_x;
    for (int ui = 0; ui < num_inputs; ++ui) {
      // When R is diagonal, we can evaluate each input independently,
      // including the input limits. (More generally, it would require solving
      // a QP; though I suspect the FastQP approach would work well here in
      // batch).
      ui_star.noalias() =
          -0.5 * R_diagonal_inv(ui) *
          (dynamics_ui.array() * dJdX.array()).colwise().sum().matrix();
      if (options.input_lower_limit.size()) {
        ui_star = ui_star.array().max(options.input_lower_limit[ui]);
      }
      if (options.input_upper_limit.size()) {
        ui_star = ui_star.array().min(options.input_upper_limit[ui]);
      }
      cost.noalias() += R_diagonal(ui) * ui_star.array().square().matrix();
      for (int xi = 0; xi < num_states; ++xi) {
        next_state.row(xi) +=
            options.time_step *
            (dynamics_ui.row(xi).array() * ui_star.array()).matrix();
      }
    }
    value.BatchOutput(*target_context, next_state, &Jnext);
    Jd.noalias() =
        options.time_step * cost + options.discount_factor * Jnext;
    loss = 0.0;
    for (int ostep = 0; ostep < options.optimization_steps_per_epoch; ++ostep) {
      for (int i = 0; i < N; ++i) {
        const int index = random_sample(*generator);
        batch_state.col(i) = state_samples.col(index);
        batch_Jd[i] = Jd[index];
      }
      loss += value.BackpropagationMeanSquaredError(*value_context, batch_state,
                                                    batch_Jd, &dloss_dparams);
      optimizer.Step(dloss_dparams);
    }
    if (options.visualization_callback &&
        epoch % options.epochs_per_visualization_callback == 0) {
      options.visualization_callback(
          epoch, loss / options.optimization_steps_per_epoch);
    }
    target_parameters.noalias() =
        options.target_network_smoothing_factor * value_parameters +
        (1 - options.target_network_smoothing_factor) * target_parameters;
  }
}

}  // namespace controllers
}  // namespace systems
}  // namespace drake
