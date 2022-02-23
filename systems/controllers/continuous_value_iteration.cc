#include "drake/systems/controllers/continuous_value_iteration.h"

#include <algorithm>
#include <future>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
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

  void Reset() {
    num_steps_ = 0;
    exp_avgs_.setZero();
    exp_avgs_sqs_.setZero();
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

// Note: This is adapted from monte_carlo.cc.  We should probably put this
// someplace more central and standardize the syntax.
int SelectNumberOfThreadsToUse(const std::optional<int>& max_threads) {
  const int hardware_concurrency =
      static_cast<int>(std::thread::hardware_concurrency());

  int num_threads = 0;

  if (!max_threads) {
    num_threads = hardware_concurrency;
  } else {
    DRAKE_DEMAND(max_threads > 0);
    num_threads = *max_threads;
    if (num_threads > hardware_concurrency) {
      drake::log()->warn(
          "Provided max_threads value of {} is greater than the "
          "value of hardware concurrency {} for this computer, this is likely "
          "to result in poor performance",
          num_threads, hardware_concurrency);
    }
  }

  return num_threads;
}

class ThreadWorker {
 public:
  ThreadWorker(const Context<double>& model_plant_context,
               const Context<double>& value_context, int segment_start,
               int segment_size, int num_states, int num_inputs,
               const Eigen::Ref<const MatrixXd>& state_samples)
      : plant_context_(model_plant_context.Clone()),
        target_context_(value_context.Clone()),
        segment_start_(segment_start),
        segment_size_(segment_size),
        state_samples_(state_samples),
        dynamics_x_(num_states, segment_size),
        dynamics_u_(num_inputs),
        J_(segment_size),
        dJdX_(num_states, segment_size),
        next_state_(num_states, segment_size),
        dynamics_ui_(num_states, segment_size),
        cost_(segment_size),
        ui_star_(segment_size) {
    for (int i = 0; i < num_inputs; ++i) {
      dynamics_u_[i].resize(num_states, segment_size_);
    }
  }

  void Precompute(const System<double>& plant,
                  const InputPort<double>& input_port) {
    const int num_inputs = input_port.size();
    VectorXd u = VectorXd::Zero(num_inputs);

    for (int si = 0; si < segment_size_; ++si) {
      plant_context_->SetContinuousState(state_samples_.col(si));
      input_port.FixValue(plant_context_.get(), u);
      dynamics_x_.col(si) =
          plant.EvalTimeDerivatives(*plant_context_).CopyToVector();
      for (int ui = 0; ui < num_inputs; ++ui) {
        u[ui] = 1.0;
        input_port.FixValue(plant_context_.get(), u);
        dynamics_u_[ui].col(si) =
            plant.EvalTimeDerivatives(*plant_context_).CopyToVector() -
            dynamics_x_.col(si);
        u[ui] = 0.0;
      }
    }
  }

  void ComputeJd(const MultilayerPerceptron<double>& value,
                 const Eigen::Ref<const RowVectorXd>& state_cost,
                 const Eigen::Ref<const VectorXd>& R_diagonal,
                 const VectorXd& R_diagonal_inv,
                 const ContinuousValueIterationOptions& options,
                 EigenPtr<RowVectorXd> Jd) {
    const int num_inputs = dynamics_u_.size();
    const int num_states = state_samples_.rows();
    value.BatchOutput(*target_context_, state_samples_, &J_, &dJdX_);
    next_state_.noalias() = state_samples_ + options.time_step * dynamics_x_;
    cost_ = state_cost.segment(segment_start_, segment_size_);
    for (int ui = 0; ui < num_inputs; ++ui) {
      // When R is diagonal, we can evaluate each input independently,
      // including the input limits. (More generally, it would require solving
      // a QP; though I suspect the FastQP approach would work well here in
      // batch).
      ui_star_.noalias() =
          -0.5 * R_diagonal_inv(ui) *
          (dynamics_u_[ui].array() * dJdX_.array()).colwise().sum().matrix();
      if (options.input_lower_limit.size()) {
        ui_star_ = ui_star_.array().max(options.input_lower_limit[ui]);
      }
      if (options.input_upper_limit.size()) {
        ui_star_ = ui_star_.array().min(options.input_upper_limit[ui]);
      }
      cost_.noalias() += R_diagonal(ui) * ui_star_.array().square().matrix();
      for (int xi = 0; xi < num_states; ++xi) {
        next_state_.row(xi) +=
            options.time_step *
            (dynamics_u_[ui].row(xi).array() * ui_star_.array()).matrix();
      }
    }
    // Compute J(next_state).
    value.BatchOutput(*target_context_, next_state_, &J_);
    Jd->segment(segment_start_, segment_size_).noalias() =
        options.time_step * cost_ + options.discount_factor * J_;
  }

  Context<double>* target_context() const { return target_context_.get(); }
  int segment_start() const { return segment_start_; }
  int segment_size() const { return segment_size_; }

 private:
  std::unique_ptr<Context<double>> plant_context_;
  std::unique_ptr<Context<double>> target_context_;
  int segment_start_;
  int segment_size_;

  MatrixXd state_samples_;
  MatrixXd dynamics_x_;
  std::vector<MatrixXd> dynamics_u_;

  // Scratch data:
  RowVectorXd J_;
  MatrixXd dJdX_;
  MatrixXd next_state_;
  MatrixXd dynamics_ui_;
  RowVectorXd cost_;
  RowVectorXd ui_star_;
};

}  // namespace

void ContinuousValueIteration(
    const System<double>& plant, const Context<double>& model_plant_context,
    const MultilayerPerceptron<double>& value,
    const Eigen::Ref<const Eigen::MatrixXd>& state_samples,
    const Eigen::Ref<const Eigen::RowVectorXd>& state_cost,
    const Eigen::Ref<const Eigen::VectorXd>& R_diagonal,
    Context<double>* value_context, RandomGenerator* generator,
    const ContinuousValueIterationOptions& options) {
  const InputPort<double>& input_port =
      plant.get_input_port(options.input_port_index);
  const int num_inputs = input_port.size();
  const int num_states = plant.num_continuous_states();
  const int num_samples = state_samples.cols();

  plant.ValidateContext(model_plant_context);
  DRAKE_DEMAND(model_plant_context.has_only_continuous_state());
  DRAKE_DEMAND(value.get_input_port().size() == num_states);
  DRAKE_DEMAND(value.layers().back() == 1);
  DRAKE_DEMAND(state_samples.rows() == num_states);
  DRAKE_DEMAND(state_cost.size() == num_samples);
  DRAKE_DEMAND(R_diagonal.size() == num_inputs);
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
  DRAKE_DEMAND(options.zero_value_states.size() == 0 ||
               options.zero_value_states.rows() == num_states);
  DRAKE_DEMAND(options.zero_value_states.size() == 0 ||
               options.zero_value_states.cols() < options.minibatch_size);
  DRAKE_DEMAND(value_context != nullptr);
  value.ValidateContext(*value_context);
  VectorXd R_diagonal_inv = (1.0 / R_diagonal.array()).matrix();

  const int num_threads = SelectNumberOfThreadsToUse(options.max_threads);
  std::vector<ThreadWorker> worker;
  std::vector<Eigen::VectorBlock<VectorXd>> target_parameters;
  std::vector<std::future<void>> future(num_threads);

  const int indices_per_thread =
      std::ceil(static_cast<double>(num_samples) / num_threads);
  int state_index = 0;
  for (int i = 0; i < num_threads; ++i) {
    int segment_size = indices_per_thread;
    if (state_index + segment_size >= num_samples) {
      segment_size = num_samples - state_index;
    }
    // Note: slicing state_samples could be inefficient if it's passed in
    // row-major (e.g. from Python); so we go ahead and copy it into the
    // per-thread data here.
    worker.push_back(
        ThreadWorker(model_plant_context, *value_context, state_index,
                     segment_size, num_states, num_inputs,
                     state_samples.middleCols(state_index, segment_size)));
    state_index += segment_size;
    target_parameters.push_back(
        value.GetMutableParameters(worker[i].target_context()));
  }

  // Precompute dynamics and cost.
  drake::log()->info("Precomputing dynamics at sample points...");
  if (num_threads == 1) {
    worker[0].Precompute(plant, input_port);
  } else {
    // Dispatch the worker threads.
    for (int i = 0; i < num_threads; ++i) {
      future[i] = std::async(std::launch::async, &ThreadWorker::Precompute,
                             &worker[i], std::ref(plant), std::ref(input_port));
    }
    // Wait for all threads to return.
    for (int i = 0; i < num_threads; ++i) {
      future[i].wait();
    }
  }

  Eigen::VectorBlock<VectorXd> value_parameters =
      value.GetMutableParameters(value_context);
  Adam optimizer(value_parameters);
  optimizer.set_learning_rate(options.learning_rate);

  RowVectorXd Jd(num_samples);
  VectorXd dloss_dparams(value.num_parameters());
  double loss{0.0};

  const int N =
      (options.minibatch_size > 0 ? options.minibatch_size : num_samples) +
      options.zero_value_states.cols();
  Eigen::VectorXi batch(N);
  MatrixXd batch_state(num_states, N);
  RowVectorXd batch_Jd(N);

  batch_state.leftCols(options.zero_value_states.cols()) =
      options.zero_value_states;
  batch_Jd.head(options.zero_value_states.cols()).setZero();

  std::uniform_int_distribution random_sample(0, num_samples - 1);

  // Note: During development, the loop below was tested will a limit malloc
  // guard to have zero allocations after one call to Backpropagation.
#if LIMIT_MALLOC
  for (int i = 0; i < num_threads; ++i) {
    worker[i].ComputeJd(value, state_cost, R_diagonal, R_diagonal_inv, options,
                        &Jd);
  }
  value.BackpropagationMeanSquaredError(*value_context, batch_state, batch_Jd,
                                        &dloss_dparams);
#endif

  drake::log()->info("Running value iteration...");
  for (int epoch = 0; epoch < options.max_epochs; ++epoch) {
#if LIMIT_MALLOC
    // std::async allocates.
    test::LimitMalloc guard(
        {.max_num_allocations = num_threads > 1 ? 3 * num_threads : 0});
#endif
    if (num_threads == 1) {
      worker[0].ComputeJd(value, state_cost, R_diagonal, R_diagonal_inv,
                          options, &Jd);
    } else {
      // Dispatch the worker threads.
      for (int i = 0; i < num_threads; ++i) {
        future[i] = std::async(
            std::launch::async, &ThreadWorker::ComputeJd, &worker[i],
            std::ref(value), std::ref(state_cost), std::ref(R_diagonal),
            std::ref(R_diagonal_inv), std::ref(options), &Jd);
      }
      // Wait for all threads to return.
      for (int i = 0; i < num_threads; ++i) {
        future[i].wait();
      }
    }
    loss = 0.0;
    for (int ostep = 0; ostep < options.optimization_steps_per_epoch; ++ostep) {
      for (int i = options.zero_value_states.cols(); i < N; ++i) {
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
    target_parameters[0].noalias() =
        options.target_network_smoothing_factor * value_parameters +
        (1 - options.target_network_smoothing_factor) * target_parameters[0];
    for (int i = 1; i < num_threads; ++i) {
      target_parameters[i] = target_parameters[0];
    }
    optimizer.Reset();
  }

  // Output the (more stable) target network. This has the added benefit of
  // allowing subsequent calls to this method to continue without any
  // discontinuities in the smoothing.
  value_parameters = target_parameters[0];
}

}  // namespace controllers
}  // namespace systems
}  // namespace drake
