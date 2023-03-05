#include "drake/systems/controllers/neural_value_iteration.h"

#include <algorithm>
#include <future>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "drake/common/text_logging.h"
#include "drake/common/proto/call_python.h"

#define LIMIT_MALLOC false

#if LIMIT_MALLOC
#include "drake/common/test_utilities/limit_malloc.h"
#endif

namespace drake {
namespace systems {
namespace controllers {

using common::CallPython;
using common::ToPythonKwargs;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::VectorXd;
using Eigen::VectorXf;

namespace {

// Adapted from the implementation in torch.optim.
// TODO(russt): Move this to drake::solvers and implement
// SolverInterface.
// TODO(russt): Implement weight decay and/or ams_grad.
template <typename T>
class Adam {
 public:
  // TODO(russt): This should be EigenPtr, but the MLP parameters currently
  // return a VectorBlock.
  explicit Adam(VectorX<T>& params) : params_{params} {
    exp_avgs_ = VectorX<T>::Zero(params_.size());
    exp_avgs_sqs_ = VectorX<T>::Zero(params_.size());
    denom_.resize(params_.size());
  }

  void set_learning_rate(T learning_rate) {
    DRAKE_DEMAND(learning_rate > 0.0);
    learning_rate_ = learning_rate;
  }

  void Reset() {
    num_steps_ = 0;
    exp_avgs_.setZero();
    exp_avgs_sqs_.setZero();
  }

  void Step(const Eigen::Ref<const VectorX<T>>& dloss_dparams) {
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
  T learning_rate_{1e-3};
  T beta1_{0.9};
  T beta2_{0.999};
  T eps_{1e-8};

  // State:
  int num_steps_{0};
  VectorX<T>& params_;
  VectorX<T> exp_avgs_;
  VectorX<T> exp_avgs_sqs_;

  // Temporary variables (to avoid dynamic memory allocations):
  VectorX<T> denom_;
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
  ThreadWorker(const Context<double>& plant_context)
      : plant_context_(plant_context.Clone()) {}

  void Precompute(const System<double>& plant,
                  const InputPort<double>& input_port,
                  const Eigen::Ref<const Eigen::MatrixXd>& state_samples,
                  const Eigen::Ref<const Eigen::MatrixXd>& input_samples,
                  const NeuralValueIterationOptions& options,
                  const std::vector<MatrixXd>::iterator next_state_iter) {
    const int batch_size = state_samples.cols();
    const int num_input_samples = input_samples.cols();

    std::vector<MatrixXd>::iterator iter = next_state_iter;
    for (int si = 0; si < batch_size; ++si) {
      plant_context_->SetContinuousState(state_samples.col(si));
      for (int ui = 0; ui < num_input_samples; ++ui) {
        input_port.FixValue(plant_context_.get(), input_samples.col(ui));
        iter->col(ui) =
            state_samples.col(si) +
            options.time_step *
                plant.EvalTimeDerivatives(*plant_context_).CopyToVector();
      }
      iter++;
    }

  }

 private:
  std::unique_ptr<Context<double>> plant_context_;
};

using BatchId = drake::Identifier<class FilterTag>;

template <typename T>
class MiniBatch {
 public:
  MiniBatch(const MultilayerPerceptron<double>& value_function,
            const Context<double>& value_context, int num_states,
            int num_input_samples, int batch_size)
      : value_function_(value_function),
        value_context_(value_context.Clone()),
        num_input_samples_{num_input_samples},
        batch_size_{batch_size},
        dloss_dparams_(value_function.num_parameters()),
        state_(num_states, batch_size),
        next_state_(num_states, batch_size * num_input_samples),
        cost_(num_input_samples, batch_size),
        Jnext_(batch_size * num_input_samples),
        Jd_(batch_size),
        batch_id_(BatchId::get_new_id()) {}

  void SetParameters(const Context<double>& value_context) {
    value_function_.SetParameters(value_context_.get(),
                                  value_function_.GetParameters(value_context));
  }

  void ComputeJd(const Eigen::Ref<const Eigen::MatrixXd>& state_samples,
                 const std::vector<MatrixXd>& next_state,
                 const Eigen::Ref<const Eigen::MatrixXd>& cost,
                 const std::vector<int>::iterator& state_index_iter,
                 const NeuralValueIterationOptions& options) {
    std::vector<int>::iterator iter = state_index_iter;
    for (int i = 0; i < batch_size_; ++i) {
      state_.col(i) = state_samples.col(*iter).cast<float>();
      next_state_.middleCols(i * num_input_samples_, num_input_samples_) =
          next_state[*iter].cast<float>(); // todo cast earlier?
      cost_.col(i) = cost.row(*iter).transpose().cast<float>();
      iter++;
    }
    value_function_.BatchOutput(*value_context_, next_state_, &Jnext_);
    Jd_ =
        (cost_ + options.discount_factor *
                     Map<MatrixX<T>>(Jnext_.data(), cost_.rows(), cost_.cols()))
            .colwise()
            .minCoeff();
  }

  double ValueIteration(const Context<double>& value_context) {
    return value_function_.BackpropagationMeanSquaredError(
        value_context, state_, Jd_, &dloss_dparams_);
  }

  int batch_size() const { return batch_size_; }
  const VectorX<T>& dloss_dparams() const { return dloss_dparams_; }

 private:
  const MultilayerPerceptron<double>& value_function_;
  std::unique_ptr<Context<double>> value_context_;
  int num_input_samples_{};
  int batch_size_{};
  VectorX<T> dloss_dparams_{};

  MatrixX<T> state_{};
  MatrixX<T> next_state_{};
  MatrixX<T> cost_{};
  RowVectorX<T> Jnext_{};
  RowVectorX<T> Jd_{};

  BatchId batch_id_;
};

int DivideAndRoundUp(int x, int y) {
  return (x + y - 1) / y;
}

}  // namespace

void NeuralValueIteration(
    const System<double>& plant, const Context<double>& plant_context,
    const MultilayerPerceptron<double>& value_function,
    const Eigen::Ref<const Eigen::MatrixXd>& state_samples,
    const Eigen::Ref<const Eigen::MatrixXd>& input_samples,
    const Eigen::Ref<const Eigen::MatrixXd>& cost,
    Context<double>* value_context, RandomGenerator* generator,
    const NeuralValueIterationOptions& options) {
  const InputPort<double>& input_port =
      plant.get_input_port(options.input_port_index);
  const int num_inputs = input_port.size();
  const int num_states = plant.num_continuous_states();
  const int num_state_samples = state_samples.cols();
  const int num_input_samples = input_samples.cols();

  plant.ValidateContext(plant_context);
  DRAKE_DEMAND(plant_context.has_only_continuous_state());
  DRAKE_DEMAND(value_function.get_input_port().size() == num_states);
  DRAKE_DEMAND(value_function.layers().back() == 1);
  DRAKE_DEMAND(state_samples.rows() == num_states);
  DRAKE_DEMAND(input_samples.rows() == num_inputs);
  DRAKE_DEMAND(cost.rows() == state_samples.cols());
  DRAKE_DEMAND(cost.cols() == input_samples.cols());
  DRAKE_DEMAND(options.time_step > 0.0);
  DRAKE_DEMAND(options.discount_factor > 0.0 && options.discount_factor <= 1.0);
  DRAKE_DEMAND(!options.minibatch_size || options.minibatch_size > 0);
  DRAKE_DEMAND(options.max_epochs > 0);
  DRAKE_DEMAND(options.optimization_steps_per_epoch > 0);
  DRAKE_DEMAND(options.learning_rate > 0);
  DRAKE_DEMAND(options.target_network_smoothing_factor >= 0 &&
               options.target_network_smoothing_factor <= 1.0);
  DRAKE_DEMAND(options.epochs_per_visualization_callback > 0);
  DRAKE_DEMAND(value_context != nullptr);
  value_function.ValidateContext(*value_context);

  if (!options.wandb_project.empty()) {
    CallPython("print('Hello!')");
    CallPython("setvars", "wandb_project", options.wandb_project);
    CallPython("eval",
               "print(f'Publishing to wandb project {wandb_project}.')");
    CallPython("exec", "import wandb");
    auto wandb_config = CallPython(
        "dict",
        ToPythonKwargs("time_step", options.time_step, "discount_factor",
                       options.discount_factor, "minibatch_size",
                       options.minibatch_size.value_or(0),
                       "optimization_steps_per_epoch",
                       options.optimization_steps_per_epoch, "learning_rate",
                       options.learning_rate, "target_network_smooth_factor",
                       options.target_network_smoothing_factor));
    CallPython("wandb.init",
              ToPythonKwargs("project", options.wandb_project, "config",
                              wandb_config, "reinit", true));
  }

  const int num_threads = SelectNumberOfThreadsToUse(options.max_threads);
  std::vector<ThreadWorker> worker;
  std::vector<std::future<void>> future(num_threads);

  for (int i = 0; i < num_threads; ++i) {
    worker.push_back(ThreadWorker(plant_context));
  }

  log()->info("Precomputing dynamics at sample points...");

  // next_state[i](:, j) = xᵢ + h*f(xᵢ, uⱼ).
  std::vector<MatrixXd> next_state(num_state_samples);
  for (int i = 0; i < num_state_samples; ++i) {
    next_state[i].resize(num_states, num_input_samples);
    next_state[i](0,0) = i;
  }

  if (num_threads == 1) {
    worker[0].Precompute(plant, input_port, state_samples, input_samples,
                         options, next_state.begin());
  } else {
    std::vector<MatrixXd>::iterator next_state_iter = next_state.begin();
    // Dispatch the worker threads.
    const int indices_per_thread =
        DivideAndRoundUp(num_state_samples, num_threads);
    int state_index = 0;
    for (int i = 0; i < num_threads; ++i) {
      int segment_size = indices_per_thread;
      if (state_index + segment_size >= num_state_samples) {
        segment_size = num_state_samples - state_index;
      }
      future[i] = std::async(
          std::launch::async, &ThreadWorker::Precompute, &worker[i],
          std::ref(plant), std::ref(input_port),
          state_samples.middleCols(state_index, segment_size),  // makes a copy
          std::ref(input_samples), std::ref(options), next_state_iter);
      next_state_iter += segment_size;
      state_index += segment_size;
    }
    // Wait for all threads to return.
    for (int i = 0; i < num_threads; ++i) {
      future[i].wait();
    }
  }

  VectorXf& value_parameters =
      value_function.GetMutableParameters(value_context);
  auto target_context  = value_context->Clone();
  VectorXf& target_parameters =
      value_function.GetMutableParameters(target_context.get());
  Adam<float> optimizer(value_parameters);
  optimizer.set_learning_rate(options.learning_rate);

  double loss{0.0};

  const int minibatch_size =
      options.minibatch_size ? *options.minibatch_size : num_state_samples;
  const int num_mini_batches =
      DivideAndRoundUp(num_state_samples, minibatch_size);
  std::vector<MiniBatch<float>> batch;
  future.resize(num_mini_batches);
  int state_index = 0;
  for (int i=0; i<num_mini_batches; ++i) {
      int batch_size = minibatch_size;
      if (state_index + minibatch_size >= num_state_samples) {
        batch_size = num_state_samples - state_index;
      }
      batch.emplace_back(MiniBatch<float>(value_function, *target_context,
                                          num_states, num_input_samples,
                                          batch_size));
      state_index += batch_size;
  }

  std::vector<int> state_indices(num_state_samples);
  std::iota(state_indices.begin(), state_indices.end(), 0);

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
    loss = 0.0;
    std::shuffle(state_indices.begin(), state_indices.end(), *generator);
    std::vector<int>::iterator state_indices_iter = state_indices.begin();
    if (num_threads == 1) {
      for (int i = 0; i < num_mini_batches; ++i) {
        batch[i].SetParameters(*target_context);
        batch[i].ComputeJd(state_samples, next_state, cost, state_indices_iter,
                           options);
        state_indices_iter += batch[i].batch_size();
      }
    } else {
      for (int i = 0; i < num_mini_batches; ++i) {
        batch[i].SetParameters(*target_context);
        future[i] =
            std::async(std::launch::async, &MiniBatch<float>::ComputeJd,
                       &batch[i], std::ref(state_samples), std::ref(next_state),
                       std::ref(cost), state_indices_iter, std::ref(options));
        state_indices_iter += batch[i].batch_size();
      }
      // Wait for all computations to return.
      for (int i = 0; i < num_mini_batches; ++i) {
        future[i].wait();
      }
    }

    for (int ostep = 0; ostep < options.optimization_steps_per_epoch; ++ostep) {
      for (int i = 0; i < num_mini_batches; ++i) {
        loss += batch[i].ValueIteration(*value_context);
        optimizer.Step(batch[i].dloss_dparams());
      }
    }
    if (options.visualization_callback &&
        epoch % options.epochs_per_visualization_callback == 0) {
      options.visualization_callback(
          epoch, loss / options.optimization_steps_per_epoch);
    }
    if (!options.wandb_project.empty()) {
      //RowVectorXd J(N);
      //value_function.BatchOutput(*value_context, batch_state, &J);
      auto wandb_log = CallPython(
          "dict",
          ToPythonKwargs("loss", loss / options.optimization_steps_per_epoch));
      CallPython("wandb.log", wandb_log);
    }

    target_parameters.noalias() =
        options.target_network_smoothing_factor * value_parameters +
        (1 - options.target_network_smoothing_factor) * target_parameters;
    
    optimizer.Reset();
  }

  // Output the (more stable) target network. This has the added benefit of
  // allowing subsequent calls to this method to continue without any
  // discontinuities in the smoothing.
  value_parameters = target_parameters;
}

}  // namespace controllers
}  // namespace systems
}  // namespace drake
