#include "drake/systems/primitives/bias.h"

#include "drake/common/default_scalars.h"

namespace drake {
namespace systems {

namespace {
constexpr int kNumStates{0};
}  // namespace

template <typename T>
Bias<T>::Bias(const Eigen::Ref<const Eigen::VectorXd>& bias)
    : AffineSystem<T>(
          SystemTypeTag<Bias>{},
          Eigen::MatrixXd::Zero(kNumStates, kNumStates),  // A
          Eigen::MatrixXd::Zero(kNumStates, bias.size()), // B
          Eigen::VectorXd::Zero(kNumStates),              // f0
          Eigen::MatrixXd::Zero(bias.size(), kNumStates), // C
          Eigen::MatrixXd::Identity(bias.size()),         // D
          bias,                                           // y0 
          0.0 /* time_period */) {}

template <typename T>
template <typename U>
Bias<T>::Bias(const Bias<U>& other)
    : Bias<T>(other.y0()) {}

}  // namespace systems
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::systems::Bias)
