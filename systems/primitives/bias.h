#pragma once

#include "drake/common/drake_copyable.h"
#include "drake/systems/primitives/affine_system.h"

namespace drake {
namespace systems {

/// A system that outputs the input with a bias (or offset) added to it. This
/// is implemented as a particularly simple AffineSystem.
///
/// @f[
///   y = u + bias
/// @f]
///
/// @system
/// name: Bias
/// input_ports:
/// - u0
/// output_ports:
/// - y0
/// @endsystem
///
/// @tparam_default_scalar
/// @ingroup primitive_systems
///
/// @see AffineSystem
template <typename T>
class Bias final : public AffineSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(Bias)

  /// Constructs the system. The input and output ports will have the same size
  /// as @p bias.
  explicit Bias(const Eigen::Ref<const Eigen::VectorXd>& bias);

  /// Scalar-converting copy constructor.  See @ref system_scalar_conversion.
  template <typename U>
  explicit Bias(const Bias<U>&);
};

}  // namespace systems
}  // namespace drake
