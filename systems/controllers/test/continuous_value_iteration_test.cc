#include "drake/systems/controllers/continuous_value_iteration.h"

#include <cmath>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/math/discrete_algebraic_riccati_equation.h"
#include "drake/systems/primitives/linear_system.h"
#include "drake/examples/acrobot/acrobot_plant.h"

namespace drake {
namespace systems {
namespace controllers {
namespace {

// Linear quadratic regulator for the double integrator.
// q̈ = u,  g(x,u) = x'Qx + u'Ru.
// Note: we only expect the numerical solution to be very approximate, due to
// discretization errors.
GTEST_TEST(ContinuousValueIterationTest, DoubleIntegrator) {
  Eigen::Matrix2d A;
  A << 0., 1., 0., 0.;
  const Eigen::Vector2d B{0., 1.};
  const Eigen::Matrix2d C = Eigen::Matrix2d::Identity();
  const Eigen::Vector2d D = Eigen::Vector2d::Zero();
  LinearSystem<double> plant(A, B, C, D);
  auto plant_context = plant.CreateDefaultContext();

  const Eigen::Matrix2d Q = Eigen::Matrix2d::Identity();
  const Vector1d R = Vector1d::Ones();

  // Quadratic regulator state cost.
  const auto state_cost = [&Q](const Context<double>& context) {
    const Eigen::Vector2d x = context.get_continuous_state().CopyToVector();
    return x.dot(Q * x);
  };

  MultilayerPerceptron<double> value(
      {2, 16, 16, 1},
      {PerceptronActivationType::kReLU, PerceptronActivationType::kReLU,
       PerceptronActivationType::kIdentity});
  auto value_context = value.CreateDefaultContext();
  RandomGenerator generator(123);
  value.SetRandomContext(value_context.get(), &generator);

  Eigen::VectorXd q_samples = Eigen::VectorXd::LinSpaced(11, -5, 5);
  Eigen::VectorXd qdot_samples = Eigen::VectorXd::LinSpaced(15, -5, 5);
  Eigen::Matrix2Xd state_samples(2, q_samples.size() * qdot_samples.size());
  int index = 0;
  for (int i = 0; i < q_samples.size(); ++i) {
    for (int j = 0; j < qdot_samples.size(); ++j) {
      state_samples(0, index) = q_samples[i];
      state_samples(1, index++) = qdot_samples[j];
    }
  }

  ContinuousValueIterationOptions options;
  options.time_step = 0.01;
  options.discount_factor = 0.9;
  options.max_epochs = 500;
  options.target_network_smoothing_factor = 1;
  options.learning_rate = 1e-4;
  options.epochs_per_visualization_callback = 10;

  int last_epoch{0};
  double last_loss{100};
  options.visualization_callback = [&last_epoch, &last_loss](int epoch,
                                                             double loss) {
    last_epoch = epoch;
    last_loss = loss;
    std::cout << "epoch " << epoch << ": loss = " << loss << std::endl;
  };

  ContinuousValueIteration(plant, *plant_context, value, state_cost, R,
                           state_samples, value_context.get(), &generator,
                           options);

  EXPECT_EQ(last_epoch, options.max_epochs-10);
  EXPECT_LE(last_loss, 1e-4);

  // Compute the optimal solution.
  Eigen::Matrix2d S = math::DiscreteAlgebraicRiccatiEquation(
      std::sqrt(options.discount_factor) *
          (Eigen::Matrix2d::Identity() + options.time_step * A),
      options.time_step * B, options.time_step * Q,
      options.time_step * R / options.discount_factor);

  Eigen::Matrix2Xd test_samples(2, 5);
  // clang-format off
  test_samples << 0, 4, 4, -4, -4,
                  0, 4, -4, 4, -4;
  // clang-format on
  Eigen::RowVectorXd Jhat(test_samples.cols());
  Eigen::RowVectorXd Jd = (test_samples.array() * (S * test_samples).array())
                              .colwise()
                              .sum()
                              .matrix();
  value.BatchOutput(*value_context, test_samples, &Jhat);
  EXPECT_TRUE(CompareMatrices(Jhat, Jd, 0.075));
}

GTEST_TEST(ContinuousValueIterationTest, Acrobot) {
  examples::acrobot::AcrobotPlant<double> plant;
  auto plant_context = plant.CreateDefaultContext();

  Eigen::Matrix4d Q = Eigen::Vector4d(10, 10, 1, 1).asDiagonal();
  // Quadratic regulator state cost.
  const auto state_cost = [&Q](const Context<double>& context) {
    Eigen::Vector4d x = context.get_continuous_state().CopyToVector();
    x[0] -= M_PI;
    return x.dot(Q * x);
  };
  Vector1d R(1.0);

  MultilayerPerceptron<double> value(
      {true, true, false, false}, {256, 256, 1},
      {PerceptronActivationType::kReLU, PerceptronActivationType::kReLU,
       PerceptronActivationType::kIdentity});
  auto value_context = value.CreateDefaultContext();
  RandomGenerator generator(123);
  value.SetRandomContext(value_context.get(), &generator);

  Eigen::VectorXd q_samples = Eigen::VectorXd::LinSpaced(21, 0, 2*M_PI);
  Eigen::VectorXd qdot_samples = Eigen::VectorXd::LinSpaced(15, -5, 5);
  Eigen::Matrix4Xd state_samples(4, q_samples.size() * q_samples.size() *
                                        qdot_samples.size() *
                                        qdot_samples.size());
  int index = 0;
  for (int i = 0; i < q_samples.size(); ++i) {
    for (int j = 0; j < q_samples.size(); ++j) {
      for (int k = 0; k < qdot_samples.size(); ++k) {
        for (int l = 0; l < qdot_samples.size(); ++l) {
          state_samples(0, index) = q_samples[i];
          state_samples(1, index) = q_samples[j];
          state_samples(2, index) = qdot_samples[k];
          state_samples(3, index++) = qdot_samples[l];
        }
      }
    }
  }

  ContinuousValueIterationOptions options;
  options.time_step = 0.01;
  options.discount_factor = 0.99;
  options.max_epochs = 500;
  options.target_network_smoothing_factor = 0.05;
  options.learning_rate = 1e-5;
  options.epochs_per_visualization_callback = 10;

  int last_epoch{0};
  double last_loss{100};
  options.visualization_callback = [&last_epoch, &last_loss](int epoch,
                                                             double loss) {
    last_epoch = epoch;
    last_loss = loss;
    std::cout << "epoch " << epoch << ": loss = " << loss << std::endl;
  };

  ContinuousValueIteration(plant, *plant_context, value, state_cost, R,
                           state_samples, value_context.get(), &generator,
                           options);
}

}  // namespace
}  // namespace controllers
}  // namespace systems
}  // namespace drake
