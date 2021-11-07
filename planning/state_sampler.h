#pragma once

#include <vector>

#include <Eigen/Dense>

namespace drake {
namespace planning {

/// Interface for samplers to use.
class StateSampler {
 public:
  explicit StateSampler(int size, bool supports_parallel_sampling = false)
      : size_(size), supports_parallel_sampling_(supports_parallel_sampling) {}

  virtual ~StateSampler() {}

  int size() const { return size_; }

  bool supports_parallel_sampling() const {
    return supports_parallel_sampling_;
  }

  /// Sample a single state.
  Eigen::VectorXd Sample(RandomGenerator* generator) const {
    return DoSample(generator);
  };

  std::vector<Eigen::VectorXd> SampleStates(RandomGenerator* generator,
                                            int num_samples) const;


 protected:
  virtual Eigen::VectorXd DoSample(RandomGenerator* generator) { 
    throw std::runtime_error("Subclasses must implement DoSample().");      
  };

 private:
  int size_;
  bool supports_parallel_sampling_;
};

// TODO(russt): UniformPositionSampler : StateSampler which just returns uniform
// within the joint limits.
// TODO(russt): RandomMultibodyStateSampler : StateSampler which calls
// plant.SetRandomState().

}  // namespace planning
}  // namespace drake
