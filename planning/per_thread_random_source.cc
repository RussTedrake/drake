#include "planning/per_thread_random_source.h"

#include <algorithm>
#include <limits>

#include <common_robotics_utilities/openmp_helpers.hpp>

#include "drake/common/drake_throw.h"
#include "drake/common/text_logging.h"

namespace drake {
namespace planning {
using common_robotics_utilities::openmp_helpers::GetContextOmpThreadNum;

PerThreadRandomSource::PerThreadRandomSource(const uint64_t seed) {
  // Make a generator and distribution for each thread.
  const int num_omp_threads =
      common_robotics_utilities::openmp_helpers::GetNumOmpThreads();
  const int max_num_omp_threads =
      common_robotics_utilities::openmp_helpers::GetMaxNumOmpThreads();
  const int omp_thread_limit =
      common_robotics_utilities::openmp_helpers::GetOmpThreadLimit();
  const bool omp_enabled_in_build =
      common_robotics_utilities::openmp_helpers::IsOmpEnabledInBuild();
  const int num_threads = std::max(num_omp_threads, max_num_omp_threads);
  drake::log()->info(
      "PerThreadRandomSource allocating random generators to support {} "
      "parallel queries given omp_num_threads {} omp_max_threads {} and "
      "omp_thread_limit {} OpenMP enabled in build? {}",
      num_threads, num_omp_threads, max_num_omp_threads, omp_thread_limit,
      omp_enabled_in_build);
  std::mt19937_64 seed_dist(seed);
  for (int thread = 0; thread < num_threads; ++thread) {
    generators_.emplace_back(std::mt19937_64(seed_dist()));
  }
}

double PerThreadRandomSource::DrawUniformUnitReal() {
  return std::generate_canonical<double, std::numeric_limits<double>::digits>(
      generators_.at(GetContextOmpThreadNum()));
}

uint64_t PerThreadRandomSource::DrawRaw() {
  return generators_.at(GetContextOmpThreadNum())();
}

void PerThreadRandomSource::ReseedGenerators(const uint64_t seed) {
  std::mt19937_64 seed_dist(seed);
  for (size_t index = 0; index < generators_.size(); ++index) {
    const uint64_t new_seed = seed_dist();
    generators_.at(index).seed(new_seed);
  }
}

void PerThreadRandomSource::ReseedGeneratorsIndividually(
    const std::vector<uint64_t>& seeds) {
  DRAKE_THROW_UNLESS(seeds.size() == generators_.size());
  for (size_t index = 0; index < generators_.size(); ++index) {
    generators_.at(index).seed(seeds.at(index));
  }
}

std::vector<uint64_t> PerThreadRandomSource::SnapshotGeneratorSeeds() {
  std::vector<uint64_t> snapshot_seeds(generators_.size(), 0);
  for (size_t index = 0; index < generators_.size(); ++index) {
    auto& generator = generators_.at(index);
    const uint64_t new_seed = generator();
    snapshot_seeds.at(index) = new_seed;
    generator.seed(new_seed);
  }
  return snapshot_seeds;
}

}  // namespace planning
}  // namespace drake
