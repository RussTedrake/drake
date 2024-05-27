#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "drake/common/drake_copyable.h"

namespace drake {
namespace planning {
/// Provides a per-OpenMP-thread source of random numbers, used in the
/// PlanningSpace API.
class PerThreadRandomSource {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(PerThreadRandomSource)

  /// Constructor.
  /// @param seed Seed to use when constructing random number generators.
  explicit PerThreadRandomSource(uint64_t seed);

  /// Draw a uniformly distributed double in [0.0, 1.0) from the current OpenMP
  /// thread's generator and uniform distribution.
  /// @return Random double in [0.0, 1.0).
  double DrawUniformUnitReal();

  /// Draw a random value from the current OpenMP thread's generator.
  /// @return Random uint64_t.
  uint64_t DrawRaw();

  /// Reseed the per-thread generators using the provided initial seed.
  /// @param seed Initial seed to reseed per-thread generators.
  void ReseedGenerators(uint64_t seed);

  /// Reseed the generators, e.g. with the return value from a prior call to
  /// SnapshotGeneratorSeeds(), reproduce a known state of the generators.
  /// @param seeds Seeds to use in the per-thread generators. @pre number of
  /// seeds must match number of per-thread generators.
  void ReseedGeneratorsIndividually(const std::vector<uint64_t>& seeds);

  /// Capture seeds from the generators to produce a known state of the
  /// generators that can be reproduced later.
  /// @return Seeds for the per-thread generators.
  std::vector<uint64_t> SnapshotGeneratorSeeds();

 private:
  std::vector<std::mt19937_64> generators_;
};

}  // namespace planning
}  // namespace drake
