#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "planning/default_state_types.h"
#include "planning/per_thread_random_source.h"
#include "planning/planning_space.h"

namespace drake {
namespace planning {
/// Base class for implementations of symmetric planning spaces, in which each
/// pair of *Forward* and *Backwards* methods is provided by a single
/// implementation.
template <typename StateType>
class SymmetricPlanningSpace : public PlanningSpace<StateType> {
 public:
  // The copy constructor is protected for use in implementing Clone().
  // Does not allow copy, move, or assignment.
  SymmetricPlanningSpace(SymmetricPlanningSpace<StateType>&&) = delete;
  SymmetricPlanningSpace& operator=(const SymmetricPlanningSpace<StateType>&) =
      delete;
  SymmetricPlanningSpace& operator=(SymmetricPlanningSpace<StateType>&&) =
      delete;

  ~SymmetricPlanningSpace() override;

  /// Implements PlanningSpace API, where each pair of *Forwards* and
  /// *Backwards* methods is provided by a single implementation.

  double NearestNeighborDistanceForwards(const StateType& from,
                                         const StateType& to) const final {
    return NearestNeighborDistance(from, to);
  }

  double NearestNeighborDistanceBackwards(const StateType& from,
                                          const StateType& to) const final {
    return NearestNeighborDistance(from, to);
  }

  double StateDistanceForwards(const StateType& from,
                               const StateType& to) const final {
    return StateDistance(from, to);
  }

  double StateDistanceBackwards(const StateType& from,
                                const StateType& to) const final {
    return StateDistance(from, to);
  }

  StateType InterpolateForwards(const StateType& from, const StateType& to,
                                double ratio) const final {
    return Interpolate(from, to, ratio);
  }

  StateType InterpolateBackwards(const StateType& from, const StateType& to,
                                 double ratio) const final {
    return Interpolate(from, to, ratio);
  }

  std::vector<StateType> PropagateForwards(
      const StateType& from, const StateType& to,
      std::map<std::string, double>* propagation_statistics) final {
    return Propagate(from, to, propagation_statistics);
  }

  std::vector<StateType> PropagateBackwards(
      const StateType& from, const StateType& to,
      std::map<std::string, double>* propagation_statistics) final {
    return Propagate(from, to, propagation_statistics);
  }

  double MotionCostForwards(const StateType& from,
                            const StateType& to) const final {
    return MotionCost(from, to);
  }

  double MotionCostBackwards(const StateType& from,
                             const StateType& to) const final {
    return MotionCost(from, to);
  }

  /// Computes the distance for use in nearest neighbors checks in (Bi)RRT
  /// planners. This method is provided to allow for faster approximate distance
  /// functions used in nearest neighbor checks.
  /// By default, the state distance function is used.
  /// @param from Starting state, from the start tree of a (Bi)RRT planner.
  /// @param to Ending state, generally a sampled state.
  /// @return Distance (potentially approximate) from start state to end state.
  virtual double NearestNeighborDistance(const StateType& from,
                                         const StateType& to) const {
    return StateDistance(from, to);
  }

  /// Computes the distance between two states. In the PRM planner, this is used
  /// for both nearest neighbor and edge weighting. In the RRT planner, this is
  /// used for goal distance checks. In the BiRRT planner, this is used for
  /// connection distance checks.
  /// @param from Starting state.
  /// @param to Ending state.
  /// @return Distance from start state to end state.
  virtual double StateDistance(const StateType& from,
                               const StateType& to) const = 0;

  /// Interpolates between the provided states. In the path processor, this is
  /// used to resample intermediate states on a path.
  /// @param from Starting state.
  /// @param to Ending state.
  /// @param ratio Interpolation ratio. @pre 0 <= ratio <= 1.
  /// @return Interpolated state.
  virtual StateType Interpolate(const StateType& from, const StateType& to,
                                double ratio) const = 0;

  /// Performs propagation from the provided start state towards the provided
  /// end state, as used in (Bi)RRT planners. All propagated states must be
  /// valid.
  /// @param from Starting state.
  /// @param to Ending state.
  /// @param propagation_statistics Map<string, double> that may be used to
  /// collect statistics about propagations for use in debugging.
  /// @return Sequence of propagated states.
  virtual std::vector<StateType> Propagate(
      const StateType& from, const StateType& to,
      std::map<std::string, double>* propagation_statistics) = 0;

  /// Computes the cost for the motion between the provided states, as would be
  /// used in a cost-sensitive (Bi)RRT (e.g. T-RRT/BiT-RRT or RRT*).
  /// By default, the state distance function is used.
  /// @param from Starting state.
  /// @param to Ending state.
  /// @return Motion cost from start state to end state.
  virtual double MotionCost(const StateType& from, const StateType& to) const {
    return StateDistance(from, to);
  }

 protected:
  // Copy constructor for use in Clone().
  SymmetricPlanningSpace(const SymmetricPlanningSpace<StateType>& other);

  /// Constructor.
  /// @param seed Seed for per-thread random source.
  /// @param supports_parallel Does the planning space support
  /// OpenMP-parallelized parallel operations?
  SymmetricPlanningSpace(uint64_t seed, bool supports_parallel_operations);
};
}  // namespace planning
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_PLANNING_STATE_TYPES(
    class ::drake::planning::SymmetricPlanningSpace)
