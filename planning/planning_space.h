#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "planning/default_state_types.h"
#include "planning/per_thread_random_source.h"
#include "planning/valid_starts.h"
#include "planning/valid_starts_and_goals.h"

namespace drake {
namespace planning {
/// Provides a single interface to state distance, state/edge validity, state
/// propagation, etc as used by motion planners. All sampling-based motion
/// planner methods and the path processor take a PlanningSpace. Generally,
/// support for OpenMP parallelism is provided by a single PlanningSpace, while
/// support for non-OpenMP parallelism is provided by cloning the planning space
/// and using one clone per thread.
// TODO(calderpg) Consider if motion cost should be wrapped into Propagate*()
// calls and/or edge evaluation? Some "expensive" planning spaces would benefit
// from this approach.
template <typename StateType>
class PlanningSpace {
 public:
  // The copy constructor is protected for use in implementing Clone().
  // Does not allow copy, move, or assignment.
  PlanningSpace(PlanningSpace<StateType>&&) = delete;
  PlanningSpace& operator=(const PlanningSpace<StateType>&) = delete;
  PlanningSpace& operator=(PlanningSpace<StateType>&&) = delete;

  virtual ~PlanningSpace();

  /// Clones the current planning space. Clones *must not* share mutable state,
  /// as cloning is used to provide thread safety in parallel (Bi)RRT planners.
  virtual std::unique_ptr<PlanningSpace<StateType>> Clone() const = 0;

  /// Computes the "forwards" distance for use in nearest neighbors checks in
  /// (Bi)RRT planners. All nearest neighbor checks in RRT planners and nearest
  /// neighbor checks between the start tree and sampled states in BiRRT
  /// planners use this method. This method is provided to allow for faster
  /// approximate distance functions used in nearest neighbor checks.
  /// By default, the forwards state distance function is used.
  /// @param from Starting state, from the start tree of a (Bi)RRT planner.
  /// @param to Ending state, generally a sampled state.
  /// @return Distance (potentially approximate) from start state to end state.
  virtual double NearestNeighborDistanceForwards(const StateType& from,
                                                 const StateType& to) const {
    return StateDistanceForwards(from, to);
  }

  /// Computes the "backwards" distance for use in nearest neighbors checks in
  /// BiRRT planners. Nearest neighbor checks between the goal tree and sampled
  /// states in BiRRT planners use this method. This method is provided to allow
  /// for faster approximate distance functions used in nearest neighbor checks.
  /// By default, the backwards state distance function is used.
  /// @param from Starting state, from the goal tree of a (Bi)RRT planner.
  /// @param to Ending state, generally a sampled state.
  /// @return Distance (potentially approximate) from start state to end state.
  virtual double NearestNeighborDistanceBackwards(const StateType& from,
                                                  const StateType& to) const {
    return StateDistanceBackwards(from, to);
  }

  /// Computes the "forwards" distance between two states. In the PRM planner,
  /// this is used for both nearest neighbor and edge weighting. In the RRT
  /// planner, this is used for goal distance checks. In the BiRRT planner, this
  /// is used for connection distance checks from start tree to goal tree.
  /// @param from Starting state.
  /// @param to Ending state.
  /// @return Distance from start state to end state.
  virtual double StateDistanceForwards(const StateType& from,
                                       const StateType& to) const = 0;

  /// Computes the "backwards" distance between two states. In the BiRRT
  /// planner, this is used for connection distance checks from goal tree to
  /// start tree.
  /// @param from Starting state.
  /// @param to Ending state.
  /// @return Distance from start state to end state.
  virtual double StateDistanceBackwards(const StateType& from,
                                        const StateType& to) const = 0;

  /// Calculates the length of the provided path, as the sum of the forward
  /// distance of each edge.
  /// @param path Provided path.
  /// @return Length of path.
  double CalcPathLength(const std::vector<StateType>& path) const;

  /// Interpolates "forwards" between the provided states. In the path
  /// processor, this is used to resample intermediate states on a path.
  /// @param from Starting state.
  /// @param to Ending state.
  /// @param ratio Interpolation ratio. @pre 0 <= ratio <= 1.
  /// @return Interpolated state.
  virtual StateType InterpolateForwards(const StateType& from,
                                        const StateType& to,
                                        double ratio) const = 0;

  /// Interpolates "backwards" between the provided states.
  /// @param from Starting state.
  /// @param to Ending state.
  /// @param ratio Interpolation ratio. @pre 0 <= ratio <= 1.
  /// @return Interpolated state.
  virtual StateType InterpolateBackwards(const StateType& from,
                                         const StateType& to,
                                         double ratio) const = 0;

  /// Performs "forwards" propagation from the provided start state towards the
  /// provided end state, as used in RRT planner and forward propagations of
  /// the BiRRT planner. All propagated states must be valid.
  /// @param from Starting state.
  /// @param to Ending state.
  /// @param propagation_statistics Map<string, double> that may be used to
  /// collect statistics about propagations for use in debugging.
  /// @return Sequence of propagated states.
  virtual std::vector<StateType> PropagateForwards(
      const StateType& from, const StateType& to,
      std::map<std::string, double>* propagation_statistics) = 0;

  /// Performs "backwards" propagation from the provided start state towards the
  /// provided end state, as used in backwards propagations of the BiRRT
  /// planner. All propagated states must be valid.
  /// @param from Starting state.
  /// @param to Ending state.
  /// @param propagation_statistics Map<string, double> that may be used to
  /// collect statistics about propagations for use in debugging.
  /// @return Sequence of propagated states.
  virtual std::vector<StateType> PropagateBackwards(
      const StateType& from, const StateType& to,
      std::map<std::string, double>* propagation_statistics) = 0;

  /// Computes the "forwards" cost for the motion between the provided states,
  /// as would be used in a cost-sensitive RRT or forwards propagations in a
  /// cost-sensitive BiRRT (e.g. T-RRT/BiT-RRT or RRT*).
  /// By default, the forwards state distance function is used.
  /// @param from Starting state.
  /// @param to Ending state.
  /// @return Motion cost from start state to end state.
  virtual double MotionCostForwards(const StateType& from,
                                    const StateType& to) const {
    return StateDistanceForwards(from, to);
  }

  /// Computes the "backwards" cost for the motion between the provided states,
  /// as would be used in backwards propagations in a cost-sensitive BiRRT (e.g.
  /// BiT-RRT or RRT*).
  /// By default, the backwards state distance function is used.
  /// @param from Starting state.
  /// @param to Ending state.
  /// @return Motion cost from start state to end state.
  virtual double MotionCostBackwards(const StateType& from,
                                     const StateType& to) const {
    return StateDistanceBackwards(from, to);
  }

  /// Checks the validity (e.g. if the state is collision free and within
  /// limits) of the provided state.
  /// @param state State to check.
  /// @return true if the provided state is valid.
  virtual bool CheckStateValidity(const StateType& state) const = 0;

  /// Extracts the valid states from the provided start states. Used in some
  /// forms of the RRT planner.
  /// @param starts Potential start states.
  /// @return Valid start states.
  ValidStarts<StateType> ExtractValidStarts(
      const std::vector<StateType>& starts) const;

  /// Extracts the valid states from the provided start and goal states. Used in
  /// PRM and (Bi)RRT planners.
  /// @param starts Potential start states.
  /// @param goals Potential goal states.
  /// @return Valid start and goal states.
  ValidStartsAndGoals<StateType> ExtractValidStartsAndGoals(
      const std::vector<StateType>& starts,
      const std::vector<StateType>& goals) const;

  /// Checks the validity of the provided directed edge.
  /// @param from Starting state of edge.
  /// @param to Ending state of edge.
  /// @return true if edge is valid.
  virtual bool CheckEdgeValidity(const StateType& from,
                                 const StateType& to) const = 0;

  /// Checks the validity of the provided path. By default, this checks the
  /// validity of each edge in the path, but may be overriden to provide a more
  /// efficient implementation (e.g. context/constraint reuse).
  /// @param path Provided path. @pre path is non-empty.
  /// @return true if entire path is valid.
  virtual bool CheckPathValidity(const std::vector<StateType>& path) const;

  /// Samples a state (not necessarily a valid state).
  /// @return Sampled state.
  virtual StateType SampleState() = 0;

  /// Attempts to sample a valid state, optionally taking up to max_attempts
  /// tries.
  /// @param max_attempts Maximum number of tries to sample a valid state.
  /// @pre > 0.
  /// @return Valid sample state or nullopt if no valid state could be sampled.
  virtual std::optional<StateType> MaybeSampleValidState(int max_attempts);

  /// Same as MaybeSampleValidState, but throws if no valid state could be
  /// sampled.
  /// @param max_attempts Maximum number of tries to sample a valid state.
  /// @pre > 0.
  /// @return Valid sample state.
  StateType SampleValidState(int max_attempts);

  /// Retrieves the per-thread random source.
  PerThreadRandomSource& random_source() { return random_source_; }

  /// Does the planning space support OpenMP-parallelized parallel operations?
  /// Used in PRM and (Bi)RRT planners to enable certain parallel operations.
  bool supports_parallel() const { return supports_parallel_; }

  /// Is the planning space symmetric (i.e. are *Forwards* and *Backwards*
  /// methods the same)?
  bool is_symmetric() const { return is_symmetric_; }

 protected:
  /// Copy constructor for use in Clone().
  PlanningSpace(const PlanningSpace<StateType>& other);

  /// Constructor.
  /// @param seed Seed for per-thread random source.
  /// @param supports_parallel Does the planning space support
  /// OpenMP-parallelized parallel operations?
  /// @param is_symmetric Is the planning space symmetric?
  PlanningSpace(uint64_t seed, bool supports_parallel, bool is_symmetric);

 private:
  PerThreadRandomSource random_source_;
  bool supports_parallel_ = false;
  bool is_symmetric_ = false;
};

}  // namespace planning
}  // namespace drake

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_PLANNING_STATE_TYPES(
    class ::drake::planning::PlanningSpace)
