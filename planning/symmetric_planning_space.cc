#include "planning/symmetric_planning_space.h"

#include "drake/common/drake_throw.h"

namespace drake {
namespace planning {

template <typename StateType>
SymmetricPlanningSpace<StateType>::~SymmetricPlanningSpace() = default;

template <typename StateType>
SymmetricPlanningSpace<StateType>::SymmetricPlanningSpace(
    const SymmetricPlanningSpace<StateType>&) = default;

template <typename StateType>
SymmetricPlanningSpace<StateType>::SymmetricPlanningSpace(
    const uint64_t seed, const bool supports_parallel)
    : PlanningSpace<StateType>(seed, supports_parallel, true) {}

}  // namespace planning
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_PLANNING_STATE_TYPES(
    class ::drake::planning::SymmetricPlanningSpace)
