#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

#include "planning/default_state_types.h"
#include "planning/path_planning_result.h"

namespace drake {
namespace planning {

/// Wrapper type for valid start configs with status. Supports structured
/// bindings with const reference accessors.
template <typename StateType>
class ValidStarts {
 public:
  /// Provides all copy/move/assign operations.
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(ValidStarts);

  explicit ValidStarts(std::vector<StateType> valid_starts);

  ValidStarts();

  const std::vector<StateType>& valid_starts() const { return valid_starts_; }

  const PathPlanningStatusSet& status() const { return status_; }

  // Accessor required for structured bindings.
  template <size_t Index>
  const std::tuple_element_t<Index, ValidStarts<StateType>>& get() const {
    if constexpr (Index == 0) {
      return valid_starts_;
    }
    if constexpr (Index == 1) {
      return status_;
    }
  }

 private:
  void SetStatus();

  std::vector<StateType> valid_starts_;
  PathPlanningStatusSet status_;
};
}  // namespace planning
}  // namespace drake

// Interface bits required for structured binding of ValidStarts.
namespace std {
template <typename StateType>
struct tuple_size<drake::planning::ValidStarts<StateType>> {
  static constexpr size_t value = 2;
};

template <size_t Index, typename StateType>
struct tuple_element<Index, drake::planning::ValidStarts<StateType>>
    : tuple_element<Index, tuple<std::vector<StateType>,
                                 drake::planning::PathPlanningStatusSet>> {};
}  // namespace std

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_PLANNING_STATE_TYPES(
    class ::drake::planning::ValidStarts)
