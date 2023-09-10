#include "drake/planning/trajectory_optimization/fast_path_planner.h"

#include <gtest/gtest.h>

#include "drake/geometry/optimization/hpolyhedron.h"

namespace drake {
namespace planning {
namespace trajectory_optimization {
namespace {

using Eigen::Vector2d;
using geometry::optimization::HPolyhedron;

// Simple example from https://github.com/cvxgrp/fastpathplanning.
GTEST_TEST(FastPathPlannerTest, Small2D) {
  // clang-format off
  const auto regions = MakeConvexSets(
      HPolyhedron::MakeBox(Vector2d(6.25,     4), Vector2d(7.25, 5.75)),
      HPolyhedron::MakeBox(Vector2d( 5.5,   2.5), Vector2d( 7.5, 4.75)),
      HPolyhedron::MakeBox(Vector2d(   1,     4), Vector2d(   6,    5)),
      HPolyhedron::MakeBox(Vector2d(0.25,   1.5), Vector2d(   7,    3)),
      HPolyhedron::MakeBox(Vector2d(2.25,   .75), Vector2d(   3,  2.5)),
      HPolyhedron::MakeBox(Vector2d( 0.5, -0.25), Vector2d( 1.5,  4.5)),
      HPolyhedron::MakeBox(Vector2d(   2,     0), Vector2d(6.25,    1)),
      HPolyhedron::MakeBox(Vector2d(4.75, -0.25), Vector2d(   6, 3.75)),
      HPolyhedron::MakeBox(Vector2d(   0,   5.2), Vector2d(   7,    6)));
  // clang-format on

  FastPathPlanner fpp(2);
  const int order =
      8;  // TODO(russt): Confirm that this is the order that actually results
          // from the python example (it's using defaults that looks to be (size
          // of alpha + 1)*2, but let's see).
  fpp.AddRegions(regions, order);
  fpp.Preprocess();
}

}  // namespace
}  // namespace trajectory_optimization
}  // namespace planning
}  // namespace drake
