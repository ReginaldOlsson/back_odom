// Copyright 2026 huzaifa.osal
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef BACK_ODOM__MATCH_CANDIDATE_HPP_
#define BACK_ODOM__MATCH_CANDIDATE_HPP_

#include <sophus/se3.hpp>

#include <limits>

namespace back_odom
{

struct MatchCandidate
{
  Sophus::SE3d pose{};
  double cost{std::numeric_limits<double>::infinity()};
  double ndt_cost{std::numeric_limits<double>::quiet_NaN()};
  double ndt_inlier_fraction{std::numeric_limits<double>::quiet_NaN()};
  double iterations{0.0};
  double last_step{0.0};
  int correspondences{0};
  double inlier_ratio{0.0};
  int degenerate_axes{0};
  bool passes{false};
  bool present{false};
  bool saturated{false};
  /// ICP hit the align-rate budget before converging.
  bool timed_out{false};
  /// The partial solve had settled enough to be scored and used like a full solve.
  bool partial_used{false};
};

}  // namespace back_odom

#endif  // BACK_ODOM__MATCH_CANDIDATE_HPP_
