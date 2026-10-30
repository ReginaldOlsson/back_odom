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

#ifndef BACK_ODOM__GROUND_TRUTH_ALIGNMENT_HPP_
#define BACK_ODOM__GROUND_TRUTH_ALIGNMENT_HPP_

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace back_odom
{

/// Sample chosen for the map-from-local lock, or the travel accumulated so far.
struct GroundTruthAlignmentChoice
{
  /// True when index is a sample that may be locked to the local pose.
  bool ready{false};
  /// True when at least one sample is covered by the local trajectory.
  bool anchor_found{false};
  std::size_t index{0};
  /// Signed body-x travel, in metres, from the anchor through the latest considered sample.
  double longitudinal_m{0.0};
};

/// Pick the ground-truth sample used to lock map-from-local.
///
/// The anchor is the earliest sample for which covered(stamp) is true. Body-x travel is the
/// sum of R_prev^T * (p_next - p_prev) over later samples. A non-positive threshold locks the
/// anchor. Otherwise the lock is the first later covered sample whose absolute travel reaches
/// min_longitudinal_m. Samples the local trajectory does not cover still add to the travel, and
/// the lock moves to the next covered sample.
template <typename Sample, typename Covered>
[[nodiscard]] GroundTruthAlignmentChoice select_ground_truth_alignment(
  const std::vector<Sample> & samples, const double min_longitudinal_m, const Covered & covered)
{
  GroundTruthAlignmentChoice choice;
  std::size_t anchor = samples.size();
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (covered(samples[index].stamp)) {
      anchor = index;
      break;
    }
  }
  if (anchor == samples.size()) {
    return choice;
  }
  choice.anchor_found = true;
  if (!(min_longitudinal_m > 0.0)) {
    choice.ready = true;
    choice.index = anchor;
    return choice;
  }

  Sophus::SE3d previous = samples[anchor].pose;
  for (std::size_t index = anchor + 1U; index < samples.size(); ++index) {
    const Eigen::Vector3d step =
      previous.so3().inverse() * (samples[index].pose.translation() - previous.translation());
    choice.longitudinal_m += step.x();
    previous = samples[index].pose;
    if (std::abs(choice.longitudinal_m) + 1.0e-9 < min_longitudinal_m) {
      continue;
    }
    if (!covered(samples[index].stamp)) {
      continue;
    }
    choice.ready = true;
    choice.index = index;
    return choice;
  }
  return choice;
}

}  // namespace back_odom

#endif  // BACK_ODOM__GROUND_TRUTH_ALIGNMENT_HPP_
