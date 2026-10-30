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

#include "back_odom/node/ground_truth_alignment.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace
{

struct Sample
{
  double stamp{0.0};
  Sophus::SE3d pose{};
};

Sample at_x(const double stamp, const double x)
{
  Sample sample;
  sample.stamp = stamp;
  sample.pose = Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(x, 0.0, 0.0));
  return sample;
}

bool always_covered(const double)
{
  return true;
}

}  // namespace

TEST(GroundTruthAlignment, zero_threshold_selects_the_first_covered_sample)
{
  const std::vector<Sample> samples = {at_x(0.0, 0.0), at_x(1.0, 1.0), at_x(2.0, 3.0)};
  const back_odom::GroundTruthAlignmentChoice uncovered_first =
    back_odom::select_ground_truth_alignment(
      samples, 0.0, [](const double stamp) { return stamp >= 1.0; });
  EXPECT_TRUE(uncovered_first.ready);
  EXPECT_TRUE(uncovered_first.anchor_found);
  EXPECT_EQ(uncovered_first.index, 1U);
  EXPECT_NEAR(uncovered_first.longitudinal_m, 0.0, 1.0e-9);

  const back_odom::GroundTruthAlignmentChoice covered =
    back_odom::select_ground_truth_alignment(samples, 0.0, always_covered);
  EXPECT_TRUE(covered.ready);
  EXPECT_EQ(covered.index, 0U);
}

TEST(GroundTruthAlignment, forward_travel_selects_the_sample_at_the_threshold)
{
  const std::vector<Sample> samples = {
    at_x(0.0, 0.0), at_x(1.0, 0.5), at_x(2.0, 1.5), at_x(3.0, 2.0), at_x(4.0, 3.0)};
  const back_odom::GroundTruthAlignmentChoice choice =
    back_odom::select_ground_truth_alignment(samples, 2.0, always_covered);
  EXPECT_TRUE(choice.ready);
  EXPECT_EQ(choice.index, 3U);
  EXPECT_NEAR(choice.longitudinal_m, 2.0, 1.0e-9);
}

TEST(GroundTruthAlignment, stationary_buffer_does_not_select)
{
  const std::vector<Sample> samples = {at_x(0.0, 1.0), at_x(1.0, 1.0), at_x(2.0, 1.0)};
  const back_odom::GroundTruthAlignmentChoice choice =
    back_odom::select_ground_truth_alignment(samples, 2.0, always_covered);
  EXPECT_FALSE(choice.ready);
  EXPECT_TRUE(choice.anchor_found);
  EXPECT_NEAR(choice.longitudinal_m, 0.0, 1.0e-9);
}

TEST(GroundTruthAlignment, reverse_travel_selects_once_abs_reaches_the_threshold)
{
  const std::vector<Sample> samples = {
    at_x(0.0, 0.0), at_x(1.0, -0.5), at_x(2.0, -1.5), at_x(3.0, -2.0)};
  const back_odom::GroundTruthAlignmentChoice choice =
    back_odom::select_ground_truth_alignment(samples, 2.0, always_covered);
  EXPECT_TRUE(choice.ready);
  EXPECT_EQ(choice.index, 3U);
  EXPECT_NEAR(choice.longitudinal_m, -2.0, 1.0e-9);
}

TEST(GroundTruthAlignment, lateral_motion_does_not_count)
{
  Sample start = at_x(0.0, 0.0);
  Sample side = at_x(1.0, 0.0);
  side.pose.translation().y() = 3.0;
  const std::vector<Sample> samples = {start, side};
  const back_odom::GroundTruthAlignmentChoice choice =
    back_odom::select_ground_truth_alignment(samples, 2.0, always_covered);
  EXPECT_FALSE(choice.ready);
  EXPECT_NEAR(choice.longitudinal_m, 0.0, 1.0e-9);
}

TEST(GroundTruthAlignment, uncovered_sample_past_the_threshold_is_skipped)
{
  const std::vector<Sample> samples = {
    at_x(0.0, 0.0), at_x(1.0, 1.0), at_x(2.0, 2.0), at_x(3.0, 3.0)};
  const back_odom::GroundTruthAlignmentChoice choice = back_odom::select_ground_truth_alignment(
    samples, 2.0, [](const double stamp) { return stamp < 0.5 || stamp > 2.5; });
  EXPECT_TRUE(choice.ready);
  EXPECT_TRUE(choice.anchor_found);
  EXPECT_EQ(choice.index, 3U);
  EXPECT_NEAR(choice.longitudinal_m, 3.0, 1.0e-9);
}
