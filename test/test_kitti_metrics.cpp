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

#include "back_odom/eval/kitti_metrics.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace
{

Eigen::Matrix4d pose_xy_yaw(const double x, const double y, const double z, const double yaw)
{
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  const double cosine = std::cos(yaw);
  const double sine = std::sin(yaw);
  pose(0, 0) = cosine;
  pose(0, 1) = -sine;
  pose(1, 0) = sine;
  pose(1, 1) = cosine;
  pose(0, 3) = x;
  pose(1, 3) = y;
  pose(2, 3) = z;
  return pose;
}

}  // namespace

TEST(KittiMetrics, one_percent_along_track_drift)
{
  constexpr double kSpacingM = 0.1;
  constexpr double kLengthM = 110.0;
  constexpr double kScale = 1.01;
  std::vector<Eigen::Matrix4d> ground_truth;
  std::vector<Eigen::Matrix4d> estimate;
  const int count = static_cast<int>(std::lround(kLengthM / kSpacingM));
  ground_truth.reserve(static_cast<std::size_t>(count) + 1U);
  estimate.reserve(static_cast<std::size_t>(count) + 1U);
  for (int index = 0; index <= count; ++index) {
    const double distance = static_cast<double>(index) * kSpacingM;
    ground_truth.push_back(pose_xy_yaw(distance, 0.0, 0.0, 0.0));
    estimate.push_back(pose_xy_yaw(kScale * distance, 0.0, 0.0, 0.0));
  }

  const back_odom::KittiSequenceError error =
    back_odom::evaluate_kitti_sequence(ground_truth, estimate);
  // The devkit ends a 100 m segment on the first sample past 100 m (100.1 m here).
  const double segment_m = 100.1;
  const double expected_percent = 100.0 * ((kScale * segment_m) - segment_m) / 100.0;
  EXPECT_EQ(error.num_segments, 10);
  EXPECT_NEAR(error.translation_percent, expected_percent, 1.0e-6);
  EXPECT_NEAR(error.rotation_deg_per_100m, 0.0, 1.0e-6);
}

TEST(KittiMetrics, heading_drift_in_degrees_per_100m)
{
  constexpr double kSpacingM = 0.1;
  constexpr double kLengthM = 110.0;
  constexpr double kYawPer100M = 0.1;
  std::vector<Eigen::Matrix4d> ground_truth;
  std::vector<Eigen::Matrix4d> estimate;
  const int count = static_cast<int>(std::lround(kLengthM / kSpacingM));
  for (int index = 0; index <= count; ++index) {
    const double distance = static_cast<double>(index) * kSpacingM;
    ground_truth.push_back(pose_xy_yaw(distance, 0.0, 0.0, 0.0));
    estimate.push_back(pose_xy_yaw(distance, 0.0, 0.0, kYawPer100M * distance / 100.0));
  }

  const back_odom::KittiSequenceError error =
    back_odom::evaluate_kitti_sequence(ground_truth, estimate);
  const double segment_m = 100.1;
  const double yaw = kYawPer100M * segment_m / 100.0;
  const double expected_deg_per_100m = yaw * (180.0 / 3.14159265358979323846) * 100.0 / 100.0;
  EXPECT_EQ(error.num_segments, 10);
  EXPECT_NEAR(error.rotation_deg_per_100m, expected_deg_per_100m, 1.0e-6);
}

TEST(KittiMetrics, body_frame_offset_after_first_pose_alignment)
{
  const Eigen::Matrix4d ground_truth_0 = pose_xy_yaw(10.0, 20.0, 1.0, 0.5 * 3.14159265358979323846);
  const Eigen::Matrix4d estimate_0 = pose_xy_yaw(5.0, -3.0, 0.0, 0.4);
  const Eigen::Matrix4d alignment = back_odom::first_pose_alignment(ground_truth_0, estimate_0);

  Eigen::Matrix4d body_offset = Eigen::Matrix4d::Identity();
  body_offset(0, 3) = 1.0;
  body_offset(1, 3) = 0.5;
  const Eigen::Matrix4d aligned = alignment * (estimate_0 * body_offset);
  const back_odom::BodyPositionError error =
    back_odom::body_frame_position_error(ground_truth_0, aligned);

  EXPECT_NEAR(error.longitudinal_m, 1.0, 1.0e-9);
  EXPECT_NEAR(error.lateral_m, 0.5, 1.0e-9);

  const back_odom::BodyPositionError origin =
    back_odom::body_frame_position_error(ground_truth_0, alignment * estimate_0);
  EXPECT_NEAR(origin.longitudinal_m, 0.0, 1.0e-9);
  EXPECT_NEAR(origin.lateral_m, 0.0, 1.0e-9);
}
