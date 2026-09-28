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

#include "back_odom/fpfh_coarse_align.hpp"
#include "back_odom/imu_processor.hpp"
#include "back_odom/lidar_imu_matcher.hpp"

#include <sophus/se3.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace back_odom
{
namespace
{

constexpr double k_pi = 3.14159265358979323846;

void add_grid(
  std::vector<Eigen::Vector3d> & cloud, const Eigen::Vector3d & origin,
  const Eigen::Vector3d & axis_a, const Eigen::Vector3d & axis_b, const int count_a,
  const int count_b, const double step)
{
  for (int i = 0; i <= count_a; ++i) {
    for (int j = 0; j <= count_b; ++j) {
      cloud.push_back(
        origin + axis_a * (step * static_cast<double>(i)) +
        axis_b * (step * static_cast<double>(j)));
    }
  }
}

void add_box(
  std::vector<Eigen::Vector3d> & cloud, const Eigen::Vector3d & origin, const double length,
  const double width, const double height, const double step)
{
  const int count_x = std::max(1, static_cast<int>(std::round(length / step)));
  const int count_y = std::max(1, static_cast<int>(std::round(width / step)));
  const int count_z = std::max(1, static_cast<int>(std::round(height / step)));
  const Eigen::Vector3d x_axis = Eigen::Vector3d::UnitX();
  const Eigen::Vector3d y_axis = Eigen::Vector3d::UnitY();
  const Eigen::Vector3d z_axis = Eigen::Vector3d::UnitZ();
  add_grid(cloud, origin, x_axis, y_axis, count_x, count_y, step);
  add_grid(cloud, origin + z_axis * height, x_axis, y_axis, count_x, count_y, step);
  add_grid(cloud, origin, x_axis, z_axis, count_x, count_z, step);
  add_grid(cloud, origin + y_axis * width, x_axis, z_axis, count_x, count_z, step);
  add_grid(cloud, origin, y_axis, z_axis, count_y, count_z, step);
  add_grid(cloud, origin + x_axis * length, y_axis, z_axis, count_y, count_z, step);
}

std::vector<Eigen::Vector3d> orthogonal_scene()
{
  std::vector<Eigen::Vector3d> cloud;
  constexpr double k_step = 0.4;
  const Eigen::Vector3d floor_origin(2.0, -6.0, -1.5);
  add_grid(cloud, floor_origin, Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(), 30, 30, k_step);
  add_grid(cloud, floor_origin, Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitZ(), 30, 10, k_step);
  add_grid(
    cloud, Eigen::Vector3d(12.0, -6.0, -1.5), Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitZ(),
    30, 12, k_step);
  add_box(cloud, Eigen::Vector3d(5.0, -2.0, -1.5), 1.6, 1.2, 2.0, 0.4);
  add_box(cloud, Eigen::Vector3d(8.0, 1.5, -1.5), 1.2, 1.6, 1.2, 0.4);
  add_box(cloud, Eigen::Vector3d(9.5, -3.5, -1.5), 2.0, 0.8, 2.4, 0.4);
  return cloud;
}

FpfhParams scene_params()
{
  FpfhParams params;
  params.enabled = true;
  params.keypoint_voxel = 0.8;
  params.normal_radius = 1.5;
  params.fpfh_radius = 3.0;
  params.max_keypoints = 1500;
  params.correspondence_distance = 4.0;
  params.min_inliers = 12;
  params.omp_threads = 2;
  return params;
}

ImuProcessor aligned_imu()
{
  ProcessorParams params;
  params.alignment_sample_count = 5;
  ImuProcessor imu(params);
  ProcessorOutput output;
  for (int i = 0; i < 5; ++i) {
    output =
      imu.process(ImuSample{0.01 * i, Eigen::Vector3d(0.0, 0.0, -9.81), Eigen::Vector3d::Zero()});
  }
  if (!output.aligned) {
    throw std::runtime_error("stationary window did not align");
  }
  return imu;
}

LidarScan scene_scan(const double stamp)
{
  LidarScan scan;
  scan.stamp = stamp;
  scan.points = orthogonal_scene();
  scan.timestamps.assign(scan.points.size(), stamp);
  return scan;
}

}  // namespace

TEST(FpfhCoarseAlign, too_few_points_returns_no_guess)
{
  const FpfhCoarseAlign aligner(scene_params());
  const std::vector<Eigen::Vector3d> points = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
  EXPECT_FALSE(aligner.estimate(points, points, Sophus::SE3d()).has_value());
  EXPECT_FALSE(aligner.estimate({}, orthogonal_scene(), Sophus::SE3d()).has_value());
}

TEST(FpfhCoarseAlign, orthogonal_planes_recover_a_rigid_offset)
{
  const std::vector<Eigen::Vector3d> target = orthogonal_scene();
  const Sophus::SE3d truth(
    Sophus::SO3d::exp(Eigen::Vector3d(0.0, 0.0, 5.0 * k_pi / 180.0)),
    Eigen::Vector3d(0.30, -0.15, 0.05));
  std::vector<Eigen::Vector3d> source;
  source.reserve(target.size());
  const Sophus::SE3d body_from_map = truth.inverse();
  for (const Eigen::Vector3d & point : target) {
    source.push_back(body_from_map * point);
  }
  const Sophus::SE3d imu_guess(Sophus::SO3d(), Eigen::Vector3d(1.20, -0.15, 0.05));

  const FpfhCoarseAlign aligner(scene_params());
  const std::optional<Sophus::SE3d> estimate = aligner.estimate(source, target, imu_guess);
  ASSERT_TRUE(estimate.has_value());
  EXPECT_LT((estimate->translation() - truth.translation()).norm(), 0.5);
  EXPECT_LT((estimate->so3().inverse() * truth.so3()).log().norm(), 10.0 * k_pi / 180.0);
}

TEST(FpfhCoarseAlign, drifted_scan_uses_the_coarse_guess)
{
  ImuProcessor imu = aligned_imu();
  ASSERT_TRUE(imu.aligned());
  LidarMatchParams params;
  params.voxel_size = 0.5;
  params.max_correspondence_distance = 2.0;
  params.fpfh = scene_params();
  LidarImuMatcher matcher(params);
  const LidarScan scan = scene_scan(imu.latest_stamp());
  ASSERT_TRUE(matcher.on_scan(scan, imu).first_scan);

  imu.reset_state(
    Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.3, 0.0, 0.0)), Eigen::Vector3d::Zero(),
    imu.latest_stamp());
  const MatchResult matched = matcher.on_scan(scan, imu);
  ASSERT_TRUE(matched.applied);
  EXPECT_TRUE(matched.used_coarse_guess);
  EXPECT_NEAR(imu.pose().translation().x(), 0.0, 0.15);
  EXPECT_NEAR(imu.pose().translation().y(), 0.0, 0.15);
  EXPECT_NEAR(imu.pose().translation().z(), 0.0, 0.15);
}

}  // namespace back_odom
