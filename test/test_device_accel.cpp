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

#include "back_odom/lidar/device_accel.hpp"
#include "back_odom/lidar/lidar_preprocess.hpp"
#include "back_odom/lidar/plane_icp.hpp"

#include <kiss_icp_cpp/core/VoxelHashMap.hpp>
#include <kiss_icp_cpp/core/VoxelUtils.hpp>
#include <sophus/se3.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace back_odom
{
namespace
{

double elapsed_ms(const std::chrono::steady_clock::time_point start)
{
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

std::vector<Eigen::Vector3d> scene_points()
{
  std::vector<Eigen::Vector3d> points;
  for (int x = -30; x <= 30; ++x) {
    for (int y = -30; y <= 30; ++y) {
      points.emplace_back(0.4 * x, 0.4 * y, 0.0);
    }
  }
  for (int y = -20; y <= 20; ++y) {
    for (int z = 1; z <= 12; ++z) {
      points.emplace_back(6.0, 0.4 * y, 0.25 * z);
    }
  }
  for (int x = -20; x <= 20; ++x) {
    for (int z = 1; z <= 12; ++z) {
      points.emplace_back(0.4 * x, -5.0, 0.25 * z);
    }
  }
  return points;
}

bool same_set(const std::vector<Eigen::Vector3d> & left, const std::vector<Eigen::Vector3d> & right)
{
  if (left.size() != right.size()) {
    return false;
  }
  std::vector<Eigen::Vector3d> a = left;
  std::vector<Eigen::Vector3d> b = right;
  const auto less = [](const Eigen::Vector3d & lhs, const Eigen::Vector3d & rhs) {
    if (lhs.x() != rhs.x()) {
      return lhs.x() < rhs.x();
    }
    if (lhs.y() != rhs.y()) {
      return lhs.y() < rhs.y();
    }
    return lhs.z() < rhs.z();
  };
  std::sort(a.begin(), a.end(), less);
  std::sort(b.begin(), b.end(), less);
  for (std::size_t index = 0; index < a.size(); ++index) {
    if ((a[index] - b[index]).norm() > 1.0e-9) {
      return false;
    }
  }
  return true;
}

}  // namespace

TEST(DeviceAccel, deskew_matches_the_cpu_interpolation)
{
  if (!device_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }
  std::vector<Eigen::Vector3d> points;
  std::vector<double> stamps;
  points.reserve(4000);
  for (int index = 0; index < 4000; ++index) {
    points.emplace_back(
      0.01 * (index % 80) - 2.0, 0.02 * ((index / 80) % 50) - 1.0, 0.05 * (index % 7));
    stamps.push_back(10.0 + 0.00002 * index);
  }
  std::vector<StampedPose> trajectory;
  trajectory.push_back(
    StampedPose{10.0, Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.0, 0.0, 0.0))});
  trajectory.push_back(
    StampedPose{10.04, Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(0.0, 0.0, 0.01)), Eigen::Vector3d(0.2, 0.0, 0.0))});
  trajectory.push_back(
    StampedPose{10.08, Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(0.0, 0.0, 0.03)), Eigen::Vector3d(0.5, 0.02, 0.0))});
  const Sophus::SE3d body_from_lidar(Sophus::SO3d::exp(Eigen::Vector3d(0.0, 0.0, -0.1)), Eigen::Vector3d(0.3, 0.0, 0.4));
  const auto device_start = std::chrono::steady_clock::now();
  const std::optional<std::vector<Eigen::Vector3d>> device =
    device_deskew_crop(points, stamps, trajectory, body_from_lidar, 20.0, 20.0);
  const double device_ms = elapsed_ms(device_start);
  const auto cpu_start = std::chrono::steady_clock::now();
  const std::vector<Eigen::Vector3d> cpu = crop_lidar_box(
    deskew_to_scan_end(points, stamps, trajectory, body_from_lidar), 20.0, 20.0);
  const double cpu_ms = elapsed_ms(cpu_start);
  std::cout << "deskew device " << device_ms << " ms, cpu " << cpu_ms << " ms" << std::endl;
  ASSERT_TRUE(device.has_value());
  ASSERT_EQ(device->size(), cpu.size());
  double worst = 0.0;
  for (std::size_t index = 0; index < cpu.size(); ++index) {
    worst = std::max(worst, ((*device)[index] - cpu[index]).norm());
  }
  EXPECT_LT(worst, 1.0e-6);
}

TEST(DeviceAccel, downsample_copies_back_only_the_kept_points)
{
  if (!device_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }
  std::vector<Eigen::Vector3d> points;
  std::vector<double> stamps;
  for (int index = 0; index < 4000; ++index) {
    points.emplace_back(0.05 * (index % 40) - 1.0, 0.05 * ((index / 40) % 20), 0.1 * (index % 3));
    stamps.push_back(10.0 + 0.00002 * index);
  }
  points.emplace_back(80.0, 0.0, 0.0);
  stamps.push_back(10.08);
  std::vector<StampedPose> trajectory;
  trajectory.push_back(StampedPose{10.0, Sophus::SE3d()});
  trajectory.push_back(
    StampedPose{10.08, Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(0.0, 0.0, 0.02)), Eigen::Vector3d(0.3, 0.0, 0.0))});
  const Sophus::SE3d body_from_lidar(Sophus::SO3d(), Eigen::Vector3d(0.2, 0.0, 0.1));
  // Intensity encodes the input index so the fine cloud can be checked for carrying it through.
  std::vector<float> intensities(points.size());
  for (std::size_t index = 0; index < points.size(); ++index) {
    intensities[index] = static_cast<float>(index);
  }
  (void)device_downsample_scan(
    points, stamps, intensities, trajectory, body_from_lidar, 20.0, 20.0, 0.25, 0.75);
  const auto device_start = std::chrono::steady_clock::now();
  const std::optional<DeviceScanClouds> device = device_downsample_scan(
    points, stamps, intensities, trajectory, body_from_lidar, 20.0, 20.0, 0.25, 0.75);
  const double device_ms = elapsed_ms(device_start);
  const auto cpu_start = std::chrono::steady_clock::now();
  const std::vector<Eigen::Vector3d> deskewed =
    deskew_to_scan_end(points, stamps, trajectory, body_from_lidar);
  const std::vector<Eigen::Vector3d> cropped = crop_lidar_box(deskewed, 20.0, 20.0);
  const std::vector<Eigen::Vector3d> fine = kiss_icp::VoxelDownsample(cropped, 0.25);
  const double cpu_ms = elapsed_ms(cpu_start);
  std::cout << "front device " << device_ms << " ms, cpu " << cpu_ms << " ms, raw " << points.size()
            << ", kept " << (device ? device->map_points.size() : 0) << std::endl;
  ASSERT_TRUE(device.has_value());
  // Same voxels survive on both sides (first point per voxel); order follows the voxel key.
  EXPECT_TRUE(same_set(device->map_points, fine));
  // The coarse level is one grid over the fine survivors.
  EXPECT_EQ(device->source.size(), kiss_icp::VoxelDownsample(device->map_points, 0.75).size());
  ASSERT_EQ(device->map_intensities.size(), device->map_points.size());
  // Each kept fine point must carry the intensity of the raw return it came from.
  for (std::size_t index = 0; index < device->map_points.size(); ++index) {
    const std::size_t raw = static_cast<std::size_t>(std::lround(device->map_intensities[index]));
    ASSERT_LT(raw, deskewed.size());
    EXPECT_LT((deskewed[raw] - device->map_points[index]).norm(), 1.0e-6);
  }
  EXPECT_LT(device->map_points.size(), points.size());
}

TEST(DeviceAccel, voxel_keeps_the_same_points_as_kiss)
{
  if (!device_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }
  const std::vector<Eigen::Vector3d> points = scene_points();
  const auto device = device_voxel_downsample(points, 0.5);
  const std::vector<Eigen::Vector3d> cpu = kiss_icp::VoxelDownsample(points, 0.5);
  ASSERT_TRUE(device.has_value());
  EXPECT_TRUE(same_set(*device, cpu));
  const auto device_coarse = device_voxel_downsample(*device, 1.5);
  const std::vector<Eigen::Vector3d> cpu_coarse = kiss_icp::VoxelDownsample(cpu, 1.5);
  ASSERT_TRUE(device_coarse.has_value());
  std::cout << "voxel second pass set " << (same_set(*device_coarse, cpu_coarse) ? "matches" : "differs")
            << std::endl;
}

TEST(DeviceAccel, plane_align_stays_within_one_millimetre)
{
  if (!device_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }
  const std::vector<Eigen::Vector3d> points = scene_points();
  kiss_icp::VoxelHashMap map(0.5, 1.0e6, 20);
  map.AddPoints(points);
  std::vector<Eigen::Vector3d> frame;
  frame.reserve(points.size() / 2);
  for (std::size_t index = 0; index < points.size(); index += 2) {
    frame.push_back(points[index]);
  }
  SurfaceVoxelMap surface;
  surface.voxel_size = 0.5;
  surface.rebuild_from(map);
  const Sophus::SE3d guess(
    Sophus::SO3d::exp(Eigen::Vector3d(0.0, 0.0, -0.02)), Eigen::Vector3d(-0.15, 0.05, 0.02));
  PlaneIcpParams params;
  params.max_correspondence_distance = 2.0;
  params.kernel_scale = 0.5;
  params.max_iterations = 15;
  params.convergence_criterion = 1.0e-4;
  // First call uploads the map; time the warm path like the live node sees it.
  (void)align_points_on_device(frame, map, surface, 1, guess, params);
  const auto device_start = std::chrono::steady_clock::now();
  const std::optional<PlaneIcpResult> device =
    align_points_on_device(frame, map, surface, 1, guess, params);
  const double device_ms = elapsed_ms(device_start);
  const auto cpu_start = std::chrono::steady_clock::now();
  const PlaneIcpResult cpu = align_to_surface(frame, map, surface, guess, params);
  const double cpu_ms = elapsed_ms(cpu_start);
  std::cout << "align device " << device_ms << " ms, cpu " << cpu_ms << " ms, frame " << frame.size()
            << ", iterations device " << (device ? device->iterations : 0.0) << " cpu " << cpu.iterations
            << std::endl;
  ASSERT_TRUE(device.has_value());
  const Sophus::SE3d delta = cpu.pose.inverse() * device->pose;
  EXPECT_LT(delta.translation().norm(), 1.0e-3);
  EXPECT_LT(delta.so3().log().norm(), 1.0e-3);
  // The float neighbour ranking must keep the same correspondences as the CPU path.
  EXPECT_NEAR(device->correspondences, cpu.correspondences, std::max(2, cpu.correspondences / 200));
  EXPECT_NEAR(device->robust_cost, cpu.robust_cost, 1.0e-3);
  // Both converge to the scene the frame was cut from.
  EXPECT_LT(device->pose.translation().norm(), 2.0e-2);
  EXPECT_FALSE(device->timed_out);
}

TEST(DeviceAccel, plane_align_returns_partial_pose_on_deadline)
{
  if (!device_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }
  const std::vector<Eigen::Vector3d> points = scene_points();
  kiss_icp::VoxelHashMap map(0.5, 1.0e6, 20);
  map.AddPoints(points);
  SurfaceVoxelMap surface;
  surface.voxel_size = 0.5;
  surface.rebuild_from(map);
  const Sophus::SE3d guess(Sophus::SO3d(), Eigen::Vector3d(-0.15, 0.05, 0.0));
  PlaneIcpParams params;
  params.max_iterations = 30;
  params.convergence_criterion = 1.0e-9;
  (void)align_points_on_device(points, map, surface, 2, guess, params);
  const auto deadline = std::chrono::steady_clock::now();
  const std::optional<PlaneIcpResult> device =
    align_points_on_device(points, map, surface, 2, guess, params, &deadline);
  ASSERT_TRUE(device.has_value());
  EXPECT_TRUE(device->timed_out);
  EXPECT_FALSE(device->saturated);
  EXPECT_DOUBLE_EQ(device->iterations, 0.0);
  // No step taken: the partial pose is the seed.
  EXPECT_LT((device->pose.inverse() * guess).log().norm(), 1.0e-12);
}

}  // namespace back_odom
