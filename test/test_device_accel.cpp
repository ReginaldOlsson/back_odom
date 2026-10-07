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
#include "back_odom/lidar/scan_window.hpp"

#include <kiss_icp_cpp/core/Registration.hpp>
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
  (void)device_downsample_scan(points, stamps, trajectory, body_from_lidar, 20.0, 20.0, 0.25, 0.75);
  const auto device_start = std::chrono::steady_clock::now();
  const std::optional<DeviceScanClouds> device =
    device_downsample_scan(points, stamps, trajectory, body_from_lidar, 20.0, 20.0, 0.25, 0.75);
  const double device_ms = elapsed_ms(device_start);
  const auto cpu_start = std::chrono::steady_clock::now();
  const std::vector<Eigen::Vector3d> cropped = crop_lidar_box(
    deskew_to_scan_end(points, stamps, trajectory, body_from_lidar), 20.0, 20.0);
  const std::vector<Eigen::Vector3d> fine = kiss_icp::VoxelDownsample(cropped, 0.25);
  const std::vector<Eigen::Vector3d> coarse = kiss_icp::VoxelDownsample(fine, 0.75);
  const double cpu_ms = elapsed_ms(cpu_start);
  std::cout << "front device " << device_ms << " ms, cpu " << cpu_ms << " ms, raw " << points.size()
            << ", kept " << (device ? device->map_points.size() : 0) << std::endl;
  ASSERT_TRUE(device.has_value());
  ASSERT_EQ(device->map_points.size(), fine.size());
  ASSERT_EQ(device->source.size(), coarse.size());
  for (std::size_t index = 0; index < fine.size(); ++index) {
    EXPECT_LT((device->map_points[index] - fine[index]).norm(), 1.0e-6);
  }
  for (std::size_t index = 0; index < coarse.size(); ++index) {
    EXPECT_LT((device->source[index] - coarse[index]).norm(), 1.0e-6);
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
  kiss_icp::VoxelHashMap map(0.5, 1.0e6, 7);
  map.AddPoints(points);
  std::vector<Eigen::Vector3d> frame;
  frame.reserve(points.size() / 2);
  for (std::size_t index = 0; index < points.size(); index += 2) {
    frame.push_back(points[index]);
  }
  const Sophus::SE3d guess(
    Sophus::SO3d::exp(Eigen::Vector3d(0.0, 0.0, -0.02)), Eigen::Vector3d(-0.15, 0.05, 0.02));
  (void)align_points_on_device(frame, map, 1, guess, 2.0, 0.5, 15, 1.0e-4);
  const auto device_start = std::chrono::steady_clock::now();
  const std::optional<DeviceAlignResult> device =
    align_points_on_device(frame, map, 1, guess, 2.0, 0.5, 15, 1.0e-4);
  const double device_ms = elapsed_ms(device_start);
  kiss_icp::Registration registration(15, 1.0e-4, 0);
  double iterations = 0.0;
  const auto cpu_start = std::chrono::steady_clock::now();
  const kiss_icp::PlaneAlignResult cpu =
    registration.AlignPointsToPlane(frame, map, guess, 2.0, 0.5, iterations);
  const double cpu_cost = robust_plane_cost(frame, cpu.pose, map, 2.0, 0.5);
  const double cpu_ms = elapsed_ms(cpu_start);
  (void)align_points_on_device(frame, map, 1, guess, 2.0, 0.5, 15, 1.0e-4, true);
  const auto float_start = std::chrono::steady_clock::now();
  const std::optional<DeviceAlignResult> ranked =
    align_points_on_device(frame, map, 1, guess, 2.0, 0.5, 15, 1.0e-4, true);
  const double float_ms = elapsed_ms(float_start);
  std::cout << "align double " << device_ms << " ms, float rank " << float_ms << " ms, cpu " << cpu_ms
            << " ms, frame " << frame.size() << std::endl;
  ASSERT_TRUE(device.has_value());
  ASSERT_TRUE(ranked.has_value());
  const Sophus::SE3d delta = cpu.pose.inverse() * device->pose;
  const Sophus::SE3d float_delta = cpu.pose.inverse() * ranked->pose;
  EXPECT_LT(delta.translation().norm(), 1.0e-3);
  EXPECT_LT(delta.so3().log().norm(), 1.0e-3);
  EXPECT_LT(float_delta.translation().norm(), 1.0e-3);
  EXPECT_LT(float_delta.so3().log().norm(), 1.0e-3);
  EXPECT_NEAR(device->robust_cost, robust_plane_cost(frame, device->pose, map, 2.0, 0.5), 1.0e-4);
  (void)cpu_cost;
}

}  // namespace back_odom
