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

#ifndef BACK_ODOM__DEVICE_ACCEL_HPP_
#define BACK_ODOM__DEVICE_ACCEL_HPP_

#include "back_odom/imu_types.hpp"

#include <Eigen/Core>
#include <kiss_icp_cpp/core/VoxelHashMap.hpp>
#include <sophus/se3.hpp>

#include <cstdint>
#include <optional>
#include <vector>

namespace back_odom
{

[[nodiscard]] bool device_available();

struct DeviceScanClouds
{
  /// Fine voxel cloud in kiss-icp order. This is the cloud that is stored and visualized.
  std::vector<Eigen::Vector3d> map_points;
  /// Coarser cloud used as the registration source.
  std::vector<Eigen::Vector3d> source;
  double deskew_ms{0.0};
  double voxel_ms{0.0};
};

/// Upload the raw scan once, deskew and downsample on the device, and copy back only the
/// downsampled points. Scan id and pass count stay with the scan on the CPU.
[[nodiscard]] std::optional<DeviceScanClouds> device_downsample_scan(
  const std::vector<Eigen::Vector3d> & points, const std::vector<double> & timestamps,
  const std::vector<StampedPose> & trajectory, const Sophus::SE3d & body_from_lidar,
  double half_longitudinal, double half_lateral, double fine_voxel, double coarse_voxel);

/// Deskew to the scan-end pose and crop to the body box. Empty when CUDA cannot run it.
[[nodiscard]] std::optional<std::vector<Eigen::Vector3d>> device_deskew_crop(
  const std::vector<Eigen::Vector3d> & points, const std::vector<double> & timestamps,
  const std::vector<StampedPose> & trajectory, const Sophus::SE3d & body_from_lidar,
  double half_longitudinal, double half_lateral);

/// One voxel grid. The kept point in each voxel is the first input point, matching
/// kiss_icp::VoxelDownsample. Output order is not the CPU hash order.
[[nodiscard]] std::optional<std::vector<Eigen::Vector3d>> device_voxel_downsample(
  const std::vector<Eigen::Vector3d> & points, double voxel_size);

struct DeviceAlignResult
{
  Sophus::SE3d pose{};
  Sophus::SE3d correction{};
  double iterations{0.0};
  /// Robust plane cost at `pose`, the same quantity as robust_plane_cost.
  double robust_cost{0.0};
};

/// Point-to-plane align. The 6x6 step stays in Eigen on the CPU.
/// `float_rank` ranks the 20 neighbors in float; the plane, residual, and Jacobian stay double.
/// Empty when CUDA cannot run it.
[[nodiscard]] std::optional<DeviceAlignResult> align_points_on_device(
  const std::vector<Eigen::Vector3d> & frame, const kiss_icp::VoxelHashMap & map,
  std::uint64_t map_epoch, const Sophus::SE3d & initial_guess, double max_distance,
  double kernel_scale, int max_iterations, double convergence_criterion, bool float_rank = false);

/// One nearest reference histogram per query, or -1. Same float L2 and spatial gate as the CPU loop.
[[nodiscard]] std::optional<std::vector<int>> device_histogram_matches(
  const float * query_hist, int query_count, const float * reference_hist, int reference_count,
  const std::vector<Eigen::Vector3d> & query_xyz,
  const std::vector<Eigen::Vector3d> & reference_xyz, double gate_squared);

}  // namespace back_odom

#endif  // BACK_ODOM__DEVICE_ACCEL_HPP_
