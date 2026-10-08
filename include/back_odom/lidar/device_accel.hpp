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

#include "back_odom/imu/imu_types.hpp"
#include "back_odom/lidar/ndt_health.hpp"
#include "back_odom/lidar/plane_icp.hpp"

#include <Eigen/Core>
#include <kiss_icp_cpp/core/VoxelHashMap.hpp>
#include <sophus/se3.hpp>

#include <chrono>
#include <cstdint>
#include <optional>
#include <vector>

namespace back_odom
{

[[nodiscard]] bool device_available();

struct DeviceScanClouds
{
  /// Fine voxel cloud (first point per voxel). Order follows the voxel key, not the CPU hash.
  std::vector<Eigen::Vector3d> map_points;
  /// Intensity of each fine point, taken from the original lidar return.
  std::vector<float> map_intensities;
  /// Coarser cloud used as the registration source.
  std::vector<Eigen::Vector3d> source;
  double deskew_ms{0.0};
  double voxel_ms{0.0};
};

/// Upload the raw scan once, deskew, crop and build both voxel levels on the device, and copy
/// back only the downsampled points. `intensities` may be empty.
[[nodiscard]] std::optional<DeviceScanClouds> device_downsample_scan(
  const std::vector<Eigen::Vector3d> & points, const std::vector<double> & timestamps,
  const std::vector<float> & intensities, const std::vector<StampedPose> & trajectory,
  const Sophus::SE3d & body_from_lidar, double half_longitudinal, double half_lateral,
  double fine_voxel, double coarse_voxel);

/// Deskew to the scan-end pose and crop to the body box. Empty when CUDA cannot run it.
[[nodiscard]] std::optional<std::vector<Eigen::Vector3d>> device_deskew_crop(
  const std::vector<Eigen::Vector3d> & points, const std::vector<double> & timestamps,
  const std::vector<StampedPose> & trajectory, const Sophus::SE3d & body_from_lidar,
  double half_longitudinal, double half_lateral);

/// One voxel grid. The kept point in each voxel is the first input point, matching
/// kiss_icp::VoxelDownsample. Output order is not the CPU hash order.
[[nodiscard]] std::optional<std::vector<Eigen::Vector3d>> device_voxel_downsample(
  const std::vector<Eigen::Vector3d> & points, double voxel_size);

/// Point-to-plane align on the device with the same association as align_to_surface: closest
/// map point in the 27-voxel neighbourhood, normal from the surface cache of that voxel.
/// Neighbour ranking runs in float; residual, Jacobian and the 6x6 system are accumulated in
/// double and solved in Eigen on the CPU. Empty when CUDA cannot run it.
[[nodiscard]] std::optional<PlaneIcpResult> align_points_on_device(
  const std::vector<Eigen::Vector3d> & frame, const kiss_icp::VoxelHashMap & map,
  const SurfaceVoxelMap & surface, std::uint64_t map_epoch, const Sophus::SE3d & initial_guess,
  const PlaneIcpParams & params, const std::chrono::steady_clock::time_point * deadline = nullptr);

}  // namespace back_odom

#endif  // BACK_ODOM__DEVICE_ACCEL_HPP_
