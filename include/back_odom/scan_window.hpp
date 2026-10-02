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

#ifndef BACK_ODOM__SCAN_WINDOW_HPP_
#define BACK_ODOM__SCAN_WINDOW_HPP_

#include <kiss_icp_cpp/core/VoxelHashMap.hpp>
#include <sophus/se3.hpp>

#include <cstdint>
#include <vector>

namespace back_odom
{

/// Refine one pose per scan. Each scan matches the frozen map and the other window scans, not
/// itself. IMU relatives keep neighboring poses from pulling apart. A positive newest sigma
/// anchors the last pose to where it was when this call started.
[[nodiscard]] double robust_plane_cost(
  const std::vector<Eigen::Vector3d> & points_body, const Sophus::SE3d & pose,
  const kiss_icp::VoxelHashMap & map, double max_correspondence_distance, double kernel_scale);

void optimize_scan_window(
  std::vector<Sophus::SE3d> & poses, const std::vector<std::vector<Eigen::Vector3d>> & query_points,
  const std::vector<std::vector<Eigen::Vector3d>> & surface_points,
  const std::vector<Sophus::SE3d> & imu_relatives, const kiss_icp::VoxelHashMap & frozen_map,
  double max_correspondence_distance, double kernel_scale, int max_iterations,
  double convergence_criterion, double newest_translation_sigma, double newest_rotation_sigma,
  double voxel_size, unsigned int max_points_per_voxel);

/// Alternating point-to-plane refinement. Settled scans stay fixed. A pose may move only within
/// the clamp around its insertion pose, so a later pass cannot walk it again.
struct ScanHorizonRefine
{
  double voxel_size{0.5};
  double max_correspondence_distance{1.5};
  double kernel_scale{0.5};
  unsigned int max_points_per_voxel{20};
  int outer_iterations{2};
  int inner_iterations{2};
  std::size_t max_query_points{300};
  double prior_translation_sigma{0.03};
  double prior_rotation_sigma{0.005};
  /// Weaker hold on the current pose so a settled scan can pull its error out in this pass.
  double current_prior_translation_sigma{0.15};
  double current_prior_rotation_sigma{0.05};
  double imu_translation_sigma{0.05};
  double imu_rotation_sigma{0.02};
  double max_translation_from_prior{0.15};
  double max_rotation_from_prior{0.05};
};

void refine_scan_horizon(
  std::vector<Sophus::SE3d> & poses,
  const std::vector<const std::vector<Eigen::Vector3d> *> & points_body,
  const std::vector<Sophus::SE3d> & pose_priors, const std::vector<Sophus::SE3d> & imu_from_previous,
  const ScanHorizonRefine & params, const std::vector<std::uint8_t> & adjustable,
  const std::vector<Sophus::SE3d> & clamp_poses = {});

}  // namespace back_odom

#endif  // BACK_ODOM__SCAN_WINDOW_HPP_
