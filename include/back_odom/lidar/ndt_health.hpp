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

#ifndef BACK_ODOM__NDT_HEALTH_HPP_
#define BACK_ODOM__NDT_HEALTH_HPP_

#include <kiss_icp_cpp/core/VoxelHashMap.hpp>
#include <sophus/se3.hpp>

#include <Eigen/Core>

#include <unordered_map>
#include <vector>

namespace back_odom
{

struct NdtVoxelEntry
{
  Eigen::Vector3d mean{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d covariance{Eigen::Matrix3d::Zero()};
  Eigen::Matrix3d inv_covariance{Eigen::Matrix3d::Zero()};
  bool valid_distribution{false};
};

struct NdtVoxelMap
{
  double voxel_size{0.5};
  double cov_regularization{1.0e-3};
  int min_points{5};
  std::unordered_map<kiss_icp::Voxel, NdtVoxelEntry> voxels;

  void clear();
  void rebuild_from(const kiss_icp::VoxelHashMap & map);
};

/// Mean Mahalanobis distance squared of body points at `pose` against `cloud`.
/// Returns NaN when fewer than `min_correspondences` valid voxel hits exist so callers can skip
/// the gate (bootstrap / sparse map).
[[nodiscard]] double ndt_mean_mahalanobis(
  const std::vector<Eigen::Vector3d> & points_body, const Sophus::SE3d & pose,
  const NdtVoxelMap & cloud, int min_correspondences);

}  // namespace back_odom

#endif  // BACK_ODOM__NDT_HEALTH_HPP_
