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

#include "back_odom/lidar/ndt_health.hpp"

#include <kiss_icp_cpp/core/VoxelUtils.hpp>

#include <cmath>
#include <limits>

namespace back_odom
{
namespace
{

[[nodiscard]] bool fill_distribution(
  const std::vector<Eigen::Vector3d> & points, const double cov_regularization,
  const int min_points, NdtVoxelEntry & entry)
{
  entry = NdtVoxelEntry{};
  if (static_cast<int>(points.size()) < min_points) {
    return false;
  }
  entry.mean.setZero();
  for (const Eigen::Vector3d & point : points) {
    entry.mean += point;
  }
  entry.mean /= static_cast<double>(points.size());

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const Eigen::Vector3d & point : points) {
    const Eigen::Vector3d delta = point - entry.mean;
    covariance += delta * delta.transpose();
  }
  covariance /= static_cast<double>(points.size());
  entry.covariance = covariance;

  const Eigen::Matrix3d regularized =
    covariance + Eigen::Matrix3d::Identity() * std::max(0.0, cov_regularization);
  Eigen::Matrix3d inverse = Eigen::Matrix3d::Zero();
  bool invertible = false;
  regularized.computeInverseWithCheck(inverse, invertible);
  if (!invertible || !inverse.allFinite()) {
    return false;
  }
  entry.inv_covariance = inverse;
  entry.valid_distribution = true;
  return true;
}

}  // namespace

void NdtVoxelMap::clear()
{
  voxels.clear();
}

void NdtVoxelMap::rebuild_from(const kiss_icp::VoxelHashMap & map)
{
  voxels.clear();
  voxel_size = map.voxel_size_;
  voxels.reserve(map.map_.size());
  for (const auto & item : map.map_) {
    NdtVoxelEntry entry;
    if (fill_distribution(item.second, cov_regularization, min_points, entry)) {
      voxels.emplace(item.first, entry);
    }
  }
}

double ndt_mean_mahalanobis(
  const std::vector<Eigen::Vector3d> & points_body, const Sophus::SE3d & pose,
  const NdtVoxelMap & cloud, const int min_correspondences)
{
  if (cloud.voxels.empty() || cloud.voxel_size <= 0.0 || points_body.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  double total = 0.0;
  int valid_points = 0;
  for (const Eigen::Vector3d & point_body : points_body) {
    const Eigen::Vector3d point_world = pose * point_body;
    const kiss_icp::Voxel voxel = kiss_icp::PointToVoxel(point_world, cloud.voxel_size);
    const auto found = cloud.voxels.find(voxel);
    if (found == cloud.voxels.end() || !found->second.valid_distribution) {
      continue;
    }
    const Eigen::Vector3d delta = point_world - found->second.mean;
    const double dist_sq = delta.transpose() * found->second.inv_covariance * delta;
    if (!std::isfinite(dist_sq)) {
      continue;
    }
    total += dist_sq;
    ++valid_points;
  }

  if (valid_points < std::max(1, min_correspondences)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return total / static_cast<double>(valid_points);
}

}  // namespace back_odom
