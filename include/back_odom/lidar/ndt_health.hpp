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
#include <tsl/robin_map.h>

#include <Eigen/Core>

#include <cstddef>
#include <vector>

namespace back_odom
{

/// Per-voxel surface statistics of the local map. One entry per occupied kiss voxel.
struct SurfaceVoxel
{
  Eigen::Vector3d mean{Eigen::Vector3d::Zero()};
  /// Unit normal (smallest covariance eigenvector). Only meaningful when `planar`.
  Eigen::Vector3d normal{Eigen::Vector3d::UnitZ()};
  Eigen::Matrix3d inv_covariance{Eigen::Matrix3d::Zero()};
  /// Number of map points the entry was computed from. Lets an update skip unchanged voxels.
  std::uint32_t point_count{0};
  /// Covariance inverted successfully: usable for the Mahalanobis gate.
  bool valid_distribution{false};
  /// Eigenvalue ratio says the points lie on a plane: usable as an ICP target normal.
  bool planar{false};
  /// The normal came from this voxel's own points. When false (sparse voxel) it was fitted on
  /// the 27-voxel neighbourhood and must be refreshed whenever a neighbour changes.
  bool own_fit{false};
};

/// Incrementally maintained surface cache over a kiss_icp::VoxelHashMap. ICP reads normals
/// from it instead of re-fitting a plane per query point per iteration, and the NDT health gate
/// reads the inverse covariances.
struct SurfaceVoxelMap
{
  double voxel_size{0.5};
  double cov_regularization{1.0e-3};
  int min_points{5};
  tsl::robin_map<kiss_icp::Voxel, SurfaceVoxel> voxels;

  void clear();
  /// Recompute every voxel from `map`. Use after the map object was replaced.
  void rebuild_from(const kiss_icp::VoxelHashMap & map);
  /// Recompute only `touched` voxels (duplicates allowed) plus the neighbours whose normal
  /// depends on them. Voxels whose point count did not change and whose normal is their own
  /// are skipped. Voxels missing from `map` are erased.
  void update_voxels(const kiss_icp::VoxelHashMap & map, const std::vector<kiss_icp::Voxel> & touched);
  /// Drop entries whose mean left the sensor-frame XY box (mirror of RemovePointsOutsideBox).
  void cull_outside_box(const Sophus::SE3d & world_to_sensor, double half_longitudinal, double half_lateral);
  /// Drop entries whose voxel no longer exists in `map`.
  void prune_missing(const kiss_icp::VoxelHashMap & map);

  [[nodiscard]] const SurfaceVoxel * find(const kiss_icp::Voxel & voxel) const
  {
    const auto found = voxels.find(voxel);
    return found == voxels.end() ? nullptr : &found->second;
  }
};

/// Backwards-compatible name used by the health tests.
using NdtVoxelMap = SurfaceVoxelMap;
using NdtVoxelEntry = SurfaceVoxel;

/// Mean Mahalanobis distance squared of body points at `pose` against `cloud`.
/// Returns NaN when fewer than `min_correspondences` valid voxel hits exist so callers can skip
/// the gate (bootstrap / sparse map).
[[nodiscard]] double ndt_mean_mahalanobis(
  const std::vector<Eigen::Vector3d> & points_body, const Sophus::SE3d & pose,
  const SurfaceVoxelMap & cloud, int min_correspondences);

struct NdtScore
{
  /// Mean D_M^2 over valid hits (NaN when too few hits).
  double mean_cost{0.0};
  /// Fraction of valid hits with D_M^2 below the threshold (NaN when too few hits).
  double inlier_fraction{0.0};
  int valid_hits{0};
};

/// Mean cost and inlier fraction in one pass.
[[nodiscard]] NdtScore ndt_score(
  const std::vector<Eigen::Vector3d> & points_body, const Sophus::SE3d & pose,
  const SurfaceVoxelMap & cloud, double inlier_threshold, int min_correspondences);

}  // namespace back_odom

#endif  // BACK_ODOM__NDT_HEALTH_HPP_
