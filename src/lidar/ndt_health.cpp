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
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace back_odom
{
namespace
{

constexpr double k_planarity_ratio = 0.25;

const std::array<kiss_icp::Voxel, 26> k_neighbours = {
  kiss_icp::Voxel{1, 0, 0},   kiss_icp::Voxel{-1, 0, 0},  kiss_icp::Voxel{0, 1, 0},
  kiss_icp::Voxel{0, -1, 0},  kiss_icp::Voxel{0, 0, 1},   kiss_icp::Voxel{0, 0, -1},
  kiss_icp::Voxel{1, 1, 0},   kiss_icp::Voxel{1, -1, 0},  kiss_icp::Voxel{-1, 1, 0},
  kiss_icp::Voxel{-1, -1, 0}, kiss_icp::Voxel{1, 0, 1},   kiss_icp::Voxel{1, 0, -1},
  kiss_icp::Voxel{-1, 0, 1},  kiss_icp::Voxel{-1, 0, -1}, kiss_icp::Voxel{0, 1, 1},
  kiss_icp::Voxel{0, 1, -1},  kiss_icp::Voxel{0, -1, 1},  kiss_icp::Voxel{0, -1, -1},
  kiss_icp::Voxel{1, 1, 1},   kiss_icp::Voxel{1, 1, -1},  kiss_icp::Voxel{1, -1, 1},
  kiss_icp::Voxel{1, -1, -1}, kiss_icp::Voxel{-1, 1, 1},  kiss_icp::Voxel{-1, 1, -1},
  kiss_icp::Voxel{-1, -1, 1}, kiss_icp::Voxel{-1, -1, -1}};

/// Running first/second moments; the plane normal is the smallest eigenvector of the covariance.
struct Moments
{
  Eigen::Vector3d sum{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d sum_sq{Eigen::Matrix3d::Zero()};
  int count{0};

  void add(const std::vector<Eigen::Vector3d> & points)
  {
    for (const Eigen::Vector3d & point : points) {
      sum += point;
      sum_sq.noalias() += point * point.transpose();
    }
    count += static_cast<int>(points.size());
  }
  [[nodiscard]] Eigen::Vector3d mean() const { return sum / static_cast<double>(std::max(1, count)); }
  [[nodiscard]] Eigen::Matrix3d covariance() const
  {
    const Eigen::Vector3d mu = mean();
    return sum_sq / static_cast<double>(std::max(1, count)) - mu * mu.transpose();
  }
};

bool planar_normal(const Eigen::Matrix3d & covariance, Eigen::Vector3d & normal)
{
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  if (solver.info() != Eigen::Success) {
    return false;
  }
  const double lambda_min = solver.eigenvalues()(0);
  const double lambda_max = solver.eigenvalues()(2);
  if (!(lambda_max > 1.0e-10) || lambda_min / lambda_max > k_planarity_ratio) {
    return false;
  }
  const Eigen::Vector3d candidate = solver.eigenvectors().col(0);
  const double norm = candidate.norm();
  if (!(norm > 1.0e-8)) {
    return false;
  }
  normal = candidate / norm;
  return true;
}

/// NDT statistics from the voxel's own points; normal from its own points when they span a
/// plane, otherwise from the 27-voxel neighbourhood (sparse voxels at range, thin structure).
[[nodiscard]] SurfaceVoxel fit_voxel(
  const kiss_icp::VoxelHashMap & map, const kiss_icp::Voxel & voxel,
  const std::vector<Eigen::Vector3d> & points, const double cov_regularization, const int min_points)
{
  SurfaceVoxel entry;
  entry.point_count = static_cast<std::uint32_t>(points.size());
  const int needed = std::max(1, min_points);
  Moments own;
  own.add(points);
  if (own.count > 0) {
    entry.mean = own.mean();
  }
  if (own.count >= needed) {
    const Eigen::Matrix3d covariance = own.covariance();
    if (planar_normal(covariance, entry.normal)) {
      entry.planar = true;
      entry.own_fit = true;
    }
    const Eigen::Matrix3d regularized =
      covariance + Eigen::Matrix3d::Identity() * std::max(0.0, cov_regularization);
    Eigen::Matrix3d inverse = Eigen::Matrix3d::Zero();
    bool invertible = false;
    regularized.computeInverseWithCheck(inverse, invertible);
    if (invertible && inverse.allFinite()) {
      entry.inv_covariance = inverse;
      entry.valid_distribution = true;
    }
  }
  if (!entry.planar) {
    Moments neighbourhood = own;
    for (const kiss_icp::Voxel & shift : k_neighbours) {
      const auto found = map.map_.find(voxel + shift);
      if (found != map.map_.end()) {
        neighbourhood.add(found.value());
      }
    }
    if (neighbourhood.count >= needed && planar_normal(neighbourhood.covariance(), entry.normal)) {
      entry.planar = true;
      entry.own_fit = false;
    }
  }
  return entry;
}

}  // namespace

void SurfaceVoxelMap::clear()
{
  voxels.clear();
}

void SurfaceVoxelMap::rebuild_from(const kiss_icp::VoxelHashMap & map)
{
  voxel_size = map.voxel_size_;
  std::vector<std::pair<kiss_icp::Voxel, const std::vector<Eigen::Vector3d> *>> items;
  items.reserve(map.map_.size());
  for (const auto & item : map.map_) {
    items.emplace_back(item.first, &item.second);
  }
  std::vector<SurfaceVoxel> fitted(items.size());
  tbb::parallel_for(
    tbb::blocked_range<std::size_t>{0, items.size(), 256},
    [&](const tbb::blocked_range<std::size_t> & range) {
      for (std::size_t index = range.begin(); index < range.end(); ++index) {
        fitted[index] =
          fit_voxel(map, items[index].first, *items[index].second, cov_regularization, min_points);
      }
    });
  voxels.clear();
  voxels.reserve(items.size());
  for (std::size_t index = 0; index < items.size(); ++index) {
    voxels.insert({items[index].first, fitted[index]});
  }
}

void SurfaceVoxelMap::update_voxels(
  const kiss_icp::VoxelHashMap & map, const std::vector<kiss_icp::Voxel> & touched)
{
  if (touched.empty()) {
    return;
  }
  voxel_size = map.voxel_size_;
  std::vector<std::pair<kiss_icp::Voxel, const std::vector<Eigen::Vector3d> *>> work;
  work.reserve(touched.size() * 2);
  tsl::robin_map<kiss_icp::Voxel, bool> seen;
  seen.reserve(touched.size() * 4);
  // A voxel is refitted when its own point count changed, or when its normal is borrowed from
  // the neighbourhood and a neighbour was touched. Own-fit voxels with the same count are kept.
  const auto consider = [&](const kiss_icp::Voxel & voxel, const bool direct) {
    if (!seen.insert({voxel, true}).second) {
      return;
    }
    const auto in_map = map.map_.find(voxel);
    if (in_map == map.map_.end()) {
      if (direct) {
        voxels.erase(voxel);
      }
      return;
    }
    const auto existing = voxels.find(voxel);
    if (existing != voxels.end()) {
      const SurfaceVoxel & entry = existing->second;
      if (entry.own_fit && entry.point_count == in_map->second.size()) {
        return;
      }
    }
    // Missing entries (e.g. culled by mean while the map voxel kept a few points) are refitted.
    work.emplace_back(voxel, &in_map->second);
  };
  for (const kiss_icp::Voxel & voxel : touched) {
    consider(voxel, true);
  }
  // Dilate by one voxel so borrowed normals around the touched region are refreshed.
  const std::size_t direct_count = work.size();
  for (std::size_t index = 0; index < direct_count; ++index) {
    const kiss_icp::Voxel centre = work[index].first;
    for (const kiss_icp::Voxel & shift : k_neighbours) {
      consider(centre + shift, false);
    }
  }
  if (work.empty()) {
    return;
  }
  std::vector<SurfaceVoxel> fitted(work.size());
  tbb::parallel_for(
    tbb::blocked_range<std::size_t>{0, work.size(), 256},
    [&](const tbb::blocked_range<std::size_t> & range) {
      for (std::size_t index = range.begin(); index < range.end(); ++index) {
        fitted[index] =
          fit_voxel(map, work[index].first, *work[index].second, cov_regularization, min_points);
      }
    });
  for (std::size_t index = 0; index < work.size(); ++index) {
    voxels.insert_or_assign(work[index].first, fitted[index]);
  }
}

void SurfaceVoxelMap::cull_outside_box(
  const Sophus::SE3d & world_to_sensor, const double half_longitudinal, const double half_lateral)
{
  for (auto it = voxels.begin(); it != voxels.end();) {
    const SurfaceVoxel & entry = it->second;
    // Empty entries keep a zero mean; place them by voxel centre instead.
    const Eigen::Vector3d world =
      entry.point_count > 0
        ? entry.mean
        : Eigen::Vector3d(
            (static_cast<double>(it->first.x()) + 0.5) * voxel_size,
            (static_cast<double>(it->first.y()) + 0.5) * voxel_size,
            (static_cast<double>(it->first.z()) + 0.5) * voxel_size);
    const Eigen::Vector3d local = world_to_sensor * world;
    if (std::abs(local.x()) > half_longitudinal || std::abs(local.y()) > half_lateral) {
      it = voxels.erase(it);
    } else {
      ++it;
    }
  }
}

void SurfaceVoxelMap::prune_missing(const kiss_icp::VoxelHashMap & map)
{
  for (auto it = voxels.begin(); it != voxels.end();) {
    if (map.map_.find(it->first) == map.map_.end()) {
      it = voxels.erase(it);
    } else {
      ++it;
    }
  }
}

NdtScore ndt_score(
  const std::vector<Eigen::Vector3d> & points_body, const Sophus::SE3d & pose,
  const SurfaceVoxelMap & cloud, const double inlier_threshold, const int min_correspondences)
{
  NdtScore score;
  score.mean_cost = std::numeric_limits<double>::quiet_NaN();
  score.inlier_fraction = std::numeric_limits<double>::quiet_NaN();
  if (cloud.voxels.empty() || cloud.voxel_size <= 0.0 || points_body.empty()) {
    return score;
  }

  double total = 0.0;
  int valid_points = 0;
  int inliers = 0;
  for (const Eigen::Vector3d & point_body : points_body) {
    const Eigen::Vector3d point_world = pose * point_body;
    const kiss_icp::Voxel voxel = kiss_icp::PointToVoxel(point_world, cloud.voxel_size);
    const SurfaceVoxel * entry = cloud.find(voxel);
    if (entry == nullptr || !entry->valid_distribution) {
      continue;
    }
    const Eigen::Vector3d delta = point_world - entry->mean;
    const double dist_sq = delta.transpose() * entry->inv_covariance * delta;
    if (!std::isfinite(dist_sq)) {
      continue;
    }
    total += dist_sq;
    ++valid_points;
    if (dist_sq < inlier_threshold) {
      ++inliers;
    }
  }

  score.valid_hits = valid_points;
  if (valid_points < std::max(1, min_correspondences)) {
    return score;
  }
  score.mean_cost = total / static_cast<double>(valid_points);
  score.inlier_fraction = static_cast<double>(inliers) / static_cast<double>(valid_points);
  return score;
}

double ndt_mean_mahalanobis(
  const std::vector<Eigen::Vector3d> & points_body, const Sophus::SE3d & pose,
  const SurfaceVoxelMap & cloud, const int min_correspondences)
{
  return ndt_score(points_body, pose, cloud, std::numeric_limits<double>::infinity(), min_correspondences)
    .mean_cost;
}

}  // namespace back_odom
