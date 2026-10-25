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

#include "back_odom/lidar/lidar_preprocess.hpp"

#include <Eigen/Eigenvalues>
#include <kiss_icp_cpp/core/VoxelUtils.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tsl/robin_map.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <random>
#include <stdexcept>
#include <utility>

namespace back_odom
{
namespace
{

struct MotionSegment
{
  double end_stamp{0.0};
  Sophus::SE3d start_pose;
  Sophus::SE3d::Tangent relative_tangent{Sophus::SE3d::Tangent::Zero()};
};

std::vector<MotionSegment> motion_segments(const std::vector<StampedPose> & trajectory)
{
  std::vector<MotionSegment> segments;
  if (trajectory.size() < 2) {
    return segments;
  }
  segments.reserve(trajectory.size() - 1);
  for (std::size_t index = 1; index < trajectory.size(); ++index) {
    const StampedPose & before = trajectory[index - 1];
    const StampedPose & after = trajectory[index];
    const Sophus::SE3d relative = before.pose.inverse() * after.pose;
    segments.push_back(MotionSegment{after.stamp, before.pose, relative.log()});
  }
  return segments;
}

Sophus::SE3d interpolate_pose(
  const std::vector<StampedPose> & trajectory, const std::vector<MotionSegment> & segments,
  const double stamp)
{
  if (trajectory.empty()) {
    throw std::runtime_error("imu trajectory is empty");
  }
  if (segments.empty() || stamp <= trajectory.front().stamp) {
    return trajectory.front().pose;
  }
  if (stamp >= trajectory.back().stamp) {
    return trajectory.back().pose;
  }
  const auto segment = std::lower_bound(
    segments.cbegin(), segments.cend(), stamp,
    [](const MotionSegment & motion, const double value) { return motion.end_stamp < value; });
  if (segment == segments.cend()) {
    return trajectory.back().pose;
  }
  const std::size_t index = static_cast<std::size_t>(segment - segments.cbegin());
  const double start_stamp = trajectory[index].stamp;
  const double interval = segment->end_stamp - start_stamp;
  const double alpha = interval > 1e-9 ? (stamp - start_stamp) / interval : 0.0;
  return segment->start_pose * Sophus::SE3d::exp(alpha * segment->relative_tangent);
}

}  // namespace

Sophus::SE3d interpolate_stamped_pose(
  const std::vector<StampedPose> & trajectory, const double stamp)
{
  return interpolate_pose(trajectory, motion_segments(trajectory), stamp);
}

bool pose_buffer_covers(
  const std::vector<StampedPose> & samples, const double t0, const double t1, const double max_gap)
{
  if (
    samples.size() < 2 || !(max_gap >= 0.0) || !std::isfinite(max_gap) || !std::isfinite(t0) ||
    !std::isfinite(t1) || t1 + 1.0e-9 < t0) {
    return false;
  }
  if (
    samples.front().stamp > t0 + max_gap + 1.0e-9 ||
    samples.back().stamp + max_gap + 1.0e-9 < t1) {
    return false;
  }
  for (std::size_t index = 1; index < samples.size(); ++index) {
    if (samples[index].stamp + 1.0e-9 < samples[index - 1].stamp) {
      return false;
    }
    const double gap = samples[index].stamp - samples[index - 1].stamp;
    const bool overlaps = samples[index].stamp > t0 && samples[index - 1].stamp < t1;
    if (overlaps && gap > max_gap + 1.0e-9) {
      return false;
    }
  }
  return true;
}

GroundPlaneFit fit_ground_plane(
  const std::vector<Eigen::Vector3d> & points, const double distance_threshold, const int iterations,
  const std::uint32_t seed)
{
  GroundPlaneFit result;
  if (!(distance_threshold > 0.0) || iterations < 1 || points.size() < 30) {
    return result;
  }
  std::vector<Eigen::Vector3d> cloud;
  const std::size_t stride = std::max<std::size_t>(1, points.size() / 4000);
  cloud.reserve(points.size() / stride + 1);
  for (std::size_t index = 0; index < points.size(); index += stride) {
    if (points[index].allFinite()) {
      cloud.push_back(points[index]);
    }
  }
  if (cloud.size() < 30) {
    return result;
  }

  constexpr double k_min_vertical = 0.8660254037844386;  // cos(30 deg)
  std::mt19937 rng(seed);
  std::uniform_int_distribution<std::size_t> pick(0, cloud.size() - 1);
  std::size_t best_count = 0;
  Eigen::Vector3d best_normal = Eigen::Vector3d::UnitZ();
  double best_offset = 0.0;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const Eigen::Vector3d & first = cloud[pick(rng)];
    const Eigen::Vector3d & second = cloud[pick(rng)];
    const Eigen::Vector3d & third = cloud[pick(rng)];
    Eigen::Vector3d normal = (second - first).cross(third - first);
    if (normal.norm() < 1.0e-4) {
      continue;
    }
    normal.normalize();
    if (normal.z() < 0.0) {
      normal = -normal;
    }
    if (normal.z() < k_min_vertical) {
      continue;
    }
    const double offset = normal.dot(first);
    std::size_t inliers = 0;
    for (const Eigen::Vector3d & point : cloud) {
      if (std::abs(normal.dot(point) - offset) <= distance_threshold) {
        ++inliers;
      }
    }
    if (inliers > best_count) {
      best_count = inliers;
      best_normal = normal;
      best_offset = offset;
    }
  }
  if (best_count < 30) {
    return result;
  }

  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  std::size_t used = 0;
  for (const Eigen::Vector3d & point : cloud) {
    if (std::abs(best_normal.dot(point) - best_offset) <= distance_threshold) {
      mean += point;
      ++used;
    }
  }
  if (used < 30) {
    return result;
  }
  mean /= static_cast<double>(used);
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const Eigen::Vector3d & point : cloud) {
    if (std::abs(best_normal.dot(point) - best_offset) > distance_threshold) {
      continue;
    }
    const Eigen::Vector3d delta = point - mean;
    covariance += delta * delta.transpose();
  }
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  if (solver.info() != Eigen::Success) {
    return result;
  }
  Eigen::Vector3d refined = solver.eigenvectors().col(0);
  if (refined.norm() < 1.0e-8) {
    return result;
  }
  refined.normalize();
  if (refined.z() < 0.0) {
    refined = -refined;
  }
  if (refined.z() < k_min_vertical) {
    return result;
  }
  result.valid = true;
  result.normal = refined;
  result.inlier_ratio = static_cast<double>(best_count) / static_cast<double>(cloud.size());
  return result;
}

std::vector<Eigen::Vector3d> crop_lidar_box(
  const std::vector<Eigen::Vector3d> & points, const double half_longitudinal,
  const double half_lateral)
{
  std::vector<Eigen::Vector3d> cropped;
  cropped.reserve(points.size());
  std::for_each(points.cbegin(), points.cend(), [&](const Eigen::Vector3d & point) {
    if (std::abs(point.x()) <= half_longitudinal && std::abs(point.y()) <= half_lateral) {
      cropped.push_back(point);
    }
  });
  return cropped;
}

IntensityCloud crop_lidar_box(
  const std::vector<Eigen::Vector3d> & points, const std::vector<float> & intensities,
  const double half_longitudinal, const double half_lateral, const std::vector<float> * const tau,
  std::vector<float> * const cropped_tau)
{
  IntensityCloud cropped;
  cropped.points.reserve(points.size());
  cropped.intensities.reserve(points.size());
  const bool have_intensity = intensities.size() == points.size();
  const bool have_tau = tau != nullptr && tau->size() == points.size() && cropped_tau != nullptr;
  if (have_tau) {
    cropped_tau->clear();
    cropped_tau->reserve(points.size());
  }
  for (std::size_t index = 0; index < points.size(); ++index) {
    const Eigen::Vector3d & point = points[index];
    if (std::abs(point.x()) > half_longitudinal || std::abs(point.y()) > half_lateral) {
      continue;
    }
    cropped.points.push_back(point);
    cropped.intensities.push_back(have_intensity ? intensities[index] : 0.0F);
    if (have_tau) {
      cropped_tau->push_back((*tau)[index]);
    }
  }
  return cropped;
}

IntensityCloud voxel_downsample_with_intensity(
  const std::vector<Eigen::Vector3d> & points, const std::vector<float> & intensities,
  const double voxel_size, const std::vector<float> * const tau, std::vector<float> * const tau_out)
{
  IntensityCloud downsampled;
  if (tau_out != nullptr) {
    tau_out->clear();
  }
  if (points.empty() || !(voxel_size > 0.0)) {
    return downsampled;
  }
  const bool have_intensity = intensities.size() == points.size();
  const bool have_tau = tau != nullptr && tau->size() == points.size() && tau_out != nullptr;
  tsl::robin_map<kiss_icp::Voxel, std::tuple<Eigen::Vector3d, float, float>> grid;
  grid.reserve(points.size());
  for (std::size_t index = 0; index < points.size(); ++index) {
    const kiss_icp::Voxel voxel = kiss_icp::PointToVoxel(points[index], voxel_size);
    if (grid.contains(voxel)) {
      continue;
    }
    grid.insert(
      {voxel,
       {points[index], have_intensity ? intensities[index] : 0.0F, have_tau ? (*tau)[index] : 0.0F}});
  }
  downsampled.points.reserve(grid.size());
  downsampled.intensities.reserve(grid.size());
  if (have_tau) {
    tau_out->reserve(grid.size());
  }
  for (const auto & item : grid) {
    downsampled.points.push_back(std::get<0>(item.second));
    downsampled.intensities.push_back(std::get<1>(item.second));
    if (have_tau) {
      tau_out->push_back(std::get<2>(item.second));
    }
  }
  return downsampled;
}

IntensityCloud randomgrid_downsample_with_intensity(
  const std::vector<Eigen::Vector3d> & points, const std::vector<float> & intensities,
  const double voxel_size, const double sampling_rate, std::mt19937 & rng)
{
  IntensityCloud downsampled;
  if (points.empty() || !(voxel_size > 0.0) || !(sampling_rate > 0.0)) {
    return downsampled;
  }
  if (sampling_rate >= 0.99) {
    downsampled.points = points;
    downsampled.intensities = intensities.size() == points.size()
                                ? intensities
                                : std::vector<float>(points.size(), 0.0F);
    return downsampled;
  }

  const bool have_intensity = intensities.size() == points.size();
  tsl::robin_map<kiss_icp::Voxel, std::vector<std::size_t>> grid;
  grid.reserve(points.size());
  for (std::size_t index = 0; index < points.size(); ++index) {
    const Eigen::Vector3d & point = points[index];
    if (!point.allFinite()) {
      continue;
    }
    grid[kiss_icp::PointToVoxel(point, voxel_size)].push_back(index);
  }
  if (grid.empty()) {
    return downsampled;
  }

  const double rate = std::min(sampling_rate, 1.0);
  const std::size_t points_per_voxel = static_cast<std::size_t>(std::ceil(
    (rate * static_cast<double>(points.size())) / static_cast<double>(grid.size())));
  const std::size_t max_points = static_cast<std::size_t>(
    std::ceil(static_cast<double>(points.size()) * rate * 1.2));

  std::vector<std::size_t> kept;
  kept.reserve(std::min(points.size(), max_points));
  std::vector<std::size_t> chosen;
  for (auto & item : grid) {
    auto & cell = item.second;
    if (cell.size() <= points_per_voxel) {
      kept.insert(kept.end(), cell.begin(), cell.end());
      continue;
    }
    chosen.clear();
    std::sample(cell.begin(), cell.end(), std::back_inserter(chosen), points_per_voxel, rng);
    kept.insert(kept.end(), chosen.begin(), chosen.end());
  }
  if (kept.size() > max_points) {
    std::vector<std::size_t> capped;
    capped.reserve(max_points);
    std::sample(kept.begin(), kept.end(), std::back_inserter(capped), max_points, rng);
    kept.swap(capped);
  }
  std::sort(kept.begin(), kept.end());
  downsampled.points.reserve(kept.size());
  downsampled.intensities.reserve(kept.size());
  for (const std::size_t index : kept) {
    downsampled.points.push_back(points[index]);
    downsampled.intensities.push_back(have_intensity ? intensities[index] : 0.0F);
  }
  return downsampled;
}

std::vector<float> intensities_for_downsampled(
  const std::vector<Eigen::Vector3d> & cropped, const std::vector<float> & cropped_intensities,
  const std::vector<Eigen::Vector3d> & downsampled, const double voxel_size)
{
  std::vector<float> intensities(downsampled.size(), 0.0F);
  if (cropped.empty() || downsampled.empty() || !(voxel_size > 0.0)) {
    return intensities;
  }
  const bool have_intensity = cropped_intensities.size() == cropped.size();
  tsl::robin_map<kiss_icp::Voxel, float> grid;
  grid.reserve(cropped.size());
  for (std::size_t index = 0; index < cropped.size(); ++index) {
    const kiss_icp::Voxel voxel = kiss_icp::PointToVoxel(cropped[index], voxel_size);
    if (grid.contains(voxel)) {
      continue;
    }
    grid.insert({voxel, have_intensity ? cropped_intensities[index] : 0.0F});
  }
  for (std::size_t index = 0; index < downsampled.size(); ++index) {
    const auto found = grid.find(kiss_icp::PointToVoxel(downsampled[index], voxel_size));
    if (found != grid.end()) {
      intensities[index] = found->second;
    }
  }
  return intensities;
}

std::vector<StampedPose> trajectory_window(
  const std::vector<StampedPose> & trajectory, const double from_stamp, const double to_stamp)
{
  if (trajectory.size() <= 2) {
    return trajectory;
  }
  const auto by_stamp = [](const StampedPose & sample, const double value) {
    return sample.stamp < value;
  };
  auto first = std::lower_bound(trajectory.cbegin(), trajectory.cend(), from_stamp, by_stamp);
  if (first != trajectory.cbegin()) {
    --first;
  }
  auto last = std::lower_bound(first, trajectory.cend(), to_stamp, by_stamp);
  if (last != trajectory.cend()) {
    ++last;
  }
  if (last - first < 2) {
    // Degenerate span: keep at least two poses so interpolation has a segment.
    if (last != trajectory.cend()) {
      ++last;
    } else if (first != trajectory.cbegin()) {
      --first;
    }
  }
  return std::vector<StampedPose>(first, last);
}

std::vector<Eigen::Vector3d> deskew_to_scan_end(
  const std::vector<Eigen::Vector3d> & points, const std::vector<double> & timestamps,
  const std::vector<StampedPose> & full_trajectory, const Sophus::SE3d & body_from_lidar,
  std::vector<float> * const tau)
{
  if (points.size() != timestamps.size() || points.empty() || full_trajectory.empty()) {
    if (tau != nullptr) {
      tau->clear();
    }
    return {};
  }
  const auto [min_it, max_it] = std::minmax_element(timestamps.cbegin(), timestamps.cend());
  const std::vector<StampedPose> trajectory = trajectory_window(full_trajectory, *min_it, *max_it);
  const std::vector<MotionSegment> segments = motion_segments(trajectory);
  const double scan_end = *max_it;
  const Sophus::SE3d end_inverse = interpolate_pose(trajectory, segments, scan_end).inverse();
  const double span = scan_end - *min_it;

  std::vector<Eigen::Vector3d> deskewed(points.size());
  if (tau != nullptr) {
    tau->assign(points.size(), 1.0F);
  }
  tbb::parallel_for(
    tbb::blocked_range<std::size_t>{0, points.size()},
    [&](const tbb::blocked_range<std::size_t> & range) {
      for (std::size_t index = range.begin(); index < range.end(); ++index) {
        const Sophus::SE3d pose_i = interpolate_pose(trajectory, segments, timestamps[index]);
        deskewed[index] = end_inverse * pose_i * body_from_lidar * points[index];
        if (tau != nullptr && span > 1.0e-6) {
          const double normalized = (timestamps[index] - *min_it) / span;
          (*tau)[index] = static_cast<float>(std::clamp(normalized, 0.0, 1.0));
        }
      }
    });
  return deskewed;
}

void apply_sweep_residual_deskew(
  const std::vector<Eigen::Vector3d> & seed_end, const std::vector<float> & tau,
  const Eigen::Matrix<double, 6, 1> & xi, std::vector<Eigen::Vector3d> & deskewed)
{
  constexpr int k_buckets = 64;
  deskewed.resize(seed_end.size());
  if (seed_end.size() != tau.size()) {
    deskewed = seed_end;
    return;
  }
  if (xi.head<3>().squaredNorm() < 1.0e-6 && xi.tail<3>().squaredNorm() < 1.0e-8) {
    deskewed = seed_end;
    return;
  }
  std::array<Sophus::SE3d, k_buckets> buckets;
  const double scale = static_cast<double>(k_buckets - 1);
  for (int bucket = 0; bucket < k_buckets; ++bucket) {
    const double fraction = static_cast<double>(bucket) / scale - 1.0;
    buckets[static_cast<std::size_t>(bucket)] = Sophus::SE3d::exp(fraction * xi);
  }
  tbb::parallel_for(
    tbb::blocked_range<std::size_t>{0, seed_end.size()},
    [&](const tbb::blocked_range<std::size_t> & range) {
      for (std::size_t index = range.begin(); index < range.end(); ++index) {
        const double clamped = std::clamp(static_cast<double>(tau[index]), 0.0, 1.0);
        const int bucket = std::clamp(
          static_cast<int>(std::lround(clamped * scale)), 0, k_buckets - 1);
        deskewed[index] = buckets[static_cast<std::size_t>(bucket)] * seed_end[index];
      }
    });
}

}  // namespace back_odom
