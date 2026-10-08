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

#include <kiss_icp_cpp/core/VoxelUtils.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tsl/robin_map.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
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
  const double half_longitudinal, const double half_lateral)
{
  IntensityCloud cropped;
  cropped.points.reserve(points.size());
  cropped.intensities.reserve(points.size());
  const bool have_intensity = intensities.size() == points.size();
  for (std::size_t index = 0; index < points.size(); ++index) {
    const Eigen::Vector3d & point = points[index];
    if (std::abs(point.x()) > half_longitudinal || std::abs(point.y()) > half_lateral) {
      continue;
    }
    cropped.points.push_back(point);
    cropped.intensities.push_back(have_intensity ? intensities[index] : 0.0F);
  }
  return cropped;
}

IntensityCloud voxel_downsample_with_intensity(
  const std::vector<Eigen::Vector3d> & points, const std::vector<float> & intensities,
  const double voxel_size)
{
  IntensityCloud downsampled;
  if (points.empty() || !(voxel_size > 0.0)) {
    return downsampled;
  }
  const bool have_intensity = intensities.size() == points.size();
  tsl::robin_map<kiss_icp::Voxel, std::pair<Eigen::Vector3d, float>> grid;
  grid.reserve(points.size());
  for (std::size_t index = 0; index < points.size(); ++index) {
    const kiss_icp::Voxel voxel = kiss_icp::PointToVoxel(points[index], voxel_size);
    if (grid.contains(voxel)) {
      continue;
    }
    grid.insert(
      {voxel, {points[index], have_intensity ? intensities[index] : 0.0F}});
  }
  downsampled.points.reserve(grid.size());
  downsampled.intensities.reserve(grid.size());
  for (const auto & item : grid) {
    downsampled.points.push_back(item.second.first);
    downsampled.intensities.push_back(item.second.second);
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
  const std::vector<StampedPose> & full_trajectory, const Sophus::SE3d & body_from_lidar)
{
  if (points.size() != timestamps.size() || points.empty() || full_trajectory.empty()) {
    return {};
  }
  const auto [min_it, max_it] = std::minmax_element(timestamps.cbegin(), timestamps.cend());
  const std::vector<StampedPose> trajectory = trajectory_window(full_trajectory, *min_it, *max_it);
  const std::vector<MotionSegment> segments = motion_segments(trajectory);
  const double scan_end = *max_it;
  const Sophus::SE3d end_inverse = interpolate_pose(trajectory, segments, scan_end).inverse();

  std::vector<Eigen::Vector3d> deskewed(points.size());
  tbb::parallel_for(
    tbb::blocked_range<std::size_t>{0, points.size()},
    [&](const tbb::blocked_range<std::size_t> & range) {
      for (std::size_t index = range.begin(); index < range.end(); ++index) {
        const Sophus::SE3d pose_i = interpolate_pose(trajectory, segments, timestamps[index]);
        deskewed[index] = end_inverse * pose_i * body_from_lidar * points[index];
      }
    });
  return deskewed;
}

}  // namespace back_odom
