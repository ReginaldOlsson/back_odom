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

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

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

std::vector<Eigen::Vector3d> deskew_to_scan_end(
  const std::vector<Eigen::Vector3d> & points, const std::vector<double> & timestamps,
  const std::vector<StampedPose> & trajectory, const Sophus::SE3d & body_from_lidar)
{
  if (points.size() != timestamps.size() || points.empty() || trajectory.empty()) {
    return {};
  }
  const std::vector<MotionSegment> segments = motion_segments(trajectory);
  const double scan_end = *std::max_element(timestamps.cbegin(), timestamps.cend());
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
