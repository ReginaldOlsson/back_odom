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

#include "back_odom/lidar_preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace back_odom
{
namespace
{

Sophus::SE3d interpolate_pose(const std::vector<StampedPose> & trajectory, const double stamp)
{
  if (trajectory.empty()) {
    throw std::runtime_error("imu trajectory is empty");
  }
  if (stamp <= trajectory.front().stamp) {
    return trajectory.front().pose;
  }
  if (stamp >= trajectory.back().stamp) {
    return trajectory.back().pose;
  }
  for (std::size_t index = 1; index < trajectory.size(); ++index) {
    if (stamp > trajectory[index].stamp) {
      continue;
    }
    const StampedPose & before = trajectory[index - 1];
    const StampedPose & after = trajectory[index];
    const double interval = after.stamp - before.stamp;
    const double alpha = interval > 1e-9 ? (stamp - before.stamp) / interval : 0.0;
    const Sophus::SE3d relative = before.pose.inverse() * after.pose;
    return before.pose * Sophus::SE3d::exp(alpha * relative.log());
  }
  return trajectory.back().pose;
}

}  // namespace

std::vector<Eigen::Vector3d> crop_lidar_box(
  const std::vector<Eigen::Vector3d> & points, const double half_longitudinal,
  const double half_lateral)
{
  std::vector<Eigen::Vector3d> cropped;
  cropped.reserve(points.size());
  for (const auto & point : points) {
    if (std::abs(point.x()) <= half_longitudinal && std::abs(point.y()) <= half_lateral) {
      cropped.push_back(point);
    }
  }
  return cropped;
}

std::vector<Eigen::Vector3d> deskew_to_scan_end(
  const std::vector<Eigen::Vector3d> & points, const std::vector<double> & timestamps,
  const std::vector<StampedPose> & trajectory)
{
  if (points.size() != timestamps.size() || points.empty() || trajectory.empty()) {
    return {};
  }
  const double scan_end = *std::max_element(timestamps.cbegin(), timestamps.cend());
  const Sophus::SE3d end_pose = interpolate_pose(trajectory, scan_end);
  const Sophus::SE3d end_inverse = end_pose.inverse();

  std::vector<Eigen::Vector3d> deskewed;
  deskewed.reserve(points.size());
  for (std::size_t index = 0; index < points.size(); ++index) {
    const Sophus::SE3d pose_i = interpolate_pose(trajectory, timestamps[index]);
    deskewed.push_back(end_inverse * pose_i * points[index]);
  }
  return deskewed;
}

}  // namespace back_odom
