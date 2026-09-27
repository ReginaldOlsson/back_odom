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

#include "back_odom/lidar_imu_matcher.hpp"

#include "back_odom/lidar_preprocess.hpp"

#include <cmath>
#include <stdexcept>

namespace back_odom
{

LidarImuMatcher::LidarImuMatcher(const LidarMatchParams & params)
: params_(params),
  half_longitudinal_(params.crop_longitudinal * 0.5),
  half_lateral_(params.crop_lateral * 0.5),
  map_(params.voxel_size, 1.0e6, static_cast<unsigned int>(params.max_points_per_voxel)),
  registration_(params.max_iterations, params.convergence_criterion, 0)
{
  if (
    params_.voxel_size <= 0.0 || params_.crop_longitudinal <= 0.0 || params_.crop_lateral <= 0.0) {
    throw std::invalid_argument("lidar crop and voxel size must be positive");
  }
  if (params_.backward_match_stride < 1) {
    throw std::invalid_argument("backward_match_stride must be positive");
  }
  if (params_.max_points_per_voxel < 1) {
    throw std::invalid_argument("max_points_per_voxel must be positive");
  }
}

bool LidarImuMatcher::has_reference() const
{
  return has_reference_;
}

MatchResult LidarImuMatcher::on_scan(const LidarScan & scan, ImuProcessor & imu)
{
  MatchResult result;
  if (!imu.aligned()) {
    return result;
  }
  const std::vector<Eigen::Vector3d> deskewed =
    deskew_to_scan_end(scan.points, scan.timestamps, imu.trajectory());
  const std::vector<Eigen::Vector3d> cropped =
    crop_lidar_box(deskewed, half_longitudinal_, half_lateral_);
  if (cropped.empty()) {
    return result;
  }

  const Sophus::SE3d imu_pose = imu.pose();
  result.imu_pose = imu_pose;
  result.applied = true;

  if (!has_reference_) {
    map_.Update(cropped, imu_pose);
    cull_map(imu_pose);
    stored_scan_ = cropped;
    has_reference_ = true;
    last_correction_stamp_ = scan.stamp;
    last_correction_position_ = imu_pose.translation();
    imu_steps_ = 0;
    result.inserted_scan = true;
    result.first_scan = true;
    result.corrected_pose = imu_pose;
    return result;
  }

  double iterations = 0.0;
  const kiss_icp::PlaneAlignResult aligned = align(cropped, imu_pose, iterations);
  result.iterations = iterations;
  result.correction = aligned.correction;
  result.corrected_pose = aligned.pose;
  result.inserted_scan = true;
  apply_correction(imu, aligned.pose, scan.stamp);
  map_.Update(cropped, aligned.pose);
  cull_map(aligned.pose);
  stored_scan_ = cropped;
  imu_steps_ = 0;
  return result;
}

MatchResult LidarImuMatcher::on_imu(ImuProcessor & imu)
{
  MatchResult result;
  if (!has_reference_ || stored_scan_.empty() || !imu.aligned()) {
    return result;
  }
  ++imu_steps_;
  if (imu_steps_ < params_.backward_match_stride) {
    return result;
  }
  imu_steps_ = 0;

  const Sophus::SE3d imu_pose = imu.pose();
  double iterations = 0.0;
  const kiss_icp::PlaneAlignResult aligned = align(stored_scan_, imu_pose, iterations);
  result.applied = true;
  result.inserted_scan = false;
  result.imu_pose = imu_pose;
  result.iterations = iterations;
  result.correction = aligned.correction;
  result.corrected_pose = aligned.pose;
  apply_correction(imu, aligned.pose, imu.latest_stamp());
  return result;
}

void LidarImuMatcher::apply_correction(
  ImuProcessor & imu, const Sophus::SE3d & corrected_pose, const double stamp)
{
  const double dt = stamp - last_correction_stamp_;
  Eigen::Vector3d velocity = imu.velocity();
  if (dt > 1e-3) {
    velocity = (corrected_pose.translation() - last_correction_position_) / dt;
  }
  imu.reset_state(corrected_pose, velocity, stamp);
  last_correction_stamp_ = stamp;
  last_correction_position_ = corrected_pose.translation();
}

void LidarImuMatcher::cull_map(const Sophus::SE3d & lidar_pose)
{
  const std::vector<Eigen::Vector3d> world_points = map_.Pointcloud();
  const Sophus::SE3d world_to_lidar = lidar_pose.inverse();
  std::vector<Eigen::Vector3d> kept;
  kept.reserve(world_points.size());
  for (const auto & point : world_points) {
    const Eigen::Vector3d local = world_to_lidar * point;
    if (std::abs(local.x()) <= half_longitudinal_ && std::abs(local.y()) <= half_lateral_) {
      kept.push_back(point);
    }
  }
  map_.Clear();
  map_.AddPoints(kept);
}

kiss_icp::PlaneAlignResult LidarImuMatcher::align(
  const std::vector<Eigen::Vector3d> & scan, const Sophus::SE3d & guess, double & iterations)
{
  return registration_.AlignPointsToPlane(
    scan, map_, guess, params_.max_correspondence_distance, params_.kernel_scale, iterations);
}

}  // namespace back_odom
