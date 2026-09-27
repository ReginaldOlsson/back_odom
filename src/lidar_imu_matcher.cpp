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

#include <kiss_icp_cpp/core/VoxelUtils.hpp>

#include <algorithm>
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

void LidarImuMatcher::set_body_from_lidar(const Sophus::SE3d & body_from_lidar)
{
  body_from_lidar_ = body_from_lidar;
}

bool LidarImuMatcher::has_reference() const
{
  return has_reference_;
}

std::vector<Eigen::Vector3d> LidarImuMatcher::local_map() const
{
  return map_.Pointcloud();
}

MatchResult LidarImuMatcher::on_scan(const LidarScan & scan, ImuProcessor & imu)
{
  MatchResult result;
  if (!imu.aligned()) {
    return result;
  }
  const std::vector<Eigen::Vector3d> deskewed =
    deskew_to_scan_end(scan.points, scan.timestamps, imu.trajectory(), body_from_lidar_);
  const std::vector<Eigen::Vector3d> cropped =
    crop_lidar_box(deskewed, half_longitudinal_, half_lateral_);
  // Same two-level voxelization as KissICP::Voxelize: finer cloud for the map, coarser for ICP.
  const std::vector<Eigen::Vector3d> map_points =
    kiss_icp::VoxelDownsample(cropped, params_.voxel_size * 0.5);
  const std::vector<Eigen::Vector3d> source =
    kiss_icp::VoxelDownsample(map_points, params_.voxel_size * 1.5);
  if (source.empty()) {
    return result;
  }

  const double scan_end = *std::max_element(scan.timestamps.cbegin(), scan.timestamps.cend());
  const Sophus::SE3d predicted = imu.interpolate_pose(scan_end);
  result.imu_pose = predicted;
  result.applied = true;

  if (!has_reference_) {
    map_.Update(map_points, predicted);
    cull_map(predicted);
    stored_scan_ = source;
    stored_pose_ = predicted;
    has_reference_ = true;
    last_correction_stamp_ = scan.stamp;
    last_correction_position_ = predicted.translation();
    imu_steps_ = 0;
    result.inserted_scan = true;
    result.first_scan = true;
    result.corrected_pose = predicted;
    return result;
  }

  double iterations = 0.0;
  const kiss_icp::PlaneAlignResult aligned = align(source, predicted, iterations);
  const Sophus::SE3d error = aligned.pose * predicted.inverse();
  constexpr double k_max_translation = 1.0;
  constexpr double k_max_rotation = 0.2;
  if (
    iterations <= 0.0 || error.translation().norm() > k_max_translation ||
    error.so3().log().norm() > k_max_rotation) {
    result.applied = false;
    return result;
  }

  result.iterations = iterations;
  result.correction = error;
  result.corrected_pose = error * imu.pose();
  result.inserted_scan = true;
  apply_correction(imu, error, scan_end);
  map_.Update(map_points, aligned.pose);
  cull_map(aligned.pose);
  stored_scan_ = source;
  stored_pose_ = aligned.pose;
  imu_steps_ = 0;
  return result;
}

MatchResult LidarImuMatcher::on_imu(ImuProcessor & /*imu*/)
{
  return {};
}

void LidarImuMatcher::apply_correction(
  ImuProcessor & imu, const Sophus::SE3d & error, const double scan_stamp)
{
  const Sophus::SE3d corrected = error * imu.pose();
  const double dt = imu.latest_stamp() - last_correction_stamp_;
  Eigen::Vector3d velocity = imu.velocity();
  if (dt > 1e-3) {
    velocity = (corrected.translation() - last_correction_position_) / dt;
  }
  imu.apply_lidar_correction(error, velocity, scan_stamp, last_correction_stamp_);
  last_correction_stamp_ = imu.latest_stamp();
  last_correction_position_ = imu.pose().translation();
}

void LidarImuMatcher::cull_map(const Sophus::SE3d & lidar_pose)
{
  map_.RemovePointsOutsideBox(lidar_pose.inverse(), half_longitudinal_, half_lateral_);
}

kiss_icp::PlaneAlignResult LidarImuMatcher::align(
  const std::vector<Eigen::Vector3d> & scan, const Sophus::SE3d & guess, double & iterations)
{
  return registration_.AlignPointsToPlane(
    scan, map_, guess, params_.max_correspondence_distance, params_.kernel_scale, iterations);
}

}  // namespace back_odom
