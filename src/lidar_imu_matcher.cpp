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
#include <cmath>
#include <optional>
#include <stdexcept>

namespace back_odom
{
namespace
{

constexpr double k_max_translation = 1.0;
constexpr double k_max_rotation = 0.2;
constexpr double k_max_gyro_bias_step = 0.002;
constexpr double k_max_accel_bias_step = 0.05;
constexpr double k_bias_leak = 0.01;

Eigen::Vector3d clamp_vector(const Eigen::Vector3d & value, const double limit)
{
  const double norm = value.norm();
  if (limit < 0.0 || norm <= limit || norm <= 1e-12) {
    return value;
  }
  return value * (limit / norm);
}

bool finite_gain(const double gain)
{
  return std::isfinite(gain) && gain >= 0.0 && gain <= 1.0;
}

bool accept_vehicle_delta(const double iteration_count, const Sophus::SE3d & vehicle_delta)
{
  return iteration_count > 0.0 && vehicle_delta.translation().norm() <= k_max_translation &&
         vehicle_delta.so3().log().norm() <= k_max_rotation;
}

}  // namespace

InertialCorrection inertial_correction_from_match(
  const Sophus::SE3d & vehicle_delta, const Sophus::SO3d & orientation_correction,
  const Sophus::SE3d & corrected_pose, const Eigen::Vector3d & imu_velocity_world,
  const Eigen::Vector3d & previous_position, const double dt, const double gyro_bias_gain,
  const double accel_bias_gain, const double speed_gain)
{
  InertialCorrection correction;
  const Eigen::Vector3d turned_velocity = orientation_correction * imu_velocity_world;
  const Sophus::SO3d & orientation = corrected_pose.so3();
  const Eigen::Vector3d imu_body = orientation.inverse() * turned_velocity;
  // The matched heading owns the direction. Lateral IMU velocity is what pulls the next guess
  // off the pose ICP just accepted.
  Eigen::Vector3d velocity_body(imu_body.x(), 0.0, imu_body.z());
  const Eigen::Vector3d displacement = corrected_pose.translation() - previous_position;
  const Eigen::Vector3d disp_body = orientation.inverse() * displacement;
  if (dt > 1e-3) {
    velocity_body.x() = (1.0 - speed_gain) * imu_body.x() + speed_gain * (disp_body.x() / dt);
    velocity_body.z() = (1.0 - speed_gain) * imu_body.z() + speed_gain * (disp_body.z() / dt);
  }
  correction.velocity_world = orientation * velocity_body;
  if (dt <= 1e-3) {
    return correction;
  }

  // vehicle_delta is the body increment ICP added to the pose. A positive yaw means the
  // integrator was short, so the subtracted gyro bias is too large and must decrease.
  correction.gyro_bias_delta =
    clamp_vector(-gyro_bias_gain * vehicle_delta.so3().log() / dt, k_max_gyro_bias_step);

  Eigen::Vector3d accel_error(
    (imu_body.x() - disp_body.x() / dt) / dt, 0.0, (imu_body.z() - disp_body.z() / dt) / dt);
  correction.accel_bias_delta = clamp_vector(accel_bias_gain * accel_error, k_max_accel_bias_step);
  return correction;
}

LidarImuMatcher::LidarImuMatcher(const LidarMatchParams & params)
: params_(params),
  half_longitudinal_(params.crop_longitudinal * 0.5),
  half_lateral_(params.crop_lateral * 0.5),
  map_(params.voxel_size, 1.0e6, static_cast<unsigned int>(params.max_points_per_voxel)),
  registration_(params.max_iterations, params.convergence_criterion, 0),
  fpfh_(params.fpfh)
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
  if (
    !finite_gain(params_.gyro_bias_gain) || !finite_gain(params_.accel_bias_gain) ||
    !finite_gain(params_.speed_correction_gain)) {
    throw std::invalid_argument("bias and speed gains must be between 0 and 1");
  }
  if (params_.max_gyro_bias < 0.0 || params_.max_accel_bias < 0.0) {
    throw std::invalid_argument("bias magnitude limits must be non-negative");
  }
  if (!params_.fpfh.enabled) {
    return;
  }
  if (params_.fpfh.keypoint_voxel <= 0.0 || params_.fpfh.normal_radius <= 0.0) {
    throw std::invalid_argument("fpfh voxel and normal radius must be positive");
  }
  if (params_.fpfh.fpfh_radius <= params_.fpfh.normal_radius) {
    throw std::invalid_argument("fpfh radius must be larger than the normal radius");
  }
  if (params_.fpfh.max_keypoints < 3 || params_.fpfh.min_inliers < 3) {
    throw std::invalid_argument("fpfh keypoint and inlier counts must be at least 3");
  }
  if (params_.fpfh.correspondence_distance <= 0.0) {
    throw std::invalid_argument("fpfh correspondence distance must be positive");
  }
  if (params_.fpfh.omp_threads < 0) {
    throw std::invalid_argument("fpfh omp thread count must be non-negative");
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
  Sophus::SE3d guess = predicted;
  if (params_.fpfh.enabled) {
    if (
      const std::optional<Sophus::SE3d> coarse =
        fpfh_.estimate(source, map_.Pointcloud(), predicted)) {
      guess = *coarse;
      result.used_coarse_guess = true;
    }
  }
  kiss_icp::PlaneAlignResult aligned = align(source, guess, iterations);
  // Gate the motion of the vehicle. The world-frame left increment grows with distance from the
  // map origin, so a small yaw fix on a curve looks like a multi-meter jump and used to be dropped.
  Sophus::SE3d vehicle_delta = predicted.inverse() * aligned.pose;
  Sophus::SE3d world_error = aligned.pose * predicted.inverse();
  if (result.used_coarse_guess && !accept_vehicle_delta(iterations, vehicle_delta)) {
    result.used_coarse_guess = false;
    aligned = align(source, predicted, iterations);
    vehicle_delta = predicted.inverse() * aligned.pose;
    world_error = aligned.pose * predicted.inverse();
  }
  result.iterations = iterations;
  result.correction = world_error;
  result.translation_error = vehicle_delta.translation().norm();
  result.rotation_error = vehicle_delta.so3().log().norm();
  if (!accept_vehicle_delta(iterations, vehicle_delta)) {
    result.applied = false;
    return result;
  }

  result.corrected_pose = imu.pose() * vehicle_delta;
  result.inserted_scan = true;
  apply_correction(imu, vehicle_delta, scan_end);
  result.gyro_bias = imu.gyro_bias();
  result.accel_bias = imu.accel_bias();
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
  ImuProcessor & imu, const Sophus::SE3d & vehicle_delta, const double scan_stamp)
{
  if (!has_bias_prior_) {
    gyro_bias_prior_ = imu.gyro_bias();
    accel_bias_prior_ = imu.accel_bias();
    has_bias_prior_ = true;
  }
  // Right multiply: +body x adds forward, -body x subtracts, +yaw turns left about the vehicle.
  const Sophus::SE3d corrected = imu.pose() * vehicle_delta;
  const Sophus::SO3d orientation_correction = corrected.so3() * imu.pose().so3().inverse();
  const double dt = imu.latest_stamp() - last_correction_stamp_;
  const InertialCorrection inertial = inertial_correction_from_match(
    vehicle_delta, orientation_correction, corrected, imu.velocity(), last_correction_position_, dt,
    params_.gyro_bias_gain, params_.accel_bias_gain, params_.speed_correction_gain);
  imu.apply_lidar_correction(
    vehicle_delta, inertial.velocity_world, scan_stamp, last_correction_stamp_);
  // Leak toward the stationary prior so registration noise cannot random-walk the bias.
  const Eigen::Vector3d gyro_bias = clamp_vector(
    imu.gyro_bias() + inertial.gyro_bias_delta + k_bias_leak * (gyro_bias_prior_ - imu.gyro_bias()),
    params_.max_gyro_bias);
  const Eigen::Vector3d accel_bias = clamp_vector(
    imu.accel_bias() + inertial.accel_bias_delta +
      k_bias_leak * (accel_bias_prior_ - imu.accel_bias()),
    params_.max_accel_bias);
  imu.set_gyro_bias(gyro_bias);
  imu.set_accel_bias(accel_bias);
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
