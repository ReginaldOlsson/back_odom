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

#include "back_odom/lidar/lidar_imu_matcher.hpp"

#include "back_odom/lidar/device_accel.hpp"
#include "back_odom/lidar/lidar_preprocess.hpp"
#include "back_odom/lidar/ndt_health.hpp"
#include "back_odom/lidar/plane_icp.hpp"
#include "back_odom/lidar/scan_window.hpp"

#include <kiss_icp_cpp/core/VoxelUtils.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace back_odom
{
namespace
{

constexpr double k_max_translation = 1.0;
constexpr double k_max_rotation = 0.2;
constexpr double k_max_gyro_bias_step = 0.002;
constexpr double k_max_accel_bias_step = 0.05;
constexpr double k_bias_leak = 0.01;
constexpr std::uint64_t k_prune_every_culls = 50;
constexpr double k_cull_every_metres = 2.0;
constexpr int k_min_device_points = 256;

Sophus::SE3d clamp_pose_to_insertion(Sophus::SE3d pose, const Sophus::SE3d & insertion)
{
  Eigen::Matrix<double, 6, 1> delta = (insertion.inverse() * pose).log();
  const double translation = delta.head<3>().norm();
  const double rotation = delta.tail<3>().norm();
  constexpr double k_max_carry_translation = 0.15;
  constexpr double k_max_carry_rotation = 0.05;
  if (translation > k_max_carry_translation && translation > 1.0e-12) {
    delta.head<3>() *= k_max_carry_translation / translation;
  }
  if (rotation > k_max_carry_rotation && rotation > 1.0e-12) {
    delta.tail<3>() *= k_max_carry_rotation / rotation;
  }
  return insertion * Sophus::SE3d::exp(delta);
}

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

double elapsed_ms(const std::chrono::steady_clock::time_point start)
{
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

bool close_poses(const Sophus::SE3d & left, const Sophus::SE3d & right)
{
  const Sophus::SE3d delta = left.inverse() * right;
  return delta.translation().norm() < 5.0e-3 && delta.so3().log().norm() < 5.0e-3;
}

bool similar_size(const std::size_t left, const std::size_t right, const double tolerance)
{
  const double larger = static_cast<double>(std::max(left, right));
  if (larger <= 0.0) {
    return true;
  }
  const double diff = std::abs(static_cast<double>(left) - static_cast<double>(right));
  return diff / larger <= tolerance;
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
  map_(params.voxel_size, 1.0e6, static_cast<unsigned int>(std::max(1, params.max_points_per_voxel)))
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
  if (params_.max_longitudinal_correction < 0.0) {
    throw std::invalid_argument("max longitudinal correction must be non-negative");
  }
  if (
    params_.limits.max_speed < 0.0 || params_.limits.max_acceleration < 0.0 ||
    params_.limits.max_yaw_rate < 0.0 || params_.limits.max_match_rejects < 1 ||
    params_.refine_min_travel < 0.0) {
    throw std::invalid_argument("vehicle limits must be non-negative and allow one rejected scan");
  }
  if (!(params_.max_ndt_cost > 0.0) || !std::isfinite(params_.max_ndt_cost)) {
    throw std::invalid_argument("max_ndt_cost must be positive and finite");
  }
  if (params_.ndt_cov_regularization < 0.0 || params_.min_ndt_correspondences < 1) {
    throw std::invalid_argument(
      "ndt_cov_regularization must be non-negative and min_ndt_correspondences positive");
  }
  if (!(params_.align_rate_hz > 0.0) || !std::isfinite(params_.align_rate_hz)) {
    throw std::invalid_argument("align_rate_hz must be positive and finite");
  }
  if (
    params_.min_ndt_inlier_fraction < 0.0 || params_.min_ndt_inlier_fraction > 1.0 ||
    params_.min_inlier_ratio < 0.0 || params_.min_inlier_ratio > 1.0) {
    throw std::invalid_argument("inlier fractions must be between 0 and 1");
  }
  if (params_.partial_min_iterations < 1 || !(params_.partial_max_step >= 0.0)) {
    throw std::invalid_argument("partial solve acceptance must be positive");
  }
  if (params_.keyframe_min_translation < 0.0 || params_.keyframe_min_rotation < 0.0) {
    throw std::invalid_argument("keyframe thresholds must be non-negative");
  }
  if (params_.max_iterations < 1 || !(params_.convergence_criterion > 0.0)) {
    throw std::invalid_argument("max_iterations and convergence_criterion must be positive");
  }
  surface_.voxel_size = params_.voxel_size;
  surface_.cov_regularization = params_.ndt_cov_regularization;
  surface_.min_points = 5;
}

void LidarImuMatcher::set_body_from_lidar(const Sophus::SE3d & body_from_lidar)
{
  body_from_lidar_ = body_from_lidar;
}

LocalizationHealth LidarImuMatcher::health() const
{
  return health_.health;
}

int LidarImuMatcher::reject_streak() const
{
  return health_.reject_streak;
}

int LidarImuMatcher::timeout_streak() const
{
  return health_.timeout_streak;
}

bool LidarImuMatcher::has_reference() const
{
  return has_reference_;
}

std::vector<Eigen::Vector3d> LidarImuMatcher::local_map() const
{
  return map_.Pointcloud();
}

std::vector<LocalMapPoint> LidarImuMatcher::annotated_local_map() const
{
  std::vector<LocalMapPoint> annotated;
  annotated.reserve(visible_scan_point_count());
  for (const HorizonScan & scan : horizon_) {
    const bool have_intensity = scan.intensities.size() == scan.points.size();
    for (std::size_t index = 0; index < scan.points.size(); ++index) {
      LocalMapPoint annotated_point;
      annotated_point.position = scan.pose * scan.points[index];
      annotated_point.intensity = have_intensity ? scan.intensities[index] : 0.0F;
      annotated_point.scan_id = static_cast<float>(scan.scan_id);
      annotated_point.collective_passes = static_cast<float>(scan.collective_passes);
      annotated.push_back(annotated_point);
    }
  }
  return annotated;
}

std::vector<StampedPose> LidarImuMatcher::visible_scan_poses() const
{
  std::vector<StampedPose> poses;
  poses.reserve(horizon_.size());
  for (const HorizonScan & scan : horizon_) {
    poses.push_back(StampedPose{scan.stamp, scan.pose});
  }
  return poses;
}

std::size_t LidarImuMatcher::visible_scan_point_count() const
{
  std::size_t count = 0;
  for (const HorizonScan & scan : horizon_) {
    count += scan.points.size();
  }
  return count;
}

void LidarImuMatcher::record_horizon_scan(
  const Sophus::SE3d & pose, const double stamp, std::vector<Eigen::Vector3d> && points,
  std::vector<float> && intensities, const ImuProcessor & imu)
{
  HorizonScan scan;
  scan.stamp = stamp;
  scan.pose = pose;
  scan.prior = pose;
  if (intensities.size() != points.size()) {
    intensities.assign(points.size(), 0.0F);
  }
  scan.points = std::move(points);
  scan.intensities = std::move(intensities);
  scan.scan_id = next_scan_id_++;
  if (!horizon_.empty()) {
    const HorizonScan & previous = horizon_.back();
    const std::vector<StampedPose> & trajectory = imu.trajectory();
    if (!trajectory.empty() && previous.stamp + 1.0e-6 >= trajectory.front().stamp) {
      scan.imu_from_previous =
        imu.interpolate_pose(previous.stamp).inverse() * imu.interpolate_pose(stamp);
    } else {
      scan.imu_from_previous = previous.pose.inverse() * pose;
    }
  }
  horizon_.push_back(std::move(scan));
  const Sophus::SE3d latest = horizon_.back().pose;
  horizon_.erase(
    std::remove_if(
      horizon_.begin(), horizon_.end(),
      [&](const HorizonScan & stored) {
        return !scan_pose_inside_visible_cloud(latest, stored.pose, half_longitudinal_, half_lateral_);
      }),
    horizon_.end());
  const std::size_t max_scans = static_cast<std::size_t>(std::max(1, params_.max_visible_scans));
  while (horizon_.size() > max_scans) {
    horizon_.pop_front();
  }
}

bool LidarImuMatcher::refine_window_if_due(MatchResult & result, Sophus::SE3d & newest_pose)
{
  if (!params_.refine_window || horizon_.size() < 3) {
    return false;
  }
  if (
    has_refined_ &&
    (newest_pose.translation() - last_refine_position_).norm() < params_.refine_min_travel) {
    return false;
  }
  std::vector<Sophus::SE3d> poses;
  std::vector<Sophus::SE3d> priors;
  std::vector<Sophus::SE3d> clamps;
  std::vector<Sophus::SE3d> relatives;
  std::vector<const std::vector<Eigen::Vector3d> *> clouds;
  std::vector<std::uint8_t> adjustable;
  poses.reserve(horizon_.size());
  priors.reserve(horizon_.size());
  clamps.reserve(horizon_.size());
  relatives.reserve(horizon_.size());
  clouds.reserve(horizon_.size());
  adjustable.reserve(horizon_.size());
  int settled = 0;
  const int settle_passes = std::max(1, params_.refine_settle_passes);
  for (const HorizonScan & scan : horizon_) {
    poses.push_back(scan.pose);
    priors.push_back(scan.pose);
    clamps.push_back(scan.prior);
    relatives.push_back(scan.imu_from_previous);
    clouds.push_back(&scan.points);
    const bool settled_scan = scan.collective_passes >= settle_passes;
    adjustable.push_back(settled_scan ? 0 : 1);
    if (settled_scan) {
      ++settled;
    }
  }
  if (settled == 0) {
    adjustable.front() = 0;
    horizon_.front().optimized = true;
    horizon_.front().collective_passes = settle_passes;
    settled = 1;
  }
  adjustable.back() = 0;
  int active_ready = 0;
  for (const std::uint8_t free : adjustable) {
    active_ready += free;
  }
  if (active_ready == 0) {
    has_refined_ = true;
    last_refine_position_ = newest_pose.translation();
    return false;
  }
  const Sophus::SE3d previous_before = horizon_[horizon_.size() - 2].pose;
  ScanHorizonRefine refine;
  refine.voxel_size = params_.voxel_size;
  refine.max_correspondence_distance = params_.max_correspondence_distance;
  refine.kernel_scale = params_.kernel_scale;
  refine.max_points_per_voxel = static_cast<unsigned int>(params_.max_points_per_voxel);
  refine_scan_horizon(poses, clouds, priors, relatives, refine, adjustable, clamps);
  double largest = 0.0;
  int active = 0;
  for (std::size_t index = 0; index + 1 < horizon_.size(); ++index) {
    if (adjustable[index] == 0) {
      continue;
    }
    ++active;
    largest = std::max(largest, (horizon_[index].pose.inverse() * poses[index]).translation().norm());
    horizon_[index].pose = poses[index];
    horizon_[index].optimized = true;
    horizon_[index].collective_passes += 1;
  }
  const Sophus::SE3d carried = clamp_pose_to_insertion(
    horizon_[horizon_.size() - 2].pose * previous_before.inverse() * newest_pose, horizon_.back().prior);
  largest = std::max(largest, (newest_pose.inverse() * carried).translation().norm());
  horizon_.back().pose = carried;
  newest_pose = carried;
  for (std::size_t index = 1; index < horizon_.size(); ++index) {
    horizon_[index].imu_from_previous = horizon_[index - 1].pose.inverse() * horizon_[index].pose;
  }
  if (largest >= 0.01) {
    rebuild_from_horizon(newest_pose);
  }
  has_refined_ = true;
  last_refine_position_ = newest_pose.translation();
  result.refined_window = true;
  result.refine_shift = largest;
  result.refine_scans = active;
  result.refine_settled = settled;
  return true;
}

void LidarImuMatcher::rebuild_from_horizon(const Sophus::SE3d & cull_pose)
{
  kiss_icp::VoxelHashMap rebuilt(
    params_.voxel_size, map_.max_distance_, static_cast<unsigned int>(params_.max_points_per_voxel));
  for (const HorizonScan & scan : horizon_) {
    if (scan.points.empty()) {
      continue;
    }
    std::vector<Eigen::Vector3d> world;
    world.reserve(scan.points.size());
    for (const Eigen::Vector3d & point : scan.points) {
      world.push_back(scan.pose * point);
    }
    rebuilt.AddPoints(world);
  }
  if (rebuilt.Empty()) {
    return;
  }
  rebuilt.RemovePointsOutsideBox(cull_pose.inverse(), half_longitudinal_, half_lateral_);
  if (rebuilt.Empty()) {
    return;
  }
  map_ = std::move(rebuilt);
  rebuild_surface();
}

void LidarImuMatcher::rebuild_surface()
{
  surface_.voxel_size = params_.voxel_size;
  surface_.cov_regularization = params_.ndt_cov_regularization;
  surface_.rebuild_from(map_);
  ++map_epoch_;
}

void LidarImuMatcher::cull_local_map(const Sophus::SE3d & pose)
{
  const Sophus::SE3d world_to_sensor = pose.inverse();
  map_.RemovePointsOutsideBox(world_to_sensor, half_longitudinal_, half_lateral_);
  surface_.cull_outside_box(world_to_sensor, half_longitudinal_, half_lateral_);
  if (++culls_since_prune_ >= k_prune_every_culls) {
    culls_since_prune_ = 0;
    surface_.prune_missing(map_);
  }
}

bool LidarImuMatcher::should_insert(const Sophus::SE3d & pose) const
{
  if (horizon_.empty()) {
    return true;
  }
  const Sophus::SE3d moved = last_keyframe_pose_.inverse() * pose;
  return moved.translation().norm() >= params_.keyframe_min_translation ||
         moved.so3().log().norm() >= params_.keyframe_min_rotation;
}

void LidarImuMatcher::insert_scan(
  const Sophus::SE3d & pose, const double stamp, std::vector<Eigen::Vector3d> && points,
  std::vector<float> && intensities, const ImuProcessor & imu, MatchResult & timing)
{
  const auto map_start = std::chrono::steady_clock::now();
  std::vector<Eigen::Vector3d> world(points.size());
  std::vector<kiss_icp::Voxel> touched;
  touched.reserve(points.size());
  for (std::size_t index = 0; index < points.size(); ++index) {
    world[index] = pose * points[index];
    touched.push_back(kiss_icp::PointToVoxel(world[index], params_.voxel_size));
  }
  map_.AddPoints(world);
  // RemovePointsOutsideBox walks the whole map; a few keyframes of slack past the crop box
  // costs nothing, so cull on distance rather than every insert.
  if (
    !has_cull_pose_ ||
    (last_cull_pose_.inverse() * pose).translation().norm() >= k_cull_every_metres) {
    cull_local_map(pose);
    last_cull_pose_ = pose;
    has_cull_pose_ = true;
  }
  surface_.update_voxels(map_, touched);
  ++map_epoch_;
  record_horizon_scan(pose, stamp, std::move(points), std::move(intensities), imu);
  last_keyframe_pose_ = pose;
  timing.map_ms += elapsed_ms(map_start);
}

ScanAlignment LidarImuMatcher::align_scan(const LidarScan & scan, const ImuProcessor & imu)
{
  ScanAlignment alignment;
  MatchResult & result = alignment.result;
  result.health = health_.health;
  if (!imu.aligned()) {
    result.skipped = true;
    result.skip_reason = "IMU is not aligned yet";
    return alignment;
  }
  if (scan.points.empty() || scan.timestamps.size() != scan.points.size()) {
    result.skipped = true;
    result.skip_reason = "scan has no xyz+time points";
    return alignment;
  }
  ++scan_count_;
  PreparedClouds clouds = prepare_scan(scan, imu.trajectory(), result);
  if (clouds.source.empty()) {
    result.skipped = true;
    result.skip_reason = "deskew/crop/voxel left no source points";
    return alignment;
  }
  const double scan_end = *std::max_element(scan.timestamps.cbegin(), scan.timestamps.cend());
  // Deskew/guess come from the buffered pose nearest the scan time, not imu.pose() (live tip).
  if (!imu.has_pose_near(scan_end, imu.max_pair_dt())) {
    result.skipped = true;
    const StampedPose closest = imu.closest_pose(scan_end);
    result.skip_reason = "no IMU pose within " + std::to_string(imu.max_pair_dt()) +
                         " s of scan_end " + std::to_string(scan_end) + " (closest " +
                         std::to_string(closest.stamp) + ", imu_latest " +
                         std::to_string(imu.latest_stamp()) + ")";
    return alignment;
  }
  const Sophus::SE3d imu_pose = imu.interpolate_pose(scan_end);
  result.imu_pose = imu_pose;
  alignment.scan_end = scan_end;
  alignment.map_points = std::move(clouds.map_points);
  alignment.map_intensities = std::move(clouds.map_intensities);
  alignment.source = std::move(clouds.source);

  if (!has_reference_) {
    alignment.first_scan = true;
    alignment.predicted = imu_pose;
    alignment.ready = true;
    return alignment;
  }

  // Seed from the last committed lidar pose plus the IMU motion since then. With the IMU
  // correction on this equals imu_pose; with it off, open-loop drift cannot walk the seed away.
  Sophus::SE3d predicted = imu_pose;
  const std::vector<StampedPose> & trajectory = imu.trajectory();
  if (!trajectory.empty() && anchor_stamp_ + 1.0e-6 >= trajectory.front().stamp &&
      anchor_stamp_ <= scan_end) {
    predicted = anchor_pose_ * (imu.interpolate_pose(anchor_stamp_).inverse() * imu_pose);
  }
  alignment.predicted = predicted;

  // Budget covers ICP + gates (target align_rate_hz). Deskew/voxel stay outside the gate.
  alignment.deadline =
    std::chrono::steady_clock::now() +
    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(1.0 / params_.align_rate_hz));
  alignment.candidate = score_alignment(alignment.source, predicted, result, &alignment.deadline);

  const MatchCandidate & candidate = alignment.candidate;
  result.has_lidar_debug = candidate.present;
  result.lidar_pose = candidate.pose;
  result.lidar_cost = candidate.cost;
  result.ndt_cost = candidate.ndt_cost;
  result.ndt_inlier_fraction = candidate.ndt_inlier_fraction;
  result.inlier_ratio = candidate.inlier_ratio;
  result.correspondences = candidate.correspondences;
  result.degenerate_axes = candidate.degenerate_axes;
  result.lidar_passed = candidate.passes;
  result.saturated = candidate.saturated;
  result.align_timed_out = candidate.timed_out;
  result.partial_used = candidate.partial_used;
  result.iterations = candidate.iterations;
  alignment.ready = true;
  return alignment;
}

MatchResult LidarImuMatcher::commit_scan(ScanAlignment && alignment, ImuProcessor & imu)
{
  MatchResult result = std::move(alignment.result);
  if (!alignment.ready) {
    return result;
  }
  const Sophus::SE3d predicted = alignment.predicted;
  const double scan_end = alignment.scan_end;

  if (alignment.first_scan) {
    insert_scan(
      predicted, scan_end, std::move(alignment.map_points), std::move(alignment.map_intensities),
      imu, result);
    has_reference_ = true;
    anchor_pose_ = predicted;
    anchor_stamp_ = scan_end;
    last_correction_stamp_ = imu.latest_stamp();
    last_correction_position_ = imu.pose().translation();
    result.applied = true;
    result.inserted_scan = true;
    result.first_scan = true;
    result.corrected_pose = predicted;
    remember_healthy(predicted, imu.velocity(), scan_end);
    note_health(imu, HealthEvent::Accepted);
    result.health = health_.health;
    return result;
  }

  const MatchCandidate & candidate = alignment.candidate;
  if (candidate.timed_out && !candidate.partial_used) {
    // Out of budget before the solve settled. Skip the scan: do not insert, do not correct.
    result.applied = false;
    result.lidar_pose = predicted;
    note_health(imu, HealthEvent::TimedOut);
    result.health = health_.health;
    result.gyro_bias = imu.gyro_bias();
    result.accel_bias = imu.accel_bias();
    return result;
  }

  if (!(candidate.present && candidate.passes)) {
    const Sophus::SE3d failed = candidate.present ? candidate.pose : predicted;
    const Sophus::SE3d vehicle_delta = predicted.inverse() * failed;
    result.applied = false;
    result.correction = failed * predicted.inverse();
    result.translation_error = vehicle_delta.translation().norm();
    result.rotation_error = vehicle_delta.so3().log().norm();
    note_health(imu, HealthEvent::Rejected);
    result.health = health_.health;
    result.gyro_bias = imu.gyro_bias();
    result.accel_bias = imu.accel_bias();
    return result;
  }

  note_health(imu, candidate.saturated ? HealthEvent::Saturated : HealthEvent::Accepted);
  const bool update_bias = health_.update_bias;

  const Sophus::SE3d limited =
    limit_longitudinal_correction(predicted, candidate.pose, params_.max_longitudinal_correction);
  Sophus::SE3d committed = limited;
  if (should_insert(committed)) {
    insert_scan(
      committed, scan_end, std::move(alignment.map_points), std::move(alignment.map_intensities),
      imu, result);
    result.inserted_scan = true;
    if (params_.refine_window && std::chrono::steady_clock::now() < alignment.deadline) {
      const auto refine_start = std::chrono::steady_clock::now();
      refine_window_if_due(result, committed);
      result.refine_ms += elapsed_ms(refine_start);
    }
  }

  // Residual against the seed (lidar anchor + IMU motion since): what the IMU step got wrong.
  const Sophus::SE3d vehicle_delta = predicted.inverse() * committed;
  result.correction = committed * predicted.inverse();
  result.translation_error = vehicle_delta.translation().norm();
  result.rotation_error = vehicle_delta.so3().log().norm();
  result.applied = true;
  if (params_.apply_imu_correction) {
    // The integrator is moved by the delta from *its own* pose at scan end, which differs from
    // the seed whenever the IMU was reset or a previous correction was clipped.
    const Sophus::SE3d imu_delta = result.imu_pose.inverse() * committed;
    apply_correction(imu, imu_delta, scan_end, last_correction_stamp_, update_bias, std::nullopt);
  } else {
    last_correction_stamp_ = imu.latest_stamp();
    last_correction_position_ = imu.pose().translation();
  }
  anchor_pose_ = committed;
  anchor_stamp_ = scan_end;
  result.corrected_pose = imu.pose();
  result.lidar_pose = committed;
  remember_healthy(result.corrected_pose, imu.velocity(), scan_end);
  result.health = health_.health;
  result.gyro_bias = imu.gyro_bias();
  result.accel_bias = imu.accel_bias();
  return result;
}

MatchResult LidarImuMatcher::on_scan(const LidarScan & scan, ImuProcessor & imu)
{
  return commit_scan(align_scan(scan, imu), imu);
}

MatchResult LidarImuMatcher::on_imu(ImuProcessor & /*imu*/)
{
  return {};
}

void LidarImuMatcher::notify_speed_limit(ImuProcessor & imu)
{
  note_health(imu, HealthEvent::SpeedClamped);
}

void LidarImuMatcher::apply_correction(
  ImuProcessor & imu, const Sophus::SE3d & vehicle_delta, const double scan_stamp,
  const double since_stamp, const bool update_bias,
  const std::optional<Eigen::Vector3d> & velocity_override)
{
  if (!has_bias_prior_) {
    gyro_bias_prior_ = imu.gyro_bias();
    accel_bias_prior_ = imu.accel_bias();
    has_bias_prior_ = true;
  }
  // Right multiply: +body x adds forward, -body x subtracts, +yaw turns left about the vehicle.
  const Sophus::SE3d corrected = imu.pose() * vehicle_delta;
  const Sophus::SO3d orientation_correction = corrected.so3() * imu.pose().so3().inverse();
  const double dt = std::max(1.0e-3, imu.latest_stamp() - last_correction_stamp_);
  const InertialCorrection inertial = inertial_correction_from_match(
    vehicle_delta, orientation_correction, corrected, imu.velocity(), last_correction_position_, dt,
    params_.gyro_bias_gain, params_.accel_bias_gain, params_.speed_correction_gain);
  Eigen::Vector3d velocity = velocity_override.value_or(inertial.velocity_world);
  const Eigen::Vector3d body = clamp_body_velocity(corrected.so3().inverse() * velocity, params_.limits.max_speed);
  velocity = corrected.so3() * body;
  imu.apply_lidar_correction(vehicle_delta, velocity, scan_stamp, since_stamp);
  if (update_bias) {
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
  }
  last_correction_stamp_ = imu.latest_stamp();
  last_correction_position_ = imu.pose().translation();
}

void LidarImuMatcher::remember_healthy(
  const Sophus::SE3d & pose, const Eigen::Vector3d & velocity, const double stamp)
{
  has_healthy_ = true;
  last_healthy_pose_ = pose;
  last_healthy_stamp_ = stamp;
  last_healthy_velocity_ = velocity;
  last_healthy_speed_ = (pose.so3().inverse() * velocity).x();
}

void LidarImuMatcher::restore_healthy_speed(ImuProcessor & imu)
{
  if (!has_healthy_) {
    imu.set_velocity(Eigen::Vector3d::Zero());
    return;
  }
  imu.set_velocity(last_healthy_velocity_);
}

void LidarImuMatcher::note_health(ImuProcessor & imu, const HealthEvent event)
{
  health_ = advance_localization_health(health_, event, params_.limits.max_match_rejects);
  if (health_.restore_speed) {
    restore_healthy_speed(imu);
  }
}

MatchCandidate LidarImuMatcher::score_alignment(
  const std::vector<Eigen::Vector3d> & source, const Sophus::SE3d & guess, MatchResult & timing,
  const std::chrono::steady_clock::time_point * const deadline)
{
  MatchCandidate candidate;
  candidate.present = true;
  const auto align_start = std::chrono::steady_clock::now();
  const PlaneIcpResult icp = align(source, guess, deadline);
  timing.align_ms += elapsed_ms(align_start);

  candidate.iterations = icp.iterations;
  candidate.last_step = icp.last_step;
  candidate.correspondences = icp.correspondences;
  candidate.inlier_ratio =
    source.empty() ? 0.0 : static_cast<double>(icp.correspondences) / static_cast<double>(source.size());
  candidate.degenerate_axes = icp.degenerate_axes;
  candidate.saturated = icp.saturated;
  candidate.timed_out = icp.timed_out;
  if (icp.timed_out) {
    candidate.partial_used = icp.iterations + 0.5 >= static_cast<double>(params_.partial_min_iterations) &&
                             icp.last_step <= params_.partial_max_step;
    if (!candidate.partial_used) {
      candidate.pose = guess;
      candidate.passes = false;
      return candidate;
    }
  }
  candidate.pose = icp.pose;
  candidate.cost = icp.robust_cost;

  const auto cost_start = std::chrono::steady_clock::now();
  const NdtScore ndt = ndt_score(
    source, candidate.pose, surface_, params_.max_ndt_cost, params_.min_ndt_correspondences);
  timing.cost_ms += elapsed_ms(cost_start);
  candidate.ndt_cost = ndt.mean_cost;
  candidate.ndt_inlier_fraction = ndt.inlier_fraction;

  const Sophus::SE3d imu_delta = guess.inverse() * candidate.pose;
  const bool imu_ok = accept_vehicle_delta(icp.iterations, imu_delta);
  const bool ndt_ok = !std::isfinite(ndt.mean_cost) ||
                      (ndt.mean_cost < params_.max_ndt_cost &&
                       ndt.inlier_fraction >= params_.min_ndt_inlier_fraction);
  const bool inliers_ok = candidate.inlier_ratio >= params_.min_inlier_ratio;
  candidate.passes =
    icp.iterations > 0.0 && imu_ok && std::isfinite(candidate.cost) && ndt_ok && inliers_ok;
  return candidate;
}

LidarImuMatcher::PreparedClouds LidarImuMatcher::prepare_scan(
  const LidarScan & scan, const std::vector<StampedPose> & full_trajectory, MatchResult & timing)
{
  const double fine_voxel = params_.voxel_size * 0.5;
  const double coarse_voxel = params_.voxel_size * 1.5;
  const auto [min_it, max_it] = std::minmax_element(scan.timestamps.cbegin(), scan.timestamps.cend());
  const std::vector<StampedPose> trajectory = trajectory_window(full_trajectory, *min_it, *max_it);

  const auto cpu_prepare = [&](PreparedClouds & out) {
    const auto deskew_start = std::chrono::steady_clock::now();
    const IntensityCloud cropped = crop_lidar_box(
      deskew_to_scan_end(scan.points, scan.timestamps, trajectory, body_from_lidar_),
      scan.intensities, half_longitudinal_, half_lateral_);
    const double deskew_ms = elapsed_ms(deskew_start);
    const auto voxel_start = std::chrono::steady_clock::now();
    IntensityCloud map_cloud =
      voxel_downsample_with_intensity(cropped.points, cropped.intensities, fine_voxel);
    out.source = kiss_icp::VoxelDownsample(map_cloud.points, coarse_voxel);
    out.map_points = std::move(map_cloud.points);
    out.map_intensities = std::move(map_cloud.intensities);
    return std::make_pair(deskew_ms, elapsed_ms(voxel_start));
  };
  const auto device_prepare = [&]() {
    return device_downsample_scan(
      scan.points, scan.timestamps, scan.intensities, trajectory, body_from_lidar_,
      half_longitudinal_, half_lateral_, fine_voxel, coarse_voxel);
  };

  const bool device_ok = params_.device_bakeoff_scan > 0 && device_available() &&
                         scan.points.size() >= 1000 && !trajectory.empty();
  const bool bakeoff_due =
    device_ok &&
    (!device_front_decided_
       ? scan_count_ >= static_cast<std::uint64_t>(params_.device_bakeoff_scan)
       : (params_.device_recheck_scans > 0 &&
          scan_count_ - device_front_decided_at_ >= static_cast<std::uint64_t>(params_.device_recheck_scans)));

  if (bakeoff_due) {
    device_front_decided_ = true;
    device_front_decided_at_ = scan_count_;
    const auto device_start = std::chrono::steady_clock::now();
    std::optional<DeviceScanClouds> device = device_prepare();
    const double device_ms = elapsed_ms(device_start);
    PreparedClouds cpu;
    const auto [cpu_deskew_ms, cpu_voxel_ms] = cpu_prepare(cpu);
    device_front_use_ = device.has_value() &&
                        similar_size(device->map_points.size(), cpu.map_points.size(), 0.02) &&
                        similar_size(device->source.size(), cpu.source.size(), 0.05) &&
                        device_ms < 0.8 * (cpu_deskew_ms + cpu_voxel_ms);
    if (device_front_use_) {
      timing.deskew_ms += device->deskew_ms;
      timing.voxel_ms += device->voxel_ms;
      return PreparedClouds{
        std::move(device->map_points), std::move(device->map_intensities), std::move(device->source)};
    }
    timing.deskew_ms += cpu_deskew_ms;
    timing.voxel_ms += cpu_voxel_ms;
    return cpu;
  }
  if (device_ok && device_front_use_) {
    if (std::optional<DeviceScanClouds> device = device_prepare()) {
      timing.deskew_ms += device->deskew_ms;
      timing.voxel_ms += device->voxel_ms;
      return PreparedClouds{
        std::move(device->map_points), std::move(device->map_intensities), std::move(device->source)};
    }
    device_front_use_ = false;
  }
  PreparedClouds cpu;
  const auto [deskew_ms, voxel_ms] = cpu_prepare(cpu);
  timing.deskew_ms += deskew_ms;
  timing.voxel_ms += voxel_ms;
  return cpu;
}

PlaneIcpResult LidarImuMatcher::align(
  const std::vector<Eigen::Vector3d> & scan, const Sophus::SE3d & guess,
  const std::chrono::steady_clock::time_point * const deadline)
{
  PlaneIcpParams icp_params;
  icp_params.max_correspondence_distance = params_.max_correspondence_distance;
  icp_params.kernel_scale = params_.kernel_scale;
  icp_params.max_iterations = params_.max_iterations;
  icp_params.convergence_criterion = params_.convergence_criterion;

  const bool device_ok = params_.device_bakeoff_scan > 0 && device_available() &&
                         scan.size() >= static_cast<std::size_t>(k_min_device_points) && !map_.Empty();
  if (!device_ok) {
    return align_to_surface(scan, map_, surface_, guess, icp_params, deadline);
  }
  const bool bakeoff_due =
    !device_align_decided_
      ? scan_count_ >= static_cast<std::uint64_t>(params_.device_bakeoff_scan)
      : (params_.device_recheck_scans > 0 &&
         scan_count_ - device_align_decided_at_ >= static_cast<std::uint64_t>(params_.device_recheck_scans));
  if (bakeoff_due) {
    // The bake-off ignores the live rate budget: both solvers run to completion on this scan.
    device_align_decided_ = true;
    device_align_decided_at_ = scan_count_;
    const auto device_start = std::chrono::steady_clock::now();
    const std::optional<PlaneIcpResult> device =
      align_points_on_device(scan, map_, surface_, map_epoch_, guess, icp_params, nullptr);
    const double device_ms = elapsed_ms(device_start);
    const auto cpu_start = std::chrono::steady_clock::now();
    const PlaneIcpResult cpu = align_to_surface(scan, map_, surface_, guess, icp_params, nullptr);
    const double cpu_ms = elapsed_ms(cpu_start);
    device_align_use_ = device.has_value() && close_poses(cpu.pose, device->pose) && device_ms < 0.8 * cpu_ms;
    return device_align_use_ ? *device : cpu;
  }
  if (device_align_use_) {
    if (const std::optional<PlaneIcpResult> device =
          align_points_on_device(scan, map_, surface_, map_epoch_, guess, icp_params, deadline)) {
      return *device;
    }
    device_align_use_ = false;
  }
  return align_to_surface(scan, map_, surface_, guess, icp_params, deadline);
}

}  // namespace back_odom
