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

bool same_sequence(const std::vector<Eigen::Vector3d> & left, const std::vector<Eigen::Vector3d> & right)
{
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.size(); ++index) {
    if ((left[index] - right[index]).norm() > 1.0e-6) {
      return false;
    }
  }
  return true;
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
  frozen_(params.voxel_size, 1.0e6, static_cast<unsigned int>(params.max_points_per_voxel)),
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
  const int settle_passes = std::max(1, params_.refine_settle_passes);
  std::vector<LocalMapPoint> annotated;
  annotated.reserve(visible_scan_point_count());
  for (const HorizonScan & scan : horizon_) {
    const bool frozen = scan.collective_passes >= settle_passes;
    const float intensity = frozen ? 255.0F : 30.0F;
    for (const Eigen::Vector3d & point : scan.points) {
      LocalMapPoint annotated_point;
      annotated_point.position = scan.pose * point;
      annotated_point.intensity = intensity;
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
  const Sophus::SE3d & pose, const double stamp, const std::vector<Eigen::Vector3d> & points,
  const ImuProcessor & imu)
{
  HorizonScan scan;
  scan.stamp = stamp;
  scan.pose = pose;
  scan.prior = pose;
  scan.points = points;
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
  const Sophus::SE3d & latest = horizon_.back().pose;
  std::deque<HorizonScan> kept;
  for (HorizonScan & stored : horizon_) {
    const bool inside = scan_pose_inside_visible_cloud(
      latest, stored.pose, half_longitudinal_, half_lateral_);
    if (inside) {
      kept.push_back(std::move(stored));
    }
  }
  const std::size_t max_scans = static_cast<std::size_t>(std::max(1, params_.max_visible_scans));
  while (kept.size() > max_scans) {
    kept.pop_front();
  }
  horizon_ = std::move(kept);
}

bool LidarImuMatcher::refine_window_if_due(MatchResult & result, Sophus::SE3d & newest_pose)
{
  // Freezing each scan at its own pose painted the same building once per scan and stalled the callback.
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
  cull_local_map(rebuilt, cull_pose);
  if (rebuilt.Empty()) {
    return;
  }
  map_ = std::move(rebuilt);
  note_map_edit();
}

void LidarImuMatcher::cull_local_map(kiss_icp::VoxelHashMap & map, const Sophus::SE3d & pose) const
{
  map.RemovePointsOutsideBox(pose.inverse(), half_longitudinal_, half_lateral_);
}

MatchResult LidarImuMatcher::on_scan(const LidarScan & scan, ImuProcessor & imu)
{
  MatchResult result;
  result.health = health_.health;
  if (!imu.aligned()) {
    return result;
  }
  const auto prepared_clouds = prepare_scan(scan, imu.trajectory(), result);
  const std::vector<Eigen::Vector3d> & map_points = prepared_clouds.first;
  const std::vector<Eigen::Vector3d> & source = prepared_clouds.second;
  if (source.empty()) {
    return result;
  }

  const double scan_end = *std::max_element(scan.timestamps.cbegin(), scan.timestamps.cend());
  const Sophus::SE3d predicted = imu.interpolate_pose(scan_end);
  result.imu_pose = predicted;
  PreparedScan prepared{scan_end, map_points, source, predicted};

  if (!has_reference_) {
    window_.push_back(WindowScan{scan_end, map_points, source, predicted});
    has_reference_ = true;
    last_correction_stamp_ = scan.stamp;
    last_correction_position_ = predicted.translation();
    rebuild_map(predicted);
    record_horizon_scan(predicted, scan_end, map_points, imu);
    result.applied = true;
    result.inserted_scan = true;
    result.first_scan = true;
    result.corrected_pose = predicted;
    remember_healthy(predicted, imu.velocity(), scan_end);
    note_health(imu, HealthEvent::Accepted);
    result.health = health_.health;
    return result;
  }

  const Sophus::SE3d imu_guess = predicted;
  const MatchCandidate imu_candidate = score_alignment(source, imu_guess, predicted, true, result);

  result.has_lidar_debug = imu_candidate.present;
  result.lidar_pose = imu_candidate.pose;
  result.lidar_cost = imu_candidate.cost;
  result.lidar_passed = imu_candidate.passes;

  if (!(imu_candidate.present && imu_candidate.passes)) {
    const Sophus::SE3d failed = imu_candidate.present ? imu_candidate.pose : predicted;
    const Sophus::SE3d vehicle_delta = predicted.inverse() * failed;
    result.applied = false;
    result.correction = failed * predicted.inverse();
    result.translation_error = vehicle_delta.translation().norm();
    result.rotation_error = vehicle_delta.so3().log().norm();
    result.iterations = imu_candidate.iterations;
    result.health = health_.health;
    return result;
  }

  result.iterations = imu_candidate.iterations;
  finish_accepted_scan(
    result, prepared, imu_candidate.pose, imu_candidate.iterations, imu, last_correction_stamp_, true,
    std::nullopt);
  note_health(imu, HealthEvent::Accepted);
  remember_healthy(result.corrected_pose, imu.velocity(), scan_end);
  result.health = health_.health;
  result.gyro_bias = imu.gyro_bias();
  result.accel_bias = imu.accel_bias();
  return result;
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

void LidarImuMatcher::finish_accepted_scan(
  MatchResult & result, PreparedScan prepared, const Sophus::SE3d & aligned_pose,
  const double iterations, ImuProcessor & imu, const double since_stamp, const bool update_bias,
  const std::optional<Eigen::Vector3d> & velocity_override)
{
  const Sophus::SE3d limited = limit_longitudinal_correction(
    prepared.predicted, aligned_pose, params_.max_longitudinal_correction);
  const auto map_start = std::chrono::steady_clock::now();
  map_.Update(prepared.map_points, limited);
  record_horizon_scan(limited, prepared.scan_end, prepared.map_points, imu);
  window_.push_back(
    WindowScan{prepared.scan_end, std::move(prepared.map_points), std::move(prepared.source), limited});
  while (static_cast<int>(window_.size()) > params_.backward_match_stride) {
    window_.pop_front();
  }
  cull_local_map(map_, limited);
  result.map_ms += elapsed_ms(map_start);
  Sophus::SE3d committed = limited;
  const auto refine_start = std::chrono::steady_clock::now();
  refine_window_if_due(result, committed);
  result.refine_ms += elapsed_ms(refine_start);
  window_.back().pose = committed;

  const Sophus::SE3d vehicle_delta = prepared.predicted.inverse() * committed;
  result.correction = window_.back().pose * prepared.predicted.inverse();
  result.translation_error = vehicle_delta.translation().norm();
  result.rotation_error = vehicle_delta.so3().log().norm();
  result.corrected_pose = imu.pose() * vehicle_delta;
  result.inserted_scan = true;
  result.applied = true;
  result.iterations = iterations;
  apply_correction(imu, vehicle_delta, prepared.scan_end, since_stamp, update_bias, velocity_override);
  result.corrected_pose = imu.pose();
  result.gyro_bias = imu.gyro_bias();
  result.accel_bias = imu.accel_bias();
  note_map_edit();
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

Sophus::SO3d LidarImuMatcher::imu_rotation_between(
  const ImuProcessor & imu, const double from_stamp, const double to_stamp) const
{
  if (to_stamp <= from_stamp + 1.0e-6) {
    return Sophus::SO3d();
  }
  return imu.interpolate_pose(from_stamp).so3().inverse() * imu.interpolate_pose(to_stamp).so3();
}

Sophus::SE3d LidarImuMatcher::recovery_pose(const double scan_end, const ImuProcessor & imu) const
{
  const Sophus::SE3d last_healthy = has_healthy_ ? last_healthy_pose_ : imu.pose();
  const double origin_stamp = has_healthy_ ? last_healthy_stamp_ : scan_end;
  const double tail_dt = std::max(0.0, scan_end - origin_stamp);
  const double limit = std::max(0.0, params_.limits.max_speed);
  const double speed = std::clamp(last_healthy_speed_, -limit, limit);
  const Sophus::SO3d tail_rotation = imu_rotation_between(imu, origin_stamp, scan_end);
  return last_healthy * Sophus::SE3d(tail_rotation, Eigen::Vector3d(speed * tail_dt, 0.0, 0.0));
}

MatchCandidate LidarImuMatcher::score_alignment(
  const std::vector<Eigen::Vector3d> & source, const Sophus::SE3d & guess,
  const Sophus::SE3d & imu_reference, const bool check_imu_gate, MatchResult & timing)
{
  MatchCandidate candidate;
  candidate.present = true;
  double iterations = 0.0;
  double device_cost = 0.0;
  const auto align_start = std::chrono::steady_clock::now();
  const kiss_icp::PlaneAlignResult aligned = align(source, guess, iterations, &device_cost);
  timing.align_ms += elapsed_ms(align_start);
  candidate.pose = aligned.pose;
  candidate.iterations = iterations;
  candidate.saturated = iterations + 0.5 >= static_cast<double>(params_.max_iterations);
  const auto cost_start = std::chrono::steady_clock::now();
  if (device_align_use_ && std::isfinite(device_cost)) {
    candidate.cost = device_cost;
  } else {
    candidate.cost = robust_plane_cost(
      source, aligned.pose, map_, params_.max_correspondence_distance, params_.kernel_scale);
  }
  timing.cost_ms += elapsed_ms(cost_start);
  const Sophus::SE3d imu_delta = imu_reference.inverse() * aligned.pose;
  const bool imu_ok = !check_imu_gate || accept_vehicle_delta(iterations, imu_delta);
  candidate.passes = iterations > 0.0 && imu_ok && std::isfinite(candidate.cost);
  return candidate;
}

void LidarImuMatcher::optimize_window(ImuProcessor & imu)
{
  if (window_.size() < 2) {
    return;
  }
  std::vector<Sophus::SE3d> poses;
  std::vector<std::vector<Eigen::Vector3d>> queries;
  std::vector<std::vector<Eigen::Vector3d>> surfaces;
  std::vector<Sophus::SE3d> relatives;
  poses.reserve(window_.size());
  queries.reserve(window_.size());
  surfaces.reserve(window_.size());
  for (const WindowScan & scan : window_) {
    poses.push_back(scan.pose);
    queries.push_back(scan.source_points);
    surfaces.push_back(scan.map_points);
  }
  for (std::size_t index = 1; index < window_.size(); ++index) {
    const Sophus::SE3d before = imu.interpolate_pose(window_[index - 1].stamp);
    const Sophus::SE3d after = imu.interpolate_pose(window_[index].stamp);
    relatives.push_back(before.inverse() * after);
  }
  const int window_iterations = std::min(params_.max_iterations, 15);
  optimize_scan_window(
    poses, queries, surfaces, relatives, frozen_, params_.max_correspondence_distance,
    params_.kernel_scale, window_iterations, params_.convergence_criterion, 0.05, 0.01,
    params_.voxel_size, static_cast<unsigned int>(params_.max_points_per_voxel));
  for (std::size_t index = 0; index < window_.size(); ++index) {
    window_[index].pose = poses[index];
  }
}

void LidarImuMatcher::freeze_oldest(const Sophus::SE3d & cull_pose)
{
  while (static_cast<int>(window_.size()) > params_.backward_match_stride) {
    frozen_.Update(window_.front().map_points, window_.front().pose);
    window_.pop_front();
  }
  if (!frozen_.Empty()) {
    frozen_.RemovePointsOutsideBox(cull_pose.inverse(), half_longitudinal_, half_lateral_);
  }
}

void LidarImuMatcher::rebuild_map(const Sophus::SE3d & cull_pose)
{
  map_.Clear();
  if (!frozen_.Empty()) {
    map_.AddPoints(frozen_.Pointcloud());
  }
  for (const WindowScan & scan : window_) {
    map_.Update(scan.map_points, scan.pose);
  }
  map_.RemovePointsOutsideBox(cull_pose.inverse(), half_longitudinal_, half_lateral_);
  note_map_edit();
}

void LidarImuMatcher::note_map_edit()
{
  ++map_epoch_;
}

std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>> LidarImuMatcher::prepare_scan(
  const LidarScan & scan, const std::vector<StampedPose> & trajectory, MatchResult & timing)
{
  const auto cpu_prepare = [&](const std::vector<Eigen::Vector3d> & cropped) {
    const std::vector<Eigen::Vector3d> map_points =
      kiss_icp::VoxelDownsample(cropped, params_.voxel_size * 0.5);
    const std::vector<Eigen::Vector3d> source =
      kiss_icp::VoxelDownsample(map_points, params_.voxel_size * 1.5);
    return std::make_pair(map_points, source);
  };
  const auto run_device = [&]() {
    return device_downsample_scan(
      scan.points, scan.timestamps, trajectory, body_from_lidar_, half_longitudinal_, half_lateral_,
      params_.voxel_size * 0.5, params_.voxel_size * 1.5);
  };
  if (!device_prepare_decided_ && scan.points.size() >= 1000 && device_available() && !trajectory.empty()) {
    device_prepare_decided_ = true;
    (void)run_device();
    const auto device_start = std::chrono::steady_clock::now();
    const std::optional<DeviceScanClouds> device = run_device();
    const double device_ms = elapsed_ms(device_start);
    const auto cpu_deskew_start = std::chrono::steady_clock::now();
    const std::vector<Eigen::Vector3d> cpu_cropped = crop_lidar_box(
      deskew_to_scan_end(scan.points, scan.timestamps, trajectory, body_from_lidar_),
      half_longitudinal_, half_lateral_);
    const double cpu_deskew_ms = elapsed_ms(cpu_deskew_start);
    const auto cpu_voxel_start = std::chrono::steady_clock::now();
    const auto cpu_clouds = cpu_prepare(cpu_cropped);
    const double cpu_voxel_ms = elapsed_ms(cpu_voxel_start);
    device_front_use_ = device.has_value() && same_sequence(device->map_points, cpu_clouds.first) &&
                        same_sequence(device->source, cpu_clouds.second) &&
                        device_ms < cpu_deskew_ms + cpu_voxel_ms;
    if (device_front_use_) {
      timing.deskew_ms += device->deskew_ms;
      timing.voxel_ms += device->voxel_ms;
      return {device->map_points, device->source};
    }
    timing.deskew_ms += cpu_deskew_ms;
    timing.voxel_ms += cpu_voxel_ms;
    return cpu_clouds;
  }
  if (device_front_use_) {
    if (const std::optional<DeviceScanClouds> device = run_device()) {
      timing.deskew_ms += device->deskew_ms;
      timing.voxel_ms += device->voxel_ms;
      return {device->map_points, device->source};
    }
    device_front_use_ = false;
  }
  const auto deskew_start = std::chrono::steady_clock::now();
  const std::vector<Eigen::Vector3d> cropped = crop_lidar_box(
    deskew_to_scan_end(scan.points, scan.timestamps, trajectory, body_from_lidar_),
    half_longitudinal_, half_lateral_);
  timing.deskew_ms += elapsed_ms(deskew_start);
  const auto voxel_start = std::chrono::steady_clock::now();
  const auto clouds = cpu_prepare(cropped);
  timing.voxel_ms += elapsed_ms(voxel_start);
  return clouds;
}

kiss_icp::PlaneAlignResult LidarImuMatcher::align(
  const std::vector<Eigen::Vector3d> & scan, const Sophus::SE3d & guess, double & iterations,
  double * const robust_cost)
{
  const auto cpu_align = [&]() {
    if (robust_cost != nullptr) {
      *robust_cost = std::numeric_limits<double>::quiet_NaN();
    }
    return registration_.AlignPointsToPlane(
      scan, map_, guess, params_.max_correspondence_distance, params_.kernel_scale, iterations);
  };
  const auto accept_device = [&](const DeviceAlignResult & device) {
    iterations = device.iterations;
    if (robust_cost != nullptr) {
      *robust_cost = device.robust_cost;
    }
    kiss_icp::PlaneAlignResult accepted;
    accepted.pose = device.pose;
    accepted.correction = device.correction;
    return accepted;
  };
  if (!device_available() || scan.size() < 256) {
    return cpu_align();
  }
  if (!device_align_decided_) {
    device_align_decided_ = true;
    const auto time_device = [&](const bool float_rank) {
      (void)align_points_on_device(
        scan, map_, map_epoch_, guess, params_.max_correspondence_distance, params_.kernel_scale,
        params_.max_iterations, params_.convergence_criterion, float_rank);
      const auto device_start = std::chrono::steady_clock::now();
      const std::optional<DeviceAlignResult> device = align_points_on_device(
        scan, map_, map_epoch_, guess, params_.max_correspondence_distance, params_.kernel_scale,
        params_.max_iterations, params_.convergence_criterion, float_rank);
      return std::make_pair(device, elapsed_ms(device_start));
    };
    const auto double_trial = time_device(false);
    double cpu_iterations = 0.0;
    const auto cpu_start = std::chrono::steady_clock::now();
    const kiss_icp::PlaneAlignResult cpu = registration_.AlignPointsToPlane(
      scan, map_, guess, params_.max_correspondence_distance, params_.kernel_scale, cpu_iterations);
    (void)robust_plane_cost(
      scan, cpu.pose, map_, params_.max_correspondence_distance, params_.kernel_scale);
    const double cpu_ms = elapsed_ms(cpu_start);
    const auto accept_if_faster = [&](const std::optional<DeviceAlignResult> & device, const double device_ms,
                                      const bool float_rank) {
      const bool close = device.has_value() &&
                         (cpu.pose.inverse() * device->pose).translation().norm() < 1.0e-3 &&
                         (cpu.pose.inverse() * device->pose).so3().log().norm() < 1.0e-3;
      const bool cost_matches =
        device.has_value() &&
        std::abs(
          device->robust_cost -
          robust_plane_cost(
            scan, device->pose, map_, params_.max_correspondence_distance, params_.kernel_scale)) <
          1.0e-4;
      if (!(close && cost_matches && device_ms < cpu_ms)) {
        return false;
      }
      device_align_use_ = true;
      device_align_float_ = float_rank;
      return true;
    };
    if (accept_if_faster(double_trial.first, double_trial.second, false)) {
      return accept_device(*double_trial.first);
    }
    const auto float_trial = time_device(true);
    if (accept_if_faster(float_trial.first, float_trial.second, true)) {
      return accept_device(*float_trial.first);
    }
    iterations = cpu_iterations;
    if (robust_cost != nullptr) {
      *robust_cost = std::numeric_limits<double>::quiet_NaN();
    }
    return cpu;
  }
  if (!device_align_use_) {
    return cpu_align();
  }
  const std::optional<DeviceAlignResult> device = align_points_on_device(
    scan, map_, map_epoch_, guess, params_.max_correspondence_distance, params_.kernel_scale,
    params_.max_iterations, params_.convergence_criterion, device_align_float_);
  if (!device) {
    device_align_use_ = false;
    return cpu_align();
  }
  iterations = device->iterations;
  if (robust_cost != nullptr) {
    *robust_cost = device->robust_cost;
  }
  kiss_icp::PlaneAlignResult result;
  result.pose = device->pose;
  result.correction = device->correction;
  return result;
}

}  // namespace back_odom
