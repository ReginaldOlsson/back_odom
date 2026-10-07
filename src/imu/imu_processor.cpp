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

#include "back_odom/imu/imu_processor.hpp"

#include "back_odom/imu/imu_alignment.hpp"
#include "back_odom/imu/kinematic_limits.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace back_odom
{

ImuProcessor::ImuProcessor(const ProcessorParams & params) : params_(params)
{
  if (params_.alignment_sample_count == 0) {
    throw std::invalid_argument("alignment_sample_count must be positive");
  }
  if (params_.gravity <= 0.0) {
    throw std::invalid_argument("gravity must be positive");
  }
  if (params_.max_dt <= 0.0) {
    throw std::invalid_argument("max_dt must be positive");
  }
  if (params_.max_speed < 0.0 || params_.max_acceleration < 0.0) {
    throw std::invalid_argument("speed and acceleration limits must be non-negative");
  }
  if (!(params_.trajectory_horizon > 0.0) || !std::isfinite(params_.trajectory_horizon)) {
    throw std::invalid_argument("trajectory_horizon must be positive and finite");
  }
  if (params_.max_trajectory_poses < 2) {
    throw std::invalid_argument("max_trajectory_poses must be at least 2");
  }
  if (!(params_.max_pair_dt > 0.0) || !std::isfinite(params_.max_pair_dt)) {
    throw std::invalid_argument("max_pair_dt must be positive and finite");
  }
}

ProcessorOutput ImuProcessor::process(const ImuSample & sample)
{
  if (!dead_reckoning_) {
    return process_collecting(sample);
  }
  return process_tracking(sample);
}

ProcessorOutput ImuProcessor::process_collecting(const ImuSample & sample)
{
  window_.push_back(sample);
  if (window_.size() > params_.alignment_sample_count) {
    window_.erase(window_.begin());
  }

  ProcessorOutput output;
  output.phase = ProcessorPhase::Collecting;
  output.aligned = false;
  output.sample_count = window_.size();
  output.specific_force_body = sample.linear_acceleration;
  output.angular_velocity_body = sample.angular_velocity;
  output.gravity = params_.gravity;

  if (
    window_.size() != params_.alignment_sample_count ||
    !is_stationary(window_, params_.stationary_gyro_thresh, params_.stationary_accel_dev_thresh)) {
    return output;
  }

  const AlignmentResult alignment = align_stationary_window(window_, params_.gravity);
  dead_reckoning_.emplace(alignment.gravity);
  dead_reckoning_->set_initial_orientation(alignment.orientation);
  dead_reckoning_->set_gyro_bias(alignment.gyro_bias);
  window_.clear();
  has_previous_stamp_ = true;
  previous_stamp_ = sample.stamp;

  output = make_tracking_output(sample);
  output.sample_count = params_.alignment_sample_count;
  record_pose(sample.stamp);
  return output;
}

ProcessorOutput ImuProcessor::process_tracking(const ImuSample & sample)
{
  const double dt = sample.stamp - previous_stamp_;
  bool speed_clamped = false;
  if (has_previous_stamp_ && dt > 0.0 && dt <= params_.max_dt) {
    const Eigen::Vector3d velocity_before = dead_reckoning_->velocity();
    const Sophus::SO3d orientation_before = dead_reckoning_->orientation();
    dead_reckoning_->integrate(sample.angular_velocity, sample.linear_acceleration, dt);
    const Eigen::Vector3d body_before = orientation_before.inverse() * velocity_before;
    const Eigen::Vector3d body_after =
      dead_reckoning_->orientation().inverse() * dead_reckoning_->velocity();
    VehicleLimits limits;
    limits.max_speed = params_.max_speed;
    limits.max_acceleration = params_.max_acceleration;
    const Eigen::Vector3d limited = limit_body_velocity(body_before, body_after, dt, limits);
    speed_clamped = std::abs(body_after.x()) > params_.max_speed + 1.0e-3 ||
                    std::abs(body_after.z()) > params_.max_speed + 1.0e-3;
    dead_reckoning_->reset_state(
      Sophus::SE3d(dead_reckoning_->orientation(), dead_reckoning_->position()),
      dead_reckoning_->orientation() * limited);
  }
  previous_stamp_ = sample.stamp;
  has_previous_stamp_ = true;

  ProcessorOutput output = make_tracking_output(sample);
  output.speed_clamped = speed_clamped;
  output.sample_count = params_.alignment_sample_count;
  record_pose(sample.stamp);
  return output;
}

ProcessorOutput ImuProcessor::output_at(const ImuSample & sample) const
{
  if (!dead_reckoning_) {
    ProcessorOutput output;
    output.specific_force_body = sample.linear_acceleration;
    output.angular_velocity_body = sample.angular_velocity;
    output.gravity = params_.gravity;
    return output;
  }
  return make_tracking_output(sample);
}

bool ImuProcessor::aligned() const
{
  return dead_reckoning_.has_value();
}

Sophus::SE3d ImuProcessor::pose() const
{
  if (!dead_reckoning_) {
    return Sophus::SE3d();
  }
  return {dead_reckoning_->orientation(), dead_reckoning_->position()};
}

Eigen::Vector3d ImuProcessor::velocity() const
{
  if (!dead_reckoning_) {
    return Eigen::Vector3d::Zero();
  }
  return dead_reckoning_->velocity();
}

double ImuProcessor::latest_stamp() const
{
  return previous_stamp_;
}

const std::vector<StampedPose> & ImuProcessor::trajectory() const
{
  return trajectory_;
}

double ImuProcessor::max_pair_dt() const
{
  return params_.max_pair_dt;
}

Sophus::SE3d ImuProcessor::interpolate_pose(const double stamp) const
{
  if (trajectory_.empty()) {
    throw std::runtime_error("imu trajectory is empty");
  }
  if (stamp <= trajectory_.front().stamp) {
    return trajectory_.front().pose;
  }
  if (stamp >= trajectory_.back().stamp) {
    return trajectory_.back().pose;
  }
  const auto after = std::lower_bound(
    trajectory_.cbegin(), trajectory_.cend(), stamp,
    [](const StampedPose & sample, const double value) { return sample.stamp < value; });
  if (after == trajectory_.cbegin()) {
    return trajectory_.front().pose;
  }
  if (after == trajectory_.cend()) {
    return trajectory_.back().pose;
  }
  const StampedPose & before = *std::prev(after);
  const double interval = after->stamp - before.stamp;
  const double alpha = interval > 1e-9 ? (stamp - before.stamp) / interval : 0.0;
  const Sophus::SE3d relative = before.pose.inverse() * after->pose;
  return before.pose * Sophus::SE3d::exp(alpha * relative.log());
}

StampedPose ImuProcessor::closest_pose(const double stamp) const
{
  if (trajectory_.empty()) {
    throw std::runtime_error("imu trajectory is empty");
  }
  const auto after = std::lower_bound(
    trajectory_.cbegin(), trajectory_.cend(), stamp,
    [](const StampedPose & sample, const double value) { return sample.stamp < value; });
  if (after == trajectory_.cbegin()) {
    return trajectory_.front();
  }
  if (after == trajectory_.cend()) {
    return trajectory_.back();
  }
  const StampedPose & before = *std::prev(after);
  return (stamp - before.stamp) <= (after->stamp - stamp) ? before : *after;
}

bool ImuProcessor::has_pose_near(const double stamp, const double max_dt) const
{
  if (trajectory_.empty() || !(max_dt >= 0.0)) {
    return false;
  }
  return std::abs(closest_pose(stamp).stamp - stamp) <= max_dt;
}

void ImuProcessor::set_velocity(const Eigen::Vector3d & velocity_world)
{
  if (!dead_reckoning_) {
    throw std::runtime_error("imu integrator is not aligned");
  }
  dead_reckoning_->reset_state(pose(), velocity_world);
}

void ImuProcessor::reset_state(
  const Sophus::SE3d & pose, const Eigen::Vector3d & velocity_world, const double stamp)
{
  if (!dead_reckoning_) {
    throw std::runtime_error("imu integrator is not aligned");
  }
  dead_reckoning_->reset_state(pose, velocity_world);
  if (!trajectory_.empty() && std::abs(trajectory_.back().stamp - stamp) < 1e-6) {
    trajectory_.back().pose = pose;
  } else {
    trajectory_.push_back(StampedPose{stamp, pose});
  }
}

void ImuProcessor::apply_lidar_correction(
  const Sophus::SE3d & body_delta, const Eigen::Vector3d & velocity_world, const double scan_stamp,
  const double since_stamp)
{
  if (!dead_reckoning_) {
    throw std::runtime_error("imu integrator is not aligned");
  }
  const Sophus::SE3d::Tangent tangent = body_delta.log();
  const double span = scan_stamp - since_stamp;
  if (tangent.norm() > 1e-8) {
    for (StampedPose & sample : trajectory_) {
      if (sample.stamp <= since_stamp) {
        continue;
      }
      const double alpha = (sample.stamp < scan_stamp && span > 1e-9)
                             ? std::clamp((sample.stamp - since_stamp) / span, 0.0, 1.0)
                             : 1.0;
      sample.pose = sample.pose * Sophus::SE3d::exp(alpha * tangent);
    }
  }
  const Sophus::SE3d corrected = pose() * body_delta;
  dead_reckoning_->reset_state(corrected, velocity_world);
  if (!trajectory_.empty() && std::abs(trajectory_.back().stamp - previous_stamp_) < 1e-6) {
    trajectory_.back().pose = corrected;
  }
}

Eigen::Vector3d ImuProcessor::gyro_bias() const
{
  if (!dead_reckoning_) {
    return Eigen::Vector3d::Zero();
  }
  return dead_reckoning_->gyro_bias();
}

Eigen::Vector3d ImuProcessor::accel_bias() const
{
  if (!dead_reckoning_) {
    return Eigen::Vector3d::Zero();
  }
  return dead_reckoning_->accel_bias();
}

void ImuProcessor::set_gyro_bias(const Eigen::Vector3d & gyro_bias)
{
  if (!dead_reckoning_) {
    throw std::runtime_error("imu integrator is not aligned");
  }
  dead_reckoning_->set_gyro_bias(gyro_bias);
}

void ImuProcessor::set_accel_bias(const Eigen::Vector3d & accel_bias)
{
  if (!dead_reckoning_) {
    throw std::runtime_error("imu integrator is not aligned");
  }
  dead_reckoning_->set_accel_bias(accel_bias);
}

void ImuProcessor::record_pose(const double stamp)
{
  trajectory_.push_back(StampedPose{stamp, pose()});
  const double cutoff = stamp - params_.trajectory_horizon;
  if (trajectory_.size() > 2 && trajectory_.front().stamp < cutoff) {
    const auto keep = std::lower_bound(
      trajectory_.cbegin(), trajectory_.cend(), cutoff,
      [](const StampedPose & sample, const double value) { return sample.stamp < value; });
    const std::size_t keep_index = static_cast<std::size_t>(keep - trajectory_.cbegin());
    const std::size_t erase_count = std::min(keep_index, trajectory_.size() - 2);
    if (erase_count > 0) {
      trajectory_.erase(
        trajectory_.begin(),
        trajectory_.begin() + static_cast<std::ptrdiff_t>(erase_count));
    }
  }
  while (trajectory_.size() > params_.max_trajectory_poses && trajectory_.size() > 2) {
    trajectory_.erase(trajectory_.begin());
  }
}

ProcessorOutput ImuProcessor::make_tracking_output(const ImuSample & sample) const
{
  ProcessorOutput output;
  output.phase = ProcessorPhase::Tracking;
  output.aligned = true;
  output.orientation = dead_reckoning_->orientation();
  output.position = dead_reckoning_->position();
  output.velocity_world = dead_reckoning_->velocity();
  output.linear_acceleration_world = dead_reckoning_->linear_acceleration_world();
  output.specific_force_body = sample.linear_acceleration;
  output.angular_velocity_body = sample.angular_velocity - dead_reckoning_->gyro_bias();
  output.gravity = dead_reckoning_->gravity();
  return output;
}

}  // namespace back_odom
