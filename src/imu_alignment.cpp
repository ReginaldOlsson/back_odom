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

#include "back_odom/imu_alignment.hpp"

#include <Eigen/Geometry>

#include <stdexcept>
#include <vector>

namespace back_odom
{
namespace
{
constexpr double k_min_plausible_gravity = 8.0;
constexpr double k_max_plausible_gravity = 11.0;
constexpr double k_pi = 3.14159265358979323846;

Sophus::SO3d rotation_aligning_vectors(const Eigen::Vector3d & from, const Eigen::Vector3d & to)
{
  const double cos_theta = from.dot(to);
  const Eigen::Vector3d axis = from.cross(to);
  if (cos_theta < -1.0 + 1e-8) {
    Eigen::Vector3d orthogonal = Eigen::Vector3d::UnitX().cross(from);
    if (orthogonal.squaredNorm() < 1e-12) {
      orthogonal = Eigen::Vector3d::UnitY().cross(from);
    }
    orthogonal.normalize();
    return Sophus::SO3d::exp(orthogonal * k_pi);
  }

  Eigen::Quaterniond quaternion;
  quaternion.w() = 1.0 + cos_theta;
  quaternion.vec() = axis;
  quaternion.normalize();
  return Sophus::SO3d(quaternion);
}

}  // namespace

Sophus::SO3d compute_initial_alignment(const std::vector<Eigen::Vector3d> & linear_accelerations)
{
  if (linear_accelerations.empty()) {
    throw std::invalid_argument("compute_initial_alignment requires at least one acceleration");
  }

  Eigen::Vector3d acceleration_mean = Eigen::Vector3d::Zero();
  for (const auto & acceleration : linear_accelerations) {
    acceleration_mean += acceleration;
  }
  acceleration_mean /= static_cast<double>(linear_accelerations.size());

  if (acceleration_mean.norm() < 1e-6) {
    throw std::invalid_argument("mean linear acceleration is too small to define an up direction");
  }
  acceleration_mean.normalize();
  return rotation_aligning_vectors(acceleration_mean, -Eigen::Vector3d::UnitZ());
}

ImuSample to_right_handed_imu(const ImuSample & sample)
{
  ImuSample right_handed = sample;
  right_handed.linear_acceleration.y() = -sample.linear_acceleration.y();
  right_handed.angular_velocity.x() = -sample.angular_velocity.x();
  right_handed.angular_velocity.z() = -sample.angular_velocity.z();
  return right_handed;
}

ImuSample transform_imu_sample(
  const ImuSample & sample, const Sophus::SE3d & target_from_imu,
  const Eigen::Vector3d & previous_angular_velocity, const double previous_stamp,
  const bool has_previous)
{
  const Sophus::SO3d & rotation = target_from_imu.so3();
  const Eigen::Vector3d angular_velocity = rotation * sample.angular_velocity;
  Eigen::Vector3d linear_acceleration = rotation * sample.linear_acceleration;

  const Eigen::Vector3d imu_to_target = -target_from_imu.translation();
  const double dt = sample.stamp - previous_stamp;
  if (has_previous && dt > 0.0) {
    const Eigen::Vector3d angular_acceleration =
      (angular_velocity - previous_angular_velocity) / dt;
    linear_acceleration += angular_acceleration.cross(imu_to_target);
  }
  linear_acceleration += angular_velocity.cross(angular_velocity.cross(imu_to_target));

  ImuSample transformed = sample;
  transformed.angular_velocity = angular_velocity;
  transformed.linear_acceleration = linear_acceleration;
  return transformed;
}

bool is_stationary(
  const std::vector<ImuSample> & samples, const double gyro_thresh, const double accel_dev_thresh)
{
  if (samples.empty()) {
    return false;
  }

  Eigen::Vector3d acceleration_sum = Eigen::Vector3d::Zero();
  for (const auto & sample : samples) {
    if (sample.angular_velocity.norm() > gyro_thresh) {
      return false;
    }
    acceleration_sum += sample.linear_acceleration;
  }

  const Eigen::Vector3d acceleration_mean = acceleration_sum / static_cast<double>(samples.size());
  for (const auto & sample : samples) {
    if ((sample.linear_acceleration - acceleration_mean).norm() > accel_dev_thresh) {
      return false;
    }
  }
  return true;
}

Eigen::Vector3d mean_angular_velocity(const std::vector<ImuSample> & samples)
{
  if (samples.empty()) {
    throw std::invalid_argument("mean_angular_velocity requires at least one sample");
  }

  Eigen::Vector3d angular_velocity_sum = Eigen::Vector3d::Zero();
  for (const auto & sample : samples) {
    angular_velocity_sum += sample.angular_velocity;
  }
  return angular_velocity_sum / static_cast<double>(samples.size());
}

AlignmentResult align_stationary_window(
  const std::vector<ImuSample> & samples, const double nominal_gravity)
{
  if (samples.empty()) {
    throw std::invalid_argument("align_stationary_window requires a non-empty IMU window");
  }

  std::vector<Eigen::Vector3d> linear_accelerations;
  linear_accelerations.reserve(samples.size());
  Eigen::Vector3d acceleration_sum = Eigen::Vector3d::Zero();
  for (const auto & sample : samples) {
    linear_accelerations.push_back(sample.linear_acceleration);
    acceleration_sum += sample.linear_acceleration;
  }

  AlignmentResult result;
  result.orientation = compute_initial_alignment(linear_accelerations);
  result.gyro_bias = mean_angular_velocity(samples);
  const double acceleration_norm = (acceleration_sum / static_cast<double>(samples.size())).norm();
  result.gravity = nominal_gravity;
  if (
    acceleration_norm >= k_min_plausible_gravity && acceleration_norm <= k_max_plausible_gravity) {
    result.gravity = acceleration_norm;
  }
  return result;
}

}  // namespace back_odom
