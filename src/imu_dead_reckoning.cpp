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

#include "back_odom/imu_dead_reckoning.hpp"

#include "back_odom/kinematic_limits.hpp"

#include <stdexcept>

namespace back_odom
{

ImuDeadReckoning::ImuDeadReckoning(const double gravity) : gravity_(gravity)
{
  if (gravity_ <= 0.0) {
    throw std::invalid_argument("gravity must be positive");
  }
}

void ImuDeadReckoning::set_initial_orientation(const Sophus::SO3d & orientation)
{
  orientation_ = orientation;
}

void ImuDeadReckoning::set_gyro_bias(const Eigen::Vector3d & gyro_bias)
{
  gyro_bias_ = gyro_bias;
}

void ImuDeadReckoning::set_accel_bias(const Eigen::Vector3d & accel_bias)
{
  accel_bias_ = accel_bias;
}

void ImuDeadReckoning::reset_state(
  const Sophus::SE3d & pose, const Eigen::Vector3d & velocity_world)
{
  orientation_ = pose.so3();
  position_ = pose.translation();
  velocity_ = velocity_world;
}

bool ImuDeadReckoning::clamp_speed(const double max_speed)
{
  return clamp_world_velocity(orientation_, velocity_, max_speed);
}

void ImuDeadReckoning::integrate(
  const Eigen::Vector3d & angular_velocity, const Eigen::Vector3d & linear_acceleration,
  const double dt)
{
  if (dt <= 0.0) {
    return;
  }

  const Sophus::SO3d previous_orientation = orientation_;
  orientation_ = orientation_ * Sophus::SO3d::exp((angular_velocity - gyro_bias_) * dt);
  // Carry the velocity with the heading, then drop the lateral part so it cannot steer the
  // next lidar guess off the forward axis.
  velocity_ = orientation_ * previous_orientation.inverse() * velocity_;
  const Eigen::Vector3d gravity_world(0.0, 0.0, -gravity_);
  linear_acceleration_world_ = orientation_ * (linear_acceleration - accel_bias_) - gravity_world;
  velocity_ += linear_acceleration_world_ * dt;
  Eigen::Vector3d velocity_body = orientation_.inverse() * velocity_;
  velocity_body.y() = 0.0;
  velocity_ = orientation_ * velocity_body;
  position_ += velocity_ * dt;
}

const Sophus::SO3d & ImuDeadReckoning::orientation() const
{
  return orientation_;
}

const Eigen::Vector3d & ImuDeadReckoning::position() const
{
  return position_;
}

const Eigen::Vector3d & ImuDeadReckoning::velocity() const
{
  return velocity_;
}

const Eigen::Vector3d & ImuDeadReckoning::linear_acceleration_world() const
{
  return linear_acceleration_world_;
}

const Eigen::Vector3d & ImuDeadReckoning::gyro_bias() const
{
  return gyro_bias_;
}

const Eigen::Vector3d & ImuDeadReckoning::accel_bias() const
{
  return accel_bias_;
}

double ImuDeadReckoning::gravity() const
{
  return gravity_;
}

}  // namespace back_odom
