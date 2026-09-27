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

void ImuDeadReckoning::integrate(
  const Eigen::Vector3d & angular_velocity, const Eigen::Vector3d & linear_acceleration,
  const double dt)
{
  if (dt <= 0.0) {
    return;
  }

  orientation_ = orientation_ * Sophus::SO3d::exp((angular_velocity - gyro_bias_) * dt);
  linear_acceleration_world_ =
    orientation_ * linear_acceleration + Eigen::Vector3d(0.0, 0.0, -gravity_);
  velocity_ += linear_acceleration_world_ * dt;
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

double ImuDeadReckoning::gravity() const
{
  return gravity_;
}

}  // namespace back_odom
