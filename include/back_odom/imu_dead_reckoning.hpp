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

#ifndef BACK_ODOM__IMU_DEAD_RECKONING_HPP_
#define BACK_ODOM__IMU_DEAD_RECKONING_HPP_

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

namespace back_odom
{

class ImuDeadReckoning
{
public:
  explicit ImuDeadReckoning(double gravity);

  void set_initial_orientation(const Sophus::SO3d & orientation);
  void set_gyro_bias(const Eigen::Vector3d & gyro_bias);
  void integrate(
    const Eigen::Vector3d & angular_velocity, const Eigen::Vector3d & linear_acceleration,
    double dt);
  void reset_state(const Sophus::SE3d & pose, const Eigen::Vector3d & velocity_world);

  [[nodiscard]] const Sophus::SO3d & orientation() const;
  [[nodiscard]] const Eigen::Vector3d & position() const;
  [[nodiscard]] const Eigen::Vector3d & velocity() const;
  [[nodiscard]] const Eigen::Vector3d & linear_acceleration_world() const;
  [[nodiscard]] const Eigen::Vector3d & gyro_bias() const;
  [[nodiscard]] double gravity() const;

private:
  Sophus::SO3d orientation_{};
  Eigen::Vector3d position_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d gyro_bias_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d linear_acceleration_world_{Eigen::Vector3d::Zero()};
  double gravity_{9.81};
};

}  // namespace back_odom

#endif  // BACK_ODOM__IMU_DEAD_RECKONING_HPP_
