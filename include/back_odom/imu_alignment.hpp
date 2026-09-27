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

#ifndef BACK_ODOM__IMU_ALIGNMENT_HPP_
#define BACK_ODOM__IMU_ALIGNMENT_HPP_

#include "back_odom/imu_types.hpp"

#include <Eigen/Core>
#include <sophus/so3.hpp>

#include <vector>

namespace back_odom
{

struct AlignmentResult
{
  Sophus::SO3d orientation{};
  Eigen::Vector3d gyro_bias{Eigen::Vector3d::Zero()};
  double gravity{9.81};
};

/// Rotation that maps the mean specific force onto world +Z (ENU up).
[[nodiscard]] Sophus::SO3d compute_initial_alignment(
  const std::vector<Eigen::Vector3d> & linear_accelerations);

[[nodiscard]] bool is_stationary(
  const std::vector<ImuSample> & samples, double gyro_thresh, double accel_dev_thresh);

[[nodiscard]] Eigen::Vector3d mean_angular_velocity(const std::vector<ImuSample> & samples);

[[nodiscard]] AlignmentResult align_stationary_window(
  const std::vector<ImuSample> & samples, double nominal_gravity);

}  // namespace back_odom

#endif  // BACK_ODOM__IMU_ALIGNMENT_HPP_
