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

#ifndef BACK_ODOM__IMU_TYPES_HPP_
#define BACK_ODOM__IMU_TYPES_HPP_

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include <cstddef>

namespace back_odom
{

struct ImuSample
{
  double stamp{0.0};
  Eigen::Vector3d linear_acceleration{Eigen::Vector3d::Zero()};
  Eigen::Vector3d angular_velocity{Eigen::Vector3d::Zero()};
};

struct ProcessorParams
{
  std::size_t alignment_sample_count{100};
  double stationary_gyro_thresh{0.05};
  double stationary_accel_dev_thresh{0.5};
  double gravity{9.81};
  double max_dt{0.1};
};

enum class ProcessorPhase { Collecting, Tracking };

struct StampedPose
{
  double stamp{0.0};
  Sophus::SE3d pose{};
};

struct ProcessorOutput
{
  ProcessorPhase phase{ProcessorPhase::Collecting};
  std::size_t sample_count{0};
  bool aligned{false};
  Sophus::SO3d orientation{};
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity_world{Eigen::Vector3d::Zero()};
  Eigen::Vector3d angular_velocity_body{Eigen::Vector3d::Zero()};
  Eigen::Vector3d linear_acceleration_world{Eigen::Vector3d::Zero()};
  Eigen::Vector3d specific_force_body{Eigen::Vector3d::Zero()};
  double gravity{9.81};
};

}  // namespace back_odom

#endif  // BACK_ODOM__IMU_TYPES_HPP_
