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

#ifndef BACK_ODOM__LIDAR_PREPROCESS_HPP_
#define BACK_ODOM__LIDAR_PREPROCESS_HPP_

#include "back_odom/imu/imu_types.hpp"

#include <Eigen/Core>

#include <vector>

namespace back_odom
{

[[nodiscard]] std::vector<Eigen::Vector3d> crop_lidar_box(
  const std::vector<Eigen::Vector3d> & points, double half_longitudinal, double half_lateral);

/// Put each lidar point in the integrator frame with `body_from_lidar`, then move it from its
/// timestamp to the scan-end pose. The trajectory and the extrinsic must share that frame.
[[nodiscard]] std::vector<Eigen::Vector3d> deskew_to_scan_end(
  const std::vector<Eigen::Vector3d> & points, const std::vector<double> & timestamps,
  const std::vector<StampedPose> & trajectory, const Sophus::SE3d & body_from_lidar = {});

}  // namespace back_odom

#endif  // BACK_ODOM__LIDAR_PREPROCESS_HPP_
