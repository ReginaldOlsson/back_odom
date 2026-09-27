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

#ifndef BACK_ODOM__LIDAR_IMU_MATCHER_HPP_
#define BACK_ODOM__LIDAR_IMU_MATCHER_HPP_

#include "back_odom/imu_processor.hpp"

#include <Eigen/Core>
#include <kiss_icp_cpp/core/Registration.hpp>
#include <kiss_icp_cpp/core/VoxelHashMap.hpp>
#include <sophus/se3.hpp>

#include <vector>

namespace back_odom
{

struct LidarScan
{
  double stamp{0.0};
  std::vector<Eigen::Vector3d> points;
  std::vector<double> timestamps;
};

struct LidarMatchParams
{
  double voxel_size{0.5};
  double crop_longitudinal{75.0};
  double crop_lateral{50.0};
  double max_correspondence_distance{2.0};
  double kernel_scale{0.5};
  double convergence_criterion{1.0e-4};
  int max_iterations{50};
  int max_points_per_voxel{20};
  int backward_match_stride{3};
};

struct MatchResult
{
  bool applied{false};
  bool inserted_scan{false};
  bool first_scan{false};
  Sophus::SE3d imu_pose{};
  Sophus::SE3d corrected_pose{};
  Sophus::SE3d correction{};
  double iterations{0.0};
};

class LidarImuMatcher
{
public:
  explicit LidarImuMatcher(const LidarMatchParams & params);

  [[nodiscard]] bool has_reference() const;

  /// Deskew, crop, and either store the first scan or match a new scan into the map.
  [[nodiscard]] MatchResult on_scan(const LidarScan & scan, ImuProcessor & imu);

  /// After `backward_match_stride` IMU samples, match the stored scan back onto the map.
  [[nodiscard]] MatchResult on_imu(ImuProcessor & imu);

private:
  void apply_correction(ImuProcessor & imu, const Sophus::SE3d & corrected_pose, double stamp);
  void cull_map(const Sophus::SE3d & lidar_pose);
  [[nodiscard]] kiss_icp::PlaneAlignResult align(
    const std::vector<Eigen::Vector3d> & scan, const Sophus::SE3d & guess, double & iterations);

  LidarMatchParams params_;
  double half_longitudinal_{37.5};
  double half_lateral_{25.0};
  kiss_icp::VoxelHashMap map_;
  kiss_icp::Registration registration_;
  std::vector<Eigen::Vector3d> stored_scan_;
  bool has_reference_{false};
  int imu_steps_{0};
  double last_correction_stamp_{0.0};
  Eigen::Vector3d last_correction_position_{Eigen::Vector3d::Zero()};
};

}  // namespace back_odom

#endif  // BACK_ODOM__LIDAR_IMU_MATCHER_HPP_
