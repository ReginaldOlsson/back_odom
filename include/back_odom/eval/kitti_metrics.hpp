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

#ifndef BACK_ODOM__KITTI_METRICS_HPP_
#define BACK_ODOM__KITTI_METRICS_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace back_odom
{

struct KittiSequenceError
{
  double translation_percent{std::numeric_limits<double>::quiet_NaN()};
  double rotation_deg_per_100m{std::numeric_limits<double>::quiet_NaN()};
  int num_segments{0};
};

struct BodyPositionError
{
  double longitudinal_m{0.0};
  double lateral_m{0.0};
};

/// SE(3) that maps the estimate pose onto the ground-truth pose.
inline Eigen::Matrix4d first_pose_alignment(
  const Eigen::Matrix4d & ground_truth, const Eigen::Matrix4d & estimate)
{
  return ground_truth * estimate.inverse();
}

/// Position error of an aligned estimate, expressed in the ground-truth body frame.
inline BodyPositionError body_frame_position_error(
  const Eigen::Matrix4d & ground_truth, const Eigen::Matrix4d & aligned_estimate)
{
  const Eigen::Vector3d delta =
    aligned_estimate.block<3, 1>(0, 3) - ground_truth.block<3, 1>(0, 3);
  const Eigen::Vector3d body = ground_truth.block<3, 3>(0, 0).transpose() * delta;
  return BodyPositionError{body.x(), body.y()};
}

/// KITTI odometry sequence error on equal-length pose lists.
/// Segments are 100 m to 800 m, started every 10 poses. Translation is the mean
/// relative position error in percent. Rotation is the mean SO(3) angle in deg/100 m
/// (the KITTI devkit prints the same average in deg/m).
inline KittiSequenceError evaluate_kitti_sequence(
  const std::vector<Eigen::Matrix4d> & poses_gt, const std::vector<Eigen::Matrix4d> & poses_result)
{
  constexpr double kPi = 3.14159265358979323846;
  constexpr double kLengthsM[] = {100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0};
  constexpr int kStep = 10;

  KittiSequenceError result;
  if (poses_gt.size() != poses_result.size() || poses_gt.size() < 2) {
    return result;
  }

  std::vector<double> distance(poses_gt.size(), 0.0);
  for (std::size_t index = 1; index < poses_gt.size(); ++index) {
    const Eigen::Vector3d step =
      poses_gt[index].block<3, 1>(0, 3) - poses_gt[index - 1].block<3, 1>(0, 3);
    distance[index] = distance[index - 1] + step.norm();
  }

  double translation_sum = 0.0;
  double rotation_sum = 0.0;
  int count = 0;
  for (std::size_t first = 0; first < poses_gt.size(); first += static_cast<std::size_t>(kStep)) {
    for (const double length_m : kLengthsM) {
      int last = -1;
      for (std::size_t index = first; index < distance.size(); ++index) {
        if (distance[index] > distance[first] + length_m) {
          last = static_cast<int>(index);
          break;
        }
      }
      if (last < 0) {
        continue;
      }
      const Eigen::Matrix4d ground_truth_delta =
        poses_gt[first].inverse() * poses_gt[static_cast<std::size_t>(last)];
      const Eigen::Matrix4d estimate_delta =
        poses_result[first].inverse() * poses_result[static_cast<std::size_t>(last)];
      const Eigen::Matrix4d error = estimate_delta.inverse() * ground_truth_delta;
      const double trace = error(0, 0) + error(1, 1) + error(2, 2);
      const double cosine = std::clamp(0.5 * (trace - 1.0), -1.0, 1.0);
      translation_sum += error.block<3, 1>(0, 3).norm() / length_m;
      rotation_sum += std::acos(cosine) / length_m;
      ++count;
    }
  }
  if (count == 0) {
    return result;
  }
  result.num_segments = count;
  result.translation_percent = 100.0 * (translation_sum / static_cast<double>(count));
  result.rotation_deg_per_100m =
    (rotation_sum / static_cast<double>(count)) * (180.0 / kPi) * 100.0;
  return result;
}

}  // namespace back_odom

#endif  // BACK_ODOM__KITTI_METRICS_HPP_
