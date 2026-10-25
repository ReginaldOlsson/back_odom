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

#include <cstdint>
#include <random>
#include <vector>

namespace back_odom
{

[[nodiscard]] std::vector<Eigen::Vector3d> crop_lidar_box(
  const std::vector<Eigen::Vector3d> & points, double half_longitudinal, double half_lateral);

struct IntensityCloud
{
  std::vector<Eigen::Vector3d> points;
  std::vector<float> intensities;
};

/// Same box crop as `crop_lidar_box`, keeping each point's intensity.
/// When `tau` matches `points`, `cropped_tau` receives the kept normalized timestamps.
[[nodiscard]] IntensityCloud crop_lidar_box(
  const std::vector<Eigen::Vector3d> & points, const std::vector<float> & intensities,
  double half_longitudinal, double half_lateral, const std::vector<float> * tau = nullptr,
  std::vector<float> * cropped_tau = nullptr);

/// First-point voxel downsample matching kiss_icp::VoxelDownsample, with intensities.
/// When `tau` matches `points`, `tau_out` receives the timestamp of each kept point.
[[nodiscard]] IntensityCloud voxel_downsample_with_intensity(
  const std::vector<Eigen::Vector3d> & points, const std::vector<float> & intensities,
  double voxel_size, const std::vector<float> * tau = nullptr,
  std::vector<float> * tau_out = nullptr);

/// GLIM-style random grid. Each voxel keeps about the same number of original points
/// (`sampling_rate` of the cloud, spread across occupied voxels). Points are sampled, not averaged.
[[nodiscard]] IntensityCloud randomgrid_downsample_with_intensity(
  const std::vector<Eigen::Vector3d> & points, const std::vector<float> & intensities,
  double voxel_size, double sampling_rate, std::mt19937 & rng);

/// Intensity of the first cropped point that occupies each downsampled point's voxel.
[[nodiscard]] std::vector<float> intensities_for_downsampled(
  const std::vector<Eigen::Vector3d> & cropped, const std::vector<float> & cropped_intensities,
  const std::vector<Eigen::Vector3d> & downsampled, double voxel_size);

/// The slice of `trajectory` that brackets [from_stamp, to_stamp], with one pose of margin on
/// each side. Deskew only needs these; building motion segments for a 30 s buffer per scan is
/// wasted work.
[[nodiscard]] std::vector<StampedPose> trajectory_window(
  const std::vector<StampedPose> & trajectory, double from_stamp, double to_stamp);

struct GroundPlaneFit
{
  bool valid{false};
  /// Unit normal in the cloud frame, with a positive z component.
  Eigen::Vector3d normal{Eigen::Vector3d::UnitZ()};
  double inlier_ratio{0.0};
};

/// RANSAC ground plane. Candidates whose normal is more than 30° from +Z are rejected.
[[nodiscard]] GroundPlaneFit fit_ground_plane(
  const std::vector<Eigen::Vector3d> & points, double distance_threshold, int iterations,
  std::uint32_t seed);

/// True when `samples` is time-ordered and covers [t0, t1] with no hole longer than `max_gap`.
[[nodiscard]] bool pose_buffer_covers(
  const std::vector<StampedPose> & samples, double t0, double t1, double max_gap);

/// SE(3) interpolation of a stamped pose buffer. Clamps outside the buffer.
[[nodiscard]] Sophus::SE3d interpolate_stamped_pose(
  const std::vector<StampedPose> & trajectory, double stamp);

/// Put each lidar point in the integrator frame with `body_from_lidar`, then move it from its
/// timestamp to the scan-end pose. The trajectory and the extrinsic must share that frame.
/// `tau`, when set, receives each point's normalized stamp in [0, 1] (1 at scan end).
[[nodiscard]] std::vector<Eigen::Vector3d> deskew_to_scan_end(
  const std::vector<Eigen::Vector3d> & points, const std::vector<double> & timestamps,
  const std::vector<StampedPose> & trajectory, const Sophus::SE3d & body_from_lidar = {},
  std::vector<float> * tau = nullptr);

/// Move seed-end points by the fractional ICP correction. `xi` is log(seed^{-1} * current),
/// and a point at normalized time `tau` is multiplied by exp((tau - 1) xi). 64 uniform buckets.
void apply_sweep_residual_deskew(
  const std::vector<Eigen::Vector3d> & seed_end, const std::vector<float> & tau,
  const Eigen::Matrix<double, 6, 1> & xi, std::vector<Eigen::Vector3d> & deskewed);

}  // namespace back_odom

#endif  // BACK_ODOM__LIDAR_PREPROCESS_HPP_
