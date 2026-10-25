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

#ifndef BACK_ODOM__PLANE_ICP_HPP_
#define BACK_ODOM__PLANE_ICP_HPP_

#include "back_odom/lidar/ndt_health.hpp"

#include <Eigen/Core>
#include <kiss_icp_cpp/core/VoxelHashMap.hpp>
#include <sophus/se3.hpp>

#include <chrono>
#include <vector>

namespace back_odom
{

struct PlaneIcpParams
{
  double max_correspondence_distance{2.0};
  /// First `coarse_iterations` use this neighbourhood so a slightly-off seed can still see the
  /// right plane. 0 disables the coarse stage (every iteration uses `max_correspondence_distance`).
  double coarse_correspondence_distance{0.0};
  int coarse_iterations{0};
  double kernel_scale{0.5};
  int max_iterations{20};
  double convergence_criterion{1.0e-3};
  /// |yaw rate| above about 0.1 rad/s gives the ground block only z, roll, and pitch.
  double yaw_rate{0.0};
  /// Rear axle in the body frame. Used with `longitudinal_sigma` as a soft along-track prior.
  Eigen::Vector3d axle_in_body{Eigen::Vector3d::Zero()};
  /// Non-positive disables the rear-axle prior and leaves the post-solve clamp in charge.
  double longitudinal_sigma{0.0};
};

[[nodiscard]] inline double correspondence_distance_for_iteration(
  const PlaneIcpParams & params, const int iteration)
{
  if (
    iteration >= 0 && iteration < params.coarse_iterations &&
    params.coarse_correspondence_distance > 0.0) {
    return params.coarse_correspondence_distance;
  }
  return params.max_correspondence_distance;
}

/// Coarse iterations accept smeared walls, then the kernel tightens after each residual deskew.
[[nodiscard]] inline double kernel_scale_for_iteration(
  const PlaneIcpParams & params, const int iteration, const bool residual_deskew)
{
  if (!residual_deskew) {
    return params.kernel_scale;
  }
  if (iteration < 2) {
    return params.kernel_scale < 1.5 ? 1.5 : params.kernel_scale;
  }
  if (iteration < 4) {
    return params.kernel_scale < 0.8 ? 0.8 : params.kernel_scale;
  }
  return params.kernel_scale;
}

struct PlaneIcpResult
{
  Sophus::SE3d pose{};
  /// pose = correction * initial_guess
  Sophus::SE3d correction{};
  double iterations{0.0};
  /// Correspondences in the last linear system.
  int correspondences{0};
  /// Norm of the last body increment (0 when no step was taken).
  double last_step{0.0};
  /// Mean Geman-McClure cost over all source points at the pose the last system was built at.
  double robust_cost{0.0};
  /// Number of 6-DoF directions the solver dropped as unobservable in the last step.
  int degenerate_axes{0};
  /// Deadline reached before convergence. `pose` is the partial solve.
  bool timed_out{false};
  /// Iteration cap reached without meeting the convergence criterion.
  bool saturated{false};
  /// Final 6x6 normal matrix (body frame), for downstream covariance / observability use.
  Eigen::Matrix<double, 6, 6> hessian{Eigen::Matrix<double, 6, 6>::Zero()};
  /// Wall block before the ground mask and the axle prior. Feeds the planar pose covariance.
  Eigen::Matrix<double, 6, 6> wall_hessian{Eigen::Matrix<double, 6, 6>::Zero()};
  /// Weighted RMS of the point-to-plane residual at the last system, in metres.
  double robust_rms{0.08};
  /// Wall correspondences divided by all plane correspondences.
  double wall_inlier_fraction{0.0};
};

/// Point-to-plane ICP. Closest point from `map`, plane normal from the `surface` cache of the
/// voxel that holds the closest point. Right-multiplied body increments.
[[nodiscard]] PlaneIcpResult align_to_surface(
  const std::vector<Eigen::Vector3d> & source, const kiss_icp::VoxelHashMap & map,
  const SurfaceVoxelMap & surface, const Sophus::SE3d & initial_guess, const PlaneIcpParams & params,
  const std::chrono::steady_clock::time_point * deadline = nullptr,
  const std::vector<float> * tau = nullptr);

/// Mean robust plane cost at `pose` with the same association as align_to_surface.
[[nodiscard]] double surface_plane_cost(
  const std::vector<Eigen::Vector3d> & source, const kiss_icp::VoxelHashMap & map,
  const SurfaceVoxelMap & surface, const Sophus::SE3d & pose, double max_correspondence_distance,
  double kernel_scale);

}  // namespace back_odom

#endif  // BACK_ODOM__PLANE_ICP_HPP_
