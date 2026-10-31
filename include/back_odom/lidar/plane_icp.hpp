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
  /// Soft penalty on rear-axle longitudinal slip away from the seed. 0 leaves the solve unchanged.
  double longitudinal_prior_weight{0.0};
  /// Rear axle in the body frame. z is negative when the axle is below the sensor.
  double rear_axle_z{-1.8};
  double rear_axle_y{0.0};
};

/// Add w * j j^T to H and w (j·ξ) j to J^T r, with j the longitudinal motion of the rear axle.
/// The eigen solver steps dx = -H^{-1} g, so a positive axle slip pulls the iterate back to the seed.
inline void add_rear_axle_longitudinal_prior(
  Eigen::Matrix<double, 6, 6> & hessian, Eigen::Matrix<double, 6, 1> & gradient,
  const Sophus::SE3d & pose, const Sophus::SE3d & seed, const double weight, const double axle_z,
  const double axle_y)
{
  if (!(weight > 0.0)) {
    return;
  }
  const Eigen::Matrix<double, 6, 1> xi = (seed.inverse() * pose).log();
  Eigen::Matrix<double, 6, 1> axle_jacobian;
  axle_jacobian << 1.0, 0.0, 0.0, 0.0, axle_z, -axle_y;
  hessian.noalias() += weight * axle_jacobian * axle_jacobian.transpose();
  gradient.noalias() += (weight * axle_jacobian.dot(xi)) * axle_jacobian;
}

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
};

/// Point-to-plane ICP. Closest point from `map`, plane normal from the `surface` cache of the
/// voxel that holds the closest point. Right-multiplied body increments.
[[nodiscard]] PlaneIcpResult align_to_surface(
  const std::vector<Eigen::Vector3d> & source, const kiss_icp::VoxelHashMap & map,
  const SurfaceVoxelMap & surface, const Sophus::SE3d & initial_guess, const PlaneIcpParams & params,
  const std::chrono::steady_clock::time_point * deadline = nullptr);

/// Mean robust plane cost at `pose` with the same association as align_to_surface.
[[nodiscard]] double surface_plane_cost(
  const std::vector<Eigen::Vector3d> & source, const kiss_icp::VoxelHashMap & map,
  const SurfaceVoxelMap & surface, const Sophus::SE3d & pose, double max_correspondence_distance,
  double kernel_scale);

}  // namespace back_odom

#endif  // BACK_ODOM__PLANE_ICP_HPP_
