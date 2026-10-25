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

#ifndef BACK_ODOM__PLANE_CLASS_HPP_
#define BACK_ODOM__PLANE_CLASS_HPP_

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>

namespace back_odom
{

/// |n·z| gates. cos(15°) is ground, sin(15°) is a vertical wall.
inline constexpr double kGroundNormalZ = 0.9659258262890683;
inline constexpr double kWallNormalZ = 0.25881904510252074;

enum class PlaneClass { Other = 0, Wall = 1, Ground = 2 };

[[nodiscard]] inline PlaneClass classify_plane_normal(const double normal_z)
{
  const double abs_z = std::abs(normal_z);
  if (abs_z >= kGroundNormalZ) {
    return PlaneClass::Ground;
  }
  if (abs_z <= kWallNormalZ) {
    return PlaneClass::Wall;
  }
  return PlaneClass::Other;
}

/// Scale that gives the ground class the same total robust weight as the walls.
/// Returns 1 when either class is missing. Clamped so a handful of ground points
/// cannot drown the walls.
[[nodiscard]] inline double ground_balance_scale(
  const double ground_weight, const double wall_weight)
{
  if (!(ground_weight > 0.0) || !(wall_weight > 0.0)) {
    return 1.0;
  }
  const double scale = wall_weight / ground_weight;
  if (scale < 0.25) {
    return 0.25;
  }
  if (scale > 20.0) {
    return 20.0;
  }
  return scale;
}

/// Yaw rate, rear-axle lever, and the body twist already taken since the ICP seed.
/// A non-positive `longitudinal_sigma` leaves the along-track direction unconstrained.
struct CurveConstraint
{
  double yaw_rate{0.0};
  Eigen::Vector3d axle_in_body{Eigen::Vector3d::Zero()};
  double longitudinal_sigma{0.0};
  Eigen::Matrix<double, 6, 1> xi_from_seed{Eigen::Matrix<double, 6, 1>::Zero()};
};

/// Ground keeps [z, roll, pitch] as the turn rate grows. Walls keep [x, y, yaw].
/// The along-track residual is a soft constraint at the rear axle, in the body tangent
/// [dx, dy, dz, droll, dpitch, dyaw].
inline void compose_class_system(
  const Eigen::Matrix<double, 6, 6> & ground, const Eigen::Matrix<double, 6, 6> & wall,
  const Eigen::Matrix<double, 6, 6> & other, const Eigen::Matrix<double, 6, 1> & ground_gradient,
  const Eigen::Matrix<double, 6, 1> & wall_gradient, const Eigen::Matrix<double, 6, 1> & other_gradient,
  const CurveConstraint & constraint, Eigen::Matrix<double, 6, 6> & hessian,
  Eigen::Matrix<double, 6, 1> & gradient)
{
  const double turn_alpha = std::clamp(std::abs(constraint.yaw_rate) / 0.10, 0.2, 1.0);
  Eigen::Matrix<double, 6, 6> tilt_mask = Eigen::Matrix<double, 6, 6>::Zero();
  tilt_mask(2, 2) = 1.0;
  tilt_mask(3, 3) = 1.0;
  tilt_mask(4, 4) = 1.0;
  const Eigen::Matrix<double, 6, 6> ground_effective =
    (1.0 - turn_alpha) * ground + turn_alpha * (tilt_mask * ground * tilt_mask);
  const Eigen::Matrix<double, 6, 1> ground_gradient_effective =
    (1.0 - turn_alpha) * ground_gradient + turn_alpha * (tilt_mask * ground_gradient);

  const double wall_planar = wall(0, 0) + wall(1, 1) + wall(5, 5);
  const double ground_tilt =
    ground_effective(2, 2) + ground_effective(3, 3) + ground_effective(4, 4);
  double ground_scale = 1.0;
  if (ground_tilt > 1.0e-6 && wall_planar > 1.0e-6) {
    ground_scale = std::clamp(wall_planar / ground_tilt, 0.25, 20.0);
  }

  hessian = other + wall + ground_scale * ground_effective;
  gradient = other_gradient + wall_gradient + ground_scale * ground_gradient_effective;
  if (!(constraint.longitudinal_sigma > 0.0)) {
    return;
  }
  Eigen::Matrix<double, 6, 1> axle_row;
  axle_row << 1.0, 0.0, 0.0, 0.0, constraint.axle_in_body.z(), -constraint.axle_in_body.y();
  const double weight =
    std::max(1.0, 0.05 * (wall(0, 0) + wall(1, 1))) /
    (constraint.longitudinal_sigma * constraint.longitudinal_sigma);
  hessian.noalias() += weight * (axle_row * axle_row.transpose());
  gradient.noalias() += weight * axle_row * axle_row.dot(constraint.xi_from_seed);
}

/// Correlated [x, y, yaw] covariance in the body frame, from the wall block of the normal
/// equations. A one-sided wall makes one planar direction weak; the inverse of that eigenvalue
/// loosens x, y, and yaw together instead of inflating yaw alone.
[[nodiscard]] inline Eigen::Matrix3d adaptive_planar_covariance(
  const Eigen::Matrix<double, 6, 6> & wall_hessian, const double robust_rms,
  const double wall_inlier_fraction, const double yaw_rate, const double lever_arm = 15.0,
  const double min_sigma_position = 0.03, const double max_sigma_position = 0.80,
  const double min_sigma_yaw = 0.015, const double max_sigma_yaw = 0.12)
{
  Eigen::Matrix3d planar = Eigen::Matrix3d::Zero();
  const int index[3] = {0, 1, 5};
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      planar(row, column) = wall_hessian(index[row], index[column]);
    }
  }
  const double lever = std::max(lever_arm, 1.0);
  const Eigen::DiagonalMatrix<double, 3> metric_scale(1.0, 1.0, 1.0 / lever);
  const Eigen::Matrix3d metric = metric_scale * planar * metric_scale;
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(metric);
  const double min_variance = min_sigma_position * min_sigma_position;
  const double max_variance = max_sigma_position * max_sigma_position;
  if (solver.info() != Eigen::Success) {
    return Eigen::Vector3d(max_variance, max_variance, max_sigma_yaw * max_sigma_yaw).asDiagonal();
  }

  const double residual_variance = std::max(0.0025, robust_rms * robust_rms);
  const double inlier_penalty = 1.0 / std::clamp(wall_inlier_fraction, 0.15, 1.0);
  const double turn_inflation =
    1.0 + 4.0 * std::clamp(std::abs(yaw_rate) / 0.30, 0.0, 2.0) *
            (1.0 - std::clamp(wall_inlier_fraction, 0.0, 1.0));
  Eigen::Vector3d inverse_eigenvalues;
  const Eigen::Vector3d eigenvalues = solver.eigenvalues();
  for (int axis = 0; axis < 3; ++axis) {
    const double raw = eigenvalues(axis) > 1.0e-6
                         ? residual_variance * inlier_penalty * turn_inflation / eigenvalues(axis)
                         : max_variance;
    inverse_eigenvalues(axis) = std::clamp(raw, min_variance, max_variance);
  }
  const Eigen::Matrix3d metric_covariance =
    solver.eigenvectors() * inverse_eigenvalues.asDiagonal() * solver.eigenvectors().transpose();
  Eigen::Matrix3d covariance = metric_scale * metric_covariance * metric_scale;

  const double yaw_min = min_sigma_yaw * min_sigma_yaw;
  const double yaw_max = max_sigma_yaw * max_sigma_yaw;
  if (covariance(2, 2) < yaw_min || covariance(2, 2) > yaw_max) {
    const double target = std::clamp(covariance(2, 2), yaw_min, yaw_max);
    const double scale = std::sqrt(target / std::max(1.0e-12, covariance(2, 2)));
    covariance.row(2) *= scale;
    covariance.col(2) *= scale;
  }
  return 0.5 * (covariance + covariance.transpose());
}

/// Map a body [x, y, yaw] covariance into the ROS pose covariance at heading `yaw`.
/// z, roll, and pitch stay on their own diagonal so a drifting lidar altitude cannot pull gravity.
[[nodiscard]] inline std::array<double, 36> pose_covariance_with_planar(
  const Eigen::Matrix3d & planar_body, const double yaw, const double variance_z,
  const double variance_roll_pitch)
{
  const double cosine = std::cos(yaw);
  const double sine = std::sin(yaw);
  Eigen::Matrix3d heading = Eigen::Matrix3d::Identity();
  heading(0, 0) = cosine;
  heading(0, 1) = -sine;
  heading(1, 0) = sine;
  heading(1, 1) = cosine;
  const Eigen::Matrix3d world = heading * planar_body * heading.transpose();
  std::array<double, 36> covariance{};
  const auto at = [&covariance](const int row, const int column) -> double & {
    return covariance[static_cast<std::size_t>(row * 6 + column)];
  };
  at(0, 0) = world(0, 0);
  at(0, 1) = world(0, 1);
  at(1, 0) = world(1, 0);
  at(1, 1) = world(1, 1);
  at(0, 5) = world(0, 2);
  at(5, 0) = world(2, 0);
  at(1, 5) = world(1, 2);
  at(5, 1) = world(2, 1);
  at(5, 5) = world(2, 2);
  at(2, 2) = variance_z;
  at(3, 3) = variance_roll_pitch;
  at(4, 4) = variance_roll_pitch;
  return covariance;
}

}  // namespace back_odom

#endif  // BACK_ODOM__PLANE_CLASS_HPP_
