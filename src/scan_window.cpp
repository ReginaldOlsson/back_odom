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

#include "back_odom/scan_window.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace back_odom
{
namespace
{

constexpr int k_neighbor_count = 20;
constexpr int k_min_plane_neighbors = 5;
constexpr double k_imu_translation_sigma = 0.05;
constexpr double k_imu_rotation_sigma = 0.02;
constexpr double k_max_step_translation = 0.5;
constexpr double k_max_step_rotation = 0.2;
constexpr double k_hessian_damping = 1.0e-3;

bool estimate_plane(const std::vector<Eigen::Vector3d> & neighbors, Eigen::Vector3d & normal)
{
  if (static_cast<int>(neighbors.size()) < k_min_plane_neighbors) {
    return false;
  }
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const Eigen::Vector3d & point : neighbors) {
    mean += point;
  }
  mean /= static_cast<double>(neighbors.size());

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const Eigen::Vector3d & point : neighbors) {
    const Eigen::Vector3d delta = point - mean;
    covariance += delta * delta.transpose();
  }
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  if (solver.info() != Eigen::Success) {
    return false;
  }
  const double lambda_min = solver.eigenvalues()(0);
  const double lambda_max = solver.eigenvalues()(2);
  if (lambda_max <= 1.0e-8 || lambda_min / lambda_max > 0.25) {
    return false;
  }
  normal = solver.eigenvectors().col(0);
  if (normal.norm() < 1.0e-8) {
    return false;
  }
  normal.normalize();
  return true;
}

void accumulate_scan(
  Eigen::MatrixXd & hessian, Eigen::VectorXd & gradient, const int pose_index,
  const std::vector<Eigen::Vector3d> & points, const Sophus::SE3d & pose,
  const kiss_icp::VoxelHashMap & frozen_map, const kiss_icp::VoxelHashMap & other_scans,
  const double max_distance, const double kernel_scale)
{
  const int offset = 6 * pose_index;
  const double kernel2 = kernel_scale * kernel_scale;
  for (const Eigen::Vector3d & point_body : points) {
    const Eigen::Vector3d point_world = pose * point_body;
    std::vector<Eigen::Vector3d> neighbors;
    if (!frozen_map.Empty()) {
      frozen_map.CollectNeighbors(point_world, k_neighbor_count, neighbors);
    }
    if (!other_scans.Empty()) {
      std::vector<Eigen::Vector3d> extra;
      other_scans.CollectNeighbors(point_world, k_neighbor_count, extra);
      neighbors.insert(neighbors.end(), extra.begin(), extra.end());
    }
    if (neighbors.empty()) {
      continue;
    }
    const auto closest = std::min_element(
      neighbors.cbegin(), neighbors.cend(),
      [&](const Eigen::Vector3d & lhs, const Eigen::Vector3d & rhs) {
        return (lhs - point_world).squaredNorm() < (rhs - point_world).squaredNorm();
      });
    if ((*closest - point_world).norm() >= max_distance) {
      continue;
    }
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    if (!estimate_plane(neighbors, normal)) {
      continue;
    }
    const Eigen::Vector3d normal_body = pose.so3().inverse() * normal;
    const double residual = normal.dot(point_world - *closest);
    Eigen::Matrix<double, 1, 6> jacobian;
    jacobian.block<1, 3>(0, 0) = normal_body.transpose();
    jacobian.block<1, 3>(0, 3) = point_body.cross(normal_body).transpose();
    const double denom = kernel_scale + residual * residual;
    const double weight = kernel2 / (denom * denom);
    hessian.block<6, 6>(offset, offset) += jacobian.transpose() * weight * jacobian;
    gradient.segment<6>(offset) += jacobian.transpose() * weight * residual;
  }
}

Eigen::Matrix<double, 6, 6> imu_weight()
{
  Eigen::Matrix<double, 6, 6> weight = Eigen::Matrix<double, 6, 6>::Zero();
  const double translation = 1.0 / (k_imu_translation_sigma * k_imu_translation_sigma);
  const double rotation = 1.0 / (k_imu_rotation_sigma * k_imu_rotation_sigma);
  weight.diagonal().head<3>().setConstant(translation);
  weight.diagonal().tail<3>().setConstant(rotation);
  return weight;
}

void accumulate_imu_edge(
  Eigen::MatrixXd & hessian, Eigen::VectorXd & gradient, const int from_index, const int to_index,
  const Sophus::SE3d & pose_from, const Sophus::SE3d & pose_to, const Sophus::SE3d & relative)
{
  const auto residual_of = [&](const Sophus::SE3d & from, const Sophus::SE3d & to) {
    return (relative.inverse() * from.inverse() * to).log();
  };
  const Eigen::Matrix<double, 6, 1> residual = residual_of(pose_from, pose_to);
  constexpr double k_epsilon = 1.0e-6;
  Eigen::Matrix<double, 6, 6> jacobian_from;
  Eigen::Matrix<double, 6, 6> jacobian_to;
  for (int axis = 0; axis < 6; ++axis) {
    Eigen::Matrix<double, 6, 1> step = Eigen::Matrix<double, 6, 1>::Zero();
    step(axis) = k_epsilon;
    const Sophus::SE3d delta = Sophus::SE3d::exp(step);
    jacobian_from.col(axis) = (residual_of(pose_from * delta, pose_to) - residual) / k_epsilon;
    jacobian_to.col(axis) = (residual_of(pose_from, pose_to * delta) - residual) / k_epsilon;
  }

  const Eigen::Matrix<double, 6, 6> weight = imu_weight();
  const int from_offset = 6 * from_index;
  const int to_offset = 6 * to_index;
  const Eigen::Matrix<double, 6, 6> from_term = jacobian_from.transpose() * weight;
  const Eigen::Matrix<double, 6, 6> to_term = jacobian_to.transpose() * weight;
  hessian.block<6, 6>(from_offset, from_offset) += from_term * jacobian_from;
  hessian.block<6, 6>(to_offset, to_offset) += to_term * jacobian_to;
  hessian.block<6, 6>(from_offset, to_offset) += from_term * jacobian_to;
  hessian.block<6, 6>(to_offset, from_offset) += to_term * jacobian_from;
  gradient.segment<6>(from_offset) += from_term * residual;
  gradient.segment<6>(to_offset) += to_term * residual;
}

void accumulate_newest_prior(
  Eigen::MatrixXd & hessian, Eigen::VectorXd & gradient, const Sophus::SE3d & original,
  const Sophus::SE3d & current, const double translation_sigma, const double rotation_sigma)
{
  const int offset = static_cast<int>(hessian.rows()) - 6;
  const Eigen::Matrix<double, 6, 1> residual = (original.inverse() * current).log();
  Eigen::Matrix<double, 6, 6> weight = Eigen::Matrix<double, 6, 6>::Zero();
  weight.diagonal().head<3>().setConstant(1.0 / (translation_sigma * translation_sigma));
  weight.diagonal().tail<3>().setConstant(1.0 / (rotation_sigma * rotation_sigma));
  hessian.block<6, 6>(offset, offset) += weight;
  gradient.segment<6>(offset) += weight * residual;
}

void clamp_step(Eigen::Matrix<double, 6, 1> & step)
{
  const double translation = step.head<3>().norm();
  if (translation > k_max_step_translation) {
    step.head<3>() *= k_max_step_translation / translation;
  }
  const double rotation = step.tail<3>().norm();
  if (rotation > k_max_step_rotation) {
    step.tail<3>() *= k_max_step_rotation / rotation;
  }
}

}  // namespace

double robust_plane_cost(
  const std::vector<Eigen::Vector3d> & points_body, const Sophus::SE3d & pose,
  const kiss_icp::VoxelHashMap & map, const double max_correspondence_distance,
  const double kernel_scale)
{
  if (points_body.empty() || kernel_scale <= 0.0 || max_correspondence_distance <= 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  const double kernel2 = kernel_scale * kernel_scale;
  double cost = 0.0;
  for (const Eigen::Vector3d & point_body : points_body) {
    const Eigen::Vector3d point_world = pose * point_body;
    const std::vector<Eigen::Vector3d> neighbors = map.Neighbors(point_world, k_neighbor_count);
    double residual = max_correspondence_distance;
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    if (estimate_plane(neighbors, normal)) {
      Eigen::Vector3d closest = neighbors.front();
      double best = (closest - point_world).squaredNorm();
      for (const Eigen::Vector3d & neighbor : neighbors) {
        const double distance = (neighbor - point_world).squaredNorm();
        if (distance < best) {
          closest = neighbor;
          best = distance;
        }
      }
      if (std::sqrt(best) <= max_correspondence_distance) {
        residual = normal.dot(point_world - closest);
      }
    }
    cost += kernel2 * residual * residual / (kernel_scale + residual * residual);
  }
  return cost / static_cast<double>(points_body.size());
}

void optimize_scan_window(
  std::vector<Sophus::SE3d> & poses, const std::vector<std::vector<Eigen::Vector3d>> & query_points,
  const std::vector<std::vector<Eigen::Vector3d>> & surface_points,
  const std::vector<Sophus::SE3d> & imu_relatives, const kiss_icp::VoxelHashMap & frozen_map,
  const double max_correspondence_distance, const double kernel_scale, const int max_iterations,
  const double convergence_criterion, const double newest_translation_sigma,
  const double newest_rotation_sigma, const double voxel_size,
  const unsigned int max_points_per_voxel)
{
  const std::size_t count = poses.size();
  if (count < 1 || count != query_points.size() || count != surface_points.size()) {
    return;
  }
  if (imu_relatives.size() + 1 != count || kernel_scale <= 0.0 || max_iterations < 1) {
    return;
  }
  const Sophus::SE3d newest_original = poses.back();
  const int dof = 6 * static_cast<int>(count);

  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(dof, dof);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(dof);
    for (std::size_t index = 0; index < count; ++index) {
      kiss_icp::VoxelHashMap others(voxel_size, 1.0e6, max_points_per_voxel);
      for (std::size_t other = 0; other < count; ++other) {
        if (other == index) {
          continue;
        }
        others.Update(surface_points[other], poses[other]);
      }
      accumulate_scan(
        hessian, gradient, static_cast<int>(index), query_points[index], poses[index], frozen_map,
        others, max_correspondence_distance, kernel_scale);
    }
    for (std::size_t index = 1; index < count; ++index) {
      accumulate_imu_edge(
        hessian, gradient, static_cast<int>(index - 1), static_cast<int>(index),
        poses[index - 1], poses[index], imu_relatives[index - 1]);
    }
    if (newest_translation_sigma > 0.0 && newest_rotation_sigma > 0.0) {
      accumulate_newest_prior(
        hessian, gradient, newest_original, poses.back(), newest_translation_sigma,
        newest_rotation_sigma);
    }
    hessian.diagonal().array() += k_hessian_damping;
    const Eigen::LDLT<Eigen::MatrixXd> solver(hessian);
    if (solver.info() != Eigen::Success) {
      break;
    }
    const Eigen::VectorXd step = -solver.solve(gradient);
    if (!step.allFinite()) {
      break;
    }
    double largest = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
      Eigen::Matrix<double, 6, 1> pose_step = step.segment<6>(6 * static_cast<int>(index));
      largest = std::max(largest, pose_step.norm());
      clamp_step(pose_step);
      poses[index] = poses[index] * Sophus::SE3d::exp(pose_step);
    }
    if (largest < convergence_criterion) {
      break;
    }
  }
}

}  // namespace back_odom
