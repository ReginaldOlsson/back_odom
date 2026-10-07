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

#include "back_odom/lidar/scan_window.hpp"

#include <kiss_icp_cpp/core/VoxelUtils.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
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

namespace
{

constexpr int k_plane_neighbors = 5;
constexpr double k_refine_step_translation = 0.05;
constexpr double k_refine_step_rotation = 0.02;
constexpr double k_refine_damping = 1.0e-3;

const std::array<kiss_icp::Voxel, 27> k_voxel_shifts = {
  kiss_icp::Voxel{0, 0, 0},   kiss_icp::Voxel{1, 0, 0},   kiss_icp::Voxel{-1, 0, 0},
  kiss_icp::Voxel{0, 1, 0},   kiss_icp::Voxel{0, -1, 0},  kiss_icp::Voxel{0, 0, 1},
  kiss_icp::Voxel{0, 0, -1},  kiss_icp::Voxel{1, 1, 0},   kiss_icp::Voxel{1, -1, 0},
  kiss_icp::Voxel{-1, 1, 0},  kiss_icp::Voxel{-1, -1, 0}, kiss_icp::Voxel{1, 0, 1},
  kiss_icp::Voxel{1, 0, -1},  kiss_icp::Voxel{-1, 0, 1},  kiss_icp::Voxel{-1, 0, -1},
  kiss_icp::Voxel{0, 1, 1},   kiss_icp::Voxel{0, 1, -1},  kiss_icp::Voxel{0, -1, 1},
  kiss_icp::Voxel{0, -1, -1}, kiss_icp::Voxel{1, 1, 1},   kiss_icp::Voxel{1, 1, -1},
  kiss_icp::Voxel{1, -1, 1},  kiss_icp::Voxel{1, -1, -1}, kiss_icp::Voxel{-1, 1, 1},
  kiss_icp::Voxel{-1, 1, -1}, kiss_icp::Voxel{-1, -1, 1}, kiss_icp::Voxel{-1, -1, -1}};

struct IndexedPoint
{
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  std::uint16_t scan{0};
};

struct VoxelEntry
{
  std::vector<IndexedPoint> points;
  Eigen::Vector3d mean{Eigen::Vector3d::Zero()};
  Eigen::Vector3d normal{Eigen::Vector3d::UnitZ()};
  Eigen::Matrix3d covariance{Eigen::Matrix3d::Zero()};
  Eigen::Matrix3d inv_covariance{Eigen::Matrix3d::Zero()};
  bool valid_distribution{false};
  bool planar{false};
  std::uint16_t dominant{0};
  int dominant_count{0};
};

struct IndexedCloud
{
  double voxel_size{0.5};
  unsigned int max_points{20};
  std::unordered_map<kiss_icp::Voxel, VoxelEntry> voxels;

  void add(const std::uint16_t scan, const Eigen::Vector3d & point)
  {
    const kiss_icp::Voxel voxel = kiss_icp::PointToVoxel(point, voxel_size);
    VoxelEntry & entry = voxels[voxel];
    if (entry.points.size() >= max_points) {
      return;
    }
    const double spacing = voxel_size / std::sqrt(static_cast<double>(std::max(1U, max_points)));
    for (const IndexedPoint & stored : entry.points) {
      if ((stored.position - point).norm() < spacing) {
        return;
      }
    }
    entry.points.push_back(IndexedPoint{point, scan});
  }

  void compute_planes()
  {
    constexpr double k_cov_regularization = 1.0e-3;
    for (auto & item : voxels) {
      VoxelEntry & entry = item.second;
      entry.planar = false;
      entry.valid_distribution = false;
      entry.covariance.setZero();
      entry.inv_covariance.setZero();
      if (static_cast<int>(entry.points.size()) < k_plane_neighbors) {
        continue;
      }
      entry.mean.setZero();
      for (const IndexedPoint & point : entry.points) {
        entry.mean += point.position;
      }
      entry.mean /= static_cast<double>(entry.points.size());
      Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
      for (const IndexedPoint & point : entry.points) {
        const Eigen::Vector3d delta = point.position - entry.mean;
        covariance += delta * delta.transpose();
      }
      // Eigenvectors are scale-invariant; keep the unnormalized matrix for the plane test.
      const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
      if (solver.info() != Eigen::Success) {
        continue;
      }
      const double lambda_min = solver.eigenvalues()(0);
      const double lambda_max = solver.eigenvalues()(2);
      if (lambda_max <= 1.0e-8 || lambda_min / lambda_max > 0.25) {
        continue;
      }
      entry.normal = solver.eigenvectors().col(0);
      if (entry.normal.norm() < 1.0e-8) {
        continue;
      }
      entry.normal.normalize();
      entry.covariance = covariance / static_cast<double>(entry.points.size());
      const Eigen::Matrix3d regularized =
        entry.covariance + Eigen::Matrix3d::Identity() * k_cov_regularization;
      Eigen::Matrix3d inverse = Eigen::Matrix3d::Zero();
      bool invertible = false;
      regularized.computeInverseWithCheck(inverse, invertible);
      if (invertible && inverse.allFinite()) {
        entry.inv_covariance = inverse;
        entry.valid_distribution = true;
      }
      entry.dominant = entry.points.front().scan;
      entry.dominant_count = 0;
      for (const IndexedPoint & point : entry.points) {
        int count = 0;
        for (const IndexedPoint & other : entry.points) {
          if (other.scan == point.scan) {
            ++count;
          }
        }
        if (count > entry.dominant_count) {
          entry.dominant_count = count;
          entry.dominant = point.scan;
        }
      }
      entry.planar = true;
    }
  }

  [[nodiscard]] bool closest_plane(
    const Eigen::Vector3d & query, const std::uint16_t exclude, const double max_distance,
    Eigen::Vector3d & mean, Eigen::Vector3d & normal) const
  {
    const kiss_icp::Voxel voxel = kiss_icp::PointToVoxel(query, voxel_size);
    double best = max_distance * max_distance;
    bool found = false;
    for (const kiss_icp::Voxel & shift : k_voxel_shifts) {
      const auto found_voxel = voxels.find(voxel + shift);
      if (found_voxel == voxels.end()) {
        continue;
      }
      const VoxelEntry & entry = found_voxel->second;
      if (!entry.planar) {
        continue;
      }
      if (entry.dominant == exclude && entry.dominant_count * 2 >= static_cast<int>(entry.points.size())) {
        continue;
      }
      const double distance = (entry.mean - query).squaredNorm();
      if (distance < best) {
        best = distance;
        mean = entry.mean;
        normal = entry.normal;
        found = true;
      }
    }
    return found;
  }
};

std::vector<Eigen::Vector3d> subsample_points(
  const std::vector<Eigen::Vector3d> & points, const std::size_t limit)
{
  if (points.size() <= limit) {
    return points;
  }
  std::vector<Eigen::Vector3d> kept;
  kept.reserve(limit);
  const double step = static_cast<double>(points.size()) / static_cast<double>(limit);
  for (std::size_t index = 0; index < limit; ++index) {
    kept.push_back(points[static_cast<std::size_t>(static_cast<double>(index) * step)]);
  }
  return kept;
}

void clamp_refine_step(Eigen::Matrix<double, 6, 1> & step)
{
  const double translation = step.head<3>().norm();
  if (translation > k_refine_step_translation) {
    step.head<3>() *= k_refine_step_translation / translation;
  }
  const double rotation = step.tail<3>().norm();
  if (rotation > k_refine_step_rotation) {
    step.tail<3>() *= k_refine_step_rotation / rotation;
  }
}

void clamp_pose_to_prior(
  Sophus::SE3d & pose, const Sophus::SE3d & prior, const double max_translation,
  const double max_rotation)
{
  Eigen::Matrix<double, 6, 1> delta = (prior.inverse() * pose).log();
  const double translation = delta.head<3>().norm();
  const double rotation = delta.tail<3>().norm();
  if (translation > max_translation && translation > 1.0e-12) {
    delta.head<3>() *= max_translation / translation;
  }
  if (rotation > max_rotation && rotation > 1.0e-12) {
    delta.tail<3>() *= max_rotation / rotation;
  }
  pose = prior * Sophus::SE3d::exp(delta);
}

Eigen::Matrix<double, 6, 6> diagonal_weight(const double translation_sigma, const double rotation_sigma)
{
  Eigen::Matrix<double, 6, 6> weight = Eigen::Matrix<double, 6, 6>::Zero();
  weight.diagonal().head<3>().setConstant(1.0 / (translation_sigma * translation_sigma));
  weight.diagonal().tail<3>().setConstant(1.0 / (rotation_sigma * rotation_sigma));
  return weight;
}

template <typename Residual>
void add_right_factor(
  Eigen::Matrix<double, 6, 6> & hessian, Eigen::Matrix<double, 6, 1> & gradient,
  const Sophus::SE3d & pose, const Eigen::Matrix<double, 6, 6> & weight, Residual && residual_of)
{
  const Eigen::Matrix<double, 6, 1> residual = residual_of(pose);
  if (!residual.allFinite()) {
    return;
  }
  constexpr double k_epsilon = 1.0e-6;
  Eigen::Matrix<double, 6, 6> jacobian;
  for (int axis = 0; axis < 6; ++axis) {
    Eigen::Matrix<double, 6, 1> step = Eigen::Matrix<double, 6, 1>::Zero();
    step(axis) = k_epsilon;
    jacobian.col(axis) = (residual_of(pose * Sophus::SE3d::exp(step)) - residual) / k_epsilon;
  }
  const Eigen::Matrix<double, 6, 6> term = jacobian.transpose() * weight;
  hessian += term * jacobian;
  gradient += term * residual;
}

Sophus::SE3d refine_one_pose(
  const std::uint16_t index, const Sophus::SE3d & start, const Sophus::SE3d & previous_pose,
  const Sophus::SE3d & next_pose, const bool has_previous, const bool has_next,
  const Sophus::SE3d & from_previous, const Sophus::SE3d & to_next, const Sophus::SE3d & prior,
  const Sophus::SE3d & clamp_pose, const std::vector<Eigen::Vector3d> & points,
  const IndexedCloud & cloud, const ScanHorizonRefine & params)
{
  Sophus::SE3d pose = start;
  const double kernel2 = params.kernel_scale * params.kernel_scale;
  const double translation_sigma =
    has_next ? params.prior_translation_sigma : params.current_prior_translation_sigma;
  const double rotation_sigma =
    has_next ? params.prior_rotation_sigma : params.current_prior_rotation_sigma;
  const Eigen::Matrix<double, 6, 6> prior_weight = diagonal_weight(
    translation_sigma > 0.0 ? translation_sigma : params.prior_translation_sigma,
    rotation_sigma > 0.0 ? rotation_sigma : params.prior_rotation_sigma);
  const Eigen::Matrix<double, 6, 6> imu_weight =
    diagonal_weight(params.imu_translation_sigma, params.imu_rotation_sigma);
  for (int iteration = 0; iteration < params.inner_iterations; ++iteration) {
    Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
    if (params.kernel_scale > 0.0) {
      for (const Eigen::Vector3d & point_body : points) {
        const Eigen::Vector3d point_world = pose * point_body;
        Eigen::Vector3d mean = Eigen::Vector3d::Zero();
        Eigen::Vector3d normal = Eigen::Vector3d::Zero();
        if (!cloud.closest_plane(point_world, index, params.max_correspondence_distance, mean, normal)) {
          continue;
        }
        const Eigen::Vector3d normal_body = pose.so3().inverse() * normal;
        const double residual = normal.dot(point_world - mean);
        Eigen::Matrix<double, 1, 6> jacobian;
        jacobian.block<1, 3>(0, 0) = normal_body.transpose();
        jacobian.block<1, 3>(0, 3) = point_body.cross(normal_body).transpose();
        const double denom = params.kernel_scale + residual * residual;
        const double weight = kernel2 / (denom * denom);
        hessian += jacobian.transpose() * weight * jacobian;
        gradient += jacobian.transpose() * weight * residual;
      }
    }
    add_right_factor(hessian, gradient, pose, prior_weight, [&](const Sophus::SE3d & value) {
      return (prior.inverse() * value).log();
    });
    if (has_previous) {
      add_right_factor(hessian, gradient, pose, imu_weight, [&](const Sophus::SE3d & value) {
        return (from_previous.inverse() * previous_pose.inverse() * value).log();
      });
    }
    if (has_next) {
      add_right_factor(hessian, gradient, pose, imu_weight, [&](const Sophus::SE3d & value) {
        return (to_next.inverse() * value.inverse() * next_pose).log();
      });
    }
    hessian.diagonal().array() += k_refine_damping;
    const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> solver(hessian);
    if (solver.info() != Eigen::Success) {
      break;
    }
    Eigen::Matrix<double, 6, 1> step = -solver.solve(gradient);
    if (!step.allFinite()) {
      break;
    }
    const double magnitude = step.norm();
    clamp_refine_step(step);
    pose = pose * Sophus::SE3d::exp(step);
    if (magnitude < 1.0e-4) {
      break;
    }
  }
  clamp_pose_to_prior(
    pose, clamp_pose, params.max_translation_from_prior, params.max_rotation_from_prior);
  return pose;
}

}  // namespace

void refine_scan_horizon(
  std::vector<Sophus::SE3d> & poses,
  const std::vector<const std::vector<Eigen::Vector3d> *> & points_body,
  const std::vector<Sophus::SE3d> & pose_priors, const std::vector<Sophus::SE3d> & imu_from_previous,
  const ScanHorizonRefine & params, const std::vector<std::uint8_t> & adjustable,
  const std::vector<Sophus::SE3d> & clamp_poses)
{
  const int count = static_cast<int>(poses.size());
  if (
    count < 2 || points_body.size() != poses.size() || pose_priors.size() != poses.size() ||
    imu_from_previous.size() != poses.size() || params.outer_iterations < 1 ||
    params.inner_iterations < 1 || params.voxel_size <= 0.0 || params.max_query_points < 1) {
    return;
  }
  const bool use_mask = static_cast<int>(adjustable.size()) == count;
  const bool use_clamp = static_cast<int>(clamp_poses.size()) == count;
  const Sophus::SE3d newest = poses.back();
  const std::vector<Sophus::SE3d> initial = poses;
  std::vector<std::vector<Eigen::Vector3d>> queries(poses.size());
  for (std::size_t index = 0; index < poses.size(); ++index) {
    if (points_body[index] == nullptr) {
      continue;
    }
    queries[index] = subsample_points(*points_body[index], params.max_query_points);
  }

  for (int outer = 0; outer < params.outer_iterations; ++outer) {
    IndexedCloud cloud;
    cloud.voxel_size = params.voxel_size;
    cloud.max_points = params.max_points_per_voxel;
    for (int index = 0; index < count; ++index) {
      for (const Eigen::Vector3d & point_body : queries[static_cast<std::size_t>(index)]) {
        cloud.add(static_cast<std::uint16_t>(index), poses[static_cast<std::size_t>(index)] * point_body);
      }
    }
    cloud.compute_planes();
    const std::vector<Sophus::SE3d> held = poses;
#pragma omp parallel for schedule(dynamic) if (count > 8)
    for (int index = 0; index < count; ++index) {
      const std::size_t scan = static_cast<std::size_t>(index);
      const bool is_newest = index + 1 == count;
      if (use_mask) {
        if (adjustable[scan] == 0) {
          continue;
        }
      } else if (is_newest) {
        continue;
      }
      const bool has_next = !is_newest;
      const Sophus::SE3d previous = index > 0 ? initial[scan - 1] : Sophus::SE3d();
      const Sophus::SE3d next = has_next ? initial[scan + 1] : Sophus::SE3d();
      const Sophus::SE3d to_next = has_next ? imu_from_previous[scan + 1] : Sophus::SE3d();
      const Sophus::SE3d clamp_pose = use_clamp ? clamp_poses[scan] : pose_priors[scan];
      poses[scan] = refine_one_pose(
        static_cast<std::uint16_t>(index), held[scan], previous, next, index > 0, has_next,
        imu_from_previous[scan], to_next, pose_priors[scan], clamp_pose, queries[scan], cloud,
        params);
    }
  }
  if (!use_mask || adjustable.back() == 0) {
    poses.back() = newest;
  }
}

}  // namespace back_odom
