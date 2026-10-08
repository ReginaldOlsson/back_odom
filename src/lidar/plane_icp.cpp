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

#include "back_odom/lidar/plane_icp.hpp"

#include <kiss_icp_cpp/core/VoxelUtils.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace back_odom
{
namespace
{

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

constexpr int k_min_correspondences = 10;
constexpr double k_max_step_translation = 0.5;
constexpr double k_max_step_rotation = 0.2;
constexpr double k_degeneracy_ratio = 1.0e-3;
constexpr double k_damping = 1.0e-6;

const std::array<kiss_icp::Voxel, 27> k_shifts = {
  kiss_icp::Voxel{0, 0, 0},   kiss_icp::Voxel{1, 0, 0},   kiss_icp::Voxel{-1, 0, 0},
  kiss_icp::Voxel{0, 1, 0},   kiss_icp::Voxel{0, -1, 0},  kiss_icp::Voxel{0, 0, 1},
  kiss_icp::Voxel{0, 0, -1},  kiss_icp::Voxel{1, 1, 0},   kiss_icp::Voxel{1, -1, 0},
  kiss_icp::Voxel{-1, 1, 0},  kiss_icp::Voxel{-1, -1, 0}, kiss_icp::Voxel{1, 0, 1},
  kiss_icp::Voxel{1, 0, -1},  kiss_icp::Voxel{-1, 0, 1},  kiss_icp::Voxel{-1, 0, -1},
  kiss_icp::Voxel{0, 1, 1},   kiss_icp::Voxel{0, 1, -1},  kiss_icp::Voxel{0, -1, 1},
  kiss_icp::Voxel{0, -1, -1}, kiss_icp::Voxel{1, 1, 1},   kiss_icp::Voxel{1, 1, -1},
  kiss_icp::Voxel{1, -1, 1},  kiss_icp::Voxel{1, -1, -1}, kiss_icp::Voxel{-1, 1, 1},
  kiss_icp::Voxel{-1, 1, -1}, kiss_icp::Voxel{-1, -1, 1}, kiss_icp::Voxel{-1, -1, -1}};

struct Accumulation
{
  Matrix6d jtj{Matrix6d::Zero()};
  Vector6d jtr{Vector6d::Zero()};
  int correspondences{0};
  double cost{0.0};
};

Accumulation join(Accumulation left, const Accumulation & right)
{
  left.jtj += right.jtj;
  left.jtr += right.jtr;
  left.correspondences += right.correspondences;
  left.cost += right.cost;
  return left;
}

/// Closest map point to `query` within the 27-voxel neighbourhood and the voxel it lives in.
bool closest_point(
  const kiss_icp::VoxelHashMap & map, const Eigen::Vector3d & query, const double max_distance,
  Eigen::Vector3d & closest, kiss_icp::Voxel & owner)
{
  const kiss_icp::Voxel centre = kiss_icp::PointToVoxel(query, map.voxel_size_);
  double best = max_distance * max_distance;
  bool found = false;
  for (const kiss_icp::Voxel & shift : k_shifts) {
    const kiss_icp::Voxel voxel = centre + shift;
    const auto search = map.map_.find(voxel);
    if (search == map.map_.end()) {
      continue;
    }
    for (const Eigen::Vector3d & point : search.value()) {
      const double distance = (point - query).squaredNorm();
      if (distance < best) {
        best = distance;
        closest = point;
        owner = voxel;
        found = true;
      }
    }
  }
  return found;
}

Accumulation build_system(
  const std::vector<Eigen::Vector3d> & source, const kiss_icp::VoxelHashMap & map,
  const SurfaceVoxelMap & surface, const Sophus::SE3d & pose, const double max_distance,
  const double kernel_scale)
{
  const double kernel2 = kernel_scale * kernel_scale;
  const Eigen::Matrix3d rotation_t = pose.so3().inverse().matrix();
  return tbb::parallel_reduce(
    tbb::blocked_range<std::size_t>{0, source.size(), 512}, Accumulation{},
    [&](const tbb::blocked_range<std::size_t> & range, Accumulation accumulation) {
      for (std::size_t index = range.begin(); index < range.end(); ++index) {
        const Eigen::Vector3d & body = source[index];
        const Eigen::Vector3d world = pose * body;
        Eigen::Vector3d closest = Eigen::Vector3d::Zero();
        kiss_icp::Voxel owner{0, 0, 0};
        double residual = max_distance;
        bool usable = false;
        Eigen::Vector3d normal = Eigen::Vector3d::Zero();
        if (closest_point(map, world, max_distance, closest, owner)) {
          const SurfaceVoxel * entry = surface.find(owner);
          if (entry != nullptr && entry->planar) {
            normal = entry->normal;
            residual = normal.dot(world - closest);
            usable = true;
          }
        }
        const double denom = kernel_scale + residual * residual;
        accumulation.cost += kernel2 * residual * residual / denom;
        if (!usable) {
          continue;
        }
        const Eigen::Vector3d normal_body = rotation_t * normal;
        Vector6d jacobian;
        jacobian.head<3>() = normal_body;
        jacobian.tail<3>() = body.cross(normal_body);
        const double weight = kernel2 / (denom * denom);
        accumulation.jtj.noalias() += (weight * jacobian) * jacobian.transpose();
        accumulation.jtr.noalias() += (weight * residual) * jacobian;
        ++accumulation.correspondences;
      }
      return accumulation;
    },
    join);
}

bool solve_increment(const Matrix6d & hessian_in, const Vector6d & gradient_in, Vector6d & dx, int & dropped)
{
  dropped = 0;
  Vector6d scale = hessian_in.diagonal().cwiseAbs().cwiseSqrt();
  for (int axis = 0; axis < 6; ++axis) {
    if (scale(axis) < 1.0e-9) {
      scale(axis) = 1.0;
    }
  }
  const Matrix6d hessian = scale.cwiseInverse().asDiagonal() * hessian_in * scale.cwiseInverse().asDiagonal();
  const Vector6d gradient = gradient_in.cwiseQuotient(scale);
  const Eigen::SelfAdjointEigenSolver<Matrix6d> solver(hessian);
  if (solver.info() != Eigen::Success) {
    return false;
  }
  const double lambda_max = solver.eigenvalues()(5);
  Vector6d y = Vector6d::Zero();
  if (lambda_max > 0.0) {
    for (int axis = 0; axis < 6; ++axis) {
      const double lambda = solver.eigenvalues()(axis);
      if (lambda < k_degeneracy_ratio * lambda_max) {
        ++dropped;
        continue;
      }
      const Vector6d direction = solver.eigenvectors().col(axis);
      y -= direction * (direction.dot(gradient) / (lambda + k_damping));
    }
  }
  dx = y.cwiseQuotient(scale);
  return dx.allFinite();
}

}  // namespace

PlaneIcpResult align_to_surface(
  const std::vector<Eigen::Vector3d> & source, const kiss_icp::VoxelHashMap & map,
  const SurfaceVoxelMap & surface, const Sophus::SE3d & initial_guess, const PlaneIcpParams & params,
  const std::chrono::steady_clock::time_point * const deadline)
{
  PlaneIcpResult result;
  result.pose = initial_guess;
  if (map.Empty() || source.empty() || !(params.kernel_scale > 0.0) || params.max_iterations < 1) {
    return result;
  }
  const auto past_deadline = [deadline]() {
    return deadline != nullptr && std::chrono::steady_clock::now() >= *deadline;
  };
  Sophus::SE3d pose = initial_guess;
  const double inverse_count = 1.0 / static_cast<double>(source.size());
  bool converged = false;
  for (int iteration = 0; iteration < params.max_iterations; ++iteration) {
    if (past_deadline()) {
      result.timed_out = true;
      break;
    }
    const Accumulation system = build_system(
      source, map, surface, pose, params.max_correspondence_distance, params.kernel_scale);
    result.correspondences = system.correspondences;
    result.robust_cost = system.cost * inverse_count;
    result.hessian = system.jtj;
    if (system.correspondences < k_min_correspondences) {
      break;
    }
    Vector6d dx = Vector6d::Zero();
    int dropped = 0;
    if (!solve_increment(system.jtj, system.jtr, dx, dropped)) {
      break;
    }
    result.degenerate_axes = dropped;
    const double translation = dx.head<3>().norm();
    if (translation > k_max_step_translation) {
      dx.head<3>() *= k_max_step_translation / translation;
    }
    const double rotation = dx.tail<3>().norm();
    if (rotation > k_max_step_rotation) {
      dx.tail<3>() *= k_max_step_rotation / rotation;
    }
    pose = pose * Sophus::SE3d::exp(dx);
    result.iterations = static_cast<double>(iteration + 1);
    result.last_step = dx.norm();
    if (result.last_step < params.convergence_criterion) {
      converged = true;
      break;
    }
  }
  result.saturated =
    !converged && !result.timed_out && result.iterations + 0.5 >= static_cast<double>(params.max_iterations);
  result.pose = pose;
  result.correction = pose * initial_guess.inverse();
  return result;
}

double surface_plane_cost(
  const std::vector<Eigen::Vector3d> & source, const kiss_icp::VoxelHashMap & map,
  const SurfaceVoxelMap & surface, const Sophus::SE3d & pose, const double max_correspondence_distance,
  const double kernel_scale)
{
  if (source.empty() || !(kernel_scale > 0.0) || !(max_correspondence_distance > 0.0)) {
    return std::numeric_limits<double>::infinity();
  }
  const Accumulation system =
    build_system(source, map, surface, pose, max_correspondence_distance, kernel_scale);
  return system.cost / static_cast<double>(source.size());
}

}  // namespace back_odom
