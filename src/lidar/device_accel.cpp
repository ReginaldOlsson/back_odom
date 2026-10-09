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

#include "back_odom/lidar/device_accel.hpp"

#include "back_odom/lidar/device_kernels.hpp"
#include "back_odom/lidar/lidar_preprocess.hpp"

#include <kiss_icp_cpp/core/VoxelUtils.hpp>

#include <Eigen/Eigenvalues>
#include <sophus/se3.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace back_odom
{
namespace
{

constexpr int k_min_correspondences = 10;
constexpr double k_max_step_translation = 0.5;
constexpr double k_max_step_rotation = 0.2;
constexpr double k_degeneracy_ratio = 1.0e-3;
constexpr double k_damping = 1.0e-6;

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

struct MotionSegment
{
  double end_stamp{0.0};
  double start_stamp{0.0};
  Sophus::SE3d start_pose{};
  Sophus::SE3d::Tangent relative_tangent{Sophus::SE3d::Tangent::Zero()};
};

void store_pose(const Sophus::SE3d & pose, double rotation[9], double translation[3])
{
  const Eigen::Matrix3d matrix = pose.so3().matrix();
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      rotation[row * 3 + column] = matrix(row, column);
    }
  }
  translation[0] = pose.translation().x();
  translation[1] = pose.translation().y();
  translation[2] = pose.translation().z();
}

std::vector<MotionSegment> motion_segments(const std::vector<StampedPose> & trajectory)
{
  std::vector<MotionSegment> segments;
  if (trajectory.size() < 2) {
    return segments;
  }
  segments.reserve(trajectory.size() - 1);
  for (std::size_t index = 1; index < trajectory.size(); ++index) {
    const StampedPose & before = trajectory[index - 1];
    const StampedPose & after = trajectory[index];
    const Sophus::SE3d relative = before.pose.inverse() * after.pose;
    segments.push_back(MotionSegment{after.stamp, before.stamp, before.pose, relative.log()});
  }
  return segments;
}

Sophus::SE3d interpolate_pose(
  const std::vector<StampedPose> & trajectory, const std::vector<MotionSegment> & segments,
  const double stamp)
{
  if (trajectory.empty()) {
    return Sophus::SE3d();
  }
  if (segments.empty() || stamp <= trajectory.front().stamp) {
    return trajectory.front().pose;
  }
  if (stamp >= trajectory.back().stamp) {
    return trajectory.back().pose;
  }
  const auto segment = std::lower_bound(
    segments.cbegin(), segments.cend(), stamp,
    [](const MotionSegment & motion, const double value) { return motion.end_stamp < value; });
  if (segment == segments.cend()) {
    return trajectory.back().pose;
  }
  const double interval = segment->end_stamp - segment->start_stamp;
  const double alpha = interval > 1.0e-9 ? (stamp - segment->start_stamp) / interval : 0.0;
  return segment->start_pose * Sophus::SE3d::exp(alpha * segment->relative_tangent);
}

/// Everything the deskew kernels need, packed once per scan.
struct DeskewUpload
{
  std::vector<CudaSegment> segments;
  std::vector<double> flat;
  double front_rotation[9];
  double front_translation[3];
  double back_rotation[9];
  double back_translation[3];
  double end_rotation[9];
  double end_translation[3];
  double body_rotation[9];
  double body_translation[3];
};

DeskewUpload pack_deskew(
  const std::vector<Eigen::Vector3d> & points, const std::vector<double> & timestamps,
  const std::vector<StampedPose> & trajectory, const Sophus::SE3d & body_from_lidar)
{
  DeskewUpload upload;
  const std::vector<MotionSegment> segments = motion_segments(trajectory);
  const double scan_end = *std::max_element(timestamps.cbegin(), timestamps.cend());
  const Sophus::SE3d end_inverse = interpolate_pose(trajectory, segments, scan_end).inverse();
  upload.segments.resize(segments.size());
  for (std::size_t index = 0; index < segments.size(); ++index) {
    CudaSegment & slot = upload.segments[index];
    slot.start_stamp = segments[index].start_stamp;
    slot.end_stamp = segments[index].end_stamp;
    store_pose(segments[index].start_pose, slot.rotation, slot.translation);
    for (int axis = 0; axis < 6; ++axis) {
      slot.tangent[axis] = segments[index].relative_tangent(axis);
    }
  }
  store_pose(trajectory.front().pose, upload.front_rotation, upload.front_translation);
  store_pose(trajectory.back().pose, upload.back_rotation, upload.back_translation);
  store_pose(end_inverse, upload.end_rotation, upload.end_translation);
  store_pose(body_from_lidar, upload.body_rotation, upload.body_translation);
  upload.flat.resize(points.size() * 3);
  for (std::size_t index = 0; index < points.size(); ++index) {
    upload.flat[3 * index] = points[index].x();
    upload.flat[3 * index + 1] = points[index].y();
    upload.flat[3 * index + 2] = points[index].z();
  }
  return upload;
}

bool solve_body_increment(const Matrix6d & hessian_in, const Vector6d & gradient_in, Vector6d & dx, int & dropped)
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
  const Eigen::SelfAdjointEigenSolver<Matrix6d> eigen_solver(hessian);
  if (eigen_solver.info() != Eigen::Success) {
    return false;
  }
  const double lambda_max = eigen_solver.eigenvalues()(5);
  Vector6d y = Vector6d::Zero();
  if (lambda_max > 0.0) {
    for (int axis = 0; axis < 6; ++axis) {
      const double lambda = eigen_solver.eigenvalues()(axis);
      if (lambda < k_degeneracy_ratio * lambda_max) {
        ++dropped;
        continue;
      }
      const Vector6d direction = eigen_solver.eigenvectors().col(axis);
      y -= direction * (direction.dot(gradient) / (lambda + k_damping));
    }
  }
  dx = y.cwiseQuotient(scale);
  return dx.allFinite();
}

/// Upload points once per map epoch, then attach the cached normals per unique voxel.
bool upload_map(const kiss_icp::VoxelHashMap & map, const SurfaceVoxelMap & surface, const std::uint64_t epoch)
{
  if (cuda_map_current(&map, epoch, map.voxel_size_)) {
    return true;
  }
  std::vector<double> xyz;
  std::vector<std::uint64_t> keys;
  const std::size_t per_voxel = static_cast<std::size_t>(std::max(1U, map.max_points_per_voxel_));
  xyz.reserve(map.map_.size() * per_voxel * 3);
  keys.reserve(map.map_.size() * per_voxel);
  for (const auto & item : map.map_) {
    const std::uint64_t key = pack_voxel_key(item.first.x(), item.first.y(), item.first.z());
    for (const Eigen::Vector3d & point : item.second) {
      xyz.push_back(point.x());
      xyz.push_back(point.y());
      xyz.push_back(point.z());
      keys.push_back(key);
    }
  }
  std::vector<std::uint64_t> unique_keys;
  if (!cuda_upload_map(
        &map, epoch, map.voxel_size_, xyz.data(), keys.data(), static_cast<int>(keys.size()), unique_keys)) {
    return false;
  }
  std::vector<CudaVoxelPlane> planes(unique_keys.size());
  for (std::size_t index = 0; index < unique_keys.size(); ++index) {
    int x = 0;
    int y = 0;
    int z = 0;
    unpack_voxel_key(unique_keys[index], x, y, z);
    const SurfaceVoxel * entry = surface.find(kiss_icp::Voxel{x, y, z});
    CudaVoxelPlane & plane = planes[index];
    if (entry != nullptr && entry->planar) {
      plane.nx = static_cast<float>(entry->normal.x());
      plane.ny = static_cast<float>(entry->normal.y());
      plane.nz = static_cast<float>(entry->normal.z());
      plane.planar = 1;
    }
  }
  return cuda_upload_planes(planes.data(), static_cast<int>(planes.size()));
}

bool plane_at(
  const double * xyz, const int count, const bool upload_frame, const Sophus::SE3d & pose,
  const double max_distance, const double kernel_scale, Matrix6d & hessian, Vector6d & gradient,
  int & correspondences, double & robust_cost)
{
  double rotation[9];
  double translation[3];
  store_pose(pose, rotation, translation);
  CudaPlaneSystem system;
  if (!cuda_plane_system(xyz, count, rotation, translation, max_distance, kernel_scale, upload_frame, system)) {
    return false;
  }
  for (int row = 0; row < 6; ++row) {
    gradient(row) = system.jtr[row];
    for (int column = 0; column < 6; ++column) {
      hessian(row, column) = system.jtj[row * 6 + column];
    }
  }
  correspondences = system.correspondences;
  robust_cost = count <= 0 ? std::numeric_limits<double>::infinity()
                           : system.robust_cost_sum / static_cast<double>(count);
  return true;
}

}  // namespace

bool device_available()
{
  return cuda_available();
}

std::optional<DeviceScanClouds> device_downsample_scan(
  const std::vector<Eigen::Vector3d> & points, const std::vector<double> & timestamps,
  const std::vector<float> & intensities, const std::vector<StampedPose> & trajectory,
  const Sophus::SE3d & body_from_lidar, const double half_longitudinal, const double half_lateral,
  const double fine_voxel, const double coarse_voxel)
{
  if (!cuda_available() || points.size() != timestamps.size() || points.empty() || trajectory.empty() ||
      !(fine_voxel > 0.0) || !(coarse_voxel > 0.0) ||
      (!intensities.empty() && intensities.size() != points.size())) {
    return std::nullopt;
  }
  const auto started = std::chrono::steady_clock::now();
  const DeskewUpload upload = pack_deskew(points, timestamps, trajectory, body_from_lidar);
  std::vector<double> fine_xyz;
  std::vector<int> fine_index;
  std::vector<double> coarse_xyz;
  if (!cuda_front_downsample(
        upload.flat.data(), timestamps.data(), static_cast<int>(points.size()), upload.segments.data(),
        static_cast<int>(upload.segments.size()), trajectory.front().stamp, trajectory.back().stamp,
        upload.front_rotation, upload.front_translation, upload.back_rotation, upload.back_translation,
        upload.end_rotation, upload.end_translation, upload.body_rotation, upload.body_translation,
        half_longitudinal, half_lateral, fine_voxel, coarse_voxel, fine_xyz, fine_index, coarse_xyz) ||
      fine_xyz.size() != fine_index.size() * 3 || coarse_xyz.size() % 3 != 0) {
    return std::nullopt;
  }
  DeviceScanClouds clouds;
  clouds.deskew_ms =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  const auto voxel_started = std::chrono::steady_clock::now();
  clouds.map_points.resize(fine_index.size());
  clouds.map_intensities.resize(intensities.empty() ? 0 : fine_index.size());
  for (std::size_t index = 0; index < fine_index.size(); ++index) {
    clouds.map_points[index] =
      Eigen::Vector3d(fine_xyz[3 * index], fine_xyz[3 * index + 1], fine_xyz[3 * index + 2]);
    if (!intensities.empty()) {
      const int source_index = fine_index[index];
      clouds.map_intensities[index] =
        source_index >= 0 && static_cast<std::size_t>(source_index) < intensities.size()
          ? intensities[static_cast<std::size_t>(source_index)]
          : 0.0F;
    }
  }
  clouds.source.resize(coarse_xyz.size() / 3);
  for (std::size_t index = 0; index < clouds.source.size(); ++index) {
    clouds.source[index] =
      Eigen::Vector3d(coarse_xyz[3 * index], coarse_xyz[3 * index + 1], coarse_xyz[3 * index + 2]);
  }
  clouds.voxel_ms =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - voxel_started).count();
  return clouds;
}

std::optional<std::vector<Eigen::Vector3d>> device_deskew_crop(
  const std::vector<Eigen::Vector3d> & points, const std::vector<double> & timestamps,
  const std::vector<StampedPose> & trajectory, const Sophus::SE3d & body_from_lidar,
  const double half_longitudinal, const double half_lateral)
{
  if (!cuda_available() || points.size() != timestamps.size() || points.empty() || trajectory.empty()) {
    return std::nullopt;
  }
  const DeskewUpload upload = pack_deskew(points, timestamps, trajectory, body_from_lidar);
  std::vector<double> deskewed;
  if (!cuda_deskew(
        upload.flat.data(), timestamps.data(), static_cast<int>(points.size()), upload.segments.data(),
        static_cast<int>(upload.segments.size()), trajectory.front().stamp, trajectory.back().stamp,
        upload.front_rotation, upload.front_translation, upload.back_rotation, upload.back_translation,
        upload.end_rotation, upload.end_translation, upload.body_rotation, upload.body_translation,
        deskewed) ||
      deskewed.size() != points.size() * 3) {
    return std::nullopt;
  }
  std::vector<Eigen::Vector3d> cloud(points.size());
  for (std::size_t index = 0; index < points.size(); ++index) {
    cloud[index] = Eigen::Vector3d(deskewed[3 * index], deskewed[3 * index + 1], deskewed[3 * index + 2]);
  }
  return crop_lidar_box(cloud, half_longitudinal, half_lateral);
}

std::optional<std::vector<Eigen::Vector3d>> device_voxel_downsample(
  const std::vector<Eigen::Vector3d> & points, const double voxel_size)
{
  if (!cuda_available() || !(voxel_size > 0.0)) {
    return std::nullopt;
  }
  if (points.empty()) {
    return std::vector<Eigen::Vector3d>{};
  }
  std::vector<double> xyz(points.size() * 3);
  std::vector<std::uint64_t> keys(points.size());
  for (std::size_t index = 0; index < points.size(); ++index) {
    xyz[3 * index] = points[index].x();
    xyz[3 * index + 1] = points[index].y();
    xyz[3 * index + 2] = points[index].z();
    const kiss_icp::Voxel voxel = kiss_icp::PointToVoxel(points[index], voxel_size);
    keys[index] = pack_voxel_key(voxel.x(), voxel.y(), voxel.z());
  }
  std::vector<double> kept;
  if (!cuda_voxel_downsample(xyz.data(), keys.data(), static_cast<int>(points.size()), kept) ||
      kept.size() % 3 != 0) {
    return std::nullopt;
  }
  std::vector<Eigen::Vector3d> cloud(kept.size() / 3);
  for (std::size_t index = 0; index < cloud.size(); ++index) {
    cloud[index] = Eigen::Vector3d(kept[3 * index], kept[3 * index + 1], kept[3 * index + 2]);
  }
  return cloud;
}

std::optional<PlaneIcpResult> align_points_on_device(
  const std::vector<Eigen::Vector3d> & frame, const kiss_icp::VoxelHashMap & map,
  const SurfaceVoxelMap & surface, const std::uint64_t map_epoch, const Sophus::SE3d & initial_guess,
  const PlaneIcpParams & params, const std::chrono::steady_clock::time_point * const deadline)
{
  if (!cuda_available() || frame.empty() || map.Empty() || !(params.kernel_scale > 0.0) ||
      params.max_iterations < 1) {
    return std::nullopt;
  }
  const auto past_deadline = [deadline]() {
    return deadline != nullptr && std::chrono::steady_clock::now() >= *deadline;
  };
  PlaneIcpResult result;
  result.pose = initial_guess;
  if (past_deadline()) {
    result.timed_out = true;
    return result;
  }
  if (!upload_map(map, surface, map_epoch)) {
    return std::nullopt;
  }
  std::vector<double> xyz(frame.size() * 3);
  for (std::size_t index = 0; index < frame.size(); ++index) {
    xyz[3 * index] = frame[index].x();
    xyz[3 * index + 1] = frame[index].y();
    xyz[3 * index + 2] = frame[index].z();
  }
  const int count = static_cast<int>(frame.size());
  Sophus::SE3d pose = initial_guess;
  bool upload_frame = true;
  bool converged = false;
  for (int iteration = 0; iteration < params.max_iterations; ++iteration) {
    if (past_deadline()) {
      result.timed_out = true;
      break;
    }
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d gradient = Vector6d::Zero();
    int correspondences = 0;
    double robust_cost = 0.0;
    if (!plane_at(
          xyz.data(), count, upload_frame, pose,
          correspondence_distance_for_iteration(params, iteration), params.kernel_scale, hessian,
          gradient, correspondences, robust_cost)) {
      return std::nullopt;
    }
    upload_frame = false;
    result.correspondences = correspondences;
    result.robust_cost = robust_cost;
    result.hessian = hessian;
    if (correspondences < k_min_correspondences) {
      break;
    }
    Vector6d dx = Vector6d::Zero();
    int dropped = 0;
    if (!solve_body_increment(hessian, gradient, dx, dropped)) {
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

}  // namespace back_odom
