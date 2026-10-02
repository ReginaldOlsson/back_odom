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

#include "back_odom/bump_map.hpp"

#include "back_odom/device_kernels.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace back_odom
{
namespace
{

constexpr double k_plane_ratio = 0.25;
constexpr double k_rotation_epsilon = 1.0e-4;
constexpr double k_translation_epsilon = 1.0e-5;
constexpr int k_lm_inner = 6;
constexpr int k_image_limit = 48;

struct IndexedPoint
{
  std::uint64_t key{0};
  std::uint32_t index{0};
};

Eigen::Matrix3d skew(const Eigen::Vector3d & point)
{
  Eigen::Matrix3d hat;
  hat << 0.0, -point.z(), point.y(), point.z(), 0.0, -point.x(), -point.y(), point.x(), 0.0;
  return hat;
}

bool finite_point(const Eigen::Vector3d & point)
{
  return point.allFinite();
}

int image_index(const int width, const int x, const int y)
{
  return y * width + x;
}

}  // namespace

BumpMap::BumpMap(const BumpMapConfig & config) : config_(config)
{
  inv_voxel_ = 1.0 / config_.voxel_size;
  inv_pixel_ = 1.0 / config_.pixel_size;
  normal_tolerance_rad_ = config_.normal_tolerance_deg * std::acos(-1.0) / 180.0;
  max_correspondence_distance_sq_ =
    config_.max_correspondence_distance * config_.max_correspondence_distance;
  constexpr float sigma = 0.8F;
  float weight_sum = 0.0F;
  for (int y = -1; y <= 1; ++y) {
    for (int x = -1; x <= 1; ++x) {
      const float value = std::exp(
        -static_cast<float>(x * x + y * y) / (2.0F * sigma * sigma));
      gauss_kernel_(y + 1, x + 1) = value;
      weight_sum += value;
    }
  }
  gauss_kernel_ /= weight_sum;
  voxels_.reserve(4096);
}

void BumpMap::clear()
{
  voxels_.clear();
  generation_ = 0;
  usable_count_ = 0;
}

std::size_t BumpMap::usable_count() const
{
  return usable_count_;
}

bool BumpMap::usable_surface(const Voxel & voxel) const
{
  return voxel.usable && voxel.planar && voxel.image.width > 0 && voxel.image.height > 0;
}

std::uint64_t BumpMap::key_of(const Eigen::Vector3d & point) const
{
  const int x = static_cast<int>(std::floor(point.x() * inv_voxel_));
  const int y = static_cast<int>(std::floor(point.y() * inv_voxel_));
  const int z = static_cast<int>(std::floor(point.z() * inv_voxel_));
  return pack_voxel_key(x, y, z);
}

const BumpMap::Voxel * BumpMap::find_surface(const Eigen::Vector3d & point) const
{
  const std::uint64_t key = key_of(point);
  const auto found = voxels_.find(key);
  if (found != voxels_.end()) {
    if (usable_surface(found->second)) {
      return &found->second;
    }
    // A planar cell that has not been seen again is a vehicle-sized cluster. Keep its
    // points out of the solve instead of sticking them to the road beside it.
    if (found->second.planar && found->second.image.width > 0) {
      return nullptr;
    }
  }

  const int base_x = static_cast<int>(std::floor(point.x() * inv_voxel_));
  const int base_y = static_cast<int>(std::floor(point.y() * inv_voxel_));
  const int base_z = static_cast<int>(std::floor(point.z() * inv_voxel_));
  const Voxel * best = nullptr;
  double best_distance = max_correspondence_distance_sq_;
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0 && dz == 0) {
          continue;
        }
        const auto neighbor = voxels_.find(pack_voxel_key(base_x + dx, base_y + dy, base_z + dz));
        if (neighbor == voxels_.end() || !usable_surface(neighbor->second)) {
          continue;
        }
        const double distance = (point - neighbor->second.mean).squaredNorm();
        if (distance < best_distance) {
          best_distance = distance;
          best = &neighbor->second;
        }
      }
    }
  }
  return best;
}

bool BumpMap::sample(
  const Image & image, const double x, const double y, const bool with_gradient, double & value,
  double & d_i_dx, double & d_i_dy) const
{
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  const int max_x = image.width - 1;
  const int max_y = image.height - 1;
  if (x0 < 0 || y0 < 0 || x1 > max_x || y1 > max_y) {
    return false;
  }

  const auto weight_at = [&image](const int row, const int col) {
    return image.weight[static_cast<std::size_t>(image_index(image.width, col, row))] > 0.0F ? 1.0 : 0.0;
  };
  const auto depth_at = [&image](const int row, const int col) {
    return static_cast<double>(
      image.smoothed[static_cast<std::size_t>(image_index(image.width, col, row))]);
  };

  const double dx = x - static_cast<double>(x0);
  const double dy = y - static_cast<double>(y0);
  const double dx1 = 1.0 - dx;
  const double dy1 = 1.0 - dy;
  const double v00 = weight_at(y0, x0);
  const double v01 = weight_at(y0, x1);
  const double v10 = weight_at(y1, x0);
  const double v11 = weight_at(y1, x1);
  const double w0 = dx1 * dy1 * v00;
  const double w1 = dx * dy1 * v01;
  const double w2 = dx1 * dy * v10;
  const double w3 = dx * dy * v11;
  const double weight_sum = w0 + w1 + w2 + w3;
  if (weight_sum <= 0.0) {
    return false;
  }
  value = (w0 * depth_at(y0, x0) + w1 * depth_at(y0, x1) + w2 * depth_at(y1, x0) + w3 * depth_at(y1, x1)) /
          weight_sum;
  d_i_dx = 0.0;
  d_i_dy = 0.0;
  if (!with_gradient) {
    return true;
  }

  if (x0 >= 1 && x1 + 1 <= max_x) {
    const int xm = x0 - 1;
    const int xp = x1 + 1;
    const double am0 = weight_at(y0, xm);
    const double am2 = weight_at(y1, xm);
    const double wm0 = dx1 * dy1 * am0;
    const double wm1 = dx * dy1 * v00;
    const double wm2 = dx1 * dy * am2;
    const double wm3 = dx * dy * v10;
    const double wms = wm0 + wm1 + wm2 + wm3;
    const double ap1 = weight_at(y0, xp);
    const double ap3 = weight_at(y1, xp);
    const double wp0 = dx1 * dy1 * v01;
    const double wp1 = dx * dy1 * ap1;
    const double wp2 = dx1 * dy * v11;
    const double wp3 = dx * dy * ap3;
    const double wps = wp0 + wp1 + wp2 + wp3;
    if (wms > 0.0 && wps > 0.0) {
      const double left =
        (wm0 * depth_at(y0, xm) + wm1 * depth_at(y0, x0) + wm2 * depth_at(y1, xm) + wm3 * depth_at(y1, x0)) /
        wms;
      const double right =
        (wp0 * depth_at(y0, x1) + wp1 * depth_at(y0, xp) + wp2 * depth_at(y1, x1) + wp3 * depth_at(y1, xp)) /
        wps;
      d_i_dx = 0.5 * (right - left);
    }
  }
  if (y0 >= 1 && y1 + 1 <= max_y) {
    const int ym = y0 - 1;
    const int yp = y1 + 1;
    const double am0 = weight_at(ym, x0);
    const double am1 = weight_at(ym, x1);
    const double wm0 = dx1 * dy1 * am0;
    const double wm1 = dx * dy1 * am1;
    const double wm2 = dx1 * dy * v00;
    const double wm3 = dx * dy * v01;
    const double wms = wm0 + wm1 + wm2 + wm3;
    const double ap2 = weight_at(yp, x0);
    const double ap3 = weight_at(yp, x1);
    const double wp0 = dx1 * dy1 * v10;
    const double wp1 = dx * dy1 * v11;
    const double wp2 = dx1 * dy * ap2;
    const double wp3 = dx * dy * ap3;
    const double wps = wp0 + wp1 + wp2 + wp3;
    if (wms > 0.0 && wps > 0.0) {
      const double down =
        (wm0 * depth_at(ym, x0) + wm1 * depth_at(ym, x1) + wm2 * depth_at(y0, x0) + wm3 * depth_at(y0, x1)) /
        wms;
      const double up =
        (wp0 * depth_at(y1, x0) + wp1 * depth_at(y1, x1) + wp2 * depth_at(yp, x0) + wp3 * depth_at(yp, x1)) /
        wps;
      d_i_dy = 0.5 * (up - down);
    }
  }
  return true;
}

bool BumpMap::residual_at(
  const Voxel & voxel, const Eigen::Vector3d & world_point, double & residual) const
{
  const Eigen::Vector3d camera = voxel.R_c_w * world_point + voxel.t_c_w;
  double depth = 0.0;
  double unused_x = 0.0;
  double unused_y = 0.0;
  if (!sample(voxel.image, camera.x() * inv_pixel_, camera.y() * inv_pixel_, false, depth, unused_x, unused_y)) {
    return false;
  }
  residual = camera.z() - depth;
  return true;
}

void BumpMap::recount()
{
  usable_count_ = 0;
  for (const auto & entry : voxels_) {
    if (usable_surface(entry.second)) {
      ++usable_count_;
    }
  }
}

void BumpMap::confirm_visible(
  const std::vector<Eigen::Vector3d> & body_points, const Sophus::SE3d & guess)
{
  std::unordered_map<std::uint64_t, int> hits;
  hits.reserve(body_points.size());
  for (const Eigen::Vector3d & body_point : body_points) {
    if (!finite_point(body_point)) {
      continue;
    }
    const Eigen::Vector3d world = guess * body_point;
    const auto found = voxels_.find(key_of(world));
    if (found == voxels_.end() || found->second.usable || !found->second.planar ||
        found->second.image.width <= 0) {
      continue;
    }
    double residual = 0.0;
    if (!residual_at(found->second, world, residual)) {
      continue;
    }
    if (std::abs(residual) > config_.max_correspondence_distance) {
      continue;
    }
    ++hits[found->first];
  }
  for (const auto & hit : hits) {
    if (hit.second < config_.min_confirm_points) {
      continue;
    }
    const auto found = voxels_.find(hit.first);
    if (found == voxels_.end()) {
      continue;
    }
    found->second.usable = true;
    found->second.last_touch = generation_;
  }
  recount();
}

BumpMap::NormalUpdate BumpMap::update_normal(Voxel & voxel, const bool allow_frame_change)
{
  if (voxel.count < config_.min_points) {
    voxel.planar = false;
    voxel.usable = false;
    return NormalUpdate::NotPlanar;
  }
  const double scale = static_cast<double>(voxel.count);
  const Eigen::Vector3d mean = voxel.sum / scale;
  const Eigen::Matrix3d covariance =
    (voxel.outer - voxel.sum * mean.transpose()) / (scale - 1.0);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  if (solver.info() != Eigen::Success) {
    voxel.planar = false;
    voxel.usable = false;
    return NormalUpdate::NotPlanar;
  }
  const double lambda_min = solver.eigenvalues()(0);
  const double lambda_max = solver.eigenvalues()(2);
  if (lambda_max <= 1.0e-8 || lambda_min / lambda_max > k_plane_ratio) {
    voxel.planar = false;
    voxel.usable = false;
    return NormalUpdate::NotPlanar;
  }

  const Eigen::Vector3d normal = solver.eigenvectors().col(0).normalized();
  if (voxel.planar && voxel.image.width > 0 && !allow_frame_change) {
    const double cosine = std::clamp(std::abs(voxel.R_w_o.col(2).dot(normal)), 0.0, 1.0);
    if (std::acos(cosine) <= normal_tolerance_rad_) {
      voxel.planar = true;
      return NormalUpdate::Unchanged;
    }
  }

  const Eigen::Vector3d z_axis = normal;
  const Eigen::Vector3d x_axis = z_axis.unitOrthogonal();
  const Eigen::Vector3d y_axis = z_axis.cross(x_axis).normalized();
  voxel.R_w_o.col(0) = x_axis;
  voxel.R_w_o.col(1) = y_axis;
  voxel.R_w_o.col(2) = z_axis;
  voxel.mean = mean;
  voxel.planar = true;
  return NormalUpdate::Changed;
}

void BumpMap::smooth(Voxel & voxel)
{
  Image & image = voxel.image;
  if (!config_.smooth) {
    image.smoothed = image.depth;
    return;
  }
  const int radius = 1;
  std::vector<float> smoothed(image.depth.size(), 0.0F);
  for (int y = 0; y < image.height; ++y) {
    for (int x = 0; x < image.width; ++x) {
      const std::size_t index = static_cast<std::size_t>(image_index(image.width, x, y));
      if (image.weight[index] <= 0.0F) {
        continue;
      }
      float weighted = 0.0F;
      float weight_sum = 0.0F;
      for (int ky = -radius; ky <= radius; ++ky) {
        for (int kx = -radius; kx <= radius; ++kx) {
          const int yy = y + ky;
          const int xx = x + kx;
          if (xx < 0 || yy < 0 || xx >= image.width || yy >= image.height) {
            continue;
          }
          const std::size_t neighbor = static_cast<std::size_t>(image_index(image.width, xx, yy));
          if (image.weight[neighbor] <= 0.0F) {
            continue;
          }
          const float weight = gauss_kernel_(ky + radius, kx + radius);
          weighted += image.depth[neighbor] * weight;
          weight_sum += weight;
        }
      }
      if (weight_sum > 0.0F) {
        smoothed[index] = weighted / weight_sum;
      }
    }
  }
  image.smoothed = std::move(smoothed);
}

void BumpMap::update_image(
  Voxel & voxel, const std::vector<Eigen::Vector4d> & points, const bool rebuild_frame)
{
  if (!voxel.planar || points.empty()) {
    return;
  }

  if (rebuild_frame || voxel.image.width <= 0) {
    const Eigen::Vector3d origin(
      std::floor(voxel.mean.x() * inv_voxel_) * config_.voxel_size,
      std::floor(voxel.mean.y() * inv_voxel_) * config_.voxel_size,
      std::floor(voxel.mean.z() * inv_voxel_) * config_.voxel_size);
    double u_min = std::numeric_limits<double>::max();
    double v_min = std::numeric_limits<double>::max();
    double u_max = std::numeric_limits<double>::lowest();
    double v_max = std::numeric_limits<double>::lowest();
    for (int corner = 0; corner < 8; ++corner) {
      const Eigen::Vector3d offset(
        (corner & 1) != 0 ? config_.voxel_size : 0.0,
        (corner & 2) != 0 ? config_.voxel_size : 0.0,
        (corner & 4) != 0 ? config_.voxel_size : 0.0);
      const Eigen::Vector3d plane = voxel.R_w_o.transpose() * (origin + offset - voxel.mean);
      u_min = std::min(u_min, plane.x());
      u_max = std::max(u_max, plane.x());
      v_min = std::min(v_min, plane.y());
      v_max = std::max(v_max, plane.y());
    }
    const int width = static_cast<int>(std::ceil((u_max - u_min) * inv_pixel_)) + 2;
    const int height = static_cast<int>(std::ceil((v_max - v_min) * inv_pixel_)) + 2;
    if (width <= 1 || height <= 1 || width > k_image_limit || height > k_image_limit) {
      voxel.planar = false;
      voxel.usable = false;
      voxel.image = Image{};
      return;
    }

    Image previous = voxel.image;
    const Eigen::Matrix3d previous_rotation = voxel.R_c_w;
    const Eigen::Vector3d previous_translation = voxel.t_c_w;
    const Eigen::Vector3d image_origin = voxel.R_w_o * Eigen::Vector3d(u_min, v_min, 0.0) + voxel.mean;
    voxel.R_c_w = voxel.R_w_o.transpose();
    voxel.t_c_w = -voxel.R_c_w * image_origin;
    voxel.image = Image{};
    voxel.image.width = width;
    voxel.image.height = height;
    const std::size_t pixels = static_cast<std::size_t>(width * height);
    voxel.image.depth.assign(pixels, 0.0F);
    voxel.image.smoothed.assign(pixels, 0.0F);
    voxel.image.weight.assign(pixels, 0.0F);

    if (previous.width > 0) {
      for (int row = 0; row < previous.height; ++row) {
        for (int col = 0; col < previous.width; ++col) {
          const std::size_t source = static_cast<std::size_t>(image_index(previous.width, col, row));
          if (previous.weight[source] <= 0.0F) {
            continue;
          }
          const Eigen::Vector3d old_camera(
            static_cast<double>(col) * config_.pixel_size,
            static_cast<double>(row) * config_.pixel_size,
            static_cast<double>(previous.depth[source]));
          const Eigen::Vector3d world =
            previous_rotation.transpose() * (old_camera - previous_translation);
          const Eigen::Vector3d camera = voxel.R_c_w * world + voxel.t_c_w;
          const int x = static_cast<int>(std::lround(camera.x() * inv_pixel_));
          const int y = static_cast<int>(std::lround(camera.y() * inv_pixel_));
          if (x < 0 || y < 0 || x >= width || y >= height) {
            continue;
          }
          const std::size_t destination =
            static_cast<std::size_t>(image_index(voxel.image.width, x, y));
          voxel.image.depth[destination] = static_cast<float>(camera.z());
          voxel.image.weight[destination] = previous.weight[source];
        }
      }
    }
  }

  Image & image = voxel.image;
  for (const Eigen::Vector4d & point : points) {
    const Eigen::Vector3d camera = voxel.R_c_w * point.head<3>() + voxel.t_c_w;
    const int x = static_cast<int>(std::lround(camera.x() * inv_pixel_));
    const int y = static_cast<int>(std::lround(camera.y() * inv_pixel_));
    if (x < 0 || y < 0 || x >= image.width || y >= image.height) {
      continue;
    }
    const std::size_t index = static_cast<std::size_t>(image_index(image.width, x, y));
    const double range = std::max(point.w(), 1.0e-3);
    const double added = config_.weighted ? std::min(0.5, 1.0 / range) : 1.0;
    const double previous_weight = static_cast<double>(image.weight[index]);
    const double previous_depth = static_cast<double>(image.depth[index]);
    image.weight[index] = static_cast<float>(previous_weight + added);
    image.depth[index] =
      static_cast<float>((previous_depth * previous_weight + added * camera.z()) / (previous_weight + added));
  }
  smooth(voxel);
}

void BumpMap::integrate(
  const std::vector<Eigen::Vector3d> & world_points, const Eigen::Vector3d & sensor_origin)
{
  if (world_points.empty() || !sensor_origin.allFinite()) {
    return;
  }
  ++generation_;
  std::vector<IndexedPoint> ordered;
  ordered.reserve(world_points.size());
  for (std::uint32_t index = 0; index < static_cast<std::uint32_t>(world_points.size()); ++index) {
    if (!finite_point(world_points[index])) {
      continue;
    }
    ordered.push_back(IndexedPoint{key_of(world_points[index]), index});
  }
  if (ordered.empty()) {
    return;
  }
  std::sort(ordered.begin(), ordered.end(), [](const IndexedPoint & left, const IndexedPoint & right) {
    if (left.key != right.key) {
      return left.key < right.key;
    }
    return left.index < right.index;
  });

  std::vector<Eigen::Vector4d> accepted;
  const auto flush = [&](const std::uint64_t key, const bool create) {
    if (accepted.empty()) {
      return;
    }
    Voxel * voxel = nullptr;
    if (create) {
      voxel = &voxels_[key];
      voxel->born_generation = generation_;
    } else {
      const auto found = voxels_.find(key);
      if (found == voxels_.end()) {
        accepted.clear();
        return;
      }
      voxel = &found->second;
    }
    const bool had_image = voxel->planar && voxel->image.width > 0;
    if (
      had_image && voxel->born_generation < generation_ &&
      static_cast<int>(accepted.size()) >= config_.min_confirm_points) {
      voxel->usable = true;
    }
    voxel->last_touch = generation_;
    for (const Eigen::Vector4d & point : accepted) {
      const Eigen::Vector3d xyz = point.head<3>();
      voxel->sum += xyz;
      voxel->outer += xyz * xyz.transpose();
      ++voxel->count;
    }
    const NormalUpdate normal = update_normal(*voxel, !had_image);
    if (normal == NormalUpdate::NotPlanar) {
      voxel->image = Image{};
      voxels_.erase(key);
    } else {
      update_image(*voxel, accepted, normal == NormalUpdate::Changed || !had_image);
      if (!voxel->planar) {
        voxels_.erase(key);
      }
    }
    accepted.clear();
  };

  std::size_t cursor = 0;
  while (cursor < ordered.size()) {
    const std::uint64_t key = ordered[cursor].key;
    std::size_t end = cursor + 1;
    while (end < ordered.size() && ordered[end].key == key) {
      ++end;
    }
    const auto found = voxels_.find(key);
    const bool exists = found != voxels_.end();
    accepted.clear();
    for (std::size_t index = cursor; index < end; ++index) {
      const Eigen::Vector3d & point = world_points[ordered[index].index];
      if (exists && found->second.planar && found->second.image.width > 0) {
        double residual = 0.0;
        if (!residual_at(found->second, point, residual) || std::abs(residual) > config_.fuse_distance) {
          continue;
        }
      }
      Eigen::Vector4d stamped;
      stamped.head<3>() = point;
      stamped.w() = (point - sensor_origin).norm();
      accepted.push_back(stamped);
    }
    flush(key, !exists);
    cursor = end;
  }

  for (auto it = voxels_.begin(); it != voxels_.end();) {
    const Voxel & voxel = it->second;
    if (!voxel.usable && voxel.last_touch < generation_) {
      it = voxels_.erase(it);
    } else {
      ++it;
    }
  }
  recount();
}

void BumpMap::cull_outside_box(
  const Sophus::SE3d & world_from_body, const double half_longitudinal, const double half_lateral)
{
  const Sophus::SE3d body_from_world = world_from_body.inverse();
  for (auto it = voxels_.begin(); it != voxels_.end();) {
    const Eigen::Vector3d local = body_from_world * it->second.mean;
    if (std::abs(local.x()) > half_longitudinal || std::abs(local.y()) > half_lateral) {
      it = voxels_.erase(it);
    } else {
      ++it;
    }
  }
  recount();
}

void BumpMap::linearize(
  const std::vector<Eigen::Vector3d> & body_points, const Sophus::SE3d & pose, const bool with_jacobian,
  Accumulator & accumulator) const
{
  const Eigen::Matrix3d rotation = pose.rotationMatrix();
  const Eigen::Vector3d translation = pose.translation();
  for (const Eigen::Vector3d & body_point : body_points) {
    if (!finite_point(body_point)) {
      ++accumulator.skipped;
      continue;
    }
    const Eigen::Vector3d world = rotation * body_point + translation;
    const Voxel * surface = find_surface(world);
    if (surface == nullptr) {
      ++accumulator.skipped;
      continue;
    }
    if ((world - surface->mean).squaredNorm() > max_correspondence_distance_sq_) {
      ++accumulator.skipped;
      continue;
    }
    const Eigen::Vector3d camera = surface->R_c_w * world + surface->t_c_w;
    const double pixel_x = camera.x() * inv_pixel_;
    const double pixel_y = camera.y() * inv_pixel_;
    double depth = 0.0;
    double d_i_dx = 0.0;
    double d_i_dy = 0.0;
    if (!sample(surface->image, pixel_x, pixel_y, with_jacobian, depth, d_i_dx, d_i_dy)) {
      ++accumulator.skipped;
      continue;
    }
    const double residual = camera.z() - depth;
    if (std::abs(residual) >= config_.max_correspondence_distance) {
      ++accumulator.skipped;
      continue;
    }

    const double abs_residual = std::abs(residual);
    const bool inlier = abs_residual <= config_.huber_delta;
    const double weight = inlier ? 1.0 : config_.huber_delta / abs_residual;
    accumulator.cost += inlier ? 0.5 * residual * residual
                               : config_.huber_delta * (abs_residual - 0.5 * config_.huber_delta);
    accumulator.abs_sum += abs_residual;
    ++accumulator.count;
    if (!with_jacobian) {
      continue;
    }

    const Eigen::Matrix3d camera_from_body = surface->R_c_w * rotation;
    Eigen::Matrix<double, 3, 6> jacobian;
    jacobian.leftCols<3>() = -camera_from_body * skew(body_point);
    jacobian.rightCols<3>() = camera_from_body;
    Eigen::Matrix<double, 1, 2> image_jacobian;
    image_jacobian << d_i_dx, d_i_dy;
    image_jacobian *= inv_pixel_;
    const Eigen::Matrix<double, 1, 6> row =
      jacobian.row(2) - image_jacobian * jacobian.topRows<2>();
    const Eigen::Matrix<double, 6, 1> weighted = weight * row.transpose();
    accumulator.hessian.noalias() += weighted * row;
    accumulator.gradient.noalias() += weighted * residual;
  }
}

BumpAlignResult BumpMap::align(
  const std::vector<Eigen::Vector3d> & body_points, const Sophus::SE3d & guess,
  const std::vector<Eigen::Vector3d> * confirm_points)
{
  BumpAlignResult result;
  result.pose = guess;
  if (body_points.empty()) {
    return result;
  }
  confirm_visible(confirm_points != nullptr ? *confirm_points : body_points, guess);
  if (usable_count_ == 0) {
    return result;
  }

  const int minimum = std::max(10, std::min(400, static_cast<int>(body_points.size() / 20)));
  Sophus::SE3d pose = guess;
  double lambda = -1.0;
  int iterations = 0;
  bool solvable = false;
  for (int outer = 0; outer < config_.max_iterations; ++outer) {
    Accumulator linear;
    linearize(body_points, pose, true, linear);
    if (
      linear.count < minimum || linear.count <= linear.skipped || !linear.hessian.allFinite() ||
      !linear.gradient.allFinite()) {
      break;
    }
    solvable = true;
    if (!(lambda > 0.0)) {
      lambda = 1.0e-9 * std::max(1.0e-12, linear.hessian.diagonal().cwiseAbs().maxCoeff());
    }
    double nu = 2.0;
    bool stepped = false;
    bool converged = false;
    for (int inner = 0; inner < k_lm_inner; ++inner) {
      const Eigen::Matrix<double, 6, 6> damped =
        linear.hessian + lambda * Eigen::Matrix<double, 6, 6>::Identity();
      const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> solver(damped);
      if (solver.info() != Eigen::Success) {
        lambda *= nu;
        nu *= 2.0;
        continue;
      }
      const Eigen::Matrix<double, 6, 1> step = solver.solve(-linear.gradient);
      if (!step.allFinite()) {
        lambda *= nu;
        nu *= 2.0;
        continue;
      }
      const Sophus::SE3d delta(Sophus::SO3d::exp(step.head<3>()), step.tail<3>());
      const Eigen::Matrix3d rotation_change = delta.rotationMatrix() - Eigen::Matrix3d::Identity();
      const double convergence = std::max(
        rotation_change.cwiseAbs().maxCoeff() / k_rotation_epsilon,
        delta.translation().cwiseAbs().maxCoeff() / k_translation_epsilon);
      if (convergence < 1.0) {
        converged = true;
        break;
      }
      const Sophus::SE3d candidate = pose * delta;
      Accumulator trial;
      linearize(body_points, candidate, false, trial);
      const double predicted = step.dot(lambda * step - linear.gradient);
      const double rho = std::abs(predicted) > 1.0e-12 ? (linear.cost - trial.cost) / predicted : -1.0;
      if (rho < 0.0 || trial.count < minimum || trial.count <= trial.skipped) {
        lambda *= nu;
        nu *= 2.0;
        continue;
      }
      pose = candidate;
      lambda *= std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1.0, 3.0));
      ++iterations;
      stepped = true;
      break;
    }
    if (!stepped || converged) {
      break;
    }
  }

  if (!solvable) {
    return result;
  }
  Accumulator final_linear;
  linearize(body_points, pose, false, final_linear);
  if (final_linear.count < minimum || final_linear.count <= final_linear.skipped) {
    return result;
  }
  result.accepted = true;
  result.pose = pose;
  result.iterations = static_cast<double>(std::max(iterations, 1));
  result.correspondences = final_linear.count;
  result.skipped = final_linear.skipped;
  result.mean_cost =
    final_linear.count > 0 ? final_linear.abs_sum / static_cast<double>(final_linear.count) : 0.0;
  return result;
}

}  // namespace back_odom
