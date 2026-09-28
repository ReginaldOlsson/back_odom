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

#include "back_odom/fpfh_coarse_align.hpp"

#include <Eigen/Geometry>
#include <kiss_icp_cpp/core/VoxelUtils.hpp>

#include <omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

namespace back_odom
{
namespace
{

constexpr int k_histogram_bins = 33;
constexpr int k_ransac_iterations = 64;
constexpr double k_ransac_inlier_distance = 1.0;

struct Correspondence
{
  Eigen::Vector3d source{Eigen::Vector3d::Zero()};
  Eigen::Vector3d target{Eigen::Vector3d::Zero()};
};

struct DescribedCloud
{
  std::vector<Eigen::Vector3d> points;
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr features;
};

int thread_count(const int requested)
{
  if (requested > 0) {
    return requested;
  }
  const int available = omp_get_max_threads();
  return available > 0 ? available : 1;
}

std::vector<Eigen::Vector3d> keypoints(
  const std::vector<Eigen::Vector3d> & points, const FpfhParams & params)
{
  std::vector<Eigen::Vector3d> finite;
  finite.reserve(points.size());
  for (const Eigen::Vector3d & point : points) {
    if (point.allFinite()) {
      finite.push_back(point);
    }
  }
  std::vector<Eigen::Vector3d> downsampled =
    kiss_icp::VoxelDownsample(finite, params.keypoint_voxel);
  if (static_cast<int>(downsampled.size()) <= params.max_keypoints) {
    return downsampled;
  }

  std::vector<Eigen::Vector3d> capped;
  capped.reserve(static_cast<std::size_t>(params.max_keypoints));
  const double step =
    static_cast<double>(downsampled.size()) / static_cast<double>(params.max_keypoints);
  for (int index = 0; index < params.max_keypoints; ++index) {
    const auto source_index = static_cast<std::size_t>(static_cast<double>(index) * step);
    capped.push_back(downsampled[source_index]);
  }
  return capped;
}

bool finite_histogram(const pcl::FPFHSignature33 & signature)
{
  for (int bin = 0; bin < k_histogram_bins; ++bin) {
    if (!std::isfinite(signature.histogram[bin])) {
      return false;
    }
  }
  return true;
}

float histogram_l2_squared(const pcl::FPFHSignature33 & left, const pcl::FPFHSignature33 & right)
{
  float sum = 0.0F;
  for (int bin = 0; bin < k_histogram_bins; ++bin) {
    const float delta = left.histogram[bin] - right.histogram[bin];
    sum += delta * delta;
  }
  return sum;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr to_pcl(const std::vector<Eigen::Vector3d> & points)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  cloud->reserve(points.size());
  for (const Eigen::Vector3d & point : points) {
    pcl::PointXYZ sample;
    sample.x = static_cast<float>(point.x());
    sample.y = static_cast<float>(point.y());
    sample.z = static_cast<float>(point.z());
    cloud->push_back(sample);
  }
  return cloud;
}

std::optional<DescribedCloud> describe(
  const std::vector<Eigen::Vector3d> & points, const Eigen::Vector3d & viewpoint,
  const FpfhParams & params, const int threads)
{
  if (points.size() < 3) {
    return std::nullopt;
  }
  const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = to_pcl(points);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

  pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normals(static_cast<unsigned int>(threads));
  normals.setInputCloud(cloud);
  normals.setSearchMethod(tree);
  normals.setRadiusSearch(params.normal_radius);
  normals.setViewPoint(
    static_cast<float>(viewpoint.x()), static_cast<float>(viewpoint.y()),
    static_cast<float>(viewpoint.z()));
  pcl::PointCloud<pcl::Normal>::Ptr normal_cloud(new pcl::PointCloud<pcl::Normal>());
  normals.compute(*normal_cloud);

  pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh(
    static_cast<unsigned int>(threads));
  fpfh.setInputCloud(cloud);
  fpfh.setInputNormals(normal_cloud);
  fpfh.setSearchMethod(tree);
  fpfh.setRadiusSearch(params.fpfh_radius);
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature_cloud(
    new pcl::PointCloud<pcl::FPFHSignature33>());
  fpfh.compute(*feature_cloud);
  if (feature_cloud->size() != points.size() || normal_cloud->size() != points.size()) {
    return std::nullopt;
  }

  DescribedCloud described;
  described.features.reset(new pcl::PointCloud<pcl::FPFHSignature33>());
  described.points.reserve(points.size());
  described.features->reserve(points.size());
  for (std::size_t index = 0; index < points.size(); ++index) {
    if (
      !pcl::isFinite(normal_cloud->points[index]) ||
      !finite_histogram(feature_cloud->points[index])) {
      continue;
    }
    described.points.push_back(points[index]);
    described.features->push_back(feature_cloud->points[index]);
  }
  if (described.points.size() < 3) {
    return std::nullopt;
  }
  return described;
}

std::vector<int> nearest_gated(
  const DescribedCloud & query, const DescribedCloud & reference,
  const std::vector<Eigen::Vector3d> & query_in_map,
  const std::vector<Eigen::Vector3d> & reference_in_map, const double gate_squared,
  const int threads)
{
  const int query_count = static_cast<int>(query.points.size());
  const int reference_count = static_cast<int>(reference.points.size());
  std::vector<int> matches(static_cast<std::size_t>(query_count), -1);
#pragma omp parallel for num_threads(threads) schedule(static)
  for (int query_index = 0; query_index < query_count; ++query_index) {
    float best_distance = std::numeric_limits<float>::infinity();
    int best_index = -1;
    const pcl::FPFHSignature33 & signature =
      query.features->points[static_cast<std::size_t>(query_index)];
    const Eigen::Vector3d & query_point = query_in_map[static_cast<std::size_t>(query_index)];
    for (int reference_index = 0; reference_index < reference_count; ++reference_index) {
      const Eigen::Vector3d delta =
        query_point - reference_in_map[static_cast<std::size_t>(reference_index)];
      if (delta.squaredNorm() > gate_squared) {
        continue;
      }
      const float distance = histogram_l2_squared(
        signature, reference.features->points[static_cast<std::size_t>(reference_index)]);
      if (distance < best_distance) {
        best_distance = distance;
        best_index = reference_index;
      }
    }
    matches[static_cast<std::size_t>(query_index)] = best_index;
  }
  return matches;
}

std::vector<Correspondence> mutual_matches(
  const DescribedCloud & source, const DescribedCloud & target, const Sophus::SE3d & imu_guess,
  const double correspondence_distance, const int threads)
{
  std::vector<Eigen::Vector3d> source_in_map(source.points.size());
  for (std::size_t index = 0; index < source.points.size(); ++index) {
    source_in_map[index] = imu_guess * source.points[index];
  }
  const double gate_squared = correspondence_distance * correspondence_distance;
  const std::vector<int> source_to_target =
    nearest_gated(source, target, source_in_map, target.points, gate_squared, threads);
  const std::vector<int> target_to_source =
    nearest_gated(target, source, target.points, source_in_map, gate_squared, threads);

  std::vector<Correspondence> pairs;
  pairs.reserve(source.points.size());
  for (std::size_t index = 0; index < source.points.size(); ++index) {
    const int target_index = source_to_target[index];
    if (target_index < 0) {
      continue;
    }
    if (target_to_source[static_cast<std::size_t>(target_index)] != static_cast<int>(index)) {
      continue;
    }
    pairs.push_back(
      Correspondence{source.points[index], target.points[static_cast<std::size_t>(target_index)]});
  }
  return pairs;
}

std::optional<Sophus::SE3d> rigid_from(const std::vector<Correspondence> & pairs)
{
  if (pairs.size() < 3) {
    return std::nullopt;
  }
  Eigen::Matrix3Xd source(3, static_cast<Eigen::Index>(pairs.size()));
  Eigen::Matrix3Xd target(3, static_cast<Eigen::Index>(pairs.size()));
  for (std::size_t index = 0; index < pairs.size(); ++index) {
    source.col(static_cast<Eigen::Index>(index)) = pairs[index].source;
    target.col(static_cast<Eigen::Index>(index)) = pairs[index].target;
  }
  const Eigen::Matrix4d transform = Eigen::umeyama(source, target, false);
  const Eigen::Matrix3d rotation = transform.topLeftCorner<3, 3>();
  const Eigen::Vector3d translation = transform.topRightCorner<3, 1>();
  if (!rotation.allFinite() || !translation.allFinite() || rotation.determinant() <= 0.0) {
    return std::nullopt;
  }
  return Sophus::SE3d(Sophus::SO3d::fitToSO3(rotation), translation);
}

std::vector<std::size_t> inlier_indices(
  const std::vector<Correspondence> & pairs, const Sophus::SE3d & pose, const double threshold)
{
  const double threshold_squared = threshold * threshold;
  std::vector<std::size_t> indices;
  indices.reserve(pairs.size());
  for (std::size_t index = 0; index < pairs.size(); ++index) {
    const Eigen::Vector3d residual = pose * pairs[index].source - pairs[index].target;
    if (residual.squaredNorm() <= threshold_squared) {
      indices.push_back(index);
    }
  }
  return indices;
}

std::vector<Correspondence> gather(
  const std::vector<Correspondence> & pairs, const std::vector<std::size_t> & indices)
{
  std::vector<Correspondence> subset;
  subset.reserve(indices.size());
  for (const std::size_t index : indices) {
    subset.push_back(pairs[index]);
  }
  return subset;
}

bool better_hypothesis(
  const int count, const int iteration, const int best_count, const int best_iteration)
{
  return count > best_count || (count == best_count && iteration < best_iteration);
}

std::vector<std::size_t> ransac_inliers(
  const std::vector<Correspondence> & pairs, const int threads)
{
  const int pair_count = static_cast<int>(pairs.size());
  struct Hypothesis
  {
    int count{-1};
    int iteration{k_ransac_iterations};
    std::vector<std::size_t> indices;
  };
  std::vector<Hypothesis> local(static_cast<std::size_t>(threads));

#pragma omp parallel num_threads(threads)
  {
    Hypothesis & best = local[static_cast<std::size_t>(omp_get_thread_num())];
#pragma omp for schedule(static)
    for (int iteration = 0; iteration < k_ransac_iterations; ++iteration) {
      std::mt19937 generator(1000U + static_cast<unsigned int>(iteration));
      std::uniform_int_distribution<int> pick(0, pair_count - 1);
      const int first = pick(generator);
      int second = pick(generator);
      int third = pick(generator);
      if (second == first || third == first || third == second) {
        continue;
      }
      const Eigen::Vector3d edge_ab = pairs[static_cast<std::size_t>(second)].source -
                                      pairs[static_cast<std::size_t>(first)].source;
      const Eigen::Vector3d edge_ac = pairs[static_cast<std::size_t>(third)].source -
                                      pairs[static_cast<std::size_t>(first)].source;
      if (edge_ab.cross(edge_ac).norm() < 0.05) {
        continue;
      }
      const std::optional<Sophus::SE3d> pose = rigid_from(
        {pairs[static_cast<std::size_t>(first)], pairs[static_cast<std::size_t>(second)],
         pairs[static_cast<std::size_t>(third)]});
      if (!pose) {
        continue;
      }
      const std::vector<std::size_t> indices =
        inlier_indices(pairs, *pose, k_ransac_inlier_distance);
      const int count = static_cast<int>(indices.size());
      if (better_hypothesis(count, iteration, best.count, best.iteration)) {
        best.count = count;
        best.iteration = iteration;
        best.indices = indices;
      }
    }
  }

  Hypothesis best;
  for (const Hypothesis & hypothesis : local) {
    if (better_hypothesis(hypothesis.count, hypothesis.iteration, best.count, best.iteration)) {
      best = hypothesis;
    }
  }
  return best.indices;
}

std::optional<Sophus::SE3d> solve_rigid(
  const std::vector<Correspondence> & pairs, const int min_inliers, const int threads)
{
  if (static_cast<int>(pairs.size()) < min_inliers) {
    return std::nullopt;
  }
  const std::optional<Sophus::SE3d> svd_pose = rigid_from(pairs);
  if (!svd_pose) {
    return std::nullopt;
  }
  std::vector<std::size_t> inliers = inlier_indices(pairs, *svd_pose, k_ransac_inlier_distance);
  if (inliers.size() < pairs.size()) {
    const std::vector<std::size_t> sampled = ransac_inliers(pairs, threads);
    if (sampled.size() > inliers.size()) {
      inliers = sampled;
    }
  }
  if (static_cast<int>(inliers.size()) < min_inliers) {
    return std::nullopt;
  }
  if (inliers.size() == pairs.size()) {
    return svd_pose;
  }
  return rigid_from(gather(pairs, inliers));
}

}  // namespace

FpfhCoarseAlign::FpfhCoarseAlign(const FpfhParams & params) : params_(params)
{
}

std::optional<Sophus::SE3d> FpfhCoarseAlign::estimate(
  const std::vector<Eigen::Vector3d> & source_body, const std::vector<Eigen::Vector3d> & target_map,
  const Sophus::SE3d & imu_guess) const
{
  if (!params_.enabled || params_.min_inliers < 3) {
    return std::nullopt;
  }
  const int threads = thread_count(params_.omp_threads);
  const std::vector<Eigen::Vector3d> source_points = keypoints(source_body, params_);
  const std::vector<Eigen::Vector3d> target_points = keypoints(target_map, params_);
  if (
    static_cast<int>(source_points.size()) < params_.min_inliers ||
    static_cast<int>(target_points.size()) < params_.min_inliers) {
    return std::nullopt;
  }

  const std::optional<DescribedCloud> source =
    describe(source_points, Eigen::Vector3d::Zero(), params_, threads);
  const std::optional<DescribedCloud> target =
    describe(target_points, imu_guess.translation(), params_, threads);
  if (!source || !target) {
    return std::nullopt;
  }
  if (
    static_cast<int>(source->points.size()) < params_.min_inliers ||
    static_cast<int>(target->points.size()) < params_.min_inliers) {
    return std::nullopt;
  }

  const std::vector<Correspondence> pairs =
    mutual_matches(*source, *target, imu_guess, params_.correspondence_distance, threads);
  return solve_rigid(pairs, params_.min_inliers, threads);
}

}  // namespace back_odom
