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

#ifndef BACK_ODOM__FPFH_COARSE_ALIGN_HPP_
#define BACK_ODOM__FPFH_COARSE_ALIGN_HPP_

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace back_odom
{

struct FpfhParams
{
  bool enabled{true};
  double keypoint_voxel{1.5};
  double normal_radius{2.0};
  double fpfh_radius{5.0};
  int max_keypoints{1500};
  double correspondence_distance{5.0};
  int min_inliers{20};
  int omp_threads{0};
};

/// Coarse map-from-body pose from OpenMP FPFH correspondences.
/// Empty when the clouds do not support a rigid guess. The point-to-plane aligner still refines it.
class FpfhCoarseAlign
{
public:
  explicit FpfhCoarseAlign(const FpfhParams & params);
  ~FpfhCoarseAlign();
  FpfhCoarseAlign(const FpfhCoarseAlign &) = delete;
  FpfhCoarseAlign & operator=(const FpfhCoarseAlign &) = delete;

  /// True when this map generation and viewpoint already have target descriptors.
  [[nodiscard]] bool target_cached(std::uint64_t map_epoch, const Eigen::Vector3d & viewpoint) const;

  /// `source_body` is the deskewed scan. `target_map` is the local map.
  /// `imu_guess` maps body points into the map and rejects distant feature pairs.
  /// A non-zero `map_epoch` reuses the target descriptors until the map or viewpoint changes.
  [[nodiscard]] std::optional<Sophus::SE3d> estimate(
    const std::vector<Eigen::Vector3d> & source_body,
    const std::vector<Eigen::Vector3d> & target_map, const Sophus::SE3d & imu_guess,
    std::uint64_t map_epoch = 0) const;

private:
  struct TargetCache;
  FpfhParams params_;
  mutable std::unique_ptr<TargetCache> target_cache_;
};

}  // namespace back_odom

#endif  // BACK_ODOM__FPFH_COARSE_ALIGN_HPP_
