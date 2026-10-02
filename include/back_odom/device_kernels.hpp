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

#ifndef BACK_ODOM__DEVICE_KERNELS_HPP_
#define BACK_ODOM__DEVICE_KERNELS_HPP_

#include <cstdint>
#include <vector>

#ifdef __CUDACC__
#define BACK_ODOM_HD __host__ __device__
#else
#define BACK_ODOM_HD
#endif

namespace back_odom
{

/// 21 bits per axis, biased so negative voxel indices stay unique.
BACK_ODOM_HD inline std::uint64_t pack_voxel_key(int x, int y, int z)
{
  constexpr int bias = 1 << 20;
  const std::uint64_t mask = (1ULL << 21) - 1ULL;
  const std::uint64_t ux = static_cast<std::uint64_t>(x + bias) & mask;
  const std::uint64_t uy = static_cast<std::uint64_t>(y + bias) & mask;
  const std::uint64_t uz = static_cast<std::uint64_t>(z + bias) & mask;
  return (ux << 42) | (uy << 21) | uz;
}

struct CudaSegment
{
  double start_stamp{0.0};
  double end_stamp{0.0};
  double rotation[9]{};
  double translation[3]{};
  double tangent[6]{};
};

struct CudaPlaneSystem
{
  double jtj[36]{};
  double jtr[6]{};
  int correspondences{0};
  double robust_cost_sum{0.0};
  int point_count{0};
};

[[nodiscard]] bool cuda_available();

[[nodiscard]] bool cuda_map_current(const void * owner, std::uint64_t epoch, double voxel_size);

[[nodiscard]] bool cuda_upload_map(
  const void * owner, std::uint64_t epoch, double voxel_size, const double * xyz,
  const std::uint64_t * keys, int count);

[[nodiscard]] bool cuda_plane_system(
  const double * frame_xyz, int frame_count, const double rotation[9], const double translation[3],
  double max_distance, double kernel_scale, bool float_rank, bool upload_frame,
  CudaPlaneSystem & system);

/// Deskew, crop, and keep the first point in each voxel. The full cloud stays on the device.
/// `kept_xyz` / `kept_index` are that downsample, in input order. `cropped_count` is the number of
/// points inside the box, which the CPU hash needs so its iteration order matches kiss-icp.
[[nodiscard]] bool cuda_front_downsample(
  const double * points_xyz, const double * stamps, int count, const CudaSegment * segments,
  int segment_count, double front_stamp, double back_stamp, const double front_rotation[9],
  const double front_translation[3], const double back_rotation[9],
  const double back_translation[3], const double end_inverse_rotation[9],
  const double end_inverse_translation[3], const double body_rotation[9],
  const double body_translation[3], double half_longitudinal, double half_lateral, double voxel_size,
  std::vector<double> & kept_xyz, std::vector<int> & kept_index, int & cropped_count);

[[nodiscard]] bool cuda_deskew(
  const double * points_xyz, const double * stamps, int count, const CudaSegment * segments,
  int segment_count, double front_stamp, double back_stamp, const double front_rotation[9],
  const double front_translation[3], const double back_rotation[9],
  const double back_translation[3], const double end_inverse_rotation[9],
  const double end_inverse_translation[3], const double body_rotation[9],
  const double body_translation[3], std::vector<double> & deskewed_xyz);

/// First input point in each voxel. Output order follows the voxel key, not the CPU hash order.
[[nodiscard]] bool cuda_voxel_downsample(
  const double * xyz, const std::uint64_t * keys, int count, std::vector<double> & kept_xyz);

[[nodiscard]] bool cuda_histogram_matches(
  const float * query_hist, int query_count, const float * reference_hist, int reference_count,
  const double * query_xyz, const double * reference_xyz, double gate_squared,
  std::vector<int> & matches);

}  // namespace back_odom

#endif  // BACK_ODOM__DEVICE_KERNELS_HPP_
