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

BACK_ODOM_HD inline void unpack_voxel_key(std::uint64_t key, int & x, int & y, int & z)
{
  constexpr int bias = 1 << 20;
  const std::uint64_t mask = (1ULL << 21) - 1ULL;
  x = static_cast<int>((key >> 42) & mask) - bias;
  y = static_cast<int>((key >> 21) & mask) - bias;
  z = static_cast<int>(key & mask) - bias;
}

struct CudaSegment
{
  double start_stamp{0.0};
  double end_stamp{0.0};
  double rotation[9]{};
  double translation[3]{};
  double tangent[6]{};
};

/// Per-voxel plane for the device map, one per unique voxel key in sorted key order.
struct CudaVoxelPlane
{
  float nx{0.0F};
  float ny{0.0F};
  float nz{1.0F};
  /// 1 when the voxel has a usable normal.
  int planar{0};
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

/// Upload the map points, sort them by voxel key on the device and build the per-voxel offsets
/// there. `unique_keys` receives the sorted unique keys so the host can attach normals.
[[nodiscard]] bool cuda_upload_map(
  const void * owner, std::uint64_t epoch, double voxel_size, const double * xyz,
  const std::uint64_t * keys, int count, std::vector<std::uint64_t> & unique_keys);

/// Attach one plane per unique key (same order as returned by cuda_upload_map).
[[nodiscard]] bool cuda_upload_planes(const CudaVoxelPlane * planes, int unique_count);

/// One Gauss-Newton system. Neighbour ranking in float, residual/Jacobian/accumulation in double.
[[nodiscard]] bool cuda_plane_system(
  const double * frame_xyz, int frame_count, const double rotation[9], const double translation[3],
  double max_distance, double kernel_scale, bool upload_frame, CudaPlaneSystem & system);

/// Deskew, crop, then build the fine and coarse voxel levels on the device. Fine points come
/// back with their original input index (for intensity lookup); coarse points come back as xyz.
[[nodiscard]] bool cuda_front_downsample(
  const double * points_xyz, const double * stamps, int count, const CudaSegment * segments,
  int segment_count, double front_stamp, double back_stamp, const double front_rotation[9],
  const double front_translation[3], const double back_rotation[9],
  const double back_translation[3], const double end_inverse_rotation[9],
  const double end_inverse_translation[3], const double body_rotation[9],
  const double body_translation[3], double half_longitudinal, double half_lateral,
  double fine_voxel, double coarse_voxel, std::vector<double> & fine_xyz,
  std::vector<int> & fine_index, std::vector<double> & coarse_xyz);

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

}  // namespace back_odom

#endif  // BACK_ODOM__DEVICE_KERNELS_HPP_
