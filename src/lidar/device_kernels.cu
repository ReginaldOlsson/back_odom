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

#include "back_odom/lidar/device_kernels.hpp"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace back_odom
{
namespace
{

constexpr int k_threads = 128;
constexpr int k_max_blocks = 1024;
constexpr double k_sophus_epsilon_sq = 1.0e-20;
constexpr std::uint64_t k_outside_key = ~0ULL;

int g_failed = 0;

struct MapPoint
{
  std::uint64_t key;
  double x;
  double y;
  double z;
  float fx;
  float fy;
  float fz;
  float pad;
};

/// Persistent device copy of the local map. Grown on demand, never shrunk.
struct MapCache
{
  const void * owner{nullptr};
  std::uint64_t epoch{0};
  double voxel_size{0.0};
  int count{0};
  int unique{0};
  int point_capacity{0};
  MapPoint * points{nullptr};
  std::uint64_t * all_keys{nullptr};
  std::uint64_t * keys{nullptr};
  int * counts{nullptr};
  int * offsets{nullptr};
  CudaVoxelPlane * planes{nullptr};
  int plane_capacity{0};
};

MapCache g_map;

__constant__ int c_shifts[27][3];

const int k_host_shifts[27][3] = {
  {0, 0, 0},   {1, 0, 0},   {-1, 0, 0},  {0, 1, 0},   {0, -1, 0}, {0, 0, 1},
  {0, 0, -1},  {1, 1, 0},   {1, -1, 0},  {-1, 1, 0},  {-1, -1, 0}, {1, 0, 1},
  {1, 0, -1},  {-1, 0, 1},  {-1, 0, -1}, {0, 1, 1},   {0, 1, -1},  {0, -1, 1},
  {0, -1, -1}, {1, 1, 1},   {1, 1, -1},  {1, -1, 1},  {1, -1, -1}, {-1, 1, 1},
  {-1, 1, -1}, {-1, -1, 1}, {-1, -1, -1}};

void cuda_fail(const char * what, cudaError_t status)
{
  if (g_failed) {
    return;
  }
  g_failed = 1;
  std::fprintf(
    stderr, "back_odom CUDA disabled after %s: %s\n", what, cudaGetErrorString(status));
}

bool check(const char * what, cudaError_t status)
{
  if (status == cudaSuccess) {
    return true;
  }
  cuda_fail(what, status);
  return false;
}

template <typename T>
bool ensure_capacity(T *& pointer, int & capacity, const int needed)
{
  if (needed <= capacity && pointer != nullptr) {
    return true;
  }
  cudaFree(pointer);
  pointer = nullptr;
  capacity = 0;
  const int grown = std::max(needed, capacity + capacity / 2);
  if (!check("cudaMalloc", cudaMalloc(&pointer, static_cast<std::size_t>(grown) * sizeof(T)))) {
    return false;
  }
  capacity = grown;
  return true;
}

void release_map()
{
  cudaFree(g_map.points);
  cudaFree(g_map.all_keys);
  cudaFree(g_map.keys);
  cudaFree(g_map.counts);
  cudaFree(g_map.offsets);
  cudaFree(g_map.planes);
  g_map = MapCache{};
}

struct ByKey
{
  __host__ __device__ bool operator()(const MapPoint & left, const MapPoint & right) const
  {
    return left.key < right.key;
  }
};

struct KeyOf
{
  __host__ __device__ std::uint64_t operator()(const MapPoint & point) const { return point.key; }
};

__device__ void mat_vec(const double rotation[9], const double translation[3], const double point[3], double out[3])
{
  out[0] = rotation[0] * point[0] + rotation[1] * point[1] + rotation[2] * point[2] + translation[0];
  out[1] = rotation[3] * point[0] + rotation[4] * point[1] + rotation[5] * point[2] + translation[1];
  out[2] = rotation[6] * point[0] + rotation[7] * point[1] + rotation[8] * point[2] + translation[2];
}

__device__ void mat_mul(const double left[9], const double right[9], double out[9])
{
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      out[row * 3 + col] = left[row * 3] * right[col] + left[row * 3 + 1] * right[3 + col] +
                           left[row * 3 + 2] * right[6 + col];
    }
  }
}

__device__ void so3_exp(const double omega[3], double rotation[9], double * theta_out, int * small)
{
  const double theta_sq = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];
  double real_factor = 1.0;
  double imag_factor = 0.5;
  double theta = 0.0;
  if (theta_sq < k_sophus_epsilon_sq) {
    const double theta_po4 = theta_sq * theta_sq;
    imag_factor = 0.5 - (1.0 / 48.0) * theta_sq + (1.0 / 3840.0) * theta_po4;
    real_factor = 1.0 - (1.0 / 8.0) * theta_sq + (1.0 / 384.0) * theta_po4;
    *small = 1;
  } else {
    theta = sqrt(theta_sq);
    const double half = 0.5 * theta;
    imag_factor = sin(half) / theta;
    real_factor = cos(half);
    *small = 0;
  }
  *theta_out = theta;
  const double x = imag_factor * omega[0];
  const double y = imag_factor * omega[1];
  const double z = imag_factor * omega[2];
  const double w = real_factor;
  rotation[0] = 1.0 - 2.0 * (y * y + z * z);
  rotation[1] = 2.0 * (x * y - z * w);
  rotation[2] = 2.0 * (x * z + y * w);
  rotation[3] = 2.0 * (x * y + z * w);
  rotation[4] = 1.0 - 2.0 * (x * x + z * z);
  rotation[5] = 2.0 * (y * z - x * w);
  rotation[6] = 2.0 * (x * z - y * w);
  rotation[7] = 2.0 * (y * z + x * w);
  rotation[8] = 1.0 - 2.0 * (x * x + y * y);
}

__device__ void se3_exp(const double tangent[6], double rotation[9], double translation[3])
{
  const double omega[3] = {tangent[3], tangent[4], tangent[5]};
  int small = 0;
  double theta = 0.0;
  so3_exp(omega, rotation, &theta, &small);
  const double wx = omega[0];
  const double wy = omega[1];
  const double wz = omega[2];
  const double omega_hat[9] = {0.0, -wz, wy, wz, 0.0, -wx, -wy, wx, 0.0};
  double omega_sq[9];
  mat_mul(omega_hat, omega_hat, omega_sq);
  double V[9];
  if (small) {
    for (int index = 0; index < 9; ++index) {
      V[index] = 0.5 * omega_hat[index];
    }
    V[0] += 1.0;
    V[4] += 1.0;
    V[8] += 1.0;
  } else {
    const double theta_sq = theta * theta;
    const double a = (1.0 - cos(theta)) / theta_sq;
    const double b = (theta - sin(theta)) / (theta_sq * theta);
    for (int index = 0; index < 9; ++index) {
      V[index] = a * omega_hat[index] + b * omega_sq[index];
    }
    V[0] += 1.0;
    V[4] += 1.0;
    V[8] += 1.0;
  }
  const double upsilon[3] = {tangent[0], tangent[1], tangent[2]};
  translation[0] = V[0] * upsilon[0] + V[1] * upsilon[1] + V[2] * upsilon[2];
  translation[1] = V[3] * upsilon[0] + V[4] * upsilon[1] + V[5] * upsilon[2];
  translation[2] = V[6] * upsilon[0] + V[7] * upsilon[1] + V[8] * upsilon[2];
}

__device__ void compose(
  const double left_rotation[9], const double left_translation[3], const double right_rotation[9],
  const double right_translation[3], double rotation[9], double translation[3])
{
  mat_mul(left_rotation, right_rotation, rotation);
  double moved[3];
  const double zero[3] = {0.0, 0.0, 0.0};
  mat_vec(left_rotation, zero, right_translation, moved);
  translation[0] = left_translation[0] + moved[0];
  translation[1] = left_translation[1] + moved[1];
  translation[2] = left_translation[2] + moved[2];
}

__global__ void deskew_kernel(
  const double * points, const double * stamps, int count, const CudaSegment * segments,
  int segment_count, double front_stamp, double back_stamp, const double * front_rotation,
  const double * front_translation, const double * back_rotation, const double * back_translation,
  const double * end_inverse_rotation, const double * end_inverse_translation,
  const double * body_rotation, const double * body_translation, double * deskewed)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const double raw[3] = {points[3 * index], points[3 * index + 1], points[3 * index + 2]};
  double body_point[3];
  mat_vec(body_rotation, body_translation, raw, body_point);

  const double stamp = stamps[index];
  double pose_rotation[9];
  double pose_translation[3];
  if (segment_count == 0 || stamp <= front_stamp) {
    for (int item = 0; item < 9; ++item) {
      pose_rotation[item] = front_rotation[item];
    }
    for (int item = 0; item < 3; ++item) {
      pose_translation[item] = front_translation[item];
    }
  } else if (stamp >= back_stamp) {
    for (int item = 0; item < 9; ++item) {
      pose_rotation[item] = back_rotation[item];
    }
    for (int item = 0; item < 3; ++item) {
      pose_translation[item] = back_translation[item];
    }
  } else {
    int low = 0;
    int high = segment_count;
    while (low < high) {
      const int mid = (low + high) >> 1;
      if (segments[mid].end_stamp < stamp) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }
    if (low >= segment_count) {
      for (int item = 0; item < 9; ++item) {
        pose_rotation[item] = back_rotation[item];
      }
      for (int item = 0; item < 3; ++item) {
        pose_translation[item] = back_translation[item];
      }
    } else {
      const CudaSegment segment = segments[low];
      const double interval = segment.end_stamp - segment.start_stamp;
      const double alpha = interval > 1.0e-9 ? (stamp - segment.start_stamp) / interval : 0.0;
      double tangent[6];
      for (int item = 0; item < 6; ++item) {
        tangent[item] = alpha * segment.tangent[item];
      }
      double delta_rotation[9];
      double delta_translation[3];
      se3_exp(tangent, delta_rotation, delta_translation);
      compose(
        segment.rotation, segment.translation, delta_rotation, delta_translation, pose_rotation,
        pose_translation);
    }
  }

  double posed[3];
  mat_vec(pose_rotation, pose_translation, body_point, posed);
  double out[3];
  mat_vec(end_inverse_rotation, end_inverse_translation, posed, out);
  deskewed[3 * index] = out[0];
  deskewed[3 * index + 1] = out[1];
  deskewed[3 * index + 2] = out[2];
}

__device__ int find_voxel(const std::uint64_t * keys, int count, std::uint64_t key)
{
  int low = 0;
  int high = count;
  while (low < high) {
    const int mid = (low + high) >> 1;
    if (keys[mid] < key) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  if (low < count && keys[low] == key) {
    return low;
  }
  return -1;
}

__device__ int upper_index(int row, int col)
{
  return row * 6 - (row * (row - 1)) / 2 + (col - row);
}

struct Partial
{
  double jtj[21];
  double jtr[6];
  int correspondences;
  double cost;
};

/// Closest map point in the 27-voxel neighbourhood (float ranking), plane from that voxel's
/// cached normal, residual and Jacobian in double. Same association as align_to_surface.
__global__ void plane_kernel(
  const double * frame, int frame_count, const double * rotation, const double * translation,
  const MapPoint * points, const std::uint64_t * keys, const int * offsets,
  const CudaVoxelPlane * planes, int unique, double voxel_size, double max_distance,
  double kernel_scale, Partial * partials)
{
  Partial local{};
  const double kernel2 = kernel_scale * kernel_scale;
  const float max_distance_sq = static_cast<float>(max_distance * max_distance);
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < frame_count;
       index += blockDim.x * gridDim.x) {
    const double body[3] = {frame[3 * index], frame[3 * index + 1], frame[3 * index + 2]};
    double world[3];
    mat_vec(rotation, translation, body, world);
    const float qx = static_cast<float>(world[0]);
    const float qy = static_cast<float>(world[1]);
    const float qz = static_cast<float>(world[2]);
    const int base_x = static_cast<int>(floor(world[0] / voxel_size));
    const int base_y = static_cast<int>(floor(world[1] / voxel_size));
    const int base_z = static_cast<int>(floor(world[2] / voxel_size));

    float best = max_distance_sq;
    int best_point = -1;
    int best_voxel = -1;
    for (int shift = 0; shift < 27; ++shift) {
      const std::uint64_t key = pack_voxel_key(
        base_x + c_shifts[shift][0], base_y + c_shifts[shift][1], base_z + c_shifts[shift][2]);
      const int found = find_voxel(keys, unique, key);
      if (found < 0) {
        continue;
      }
      const int begin = offsets[found];
      const int end = offsets[found + 1];
      for (int cursor = begin; cursor < end; ++cursor) {
        const MapPoint point = points[cursor];
        const float dx = point.fx - qx;
        const float dy = point.fy - qy;
        const float dz = point.fz - qz;
        const float distance = dx * dx + dy * dy + dz * dz;
        if (distance < best) {
          best = distance;
          best_point = cursor;
          best_voxel = found;
        }
      }
    }

    double residual = max_distance;
    bool usable = false;
    double normal[3] = {0.0, 0.0, 0.0};
    if (best_point >= 0) {
      const CudaVoxelPlane plane = planes[best_voxel];
      if (plane.planar != 0) {
        const MapPoint closest = points[best_point];
        normal[0] = plane.nx;
        normal[1] = plane.ny;
        normal[2] = plane.nz;
        residual = normal[0] * (world[0] - closest.x) + normal[1] * (world[1] - closest.y) +
                   normal[2] * (world[2] - closest.z);
        usable = true;
      }
    }
    const double denom = kernel_scale + residual * residual;
    local.cost += kernel2 * residual * residual / denom;
    if (!usable) {
      continue;
    }
    const double normal_body[3] = {
      rotation[0] * normal[0] + rotation[3] * normal[1] + rotation[6] * normal[2],
      rotation[1] * normal[0] + rotation[4] * normal[1] + rotation[7] * normal[2],
      rotation[2] * normal[0] + rotation[5] * normal[1] + rotation[8] * normal[2]};
    const double jacobian[6] = {
      normal_body[0],
      normal_body[1],
      normal_body[2],
      body[1] * normal_body[2] - body[2] * normal_body[1],
      body[2] * normal_body[0] - body[0] * normal_body[2],
      body[0] * normal_body[1] - body[1] * normal_body[0]};
    const double weight = kernel2 / (denom * denom);
    ++local.correspondences;
    for (int row = 0; row < 6; ++row) {
      local.jtr[row] += jacobian[row] * weight * residual;
      for (int col = row; col < 6; ++col) {
        local.jtj[upper_index(row, col)] += weight * jacobian[row] * jacobian[col];
      }
    }
  }

  __shared__ double shared_jtj[k_threads][21];
  __shared__ double shared_jtr[k_threads][6];
  __shared__ int shared_count[k_threads];
  __shared__ double shared_cost[k_threads];
  const int thread = threadIdx.x;
  for (int item = 0; item < 21; ++item) {
    shared_jtj[thread][item] = local.jtj[item];
  }
  for (int item = 0; item < 6; ++item) {
    shared_jtr[thread][item] = local.jtr[item];
  }
  shared_count[thread] = local.correspondences;
  shared_cost[thread] = local.cost;
  __syncthreads();
  for (int stride = k_threads / 2; stride > 0; stride >>= 1) {
    if (thread < stride) {
      for (int item = 0; item < 21; ++item) {
        shared_jtj[thread][item] += shared_jtj[thread + stride][item];
      }
      for (int item = 0; item < 6; ++item) {
        shared_jtr[thread][item] += shared_jtr[thread + stride][item];
      }
      shared_count[thread] += shared_count[thread + stride];
      shared_cost[thread] += shared_cost[thread + stride];
    }
    __syncthreads();
  }
  if (thread == 0) {
    Partial & out = partials[blockIdx.x];
    for (int item = 0; item < 21; ++item) {
      out.jtj[item] = shared_jtj[0][item];
    }
    for (int item = 0; item < 6; ++item) {
      out.jtr[item] = shared_jtr[0][item];
    }
    out.correspondences = shared_count[0];
    out.cost = shared_cost[0];
  }
}

int upper_host(int row, int col)
{
  return row * 6 - (row * (row - 1)) / 2 + (col - row);
}

struct FrontPoint
{
  std::uint64_t key;
  double x;
  double y;
  double z;
  int index;
  int pad;
};

struct FrontScratch
{
  double * raw{nullptr};
  double * stamps{nullptr};
  double * deskewed{nullptr};
  FrontPoint * tagged{nullptr};
  FrontPoint * coarse{nullptr};
  CudaSegment * segments{nullptr};
  double * pose{nullptr};
  int point_capacity{0};
  int segment_capacity{0};
};

FrontScratch g_front;

struct ByFrontKey
{
  __host__ __device__ bool operator()(const FrontPoint & left, const FrontPoint & right) const
  {
    return left.key < right.key;
  }
};

struct SameFrontKey
{
  __host__ __device__ bool operator()(const FrontPoint & left, const FrontPoint & right) const
  {
    return left.key == right.key;
  }
};

__global__ void tag_front_kernel(
  const double * deskewed, int count, double voxel_size, double half_longitudinal, double half_lateral,
  FrontPoint * tagged)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const double x = deskewed[3 * index];
  const double y = deskewed[3 * index + 1];
  const double z = deskewed[3 * index + 2];
  FrontPoint & point = tagged[index];
  point.x = x;
  point.y = y;
  point.z = z;
  point.index = index;
  if (fabs(x) > half_longitudinal || fabs(y) > half_lateral) {
    point.key = k_outside_key;
    return;
  }
  point.key = pack_voxel_key(
    static_cast<int>(floor(x / voxel_size)), static_cast<int>(floor(y / voxel_size)),
    static_cast<int>(floor(z / voxel_size)));
}

__global__ void retag_kernel(const FrontPoint * fine, int count, double voxel_size, FrontPoint * coarse)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  FrontPoint point = fine[index];
  if (point.key != k_outside_key) {
    point.key = pack_voxel_key(
      static_cast<int>(floor(point.x / voxel_size)), static_cast<int>(floor(point.y / voxel_size)),
      static_cast<int>(floor(point.z / voxel_size)));
  }
  coarse[index] = point;
}

bool ensure_front(const int count, const int segment_count)
{
  if (g_front.pose == nullptr &&
      !check("cudaMalloc", cudaMalloc(&g_front.pose, 48 * sizeof(double)))) {
    return false;
  }
  if (count > g_front.point_capacity) {
    cudaFree(g_front.raw);
    cudaFree(g_front.stamps);
    cudaFree(g_front.deskewed);
    cudaFree(g_front.tagged);
    cudaFree(g_front.coarse);
    g_front.raw = nullptr;
    g_front.stamps = nullptr;
    g_front.deskewed = nullptr;
    g_front.tagged = nullptr;
    g_front.coarse = nullptr;
    g_front.point_capacity = 0;
    const int grown = count + count / 4;
    const bool allocated =
      check("cudaMalloc", cudaMalloc(&g_front.raw, static_cast<std::size_t>(grown) * 3 * sizeof(double))) &&
      check("cudaMalloc", cudaMalloc(&g_front.stamps, static_cast<std::size_t>(grown) * sizeof(double))) &&
      check(
        "cudaMalloc", cudaMalloc(&g_front.deskewed, static_cast<std::size_t>(grown) * 3 * sizeof(double))) &&
      check("cudaMalloc", cudaMalloc(&g_front.tagged, static_cast<std::size_t>(grown) * sizeof(FrontPoint))) &&
      check("cudaMalloc", cudaMalloc(&g_front.coarse, static_cast<std::size_t>(grown) * sizeof(FrontPoint)));
    if (!allocated) {
      return false;
    }
    g_front.point_capacity = grown;
  }
  if (segment_count > g_front.segment_capacity) {
    cudaFree(g_front.segments);
    g_front.segments = nullptr;
    g_front.segment_capacity = 0;
    if (segment_count > 0 &&
        !check(
          "cudaMalloc",
          cudaMalloc(&g_front.segments, static_cast<std::size_t>(segment_count) * sizeof(CudaSegment)))) {
      return false;
    }
    g_front.segment_capacity = segment_count;
  }
  return true;
}

}  // namespace

bool cuda_available()
{
  if (g_failed) {
    return false;
  }
  static int ready = -1;
  if (ready >= 0) {
    return ready == 1;
  }
  int count = 0;
  if (!check("cudaGetDeviceCount", cudaGetDeviceCount(&count)) || count < 1) {
    ready = 0;
    g_failed = 0;
    return false;
  }
  if (!check("cudaSetDevice", cudaSetDevice(0)) || !check("cudaFree", cudaFree(nullptr))) {
    ready = 0;
    return false;
  }
  if (!check(
        "cudaMemcpyToSymbol",
        cudaMemcpyToSymbol(c_shifts, k_host_shifts, sizeof(k_host_shifts)))) {
    ready = 0;
    return false;
  }
  ready = 1;
  return true;
}

bool cuda_map_current(const void * owner, const std::uint64_t epoch, const double voxel_size)
{
  return g_map.owner == owner && g_map.epoch == epoch && g_map.voxel_size == voxel_size &&
         g_map.points != nullptr;
}

bool cuda_upload_map(
  const void * owner, const std::uint64_t epoch, const double voxel_size, const double * xyz,
  const std::uint64_t * keys, const int count, std::vector<std::uint64_t> & unique_keys)
{
  unique_keys.clear();
  if (!cuda_available()) {
    return false;
  }
  if (count <= 0) {
    g_map.owner = owner;
    g_map.epoch = epoch;
    g_map.voxel_size = voxel_size;
    g_map.count = 0;
    g_map.unique = 0;
    return true;
  }
  std::vector<MapPoint> host(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    MapPoint & point = host[static_cast<std::size_t>(index)];
    point.key = keys[index];
    point.x = xyz[3 * index];
    point.y = xyz[3 * index + 1];
    point.z = xyz[3 * index + 2];
    point.fx = static_cast<float>(point.x);
    point.fy = static_cast<float>(point.y);
    point.fz = static_cast<float>(point.z);
    point.pad = 0.0F;
  }
  // Persistent buffers sized to the map; all per-point arrays share one capacity.
  if (count > g_map.point_capacity || g_map.points == nullptr) {
    const int grown = std::max(count + count / 4, g_map.point_capacity);
    CudaVoxelPlane * planes = g_map.planes;
    const int plane_capacity = g_map.plane_capacity;
    g_map.planes = nullptr;
    release_map();
    g_map.planes = planes;
    g_map.plane_capacity = plane_capacity;
    const std::size_t capacity = static_cast<std::size_t>(grown) + 1;
    if (
      !check("cudaMalloc", cudaMalloc(&g_map.points, capacity * sizeof(MapPoint))) ||
      !check("cudaMalloc", cudaMalloc(&g_map.all_keys, capacity * sizeof(std::uint64_t))) ||
      !check("cudaMalloc", cudaMalloc(&g_map.keys, capacity * sizeof(std::uint64_t))) ||
      !check("cudaMalloc", cudaMalloc(&g_map.counts, capacity * sizeof(int))) ||
      !check("cudaMalloc", cudaMalloc(&g_map.offsets, capacity * sizeof(int)))) {
      release_map();
      return false;
    }
    g_map.point_capacity = grown;
  }
  g_map.owner = owner;
  g_map.epoch = epoch;
  g_map.voxel_size = voxel_size;
  g_map.count = count;
  if (!check(
        "cudaMemcpy",
        cudaMemcpy(
          g_map.points, host.data(), static_cast<std::size_t>(count) * sizeof(MapPoint),
          cudaMemcpyHostToDevice))) {
    release_map();
    return false;
  }
  try {
    thrust::device_ptr<MapPoint> points(g_map.points);
    thrust::device_ptr<std::uint64_t> all_keys(g_map.all_keys);
    thrust::device_ptr<std::uint64_t> keys_out(g_map.keys);
    thrust::device_ptr<int> counts(g_map.counts);
    thrust::device_ptr<int> offsets(g_map.offsets);
    thrust::stable_sort(thrust::device, points, points + count, ByKey{});
    thrust::transform(thrust::device, points, points + count, all_keys, KeyOf{});
    const auto ends = thrust::reduce_by_key(
      thrust::device, all_keys, all_keys + count, thrust::constant_iterator<int>(1), keys_out, counts);
    const int unique = static_cast<int>(ends.first - keys_out);
    thrust::exclusive_scan(thrust::device, counts, counts + unique, offsets);
    // offsets[unique] = count
    if (!check("cudaMemcpy", cudaMemcpy(g_map.offsets + unique, &count, sizeof(int), cudaMemcpyHostToDevice))) {
      release_map();
      return false;
    }
    g_map.unique = unique;
    unique_keys.resize(static_cast<std::size_t>(unique));
    if (!check(
          "cudaMemcpy",
          cudaMemcpy(
            unique_keys.data(), g_map.keys, static_cast<std::size_t>(unique) * sizeof(std::uint64_t),
            cudaMemcpyDeviceToHost))) {
      release_map();
      return false;
    }
  } catch (const std::exception & error) {
    std::fprintf(stderr, "back_odom CUDA disabled after map sort: %s\n", error.what());
    g_failed = 1;
    release_map();
    return false;
  }
  return true;
}

bool cuda_upload_planes(const CudaVoxelPlane * planes, const int unique_count)
{
  if (!cuda_available() || unique_count != g_map.unique) {
    return false;
  }
  if (unique_count <= 0) {
    return true;
  }
  if (!ensure_capacity(g_map.planes, g_map.plane_capacity, unique_count)) {
    return false;
  }
  return check(
    "cudaMemcpy",
    cudaMemcpy(
      g_map.planes, planes, static_cast<std::size_t>(unique_count) * sizeof(CudaVoxelPlane),
      cudaMemcpyHostToDevice));
}

namespace
{

struct PlaneScratch
{
  double * frame{nullptr};
  int frame_capacity{0};
  int frame_count{0};
  Partial * partials{nullptr};
  double * pose{nullptr};
  std::vector<Partial> host_partials;
};

PlaneScratch g_plane;

bool plane_scratch(const int frame_count)
{
  if (g_plane.pose == nullptr) {
    if (!check("cudaMalloc", cudaMalloc(&g_plane.pose, 12 * sizeof(double))) ||
        !check("cudaMalloc", cudaMalloc(&g_plane.partials, static_cast<std::size_t>(k_max_blocks) * sizeof(Partial)))) {
      return false;
    }
  }
  if (frame_count > g_plane.frame_capacity) {
    cudaFree(g_plane.frame);
    g_plane.frame = nullptr;
    g_plane.frame_capacity = 0;
    g_plane.frame_count = 0;
    const int grown = frame_count + frame_count / 4;
    if (!check(
          "cudaMalloc",
          cudaMalloc(&g_plane.frame, static_cast<std::size_t>(grown) * 3 * sizeof(double)))) {
      return false;
    }
    g_plane.frame_capacity = grown;
  }
  return g_plane.frame != nullptr && g_plane.partials != nullptr;
}

}  // namespace

bool cuda_plane_system(
  const double * frame_xyz, const int frame_count, const double rotation[9],
  const double translation[3], const double max_distance, const double kernel_scale,
  const bool upload_frame, CudaPlaneSystem & system)
{
  system = CudaPlaneSystem{};
  system.point_count = frame_count;
  if (!cuda_available() || frame_count <= 0) {
    return false;
  }
  if (g_map.unique <= 0) {
    return true;  // empty map: no correspondences, zero system
  }
  if (g_map.points == nullptr || g_map.planes == nullptr) {
    return false;
  }
  const int blocks = std::max(1, std::min(k_max_blocks, (frame_count + k_threads - 1) / k_threads));
  if (!plane_scratch(frame_count)) {
    return false;
  }
  if (upload_frame || g_plane.frame_count != frame_count) {
    if (!check(
          "cudaMemcpy",
          cudaMemcpy(
            g_plane.frame, frame_xyz, static_cast<std::size_t>(frame_count) * 3 * sizeof(double),
            cudaMemcpyHostToDevice))) {
      g_plane.frame_count = 0;
      return false;
    }
    g_plane.frame_count = frame_count;
  }
  double pose[12];
  std::memcpy(pose, rotation, 9 * sizeof(double));
  std::memcpy(pose + 9, translation, 3 * sizeof(double));
  if (!check("cudaMemcpy", cudaMemcpy(g_plane.pose, pose, 12 * sizeof(double), cudaMemcpyHostToDevice))) {
    return false;
  }
  plane_kernel<<<blocks, k_threads>>>(
    g_plane.frame, frame_count, g_plane.pose, g_plane.pose + 9, g_map.points, g_map.keys,
    g_map.offsets, g_map.planes, g_map.unique, g_map.voxel_size, max_distance, kernel_scale,
    g_plane.partials);
  if (!check("plane_kernel", cudaGetLastError())) {
    return false;
  }
  g_plane.host_partials.resize(static_cast<std::size_t>(blocks));
  if (!check(
        "cudaMemcpy",
        cudaMemcpy(
          g_plane.host_partials.data(), g_plane.partials, static_cast<std::size_t>(blocks) * sizeof(Partial),
          cudaMemcpyDeviceToHost))) {
    return false;
  }
  double upper[21] = {};
  for (const Partial & partial : g_plane.host_partials) {
    for (int item = 0; item < 21; ++item) {
      upper[item] += partial.jtj[item];
    }
    for (int item = 0; item < 6; ++item) {
      system.jtr[item] += partial.jtr[item];
    }
    system.correspondences += partial.correspondences;
    system.robust_cost_sum += partial.cost;
  }
  for (int row = 0; row < 6; ++row) {
    for (int col = row; col < 6; ++col) {
      const double value = upper[upper_host(row, col)];
      system.jtj[row * 6 + col] = value;
      system.jtj[col * 6 + row] = value;
    }
  }
  return true;
}

bool cuda_deskew(
  const double * points_xyz, const double * stamps, const int count, const CudaSegment * segments,
  const int segment_count, const double front_stamp, const double back_stamp,
  const double front_rotation[9], const double front_translation[3], const double back_rotation[9],
  const double back_translation[3], const double end_inverse_rotation[9],
  const double end_inverse_translation[3], const double body_rotation[9],
  const double body_translation[3], std::vector<double> & deskewed_xyz)
{
  deskewed_xyz.clear();
  if (!cuda_available() || count <= 0) {
    return cuda_available();
  }
  if (!ensure_front(count, segment_count)) {
    return false;
  }
  auto upload_pose = [&](const double rotation[9], const double translation[3], double * slot) {
    return check("cudaMemcpy", cudaMemcpy(slot, rotation, 9 * sizeof(double), cudaMemcpyHostToDevice)) &&
           check("cudaMemcpy", cudaMemcpy(slot + 9, translation, 3 * sizeof(double), cudaMemcpyHostToDevice));
  };
  double * front = g_front.pose;
  double * back = g_front.pose + 12;
  double * end_inverse = g_front.pose + 24;
  double * body = g_front.pose + 36;
  const bool copied =
    check(
      "cudaMemcpy",
      cudaMemcpy(
        g_front.raw, points_xyz, static_cast<std::size_t>(count) * 3 * sizeof(double),
        cudaMemcpyHostToDevice)) &&
    check(
      "cudaMemcpy",
      cudaMemcpy(
        g_front.stamps, stamps, static_cast<std::size_t>(count) * sizeof(double), cudaMemcpyHostToDevice)) &&
    upload_pose(front_rotation, front_translation, front) &&
    upload_pose(back_rotation, back_translation, back) &&
    upload_pose(end_inverse_rotation, end_inverse_translation, end_inverse) &&
    upload_pose(body_rotation, body_translation, body) &&
    (segment_count <= 0 ||
     check(
       "cudaMemcpy",
       cudaMemcpy(
         g_front.segments, segments, static_cast<std::size_t>(segment_count) * sizeof(CudaSegment),
         cudaMemcpyHostToDevice)));
  if (!copied) {
    return false;
  }
  const int blocks = (count + k_threads - 1) / k_threads;
  deskew_kernel<<<blocks, k_threads>>>(
    g_front.raw, g_front.stamps, count, g_front.segments, segment_count, front_stamp, back_stamp, front,
    front + 9, back, back + 9, end_inverse, end_inverse + 9, body, body + 9, g_front.deskewed);
  deskewed_xyz.resize(static_cast<std::size_t>(count) * 3);
  const bool ok = check("deskew_kernel", cudaGetLastError()) &&
                  check(
                    "cudaMemcpy",
                    cudaMemcpy(
                      deskewed_xyz.data(), g_front.deskewed,
                      static_cast<std::size_t>(count) * 3 * sizeof(double), cudaMemcpyDeviceToHost));
  if (!ok) {
    deskewed_xyz.clear();
  }
  return ok;
}

bool cuda_front_downsample(
  const double * points_xyz, const double * stamps, const int count, const CudaSegment * segments,
  const int segment_count, const double front_stamp, const double back_stamp,
  const double front_rotation[9], const double front_translation[3], const double back_rotation[9],
  const double back_translation[3], const double end_inverse_rotation[9],
  const double end_inverse_translation[3], const double body_rotation[9],
  const double body_translation[3], const double half_longitudinal, const double half_lateral,
  const double fine_voxel, const double coarse_voxel, std::vector<double> & fine_xyz,
  std::vector<int> & fine_index, std::vector<double> & coarse_xyz)
{
  fine_xyz.clear();
  fine_index.clear();
  coarse_xyz.clear();
  if (!cuda_available() || count <= 0 || !(fine_voxel > 0.0) || !(coarse_voxel > 0.0)) {
    return cuda_available() && count > 0 && fine_voxel > 0.0 && coarse_voxel > 0.0;
  }
  if (!ensure_front(count, segment_count)) {
    return false;
  }
  auto upload_pose = [&](const double rotation[9], const double translation[3], double * slot) {
    return check("cudaMemcpy", cudaMemcpy(slot, rotation, 9 * sizeof(double), cudaMemcpyHostToDevice)) &&
           check("cudaMemcpy", cudaMemcpy(slot + 9, translation, 3 * sizeof(double), cudaMemcpyHostToDevice));
  };
  double * front = g_front.pose;
  double * back = g_front.pose + 12;
  double * end_inverse = g_front.pose + 24;
  double * body = g_front.pose + 36;
  const bool copied =
    check(
      "cudaMemcpy",
      cudaMemcpy(
        g_front.raw, points_xyz, static_cast<std::size_t>(count) * 3 * sizeof(double),
        cudaMemcpyHostToDevice)) &&
    check(
      "cudaMemcpy",
      cudaMemcpy(
        g_front.stamps, stamps, static_cast<std::size_t>(count) * sizeof(double), cudaMemcpyHostToDevice)) &&
    upload_pose(front_rotation, front_translation, front) &&
    upload_pose(back_rotation, back_translation, back) &&
    upload_pose(end_inverse_rotation, end_inverse_translation, end_inverse) &&
    upload_pose(body_rotation, body_translation, body) &&
    (segment_count <= 0 ||
     check(
       "cudaMemcpy",
       cudaMemcpy(
         g_front.segments, segments, static_cast<std::size_t>(segment_count) * sizeof(CudaSegment),
         cudaMemcpyHostToDevice)));
  if (!copied) {
    return false;
  }
  const int blocks = (count + k_threads - 1) / k_threads;
  deskew_kernel<<<blocks, k_threads>>>(
    g_front.raw, g_front.stamps, count, g_front.segments, segment_count, front_stamp, back_stamp, front,
    front + 9, back, back + 9, end_inverse, end_inverse + 9, body, body + 9, g_front.deskewed);
  tag_front_kernel<<<blocks, k_threads>>>(
    g_front.deskewed, count, fine_voxel, half_longitudinal, half_lateral, g_front.tagged);
  if (!check("front_kernels", cudaGetLastError())) {
    return false;
  }
  try {
    thrust::device_ptr<FrontPoint> tagged(g_front.tagged);
    thrust::device_ptr<FrontPoint> coarse(g_front.coarse);
    // Fine level: stable sort keeps the first input point per voxel, like kiss VoxelDownsample.
    thrust::stable_sort(thrust::device, tagged, tagged + count, ByFrontKey{});
    const auto fine_end = thrust::unique(thrust::device, tagged, tagged + count, SameFrontKey{});
    int fine_count = static_cast<int>(fine_end - tagged);
    // Points outside the box share k_outside_key, which sorts last; one survivor is left.
    std::vector<FrontPoint> host(static_cast<std::size_t>(std::max(fine_count, 0)));
    if (fine_count > 0 &&
        !check(
          "cudaMemcpy",
          cudaMemcpy(
            host.data(), g_front.tagged, static_cast<std::size_t>(fine_count) * sizeof(FrontPoint),
            cudaMemcpyDeviceToHost))) {
      return false;
    }
    if (!host.empty() && host.back().key == k_outside_key) {
      host.pop_back();
      --fine_count;
    }
    fine_xyz.reserve(host.size() * 3);
    fine_index.reserve(host.size());
    for (const FrontPoint & point : host) {
      fine_xyz.push_back(point.x);
      fine_xyz.push_back(point.y);
      fine_xyz.push_back(point.z);
      fine_index.push_back(point.index);
    }
    if (fine_count <= 0) {
      return true;
    }
    // Coarse level from the fine survivors.
    const int coarse_blocks = (fine_count + k_threads - 1) / k_threads;
    retag_kernel<<<coarse_blocks, k_threads>>>(g_front.tagged, fine_count, coarse_voxel, g_front.coarse);
    if (!check("retag_kernel", cudaGetLastError())) {
      return false;
    }
    thrust::stable_sort(thrust::device, coarse, coarse + fine_count, ByFrontKey{});
    const auto coarse_end = thrust::unique(thrust::device, coarse, coarse + fine_count, SameFrontKey{});
    const int coarse_count = static_cast<int>(coarse_end - coarse);
    std::vector<FrontPoint> coarse_host(static_cast<std::size_t>(std::max(coarse_count, 0)));
    if (coarse_count > 0 &&
        !check(
          "cudaMemcpy",
          cudaMemcpy(
            coarse_host.data(), g_front.coarse, static_cast<std::size_t>(coarse_count) * sizeof(FrontPoint),
            cudaMemcpyDeviceToHost))) {
      return false;
    }
    coarse_xyz.reserve(coarse_host.size() * 3);
    for (const FrontPoint & point : coarse_host) {
      coarse_xyz.push_back(point.x);
      coarse_xyz.push_back(point.y);
      coarse_xyz.push_back(point.z);
    }
  } catch (const std::exception & error) {
    std::fprintf(stderr, "back_odom CUDA disabled after front downsample: %s\n", error.what());
    g_failed = 1;
    fine_xyz.clear();
    fine_index.clear();
    coarse_xyz.clear();
    return false;
  }
  return true;
}

bool cuda_voxel_downsample(
  const double * xyz, const std::uint64_t * keys, const int count, std::vector<double> & kept_xyz)
{
  kept_xyz.clear();
  if (!cuda_available() || count <= 0) {
    return cuda_available();
  }
  std::vector<MapPoint> host(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    host[static_cast<std::size_t>(index)].key = keys[index];
    host[static_cast<std::size_t>(index)].x = xyz[3 * index];
    host[static_cast<std::size_t>(index)].y = xyz[3 * index + 1];
    host[static_cast<std::size_t>(index)].z = xyz[3 * index + 2];
  }
  MapPoint * device = nullptr;
  if (!check("cudaMalloc", cudaMalloc(&device, static_cast<std::size_t>(count) * sizeof(MapPoint)))) {
    return false;
  }
  if (!check(
        "cudaMemcpy",
        cudaMemcpy(
          device, host.data(), static_cast<std::size_t>(count) * sizeof(MapPoint),
          cudaMemcpyHostToDevice))) {
    cudaFree(device);
    return false;
  }
  try {
    thrust::stable_sort(thrust::device, device, device + count, ByKey{});
  } catch (const std::exception & error) {
    std::fprintf(stderr, "back_odom CUDA disabled after voxel sort: %s\n", error.what());
    g_failed = 1;
    cudaFree(device);
    return false;
  }
  if (!check(
        "cudaMemcpy",
        cudaMemcpy(
          host.data(), device, static_cast<std::size_t>(count) * sizeof(MapPoint),
          cudaMemcpyDeviceToHost))) {
    cudaFree(device);
    return false;
  }
  cudaFree(device);
  kept_xyz.reserve(static_cast<std::size_t>(count) * 3);
  std::uint64_t previous = 0;
  for (int index = 0; index < count; ++index) {
    const MapPoint & point = host[static_cast<std::size_t>(index)];
    if (index > 0 && point.key == previous) {
      continue;
    }
    previous = point.key;
    kept_xyz.push_back(point.x);
    kept_xyz.push_back(point.y);
    kept_xyz.push_back(point.z);
  }
  return true;
}

}  // namespace back_odom
