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

#include "back_odom/device_kernels.hpp"

namespace back_odom
{

bool cuda_available()
{
  return false;
}

bool cuda_map_current(const void *, std::uint64_t, double)
{
  return false;
}

bool cuda_upload_map(const void *, std::uint64_t, double, const double *, const std::uint64_t *, int)
{
  return false;
}

bool cuda_plane_system(
  const double *, int frame_count, const double[9], const double[3], double, double, bool, bool,
  CudaPlaneSystem & system)
{
  system = CudaPlaneSystem{};
  system.point_count = frame_count;
  return false;
}

bool cuda_front_downsample(
  const double *, const double *, int, const CudaSegment *, int, double, double, const double[9],
  const double[3], const double[9], const double[3], const double[9], const double[3], const double[9],
  const double[3], double, double, double, std::vector<double> & kept_xyz, std::vector<int> & kept_index,
  int & cropped_count)
{
  kept_xyz.clear();
  kept_index.clear();
  cropped_count = 0;
  return false;
}

bool cuda_deskew(
  const double *, const double *, int, const CudaSegment *, int, double, double, const double[9],
  const double[3], const double[9], const double[3], const double[9], const double[3], const double[9],
  const double[3], std::vector<double> & deskewed_xyz)
{
  deskewed_xyz.clear();
  return false;
}

bool cuda_voxel_downsample(const double *, const std::uint64_t *, int, std::vector<double> & kept_xyz)
{
  kept_xyz.clear();
  return false;
}

}  // namespace back_odom
