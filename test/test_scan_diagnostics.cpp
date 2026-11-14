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

#include "back_odom/node/scan_diagnostics.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace
{

std::string value_of(
  const std::vector<diagnostic_msgs::msg::DiagnosticStatus> & statuses, const std::string & name,
  const std::string & key)
{
  for (const diagnostic_msgs::msg::DiagnosticStatus & status : statuses) {
    if (status.name != name) {
      continue;
    }
    for (const diagnostic_msgs::msg::KeyValue & entry : status.values) {
      if (entry.key == key) {
        return entry.value;
      }
    }
  }
  return {};
}

}  // namespace

TEST(ScanDiagnostics, publishes_timing_icp_ndt_device_and_map_keys)
{
  back_odom::MatchResult match;
  match.has_lidar_debug = true;
  match.lidar_passed = true;
  match.applied = true;
  match.inserted_scan = true;
  match.iterations = 7.0;
  match.last_step = 0.01;
  match.correspondences = 40;
  match.inlier_ratio = 0.8;
  match.degenerate_axes = 1;
  match.lidar_cost = 0.25;
  match.ndt_cost = 3.5;
  match.ndt_inlier_fraction = 0.9;
  match.ndt_valid_hits = 12;
  match.ndt_ms = 2.25;
  match.cost_ms = 2.25;
  match.deskew_ms = 1.5;
  match.voxel_ms = 3.0;
  match.align_ms = 8.0;
  match.refine_ms = 0.0;
  match.map_ms = 4.0;
  match.cloud_parse_ms = 0.4;
  match.imu_wait_ms = 0.2;
  match.pipeline_ms = 20.0;
  match.kiss_downsample_ms = 0.7;
  match.kiss_map_ms = 1.1;
  match.translation_error = 0.05;
  match.rotation_error = 0.01;
  match.guess_distance_m = 0.12;
  match.guess_rotation_rad = 0.02;
  match.guess_dx_m = 0.1;
  match.guess_dy_m = 0.02;
  match.guess_dz_m = -0.01;
  match.coarse_iterations = 6;
  match.raw_points = 1000;
  match.source_points = 200;
  match.inserted_points = 180;
  match.map_voxels = 40;
  match.surface_voxels = 38;
  match.front_on_device = false;
  match.align_on_device = false;
  match.gpu_h2d_ms = 0.0;
  match.gpu_kernel_ms = 0.0;
  match.gpu_d2h_ms = 0.0;
  match.gpu_pack_ms = 0.0;
  match.gpu_unpack_ms = 0.0;

  back_odom::LidarMatchParams params;
  params.max_iterations = 20;
  params.partial_min_iterations = 3;
  params.convergence_criterion = 0.001;
  params.max_correspondence_distance = 2.0;
  params.kernel_scale = 0.5;
  params.voxel_size = 0.5;
  params.max_ndt_cost = 7.81;
  params.min_ndt_correspondences = 20;
  params.min_ndt_inlier_fraction = 0.5;
  params.ndt_cov_regularization = 0.001;
  params.max_points_per_voxel = 20;
  params.align_rate_hz = 10.0;

  const std::vector<diagnostic_msgs::msg::DiagnosticStatus> statuses =
    back_odom::scan_diagnostic_statuses(
      match, params, back_odom::LocalizationHealth::Healthy, 1, 2);
  ASSERT_EQ(statuses.size(), 6U);

  EXPECT_EQ(value_of(statuses, "back_odom: health", "reject_streak"), "1");
  EXPECT_EQ(value_of(statuses, "back_odom: health", "timeout_streak"), "2");
  EXPECT_EQ(value_of(statuses, "back_odom: timing", "deskew_ms"), "1.500");
  EXPECT_EQ(value_of(statuses, "back_odom: timing", "ndt_ms"), "2.250");
  EXPECT_EQ(value_of(statuses, "back_odom: timing", "cost_ms"), "2.250");
  EXPECT_EQ(value_of(statuses, "back_odom: timing", "cloud_parse_ms"), "0.400");
  EXPECT_EQ(value_of(statuses, "back_odom: timing", "imu_wait_ms"), "0.200");
  EXPECT_EQ(value_of(statuses, "back_odom: timing", "pipeline_ms"), "20.000");
  EXPECT_EQ(value_of(statuses, "back_odom: timing", "kiss_downsample_ms"), "0.700");
  EXPECT_EQ(value_of(statuses, "back_odom: timing", "kiss_map_ms"), "1.100");
  EXPECT_EQ(value_of(statuses, "back_odom: icp", "iterations"), "7.000");
  EXPECT_EQ(value_of(statuses, "back_odom: icp", "max_iterations"), "20");
  EXPECT_EQ(value_of(statuses, "back_odom: icp", "coarse_iterations"), "6");
  EXPECT_EQ(value_of(statuses, "back_odom: icp", "kernel_scale"), "0.500");
  EXPECT_EQ(value_of(statuses, "back_odom: icp", "guess_distance_m"), "0.120");
  EXPECT_EQ(value_of(statuses, "back_odom: icp", "guess_rotation_rad"), "0.020");
  EXPECT_EQ(value_of(statuses, "back_odom: icp", "guess_dx_m"), "0.100");
  EXPECT_EQ(value_of(statuses, "back_odom: icp", "guess_dy_m"), "0.020");
  EXPECT_EQ(value_of(statuses, "back_odom: icp", "guess_dz_m"), "-0.010");
  EXPECT_EQ(value_of(statuses, "back_odom: ndt", "ndt_cost"), "3.500");
  EXPECT_EQ(value_of(statuses, "back_odom: ndt", "ndt_valid_hits"), "12");
  EXPECT_EQ(value_of(statuses, "back_odom: ndt", "max_ndt_cost"), "7.810");
  EXPECT_EQ(value_of(statuses, "back_odom: ndt", "min_ndt_correspondences"), "20");
  EXPECT_EQ(value_of(statuses, "back_odom: device", "front_on_device"), "false");
  EXPECT_EQ(value_of(statuses, "back_odom: device", "align_on_device"), "false");
  EXPECT_EQ(value_of(statuses, "back_odom: device", "gpu_h2d_ms"), "0.000");
  EXPECT_EQ(value_of(statuses, "back_odom: device", "gpu_kernel_ms"), "0.000");
  EXPECT_EQ(value_of(statuses, "back_odom: device", "gpu_d2h_ms"), "0.000");
  EXPECT_EQ(value_of(statuses, "back_odom: device", "gpu_pack_ms"), "0.000");
  EXPECT_EQ(value_of(statuses, "back_odom: device", "gpu_unpack_ms"), "0.000");
  EXPECT_EQ(value_of(statuses, "back_odom: device", "bakeoff"), "false");
  EXPECT_EQ(value_of(statuses, "back_odom: map", "map_voxels"), "40");
  EXPECT_EQ(value_of(statuses, "back_odom: map", "surface_voxels"), "38");
  EXPECT_EQ(value_of(statuses, "back_odom: map", "voxel_size"), "0.500");
  EXPECT_EQ(value_of(statuses, "back_odom: map", "max_points_per_voxel"), "20");
  EXPECT_EQ(value_of(statuses, "back_odom: map", "raw_points"), "1000");
  EXPECT_EQ(value_of(statuses, "back_odom: map", "source_points"), "200");
  EXPECT_EQ(value_of(statuses, "back_odom: map", "inserted_points"), "180");

  const diagnostic_msgs::msg::DiagnosticStatus flat = back_odom::flatten_diagnostics(statuses);
  EXPECT_EQ(flat.name, "back_odom");
  EXPECT_EQ(flat.message, "healthy");
  EXPECT_EQ(value_of({flat}, "back_odom", "deskew_ms"), "1.500");
  EXPECT_EQ(value_of({flat}, "back_odom", "ndt_valid_hits"), "12");
  EXPECT_EQ(value_of({flat}, "back_odom", "gpu_h2d_ms"), "0.000");
}
