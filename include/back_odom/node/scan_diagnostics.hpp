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

#ifndef BACK_ODOM__SCAN_DIAGNOSTICS_HPP_
#define BACK_ODOM__SCAN_DIAGNOSTICS_HPP_

#include "back_odom/lidar/lidar_imu_matcher.hpp"

#include <diagnostic_msgs/msg/diagnostic_status.hpp>

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace back_odom
{

inline std::string diagnostic_number(const double value)
{
  if (!std::isfinite(value)) {
    return "none";
  }
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(3) << value;
  return stream.str();
}

inline std::string diagnostic_bool(const bool value)
{
  return value ? "true" : "false";
}

inline void add_diagnostic_value(
  diagnostic_msgs::msg::DiagnosticStatus & status, const std::string & key, const std::string & value)
{
  diagnostic_msgs::msg::KeyValue entry;
  entry.key = key;
  entry.value = value;
  status.values.push_back(entry);
}

inline diagnostic_msgs::msg::DiagnosticStatus make_diagnostic_status(
  const std::string & name, const std::uint8_t level, const std::string & message)
{
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = name;
  status.hardware_id = "back_odom";
  status.level = level;
  status.message = message;
  return status;
}

/// Six statuses for one scan: health, timing, ICP, NDT, device, and the kiss_icp map.
inline std::vector<diagnostic_msgs::msg::DiagnosticStatus> scan_diagnostic_statuses(
  const MatchResult & match, const LidarMatchParams & params, const LocalizationHealth health,
  const int reject_streak, const int timeout_streak)
{
  using Status = diagnostic_msgs::msg::DiagnosticStatus;
  const std::uint8_t ok = Status::OK;
  const std::uint8_t warn = Status::WARN;
  const std::uint8_t error = Status::ERROR;

  std::uint8_t health_level = ok;
  if (health == LocalizationHealth::Degraded) {
    health_level = warn;
  } else if (health == LocalizationHealth::Diverged) {
    health_level = error;
  }
  diagnostic_msgs::msg::DiagnosticStatus health_status =
    make_diagnostic_status("back_odom: health", health_level, health_name(health));
  add_diagnostic_value(health_status, "reject_streak", std::to_string(reject_streak));
  add_diagnostic_value(health_status, "timeout_streak", std::to_string(timeout_streak));
  add_diagnostic_value(health_status, "ekf_twist_prior", diagnostic_bool(match.ekf_twist_prior));
  add_diagnostic_value(health_status, "inserted_scan", diagnostic_bool(match.inserted_scan));
  add_diagnostic_value(health_status, "skipped", diagnostic_bool(match.skipped));
  add_diagnostic_value(health_status, "applied", diagnostic_bool(match.applied));

  const double budget_ms = params.align_rate_hz > 0.0 ? 1000.0 / params.align_rate_hz : 0.0;
  const bool over_budget = budget_ms > 0.0 && match.pipeline_ms > budget_ms;
  diagnostic_msgs::msg::DiagnosticStatus timing = make_diagnostic_status(
    "back_odom: timing", over_budget ? warn : ok,
    "pipeline " + diagnostic_number(match.pipeline_ms) + " ms");
  add_diagnostic_value(timing, "cloud_parse_ms", diagnostic_number(match.cloud_parse_ms));
  add_diagnostic_value(timing, "imu_wait_ms", diagnostic_number(match.imu_wait_ms));
  add_diagnostic_value(timing, "deskew_ms", diagnostic_number(match.deskew_ms));
  add_diagnostic_value(timing, "voxel_ms", diagnostic_number(match.voxel_ms));
  add_diagnostic_value(timing, "align_ms", diagnostic_number(match.align_ms));
  add_diagnostic_value(timing, "ndt_ms", diagnostic_number(match.ndt_ms));
  add_diagnostic_value(timing, "cost_ms", diagnostic_number(match.cost_ms));
  add_diagnostic_value(timing, "refine_ms", diagnostic_number(match.refine_ms));
  add_diagnostic_value(timing, "map_ms", diagnostic_number(match.map_ms));
  add_diagnostic_value(timing, "pipeline_ms", diagnostic_number(match.pipeline_ms));
  add_diagnostic_value(timing, "kiss_downsample_ms", diagnostic_number(match.kiss_downsample_ms));
  add_diagnostic_value(timing, "kiss_map_ms", diagnostic_number(match.kiss_map_ms));

  const bool icp_warn =
    match.has_lidar_debug && (match.saturated || match.align_timed_out || !match.lidar_passed);
  diagnostic_msgs::msg::DiagnosticStatus icp = make_diagnostic_status(
    "back_odom: icp", icp_warn ? warn : ok,
    std::to_string(static_cast<int>(match.iterations)) + " iterations");
  add_diagnostic_value(icp, "max_iterations", std::to_string(params.max_iterations));
  add_diagnostic_value(icp, "partial_min_iterations", std::to_string(params.partial_min_iterations));
  add_diagnostic_value(icp, "convergence_criterion", diagnostic_number(params.convergence_criterion));
  add_diagnostic_value(
    icp, "max_correspondence_distance", diagnostic_number(params.max_correspondence_distance));
  add_diagnostic_value(icp, "kernel_scale", diagnostic_number(params.kernel_scale));
  add_diagnostic_value(icp, "voxel_size", diagnostic_number(params.voxel_size));
  add_diagnostic_value(icp, "coarse_iterations", std::to_string(match.coarse_iterations));
  add_diagnostic_value(icp, "iterations", diagnostic_number(match.iterations));
  add_diagnostic_value(icp, "last_step", diagnostic_number(match.last_step));
  add_diagnostic_value(icp, "correspondences", std::to_string(match.correspondences));
  add_diagnostic_value(icp, "inlier_ratio", diagnostic_number(match.inlier_ratio));
  add_diagnostic_value(icp, "degenerate_axes", std::to_string(match.degenerate_axes));
  add_diagnostic_value(icp, "lidar_cost", diagnostic_number(match.lidar_cost));
  add_diagnostic_value(icp, "saturated", diagnostic_bool(match.saturated));
  add_diagnostic_value(icp, "align_timed_out", diagnostic_bool(match.align_timed_out));
  add_diagnostic_value(icp, "partial_used", diagnostic_bool(match.partial_used));
  add_diagnostic_value(icp, "lidar_passed", diagnostic_bool(match.lidar_passed));
  add_diagnostic_value(icp, "translation_error_m", diagnostic_number(match.translation_error));
  add_diagnostic_value(icp, "rotation_error_rad", diagnostic_number(match.rotation_error));
  add_diagnostic_value(icp, "guess_distance_m", diagnostic_number(match.guess_distance_m));
  add_diagnostic_value(icp, "guess_rotation_rad", diagnostic_number(match.guess_rotation_rad));
  add_diagnostic_value(icp, "guess_dx_m", diagnostic_number(match.guess_dx_m));
  add_diagnostic_value(icp, "guess_dy_m", diagnostic_number(match.guess_dy_m));
  add_diagnostic_value(icp, "guess_dz_m", diagnostic_number(match.guess_dz_m));

  const bool ndt_warn = match.has_lidar_debug && std::isfinite(match.ndt_cost) &&
                        match.ndt_cost >= params.max_ndt_cost;
  diagnostic_msgs::msg::DiagnosticStatus ndt = make_diagnostic_status(
    "back_odom: ndt", ndt_warn ? warn : ok, "ndt " + diagnostic_number(match.ndt_cost));
  add_diagnostic_value(ndt, "ndt_ms", diagnostic_number(match.ndt_ms));
  add_diagnostic_value(ndt, "ndt_cost", diagnostic_number(match.ndt_cost));
  add_diagnostic_value(ndt, "ndt_inlier_fraction", diagnostic_number(match.ndt_inlier_fraction));
  add_diagnostic_value(ndt, "ndt_valid_hits", std::to_string(match.ndt_valid_hits));
  add_diagnostic_value(ndt, "max_ndt_cost", diagnostic_number(params.max_ndt_cost));
  add_diagnostic_value(ndt, "min_ndt_correspondences", std::to_string(params.min_ndt_correspondences));
  add_diagnostic_value(ndt, "min_ndt_inlier_fraction", diagnostic_number(params.min_ndt_inlier_fraction));
  add_diagnostic_value(ndt, "ndt_cov_regularization", diagnostic_number(params.ndt_cov_regularization));

  const bool bakeoff = match.front_bakeoff || match.align_bakeoff;
  diagnostic_msgs::msg::DiagnosticStatus device =
    make_diagnostic_status("back_odom: device", ok, match.align_on_device ? "gpu" : "cpu");
  add_diagnostic_value(device, "front_on_device", diagnostic_bool(match.front_on_device));
  add_diagnostic_value(device, "align_on_device", diagnostic_bool(match.align_on_device));
  add_diagnostic_value(device, "bakeoff", diagnostic_bool(bakeoff));
  add_diagnostic_value(device, "front_bakeoff", diagnostic_bool(match.front_bakeoff));
  add_diagnostic_value(device, "align_bakeoff", diagnostic_bool(match.align_bakeoff));
  add_diagnostic_value(device, "front_bakeoff_device_ms", diagnostic_number(match.front_bakeoff_device_ms));
  add_diagnostic_value(device, "front_bakeoff_cpu_ms", diagnostic_number(match.front_bakeoff_cpu_ms));
  add_diagnostic_value(device, "align_bakeoff_device_ms", diagnostic_number(match.align_bakeoff_device_ms));
  add_diagnostic_value(device, "align_bakeoff_cpu_ms", diagnostic_number(match.align_bakeoff_cpu_ms));
  add_diagnostic_value(device, "gpu_h2d_ms", diagnostic_number(match.gpu_h2d_ms));
  add_diagnostic_value(device, "gpu_kernel_ms", diagnostic_number(match.gpu_kernel_ms));
  add_diagnostic_value(device, "gpu_d2h_ms", diagnostic_number(match.gpu_d2h_ms));
  add_diagnostic_value(device, "gpu_pack_ms", diagnostic_number(match.gpu_pack_ms));
  add_diagnostic_value(device, "gpu_unpack_ms", diagnostic_number(match.gpu_unpack_ms));

  diagnostic_msgs::msg::DiagnosticStatus map = make_diagnostic_status(
    "back_odom: map", ok, std::to_string(match.map_voxels) + " voxels");
  add_diagnostic_value(map, "map_voxels", std::to_string(match.map_voxels));
  add_diagnostic_value(map, "surface_voxels", std::to_string(match.surface_voxels));
  add_diagnostic_value(map, "voxel_size", diagnostic_number(params.voxel_size));
  add_diagnostic_value(map, "max_points_per_voxel", std::to_string(params.max_points_per_voxel));
  add_diagnostic_value(map, "raw_points", std::to_string(match.raw_points));
  add_diagnostic_value(map, "source_points", std::to_string(match.source_points));
  add_diagnostic_value(map, "inserted_points", std::to_string(match.inserted_points));

  return {health_status, timing, icp, ndt, device, map};
}

inline diagnostic_msgs::msg::DiagnosticStatus flatten_diagnostics(
  const std::vector<diagnostic_msgs::msg::DiagnosticStatus> & statuses)
{
  diagnostic_msgs::msg::DiagnosticStatus flat =
    make_diagnostic_status("back_odom", diagnostic_msgs::msg::DiagnosticStatus::OK, "back_odom");
  for (const diagnostic_msgs::msg::DiagnosticStatus & status : statuses) {
    if (status.name == "back_odom: health") {
      flat.level = status.level;
      flat.message = status.message;
    }
    flat.values.insert(flat.values.end(), status.values.begin(), status.values.end());
  }
  return flat;
}

}  // namespace back_odom

#endif  // BACK_ODOM__SCAN_DIAGNOSTICS_HPP_
