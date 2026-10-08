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

#ifndef BACK_ODOM__LIDAR_IMU_MATCHER_HPP_
#define BACK_ODOM__LIDAR_IMU_MATCHER_HPP_

#include "back_odom/imu/imu_processor.hpp"
#include "back_odom/imu/kinematic_limits.hpp"
#include "back_odom/lidar/match_candidate.hpp"
#include "back_odom/lidar/ndt_health.hpp"
#include "back_odom/lidar/plane_icp.hpp"

#include <Eigen/Core>
#include <kiss_icp_cpp/core/VoxelHashMap.hpp>
#include <sophus/se3.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace back_odom
{

struct LidarScan
{
  double stamp{0.0};
  std::vector<Eigen::Vector3d> points;
  std::vector<double> timestamps;
  /// Per-point lidar return strength. Empty means intensity was not present on the cloud.
  std::vector<float> intensities;
};

struct LidarMatchParams
{
  double voxel_size{0.5};
  double crop_longitudinal{75.0};
  double crop_lateral{50.0};
  /// Scan copies kept for the debug cloud. The live map is still the cropped ICP cloud.
  int max_visible_scans{120};
  double max_correspondence_distance{2.0};
  double kernel_scale{0.5};
  double convergence_criterion{1.0e-3};
  int max_iterations{20};
  int max_points_per_voxel{20};
  /// Kept for parameter compatibility; the live path no longer keeps a scan window.
  int backward_match_stride{4};
  /// Off on the live path: freezing each scan into the map duplicates a building once per pose.
  bool refine_window{false};
  double refine_min_travel{2.0};
  /// After this many collective passes a scan is trusted and left out of later solves.
  int refine_settle_passes{2};
  /// When false, ICP still builds the local map but does not move IMU pose/velocity/biases.
  bool apply_imu_correction{true};
  // Fraction of the vehicle-frame residual turned into a bias step each accepted scan.
  double gyro_bias_gain{0.1};
  double accel_bias_gain{0.02};
  // How much of the speed comes from the lidar displacement. Direction stays with the IMU.
  double speed_correction_gain{0.5};
  /// ICP may nudge the IMU step by at most this many metres along the vehicle.
  /// Lateral shift and yaw stay with the lidar match.
  double max_longitudinal_correction{0.15};
  double max_gyro_bias{0.05};
  double max_accel_bias{1.0};
  /// Mean Mahalanobis D_M^2 threshold (χ² 3-DoF ~95% heuristic). Also the per-point inlier cut.
  double max_ndt_cost{7.81};
  double ndt_cov_regularization{1.0e-3};
  /// Skip the NDT gate when fewer valid voxel hits than this are available.
  int min_ndt_correspondences{20};
  /// Minimum fraction of valid NDT hits below max_ndt_cost for a scan to be accepted.
  double min_ndt_inlier_fraction{0.5};
  /// Minimum plane correspondences / source points for a scan to be accepted.
  double min_inlier_ratio{0.3};
  /// Target scan match rate. ICP stops when the 1/hz budget is spent. A partial solve with at
  /// least `partial_min_iterations` steps and a last step below `partial_max_step` is still used;
  /// otherwise the scan is skipped (nothing is inserted).
  double align_rate_hz{10.0};
  int partial_min_iterations{3};
  double partial_max_step{0.01};
  /// A scan is inserted into the map only when the vehicle moved this far since the last
  /// inserted scan. Stationary scans are matched but not inserted.
  double keyframe_min_translation{0.5};
  double keyframe_min_rotation{0.05};
  /// Scan index at which the GPU vs CPU bake-off runs (the map must be representative), and
  /// how often it is re-run. 0 disables the device entirely.
  int device_bakeoff_scan{10};
  int device_recheck_scans{200};
  VehicleLimits limits{};
};

/// A scan still contributes to the visible cloud while its pose is inside the crop around the latest scan.
[[nodiscard]] inline bool scan_pose_inside_visible_cloud(
  const Sophus::SE3d & latest, const Sophus::SE3d & scan, const double longitudinal,
  const double lateral)
{
  const Eigen::Vector3d local = latest.inverse() * scan.translation();
  return std::abs(local.x()) <= longitudinal && std::abs(local.y()) <= lateral;
}

/// The road does not observe forward motion, so a match may only nudge the IMU step along the vehicle.
[[nodiscard]] inline Sophus::SE3d limit_longitudinal_correction(
  const Sophus::SE3d & predicted, const Sophus::SE3d & aligned, const double max_forward)
{
  if (!(max_forward >= 0.0) || !std::isfinite(max_forward)) {
    return aligned;
  }
  Sophus::SE3d body = predicted.inverse() * aligned;
  Eigen::Vector3d translation = body.translation();
  translation.x() = std::clamp(translation.x(), -max_forward, max_forward);
  body.translation() = translation;
  return predicted * body;
}

/// Velocity and bias step implied by one accepted ICP residual.
/// `vehicle_delta` is predicted.inverse() * aligned, so rotation is about the vehicle.
struct InertialCorrection
{
  Eigen::Vector3d velocity_world{Eigen::Vector3d::Zero()};
  Eigen::Vector3d gyro_bias_delta{Eigen::Vector3d::Zero()};
  Eigen::Vector3d accel_bias_delta{Eigen::Vector3d::Zero()};
};

[[nodiscard]] InertialCorrection inertial_correction_from_match(
  const Sophus::SE3d & vehicle_delta, const Sophus::SO3d & orientation_correction,
  const Sophus::SE3d & corrected_pose, const Eigen::Vector3d & imu_velocity_world,
  const Eigen::Vector3d & previous_position, double dt, double gyro_bias_gain,
  double accel_bias_gain, double speed_gain);

struct MatchResult
{
  bool applied{false};
  bool inserted_scan{false};
  bool first_scan{false};
  /// The scan could not be processed (IMU not aligned, no IMU pose near the scan, empty crop).
  bool skipped{false};
  /// Set when `skipped` is true so the node can log the reason at INFO.
  std::string skip_reason;
  Sophus::SE3d imu_pose{};
  Sophus::SE3d corrected_pose{};
  /// World-frame left increment applied to the pose. `correction * predicted = aligned`.
  Sophus::SE3d correction{};
  /// Vehicle-frame residual used for the gate and the bias update.
  double translation_error{0.0};
  double rotation_error{0.0};
  double iterations{0.0};
  Eigen::Vector3d gyro_bias{Eigen::Vector3d::Zero()};
  Eigen::Vector3d accel_bias{Eigen::Vector3d::Zero()};
  LocalizationHealth health{LocalizationHealth::Healthy};
  bool has_lidar_debug{false};
  Sophus::SE3d lidar_pose{};
  double lidar_cost{0.0};
  double ndt_cost{std::numeric_limits<double>::quiet_NaN()};
  double ndt_inlier_fraction{std::numeric_limits<double>::quiet_NaN()};
  double inlier_ratio{0.0};
  int correspondences{0};
  int degenerate_axes{0};
  bool lidar_passed{false};
  bool saturated{false};
  /// ICP hit the rate budget. `partial_used` says whether the partial solve was good enough.
  bool align_timed_out{false};
  bool partial_used{false};
  bool refined_window{false};
  double refine_shift{0.0};
  int refine_scans{0};
  int refine_settled{0};
  double deskew_ms{0.0};
  double voxel_ms{0.0};
  double align_ms{0.0};
  double cost_ms{0.0};
  double refine_ms{0.0};
  double map_ms{0.0};
};

struct LocalMapPoint
{
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  /// Original lidar intensity carried through deskew/downsample into the local cloud.
  float intensity{0.0F};
  float scan_id{0.0F};
  float collective_passes{0.0F};
};

/// Output of the heavy half of a scan match. Produced from a const IMU snapshot, consumed by
/// commit_scan under the state lock.
struct ScanAlignment
{
  MatchResult result;
  double scan_end{0.0};
  std::vector<Eigen::Vector3d> map_points;
  std::vector<float> map_intensities;
  std::vector<Eigen::Vector3d> source;
  /// Seed pose used for ICP (lidar anchor + IMU relative motion).
  Sophus::SE3d predicted{};
  MatchCandidate candidate;
  bool first_scan{false};
  bool ready{false};
  std::chrono::steady_clock::time_point deadline{};
};

class LidarImuMatcher
{
public:
  explicit LidarImuMatcher(const LidarMatchParams & params);

  [[nodiscard]] bool has_reference() const;
  [[nodiscard]] std::vector<Eigen::Vector3d> local_map() const;
  /// Horizon points in the map frame, with scan id and collective-pass count for RViz.
  [[nodiscard]] std::vector<LocalMapPoint> annotated_local_map() const;
  /// Lidar poses of scans whose points can still lie in the cropped cloud.
  [[nodiscard]] std::vector<StampedPose> visible_scan_poses() const;
  [[nodiscard]] std::size_t visible_scan_point_count() const;

  /// Static pose of the lidar in the integrator frame. Identity until tf_static provides it.
  void set_body_from_lidar(const Sophus::SE3d & body_from_lidar);

  [[nodiscard]] LocalizationHealth health() const;
  [[nodiscard]] int reject_streak() const;
  [[nodiscard]] int timeout_streak() const;

  /// Heavy half: deskew, crop, downsample, ICP, gates. Reads the IMU only; safe to run on a
  /// snapshot while the live integrator keeps going on another thread.
  [[nodiscard]] ScanAlignment align_scan(const LidarScan & scan, const ImuProcessor & imu);
  /// Light half: insert into the map, apply the IMU correction, advance health.
  [[nodiscard]] MatchResult commit_scan(ScanAlignment && alignment, ImuProcessor & imu);
  /// align_scan followed by commit_scan.
  [[nodiscard]] MatchResult on_scan(const LidarScan & scan, ImuProcessor & imu);

  /// The integrator crossed the speed limit. Hold the last healthy speed.
  void notify_speed_limit(ImuProcessor & imu);

  /// IMU samples only integrate. A repeated scan cannot observe drift, so it does not move the
  /// pose.
  [[nodiscard]] MatchResult on_imu(ImuProcessor & imu);

private:
  struct PreparedClouds
  {
    std::vector<Eigen::Vector3d> map_points;
    std::vector<float> map_intensities;
    std::vector<Eigen::Vector3d> source;
  };

  void apply_correction(
    ImuProcessor & imu, const Sophus::SE3d & vehicle_delta, double scan_stamp, double since_stamp,
    bool update_bias, const std::optional<Eigen::Vector3d> & velocity_override);
  [[nodiscard]] PlaneIcpResult align(
    const std::vector<Eigen::Vector3d> & scan, const Sophus::SE3d & guess,
    const std::chrono::steady_clock::time_point * deadline);
  [[nodiscard]] PreparedClouds prepare_scan(
    const LidarScan & scan, const std::vector<StampedPose> & trajectory, MatchResult & timing);
  void remember_healthy(const Sophus::SE3d & pose, const Eigen::Vector3d & velocity, double stamp);
  void restore_healthy_speed(ImuProcessor & imu);
  void note_health(ImuProcessor & imu, HealthEvent event);
  [[nodiscard]] MatchCandidate score_alignment(
    const std::vector<Eigen::Vector3d> & source, const Sophus::SE3d & guess, MatchResult & timing,
    const std::chrono::steady_clock::time_point * deadline);
  [[nodiscard]] bool should_insert(const Sophus::SE3d & pose) const;
  void insert_scan(
    const Sophus::SE3d & pose, double stamp, std::vector<Eigen::Vector3d> && points,
    std::vector<float> && intensities, const ImuProcessor & imu, MatchResult & timing);
  void record_horizon_scan(
    const Sophus::SE3d & pose, double stamp, std::vector<Eigen::Vector3d> && points,
    std::vector<float> && intensities, const ImuProcessor & imu);
  bool refine_window_if_due(MatchResult & result, Sophus::SE3d & newest_pose);
  void rebuild_from_horizon(const Sophus::SE3d & cull_pose);
  void cull_local_map(const Sophus::SE3d & pose);
  void rebuild_surface();

  Sophus::SE3d body_from_lidar_{};
  LidarMatchParams params_;
  double half_longitudinal_{37.5};
  double half_lateral_{25.0};
  kiss_icp::VoxelHashMap map_;
  SurfaceVoxelMap surface_;
  struct HorizonScan
  {
    double stamp{0.0};
    Sophus::SE3d pose{};
    Sophus::SE3d prior{};
    Sophus::SE3d imu_from_previous{};
    std::vector<Eigen::Vector3d> points;
    std::vector<float> intensities;
    bool optimized{false};
    int collective_passes{0};
    std::uint32_t scan_id{0};
  };

  std::deque<HorizonScan> horizon_;
  std::uint32_t next_scan_id_{0};
  std::uint64_t scan_count_{0};
  bool has_refined_{false};
  Eigen::Vector3d last_refine_position_{Eigen::Vector3d::Zero()};
  bool has_reference_{false};
  /// Last committed lidar pose and its scan-end stamp. Seeds the next ICP together with the
  /// IMU motion since then, so open-loop IMU drift cannot walk the seed off the map.
  Sophus::SE3d anchor_pose_{};
  double anchor_stamp_{0.0};
  Sophus::SE3d last_keyframe_pose_{};
  double last_correction_stamp_{0.0};
  Eigen::Vector3d last_correction_position_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d gyro_bias_prior_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d accel_bias_prior_{Eigen::Vector3d::Zero()};
  bool has_bias_prior_{false};
  HealthState health_{};
  bool has_healthy_{false};
  Sophus::SE3d last_healthy_pose_{};
  double last_healthy_stamp_{0.0};
  Eigen::Vector3d last_healthy_velocity_{Eigen::Vector3d::Zero()};
  double last_healthy_speed_{0.0};
  std::uint64_t map_epoch_{1};
  std::uint64_t culls_since_prune_{0};
  Sophus::SE3d last_cull_pose_{};
  bool has_cull_pose_{false};
  bool device_front_decided_{false};
  bool device_front_use_{false};
  std::uint64_t device_front_decided_at_{0};
  bool device_align_decided_{false};
  bool device_align_use_{false};
  std::uint64_t device_align_decided_at_{0};
};

}  // namespace back_odom

#endif  // BACK_ODOM__LIDAR_IMU_MATCHER_HPP_
