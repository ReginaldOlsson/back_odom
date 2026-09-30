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

#include "back_odom/fpfh_coarse_align.hpp"
#include "back_odom/imu_processor.hpp"
#include "back_odom/kinematic_limits.hpp"
#include "back_odom/visual_motion.hpp"

#include <Eigen/Core>
#include <kiss_icp_cpp/core/Registration.hpp>
#include <kiss_icp_cpp/core/VoxelHashMap.hpp>
#include <sophus/se3.hpp>

#include <cmath>
#include <deque>
#include <optional>
#include <vector>

namespace back_odom
{

struct LidarScan
{
  double stamp{0.0};
  std::vector<Eigen::Vector3d> points;
  std::vector<double> timestamps;
};

struct LidarMatchParams
{
  double voxel_size{0.5};
  double crop_longitudinal{75.0};
  double crop_lateral{50.0};
  double max_correspondence_distance{2.0};
  double kernel_scale{0.5};
  double convergence_criterion{1.0e-4};
  int max_iterations{50};
  int max_points_per_voxel{20};
  /// Recent scans kept with their lidar poses. The live map is updated in place.
  int backward_match_stride{4};
  // Fraction of the vehicle-frame residual turned into a bias step each accepted scan.
  double gyro_bias_gain{0.1};
  double accel_bias_gain{0.02};
  // How much of the speed comes from the lidar displacement. Direction stays with the IMU.
  double speed_correction_gain{0.5};
  double max_gyro_bias{0.05};
  double max_accel_bias{1.0};
  VehicleLimits limits{};
  bool visual_enabled{true};
  FpfhParams fpfh{};
};

/// A scan still contributes to the visible cloud while its pose is inside the crop around the latest scan.
[[nodiscard]] inline bool scan_pose_inside_visible_cloud(
  const Sophus::SE3d & latest, const Sophus::SE3d & scan, const double longitudinal,
  const double lateral)
{
  const Eigen::Vector3d local = latest.inverse() * scan.translation();
  return std::abs(local.x()) <= longitudinal && std::abs(local.y()) <= lateral;
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
  bool used_coarse_guess{false};
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
  bool used_visual_guess{false};
  bool has_lidar_debug{false};
  Sophus::SE3d lidar_pose{};
  double lidar_cost{0.0};
  bool lidar_passed{false};
  bool has_camera_debug{false};
  Sophus::SE3d camera_pose{};
  double camera_cost{0.0};
  bool camera_passed{false};
};

class LidarImuMatcher
{
public:
  explicit LidarImuMatcher(const LidarMatchParams & params);

  [[nodiscard]] bool has_reference() const;
  [[nodiscard]] std::vector<Eigen::Vector3d> local_map() const;
  /// Lidar poses of scans whose points can still lie in the cropped cloud.
  [[nodiscard]] std::vector<StampedPose> visible_scan_poses() const;

  /// Static pose of the lidar in the integrator frame. Identity until tf_static provides it.
  void set_body_from_lidar(const Sophus::SE3d & body_from_lidar);
  void set_body_from_camera(const Sophus::SE3d & body_from_camera);

  /// Monocular camera pose in the visual world. Scale is recovered from healthy lidar motion.
  void push_visual_pose(double stamp, const Sophus::SE3d & world_from_camera);

  [[nodiscard]] LocalizationHealth health() const;
  [[nodiscard]] int reject_streak() const;
  [[nodiscard]] bool scale_frozen() const;
  [[nodiscard]] double visual_scale() const;
  [[nodiscard]] bool has_visual_scale() const;

  /// Deskew, crop, and either store the first scan or match a new scan into the map.
  [[nodiscard]] MatchResult on_scan(const LidarScan & scan, ImuProcessor & imu);

  /// The integrator crossed the speed limit. Hold the last healthy speed.
  void notify_speed_limit(ImuProcessor & imu);

  /// IMU samples only integrate. A repeated scan cannot observe drift, so it does not move the
  /// pose.
  [[nodiscard]] MatchResult on_imu(ImuProcessor & imu);

private:
  struct WindowScan
  {
    double stamp{0.0};
    std::vector<Eigen::Vector3d> map_points;
    std::vector<Eigen::Vector3d> source_points;
    Sophus::SE3d pose{};
  };

  struct PreparedScan
  {
    double scan_end{0.0};
    std::vector<Eigen::Vector3d> map_points;
    std::vector<Eigen::Vector3d> source;
    Sophus::SE3d predicted{};
  };

  void apply_correction(
    ImuProcessor & imu, const Sophus::SE3d & vehicle_delta, double scan_stamp, double since_stamp,
    bool update_bias, const std::optional<Eigen::Vector3d> & velocity_override);
  void optimize_window(ImuProcessor & imu);
  void freeze_oldest(const Sophus::SE3d & cull_pose);
  void rebuild_map(const Sophus::SE3d & cull_pose);
  [[nodiscard]] kiss_icp::PlaneAlignResult align(
    const std::vector<Eigen::Vector3d> & scan, const Sophus::SE3d & guess, double & iterations);
  void remember_healthy(const Sophus::SE3d & pose, const Eigen::Vector3d & velocity, double stamp);
  void restore_healthy_speed(ImuProcessor & imu);
  void note_health(ImuProcessor & imu, HealthEvent event);
  struct VisualSegment
  {
    Sophus::SE3d body{};
    double t0{0.0};
    double t1{0.0};
  };

  [[nodiscard]] std::optional<StampedPose> visual_at_or_before(double stamp) const;
  [[nodiscard]] std::optional<VisualSegment> visual_segment(double from_stamp, double to_stamp) const;
  [[nodiscard]] Sophus::SO3d imu_rotation_between(
    const ImuProcessor & imu, double from_stamp, double to_stamp) const;
  [[nodiscard]] Sophus::SE3d recovery_pose(double scan_end, const ImuProcessor & imu) const;
  [[nodiscard]] MatchCandidate score_alignment(
    const std::vector<Eigen::Vector3d> & source, const Sophus::SE3d & guess,
    const Sophus::SE3d & imu_reference, bool check_imu_gate);
  void finish_accepted_scan(
    MatchResult & result, PreparedScan prepared, const Sophus::SE3d & aligned_pose,
    double iterations, ImuProcessor & imu, double since_stamp, bool update_bias,
    const std::optional<Eigen::Vector3d> & velocity_override);
  void update_scale(const Sophus::SE3d & lidar_pose, double stamp);
  void record_scan_pose(const Sophus::SE3d & pose, double stamp);

  Sophus::SE3d body_from_lidar_{};
  LidarMatchParams params_;
  double half_longitudinal_{37.5};
  double half_lateral_{25.0};
  kiss_icp::VoxelHashMap map_;
  kiss_icp::VoxelHashMap frozen_;
  kiss_icp::Registration registration_;
  FpfhCoarseAlign fpfh_;
  std::deque<WindowScan> window_;
  std::deque<StampedPose> scan_poses_;
  bool has_reference_{false};
  double last_correction_stamp_{0.0};
  Eigen::Vector3d last_correction_position_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d gyro_bias_prior_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d accel_bias_prior_{Eigen::Vector3d::Zero()};
  bool has_bias_prior_{false};
  HealthState health_{};
  Sophus::SE3d body_from_camera_{};
  bool has_camera_{false};
  std::deque<StampedPose> visual_poses_;
  ScaleEstimator scale_{};
  bool has_healthy_{false};
  Sophus::SE3d last_healthy_pose_{};
  double last_healthy_stamp_{0.0};
  Eigen::Vector3d last_healthy_velocity_{Eigen::Vector3d::Zero()};
  double last_healthy_speed_{0.0};
};

}  // namespace back_odom

#endif  // BACK_ODOM__LIDAR_IMU_MATCHER_HPP_
