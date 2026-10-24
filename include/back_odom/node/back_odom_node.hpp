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

#ifndef BACK_ODOM__BACK_ODOM_NODE_HPP_
#define BACK_ODOM__BACK_ODOM_NODE_HPP_

#include "back_odom/imu/imu_processor.hpp"
#include "back_odom/lidar/lidar_imu_matcher.hpp"

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>

#include <back_odom/msg/icp_debug.hpp>
#include <back_odom/msg/pose_refine.hpp>
#include <back_odom/msg/scan_stages.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <tf2_ros/transform_broadcaster.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace back_odom
{

/// IMU integration runs on the executor thread; every scan is matched on a dedicated worker
/// thread fed by a single-slot mailbox (the newest scan wins). `state_mutex_` guards the IMU
/// integrator and matcher health/pose; ICP runs on an IMU snapshot, and map insert is flushed
/// after that lock is released so the IMU path keeps publishing.
class BackOdomNode : public rclcpp::Node
{
public:
  explicit BackOdomNode(const rclcpp::NodeOptions & options);
  ~BackOdomNode() override;

private:
  void callback_imu(const sensor_msgs::msg::Imu::ConstSharedPtr msg);
  void callback_pointcloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
  void callback_ground_truth(const nav_msgs::msg::Odometry::ConstSharedPtr msg);
  /// Requires state_mutex_. Locks map_from_local_ on the earliest ground-truth stamp the IMU covers.
  void try_lock_ground_truth_alignment();
  [[nodiscard]] bool map_frame_ready() const;
  [[nodiscard]] Sophus::SE3d map_from_local() const;
  [[nodiscard]] ProcessorOutput to_map_frame(const ProcessorOutput & output) const;
  void worker_loop();
  void process_cloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg);
  void watch_lidar_start();
  void publish_match(
    const MatchResult & matched, const ProcessorOutput & prior, const ProcessorOutput & corrected,
    const rclcpp::Time & stamp);
  void publish_health(LocalizationHealth health, int reject_streak, int timeout_streak);
  void publish_debug_odometry(const MatchResult & match, const rclcpp::Time & stamp);
  void publish_pipeline_debug(const MatchResult & matched, const rclcpp::Time & stamp);
  void note_skipped_scan(const rclcpp::Time & stamp);
  void publish_odometry(
    const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr & publisher,
    const ProcessorOutput & output, const rclcpp::Time & stamp);
  void publish_twist(const ProcessorOutput & output, const rclcpp::Time & stamp);
  void publish_pose(const ProcessorOutput & output, const rclcpp::Time & stamp);
  [[nodiscard]] LidarScan scan_from_cloud(const sensor_msgs::msg::PointCloud2 & cloud) const;
  [[nodiscard]] bool base_from_frame(
    const std::string & source_frame, Sophus::SE3d & target_from_source);
  void publish_path(const ProcessorOutput & output, const rclcpp::Time & stamp);
  void publish_tf(const ProcessorOutput & output, const rclcpp::Time & stamp);
  void publish_markers(const ProcessorOutput & output, LocalizationHealth health, const rclcpp::Time & stamp);
  void publish_scan_poses(const std::vector<StampedPose> & poses, const rclcpp::Time & stamp);
  void publish_local_map();

  /// Guards imu_processor_ and the committed matcher state (map, horizon, health).
  std::mutex state_mutex_;
  std::condition_variable imu_cv_;
  std::unique_ptr<ImuProcessor> imu_processor_;
  std::unique_ptr<LidarImuMatcher> lidar_matcher_;

  /// Single-slot scan mailbox.
  std::mutex scan_mutex_;
  std::condition_variable scan_cv_;
  sensor_msgs::msg::PointCloud2::ConstSharedPtr pending_cloud_;
  std::uint64_t dropped_scans_{0};
  std::atomic<bool> stop_{false};
  std::thread worker_;

  /// Guards the publisher-side bookkeeping (paths, last match, throttles).
  std::mutex publish_mutex_;
  std::mutex extrinsics_mutex_;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr ground_truth_sub_;
  rclcpp::TimerBase::SharedPtr map_timer_;
  rclcpp::TimerBase::SharedPtr lidar_watch_timer_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr imu_odom_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr twist_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initial_pose_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr scan_pose_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticStatus>::SharedPtr health_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr lidar_debug_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr lidar_debug_path_pub_;
  rclcpp::Publisher<back_odom::msg::ScanStages>::SharedPtr scan_stages_pub_;
  rclcpp::Publisher<back_odom::msg::IcpDebug>::SharedPtr icp_debug_pub_;
  rclcpp::Publisher<back_odom::msg::PoseRefine>::SharedPtr pose_refine_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unordered_map<std::string, Sophus::SE3d> extrinsics_;

  nav_msgs::msg::Path path_;
  nav_msgs::msg::Path lidar_debug_path_;
  MatchResult last_match_{};
  bool has_last_match_{false};
  rclcpp::Time last_path_stamp_{0, 0, RCL_ROS_TIME};
  bool has_published_path_{false};
  rclcpp::Time last_marker_stamp_{0, 0, RCL_ROS_TIME};
  bool has_published_markers_{false};
  rclcpp::Time last_health_stamp_{0, 0, RCL_ROS_TIME};
  bool has_published_health_{false};
  std::atomic<bool> aligned_{false};
  std::atomic<std::uint64_t> clouds_received_{0};
  std::atomic<std::uint64_t> clouds_before_align_{0};
  std::atomic<std::uint64_t> scans_processed_{0};
  /// Worker-thread count of scans that produced no pose update.
  std::uint32_t skipped_scans_{0};
  std::atomic<bool> logged_first_cloud_{false};

  std::string imu_topic_;
  std::string pointcloud_topic_;
  bool pointcloud_reliable_{true};
  std::string parent_frame_;
  std::string child_frame_;
  Eigen::Vector3d previous_angular_velocity_{Eigen::Vector3d::Zero()};
  double previous_angular_velocity_stamp_{0.0};
  bool has_previous_angular_velocity_{false};
  std::size_t alignment_sample_count_{100};
  std::size_t path_max_poses_{1000};
  double imu_max_pair_dt_{0.05};
  double imu_wait_timeout_{0.5};
  double path_min_dt_{0.1};
  double marker_min_dt_{0.1};
  double health_min_dt_{0.1};
  double map_publish_period_{1.0};
  double min_range_{1.0};
  double max_range_{0.0};
  bool publish_tf_{true};
  bool initial_pose_sent_{false};
  double twist_variance_vx_{0.0225};
  double twist_variance_vy_{0.0025};
  double twist_variance_vz_{0.0025};
  double twist_variance_wz_{0.01};
  double pose_variance_xy_{0.04};
  double pose_variance_z_{0.40};
  double pose_variance_roll_pitch_{0.01};
  double pose_variance_yaw_{0.01};
  bool imu_left_handed_{false};
  std::string time_field_;
  bool align_to_ground_truth_{false};
  std::string ground_truth_odom_topic_{"/boreas/ground_truth/odom"};

  struct GroundTruthSample
  {
    double stamp{0.0};
    Sophus::SE3d pose{};
  };
  /// Guards ground-truth samples and map_from_local_. Never acquire state_mutex_ while holding this.
  mutable std::mutex align_mutex_;
  std::vector<GroundTruthSample> ground_truth_samples_;
  Sophus::SE3d map_from_local_{};
  std::atomic<bool> map_from_local_locked_{false};
};

}  // namespace back_odom

#endif  // BACK_ODOM__BACK_ODOM_NODE_HPP_
