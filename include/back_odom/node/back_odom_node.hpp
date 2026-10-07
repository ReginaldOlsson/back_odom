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

#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <tf2_ros/transform_broadcaster.h>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace back_odom
{

class BackOdomNode : public rclcpp::Node
{
public:
  explicit BackOdomNode(const rclcpp::NodeOptions & options);
  ~BackOdomNode() override;

private:
  void callback_imu(const sensor_msgs::msg::Imu::ConstSharedPtr msg);
  void callback_pointcloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
  void match_pending_scan(bool force);
  void publish_health(const rclcpp::Time & stamp);
  void publish_debug_odometry(const MatchResult & match, const rclcpp::Time & stamp);
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
  void publish_markers(const ProcessorOutput & output, const rclcpp::Time & stamp);
  void publish_scan_poses(const rclcpp::Time & stamp);
  void publish_local_map(const rclcpp::Time & stamp);

  std::unique_ptr<ImuProcessor> imu_processor_;
  std::unique_ptr<LidarImuMatcher> lidar_matcher_;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;

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
  bool aligned_{false};
  struct PendingScan
  {
    LidarScan scan;
    rclcpp::Time header_stamp;
    std::string source_frame;
  };
  std::optional<PendingScan> pending_scan_;

  std::string parent_frame_;
  std::string child_frame_;
  Eigen::Vector3d previous_angular_velocity_{Eigen::Vector3d::Zero()};
  double previous_angular_velocity_stamp_{0.0};
  bool has_previous_angular_velocity_{false};
  std::size_t alignment_sample_count_{100};
  std::size_t path_max_poses_{1000};
  double imu_max_pair_dt_{0.05};
  double path_min_dt_{0.1};
  bool publish_tf_{true};
  bool initial_pose_sent_{false};
  double twist_variance_vx_{0.05};
  double twist_variance_wz_{0.01};
  double pose_variance_xy_{0.05};
  double pose_variance_yaw_{0.01};
  bool imu_left_handed_{false};
  std::string time_field_;
};

}  // namespace back_odom

#endif  // BACK_ODOM__BACK_ODOM_NODE_HPP_
