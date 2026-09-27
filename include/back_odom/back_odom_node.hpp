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

#include "back_odom/imu_processor.hpp"

#include <rclcpp/rclcpp.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <tf2_ros/transform_broadcaster.h>

#include <memory>
#include <string>

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
  void publish_odometry(const ProcessorOutput & output, const rclcpp::Time & stamp);
  void publish_path(const ProcessorOutput & output, const rclcpp::Time & stamp);
  void publish_tf(const ProcessorOutput & output, const rclcpp::Time & stamp);
  void publish_markers(const ProcessorOutput & output, const rclcpp::Time & stamp);

  std::unique_ptr<ImuProcessor> imu_processor_;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  nav_msgs::msg::Path path_;
  rclcpp::Time last_path_stamp_{0, 0, RCL_ROS_TIME};
  bool has_published_path_{false};
  bool aligned_{false};

  std::string parent_frame_;
  std::string child_frame_;
  std::size_t alignment_sample_count_{100};
  std::size_t path_max_poses_{1000};
  double path_min_dt_{0.1};
  bool publish_tf_{true};
};

}  // namespace back_odom

#endif  // BACK_ODOM__BACK_ODOM_NODE_HPP_
