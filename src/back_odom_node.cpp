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

#include "back_odom/back_odom_node.hpp"

#include <Eigen/Geometry>
#include <rclcpp_components/register_node_macro.hpp>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <cstddef>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace back_odom
{
namespace
{

Eigen::Vector3d to_vector3(const double x, const double y, const double z)
{
  return Eigen::Vector3d(x, y, z);
}

void set_quaternion(geometry_msgs::msg::Quaternion & orientation, const Sophus::SO3d & rotation)
{
  const Eigen::Quaterniond quaternion = rotation.unit_quaternion();
  orientation.x = quaternion.x();
  orientation.y = quaternion.y();
  orientation.z = quaternion.z();
  orientation.w = quaternion.w();
}

geometry_msgs::msg::Point to_point(const Eigen::Vector3d & vector)
{
  geometry_msgs::msg::Point point;
  point.x = vector.x();
  point.y = vector.y();
  point.z = vector.z();
  return point;
}

visualization_msgs::msg::Marker make_arrow(
  const rclcpp::Time & stamp, const std::string & frame_id, const int id,
  const Eigen::Vector3d & start, const Eigen::Vector3d & end, const float red, const float green,
  const float blue)
{
  visualization_msgs::msg::Marker marker;
  marker.header.stamp = stamp;
  marker.header.frame_id = frame_id;
  marker.ns = "back_odom";
  marker.id = id;
  marker.type = visualization_msgs::msg::Marker::ARROW;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.05;
  marker.scale.y = 0.1;
  marker.scale.z = 0.15;
  marker.color.r = red;
  marker.color.g = green;
  marker.color.b = blue;
  marker.color.a = 1.0F;
  if ((end - start).norm() < 1e-3) {
    marker.action = visualization_msgs::msg::Marker::DELETE;
    return marker;
  }
  marker.points.push_back(to_point(start));
  marker.points.push_back(to_point(end));
  return marker;
}

}  // namespace

BackOdomNode::BackOdomNode(const rclcpp::NodeOptions & options) : Node("back_odom_node", options)
{
  const auto imu_topic = this->declare_parameter<std::string>("imu_topic", "/imu/data");
  const auto pointcloud_topic =
    this->declare_parameter<std::string>("pointcloud_topic", "/pointcloud");
  publish_tf_ = this->declare_parameter<bool>("publish_tf", true);
  parent_frame_ = this->declare_parameter<std::string>("parent_frame", "map");
  child_frame_ = this->declare_parameter<std::string>("child_frame", "pose_estimator_base_link");
  const int alignment_sample_count = this->declare_parameter<int>("alignment_sample_count", 100);
  const double stationary_gyro_thresh =
    this->declare_parameter<double>("stationary_gyro_thresh", 0.05);
  const double stationary_accel_dev_thresh =
    this->declare_parameter<double>("stationary_accel_dev_thresh", 0.5);
  const double gravity = this->declare_parameter<double>("gravity", 9.81);
  const double max_dt = this->declare_parameter<double>("max_dt", 0.1);
  path_min_dt_ = this->declare_parameter<double>("path_min_dt", 0.1);
  const int path_max_poses = this->declare_parameter<int>("path_max_poses", 1000);

  if (alignment_sample_count <= 0) {
    throw std::invalid_argument("alignment_sample_count must be positive");
  }
  if (path_max_poses <= 0) {
    throw std::invalid_argument("path_max_poses must be positive");
  }

  alignment_sample_count_ = static_cast<std::size_t>(alignment_sample_count);
  path_max_poses_ = static_cast<std::size_t>(path_max_poses);

  ProcessorParams params;
  params.alignment_sample_count = alignment_sample_count_;
  params.stationary_gyro_thresh = stationary_gyro_thresh;
  params.stationary_accel_dev_thresh = stationary_accel_dev_thresh;
  params.gravity = gravity;
  params.max_dt = max_dt;
  imu_processor_ = std::make_unique<ImuProcessor>(params);

  imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
    imu_topic, rclcpp::QoS(10),
    [this](const sensor_msgs::msg::Imu::ConstSharedPtr msg) { this->callback_imu(msg); });
  pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    pointcloud_topic, rclcpp::QoS(10),
    [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
      this->callback_pointcloud(msg);
    });

  odometry_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("back_odom", rclcpp::QoS(10));
  path_pub_ = this->create_publisher<nav_msgs::msg::Path>("back_odom/path", rclcpp::QoS(10));
  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "back_odom/markers", rclcpp::QoS(10));
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  RCLCPP_INFO(this->get_logger(), "BackOdomNode initialized.");
}

BackOdomNode::~BackOdomNode()
{
  RCLCPP_INFO(this->get_logger(), "BackOdomNode destroyed.");
}

void BackOdomNode::callback_imu(const sensor_msgs::msg::Imu::ConstSharedPtr msg)
{
  const rclcpp::Time stamp(msg->header.stamp);
  ImuSample sample;
  sample.stamp = stamp.seconds();
  sample.linear_acceleration =
    to_vector3(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
  sample.angular_velocity =
    to_vector3(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

  const ProcessorOutput output = imu_processor_->process(sample);
  publish_markers(output, stamp);

  if (!output.aligned && output.sample_count >= alignment_sample_count_) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "IMU window is full but not stationary. Waiting to align.");
  }
  if (!aligned_ && output.aligned) {
    RCLCPP_INFO(this->get_logger(), "Initial IMU alignment finished. gravity=%.3f", output.gravity);
  }
  aligned_ = output.aligned;
  if (!output.aligned) {
    return;
  }

  publish_odometry(output, stamp);
  publish_path(output, stamp);
  if (publish_tf_) {
    publish_tf(output, stamp);
  }
}

void BackOdomNode::callback_pointcloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  RCLCPP_DEBUG(
    this->get_logger(), "Received PointCloud2 data with width: %u, height: %u", msg->width,
    msg->height);
}

void BackOdomNode::publish_odometry(const ProcessorOutput & output, const rclcpp::Time & stamp)
{
  nav_msgs::msg::Odometry odometry;
  odometry.header.stamp = stamp;
  odometry.header.frame_id = parent_frame_;
  odometry.child_frame_id = child_frame_;
  odometry.pose.pose.position.x = output.position.x();
  odometry.pose.pose.position.y = output.position.y();
  odometry.pose.pose.position.z = output.position.z();
  set_quaternion(odometry.pose.pose.orientation, output.orientation);

  const Eigen::Vector3d velocity_body = output.orientation.inverse() * output.velocity_world;
  odometry.twist.twist.linear.x = velocity_body.x();
  odometry.twist.twist.linear.y = velocity_body.y();
  odometry.twist.twist.linear.z = velocity_body.z();
  odometry.twist.twist.angular.x = output.angular_velocity_body.x();
  odometry.twist.twist.angular.y = output.angular_velocity_body.y();
  odometry.twist.twist.angular.z = output.angular_velocity_body.z();
  odometry_pub_->publish(odometry);
}

void BackOdomNode::publish_path(const ProcessorOutput & output, const rclcpp::Time & stamp)
{
  const bool append_pose =
    !has_published_path_ || (stamp - last_path_stamp_).seconds() >= path_min_dt_;
  if (append_pose) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.stamp = stamp;
    pose.header.frame_id = parent_frame_;
    pose.pose.position.x = output.position.x();
    pose.pose.position.y = output.position.y();
    pose.pose.position.z = output.position.z();
    set_quaternion(pose.pose.orientation, output.orientation);
    path_.poses.push_back(pose);
    if (path_.poses.size() > path_max_poses_) {
      path_.poses.erase(path_.poses.begin());
    }
    last_path_stamp_ = stamp;
    has_published_path_ = true;
  }

  path_.header.stamp = stamp;
  path_.header.frame_id = parent_frame_;
  path_pub_->publish(path_);
}

void BackOdomNode::publish_tf(const ProcessorOutput & output, const rclcpp::Time & stamp)
{
  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = stamp;
  transform.header.frame_id = parent_frame_;
  transform.child_frame_id = child_frame_;
  transform.transform.translation.x = output.position.x();
  transform.transform.translation.y = output.position.y();
  transform.transform.translation.z = output.position.z();
  set_quaternion(transform.transform.rotation, output.orientation);
  tf_broadcaster_->sendTransform(transform);
}

void BackOdomNode::publish_markers(const ProcessorOutput & output, const rclcpp::Time & stamp)
{
  const Eigen::Vector3d origin = output.position;
  const Sophus::SO3d & rotation = output.orientation;
  visualization_msgs::msg::MarkerArray markers;
  markers.markers.push_back(make_arrow(
    stamp, parent_frame_, 0, origin, origin + rotation * Eigen::Vector3d::UnitX(), 1.0F, 0.0F,
    0.0F));
  markers.markers.push_back(make_arrow(
    stamp, parent_frame_, 1, origin, origin + rotation * Eigen::Vector3d::UnitY(), 0.0F, 1.0F,
    0.0F));
  markers.markers.push_back(make_arrow(
    stamp, parent_frame_, 2, origin, origin + rotation * Eigen::Vector3d::UnitZ(), 0.0F, 0.0F,
    1.0F));
  markers.markers.push_back(
    make_arrow(stamp, parent_frame_, 3, origin, origin + output.velocity_world, 1.0F, 1.0F, 0.0F));

  const Eigen::Vector3d specific_force_world = rotation * output.specific_force_body;
  const double gravity_scale = output.gravity > 1e-3 ? output.gravity : 1.0;
  markers.markers.push_back(make_arrow(
    stamp, parent_frame_, 4, origin, origin + specific_force_world / gravity_scale, 1.0F, 0.0F,
    1.0F));
  markers.markers.push_back(make_arrow(
    stamp, parent_frame_, 5, origin, origin + Eigen::Vector3d(0.0, 0.0, -1.0), 1.0F, 0.5F, 0.0F));

  visualization_msgs::msg::Marker text;
  text.header.stamp = stamp;
  text.header.frame_id = parent_frame_;
  text.ns = "back_odom";
  text.id = 6;
  text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  text.action = visualization_msgs::msg::Marker::ADD;
  text.pose.position.x = origin.x();
  text.pose.position.y = origin.y();
  text.pose.position.z = origin.z() + 1.5;
  text.pose.orientation.w = 1.0;
  text.scale.z = 0.3;
  text.color.r = 1.0F;
  text.color.g = 1.0F;
  text.color.b = 1.0F;
  text.color.a = 1.0F;

  std::ostringstream stream;
  stream << std::fixed << std::setprecision(2);
  if (output.aligned) {
    stream << "tracking |f|=" << output.specific_force_body.norm()
           << " |w|=" << output.angular_velocity_body.norm();
  } else {
    stream << "collecting " << output.sample_count << "/" << alignment_sample_count_;
  }
  text.text = stream.str();
  markers.markers.push_back(text);
  marker_pub_->publish(markers);
}

}  // namespace back_odom

RCLCPP_COMPONENTS_REGISTER_NODE(back_odom::BackOdomNode)
