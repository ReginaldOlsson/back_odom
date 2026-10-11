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

#include "back_odom/eval/kitti_metrics.hpp"
#include "back_odom/eval/trajectory_sync.hpp"

#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include <back_odom/msg/kitti_error.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace back_odom
{
namespace
{

Sophus::SE3d pose_from_odometry(const nav_msgs::msg::Odometry & odometry)
{
  Eigen::Quaterniond quaternion(
    odometry.pose.pose.orientation.w, odometry.pose.pose.orientation.x,
    odometry.pose.pose.orientation.y, odometry.pose.pose.orientation.z);
  if (quaternion.norm() < 1.0e-9) {
    quaternion = Eigen::Quaterniond::Identity();
  } else {
    quaternion.normalize();
  }
  return Sophus::SE3d(
    Sophus::SO3d(quaternion),
    Eigen::Vector3d(
      odometry.pose.pose.position.x, odometry.pose.pose.position.y, odometry.pose.pose.position.z));
}

double stamp_seconds(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1.0e-9;
}

}  // namespace

class KittiErrorNode : public rclcpp::Node
{
public:
  explicit KittiErrorNode(const rclcpp::NodeOptions & options) : Node("kitti_error_node", options)
  {
    const std::string ground_truth_topic =
      this->declare_parameter<std::string>("ground_truth_topic", "/boreas/ground_truth/odom");
    const std::string estimate_topic =
      this->declare_parameter<std::string>("estimate_topic", "/back_odom");
    const std::string error_topic =
      this->declare_parameter<std::string>("error_topic", "/back_odom/kpi/kitti");
    sync_.set_max_dt(this->declare_parameter<double>("max_association_dt", 0.05));

    const auto qos = rclcpp::QoS(rclcpp::KeepLast(50)).reliable();
    ground_truth_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      ground_truth_topic, qos,
      [this](const nav_msgs::msg::Odometry::ConstSharedPtr msg) { this->on_ground_truth(msg); });
    estimate_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      estimate_topic, qos,
      [this](const nav_msgs::msg::Odometry::ConstSharedPtr msg) { this->on_estimate(msg); });
    error_pub_ = this->create_publisher<back_odom::msg::KittiError>(error_topic, rclcpp::QoS(10));
    RCLCPP_INFO(
      this->get_logger(), "KITTI KPI node listening to %s and %s", ground_truth_topic.c_str(),
      estimate_topic.c_str());
  }

private:
  void on_estimate(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    absorb(sync_.push_estimate(stamp_seconds(msg->header.stamp), pose_from_odometry(*msg)));
  }

  void on_ground_truth(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    absorb(sync_.push_ground_truth(
      stamp_seconds(msg->header.stamp), msg->header.stamp.sec, msg->header.stamp.nanosec,
      pose_from_odometry(*msg)));
  }

  void absorb(const std::vector<AssociatedPose> & pairs)
  {
    if (pairs.empty()) {
      return;
    }
    for (const AssociatedPose & pair : pairs) {
      ground_truth_.push_back(pair.ground_truth.matrix());
      estimate_.push_back(pair.estimate.matrix());
    }
    const KittiSequenceError error = evaluate_kitti_sequence(ground_truth_, estimate_);
    if (error.num_segments == 0 || error.num_segments == published_segments_) {
      return;
    }
    published_segments_ = error.num_segments;
    back_odom::msg::KittiError message;
    message.header.stamp = this->now();
    message.header.frame_id = "map";
    message.translation_percent = error.translation_percent;
    message.rotation_deg_per_100m = error.rotation_deg_per_100m;
    message.num_segments = static_cast<std::uint32_t>(error.num_segments);
    error_pub_->publish(message);
    RCLCPP_INFO(
      this->get_logger(), "KITTI translation %.3f %%  rotation %.3f deg/100m  segments %d",
      error.translation_percent, error.rotation_deg_per_100m, error.num_segments);
  }

  TrajectorySync sync_;
  std::mutex mutex_;
  std::vector<Eigen::Matrix4d> ground_truth_;
  std::vector<Eigen::Matrix4d> estimate_;
  int published_segments_{0};
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr ground_truth_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr estimate_sub_;
  rclcpp::Publisher<back_odom::msg::KittiError>::SharedPtr error_pub_;
};

}  // namespace back_odom

RCLCPP_COMPONENTS_REGISTER_NODE(back_odom::KittiErrorNode)
