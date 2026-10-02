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

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <gtest/gtest.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include <chrono>
#include <memory>

namespace
{

void spin_for(
  rclcpp::executors::SingleThreadedExecutor & executor, const std::chrono::milliseconds budget)
{
  const auto deadline = std::chrono::steady_clock::now() + budget;
  while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline) {
    executor.spin_some(std::chrono::milliseconds(10));
  }
}

sensor_msgs::msg::Imu make_imu(
  const int32_t sec, const uint32_t nanosec, const double ax, const double wz)
{
  sensor_msgs::msg::Imu imu;
  imu.header.stamp.sec = sec;
  imu.header.stamp.nanosec = nanosec;
  imu.header.frame_id = "imu";
  imu.linear_acceleration.z = -9.81;
  imu.linear_acceleration.x = ax;
  imu.angular_velocity.z = wz;
  return imu;
}

}  // namespace

TEST(EkfMessages, imu_publishes_body_twist_and_not_the_pose)
{
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }

  rclcpp::NodeOptions options;
  options.parameter_overrides({
    {"alignment_sample_count", 5},
    {"publish_tf", false},
    {"imu_topic", std::string("/imu/data")},
  });
  auto odom = std::make_shared<back_odom::BackOdomNode>(options);
  auto helper = std::make_shared<rclcpp::Node>("ekf_message_helper");
  tf2_ros::StaticTransformBroadcaster broadcaster(helper);

  geometry_msgs::msg::TransformStamped mount;
  mount.header.frame_id = "base_link";
  mount.child_frame_id = "imu";
  mount.transform.rotation.w = 1.0;
  broadcaster.sendTransform(mount);

  geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr latest_twist;
  std::size_t twist_count = 0;
  std::size_t pose_count = 0;
  std::size_t initial_count = 0;
  auto twist_sub = helper->create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
    "/back_odom/twist_with_covariance", 10,
    [&](const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr msg) {
      latest_twist = msg;
      ++twist_count;
    });
  auto pose_sub = helper->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
    "/back_odom/pose_with_covariance", 10,
    [&](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr) { ++pose_count; });
  auto initial_sub = helper->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
    "/back_odom/initialpose", 10,
    [&](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr) { ++initial_count; });
  auto imu_pub = helper->create_publisher<sensor_msgs::msg::Imu>("/imu/data", 10);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(odom);
  executor.add_node(helper);
  (void)twist_sub;
  (void)pose_sub;
  (void)initial_sub;
  spin_for(executor, std::chrono::milliseconds(300));

  for (int sample = 0; sample < 5; ++sample) {
    imu_pub->publish(make_imu(1, static_cast<uint32_t>(sample) * 10000000U, 0.0, 0.0));
    spin_for(executor, std::chrono::milliseconds(50));
  }
  imu_pub->publish(make_imu(1, 140000000U, 1.0, 0.25));
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (twist_count < 2U && rclcpp::ok() && std::chrono::steady_clock::now() < deadline) {
    executor.spin_some(std::chrono::milliseconds(10));
  }

  ASSERT_NE(latest_twist, nullptr);
  EXPECT_EQ(latest_twist->header.frame_id, "base_link");
  EXPECT_NEAR(latest_twist->twist.twist.linear.x, 0.1, 1.0e-3);
  EXPECT_NEAR(latest_twist->twist.twist.angular.z, 0.25, 1.0e-6);
  EXPECT_DOUBLE_EQ(latest_twist->twist.covariance[0], 0.05);
  EXPECT_DOUBLE_EQ(latest_twist->twist.covariance[35], 0.01);
  EXPECT_EQ(pose_count, 0U);
  EXPECT_EQ(initial_count, 0U);

  executor.remove_node(helper);
  executor.remove_node(odom);
}
