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

#include "back_odom/imu_alignment.hpp"

#include <Eigen/Geometry>
#include <rclcpp_components/register_node_macro.hpp>
#include <tf2/exceptions.hpp>
#include <tf2/time.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
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

sensor_msgs::msg::PointCloud2 to_pointcloud(
  const rclcpp::Time & stamp, const std::string & frame_id,
  const std::vector<Eigen::Vector3d> & points)
{
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.stamp = stamp;
  cloud.header.frame_id = frame_id;
  cloud.height = 1;
  cloud.width = static_cast<std::uint32_t>(points.size());
  cloud.is_dense = true;
  cloud.is_bigendian = false;

  sensor_msgs::msg::PointField field_x;
  field_x.name = "x";
  field_x.offset = 0;
  field_x.datatype = sensor_msgs::msg::PointField::FLOAT32;
  field_x.count = 1;
  sensor_msgs::msg::PointField field_y = field_x;
  field_y.name = "y";
  field_y.offset = 4;
  sensor_msgs::msg::PointField field_z = field_x;
  field_z.name = "z";
  field_z.offset = 8;
  cloud.fields = {field_x, field_y, field_z};
  cloud.point_step = 12;
  cloud.row_step = cloud.point_step * cloud.width;
  cloud.data.resize(static_cast<std::size_t>(cloud.row_step));
  for (std::size_t index = 0; index < points.size(); ++index) {
    const float values[3] = {
      static_cast<float>(points[index].x()), static_cast<float>(points[index].y()),
      static_cast<float>(points[index].z())};
    std::memcpy(cloud.data.data() + index * cloud.point_step, values, sizeof(values));
  }
  return cloud;
}

}  // namespace

BackOdomNode::BackOdomNode(const rclcpp::NodeOptions & options) : Node("back_odom_node", options)
{
  const auto imu_topic = this->declare_parameter<std::string>("imu_topic", "/imu/data");
  const auto pointcloud_topic =
    this->declare_parameter<std::string>("pointcloud_topic", "/pointcloud");
  publish_tf_ = this->declare_parameter<bool>("publish_tf", true);
  imu_left_handed_ = this->declare_parameter<bool>("imu_left_handed", false);
  parent_frame_ = this->declare_parameter<std::string>("parent_frame", "map");
  child_frame_ = this->declare_parameter<std::string>("child_frame", "base_link");
  const int alignment_sample_count = this->declare_parameter<int>("alignment_sample_count", 100);
  const double stationary_gyro_thresh =
    this->declare_parameter<double>("stationary_gyro_thresh", 0.05);
  const double stationary_accel_dev_thresh =
    this->declare_parameter<double>("stationary_accel_dev_thresh", 0.5);
  const double gravity = this->declare_parameter<double>("gravity", 9.81);
  const double max_dt = this->declare_parameter<double>("max_dt", 0.1);
  path_min_dt_ = this->declare_parameter<double>("path_min_dt", 0.1);
  const int path_max_poses = this->declare_parameter<int>("path_max_poses", 1000);
  time_field_ = this->declare_parameter<std::string>("time_field", "time");
  LidarMatchParams lidar_params;
  lidar_params.voxel_size = this->declare_parameter<double>("voxel_size", 0.5);
  lidar_params.crop_longitudinal = this->declare_parameter<double>("crop_longitudinal", 75.0);
  lidar_params.crop_lateral = this->declare_parameter<double>("crop_lateral", 50.0);
  lidar_params.max_correspondence_distance =
    this->declare_parameter<double>("max_correspondence_distance", 2.0);
  lidar_params.kernel_scale = this->declare_parameter<double>("kernel_scale", 0.5);
  lidar_params.convergence_criterion =
    this->declare_parameter<double>("convergence_criterion", 1.0e-4);
  lidar_params.max_iterations = this->declare_parameter<int>("max_iterations", 50);
  lidar_params.max_points_per_voxel = this->declare_parameter<int>("max_points_per_voxel", 20);
  lidar_params.backward_match_stride = this->declare_parameter<int>("backward_match_stride", 3);

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
  lidar_matcher_ = std::make_unique<LidarImuMatcher>(lidar_params);

  imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
    imu_topic, rclcpp::QoS(10),
    [this](const sensor_msgs::msg::Imu::ConstSharedPtr msg) { this->callback_imu(msg); });
  pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    pointcloud_topic, rclcpp::QoS(10),
    [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
      this->callback_pointcloud(msg);
    });

  odometry_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("back_odom", rclcpp::QoS(10));
  imu_odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("back_odom/imu", rclcpp::QoS(10));
  path_pub_ = this->create_publisher<nav_msgs::msg::Path>("back_odom/path", rclcpp::QoS(10));
  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "back_odom/markers", rclcpp::QoS(10));
  map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("back_odom/map", rclcpp::QoS(1));
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_, this, false);

  RCLCPP_INFO(
    this->get_logger(), "BackOdomNode initialized. imu_left_handed=%s",
    imu_left_handed_ ? "true" : "false");
}

BackOdomNode::~BackOdomNode()
{
  RCLCPP_INFO(this->get_logger(), "BackOdomNode destroyed.");
}

void BackOdomNode::callback_imu(const sensor_msgs::msg::Imu::ConstSharedPtr msg)
{
  if (msg->header.frame_id.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "IMU frame_id is empty, so it cannot be looked up in the TF tree.");
    return;
  }
  const std::string source_frame = msg->header.frame_id;
  Sophus::SE3d base_from_imu;
  if (!base_from_frame(source_frame, base_from_imu)) {
    return;
  }

  const rclcpp::Time stamp(msg->header.stamp);
  ImuSample measured;
  measured.stamp = stamp.seconds();
  measured.linear_acceleration =
    to_vector3(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
  measured.angular_velocity =
    to_vector3(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
  if (imu_left_handed_) {
    measured = to_right_handed_imu(measured);
  }
  const ImuSample sample = transform_imu_sample(
    measured, base_from_imu, previous_angular_velocity_, previous_angular_velocity_stamp_,
    has_previous_angular_velocity_);
  previous_angular_velocity_ = sample.angular_velocity;
  previous_angular_velocity_stamp_ = sample.stamp;
  has_previous_angular_velocity_ = true;

  ProcessorOutput output = imu_processor_->process(sample);

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
    publish_markers(output, stamp);
    return;
  }

  const MatchResult backward = lidar_matcher_->on_imu(*imu_processor_);
  if (backward.applied) {
    publish_odometry(imu_odom_pub_, output, stamp);
    output = imu_processor_->output_at(sample);
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Backward IMU correction. translation error %.3f m",
      backward.correction.translation().norm());
  }

  publish_odometry(odometry_pub_, output, stamp);
  publish_path(output, stamp);
  publish_markers(output, stamp);
  if (publish_tf_) {
    publish_tf(output, stamp);
  }
}

void BackOdomNode::callback_pointcloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  if (!aligned_ || !imu_processor_->aligned()) {
    return;
  }
  LidarScan scan = scan_from_cloud(*msg);
  if (scan.points.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "Skipping lidar scan. Need xyz points and a timestamp field.");
    return;
  }
  if (msg->header.frame_id.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "Point cloud frame_id is empty, so it cannot be looked up in the TF tree.");
    return;
  }
  const std::string source_frame = msg->header.frame_id;
  Sophus::SE3d base_from_lidar;
  if (!base_from_frame(source_frame, base_from_lidar)) {
    return;
  }
  lidar_matcher_->set_body_from_lidar(base_from_lidar);
  const rclcpp::Time stamp(msg->header.stamp);
  const ProcessorOutput prior = imu_processor_->output_at(
    ImuSample{scan.stamp, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()});
  const MatchResult matched = lidar_matcher_->on_scan(scan, *imu_processor_);
  if (!matched.applied) {
    return;
  }
  publish_odometry(imu_odom_pub_, prior, stamp);
  const ProcessorOutput corrected = imu_processor_->output_at(
    ImuSample{scan.stamp, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()});
  publish_odometry(odometry_pub_, corrected, stamp);
  publish_path(corrected, stamp);
  publish_markers(corrected, stamp);
  publish_local_map(stamp);
  if (publish_tf_) {
    publish_tf(corrected, stamp);
  }
  if (matched.first_scan) {
    RCLCPP_INFO(this->get_logger(), "Stored the first lidar scan in the local map.");
  }
}

bool BackOdomNode::base_from_frame(
  const std::string & source_frame, Sophus::SE3d & target_from_source)
{
  const auto found = extrinsics_.find(source_frame);
  if (found != extrinsics_.end()) {
    target_from_source = found->second;
    return true;
  }

  try {
    const geometry_msgs::msg::TransformStamped transform = tf_buffer_->lookupTransform(
      child_frame_, source_frame, tf2::TimePointZero, tf2::durationFromSec(0.0));
    const Eigen::Isometry3d isometry = tf2::transformToEigen(transform);
    target_from_source =
      Sophus::SE3d(Sophus::SO3d(Eigen::Quaterniond(isometry.rotation())), isometry.translation());
  } catch (const tf2::TransformException & exception) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000, "Waiting for TF %s <- %s: %s",
      child_frame_.c_str(), source_frame.c_str(), exception.what());
    return false;
  }

  RCLCPP_INFO(
    this->get_logger(), "TF %s <- %s xyz=(%.3f, %.3f, %.3f)", child_frame_.c_str(),
    source_frame.c_str(), target_from_source.translation().x(),
    target_from_source.translation().y(), target_from_source.translation().z());
  extrinsics_.emplace(source_frame, target_from_source);
  return true;
}

LidarScan BackOdomNode::scan_from_cloud(const sensor_msgs::msg::PointCloud2 & cloud) const
{
  LidarScan scan;
  scan.stamp = rclcpp::Time(cloud.header.stamp).seconds();
  const auto x_field = std::find_if(
    cloud.fields.cbegin(), cloud.fields.cend(),
    [](const sensor_msgs::msg::PointField & field) { return field.name == "x"; });
  const auto y_field = std::find_if(
    cloud.fields.cbegin(), cloud.fields.cend(),
    [](const sensor_msgs::msg::PointField & field) { return field.name == "y"; });
  const auto z_field = std::find_if(
    cloud.fields.cbegin(), cloud.fields.cend(),
    [](const sensor_msgs::msg::PointField & field) { return field.name == "z"; });
  auto time_field = std::find_if(
    cloud.fields.cbegin(), cloud.fields.cend(),
    [this](const sensor_msgs::msg::PointField & field) { return field.name == time_field_; });
  if (time_field == cloud.fields.cend()) {
    for (const char * fallback : {"time", "t", "timestamp"}) {
      time_field = std::find_if(
        cloud.fields.cbegin(), cloud.fields.cend(),
        [fallback](const sensor_msgs::msg::PointField & field) { return field.name == fallback; });
      if (time_field != cloud.fields.cend()) {
        break;
      }
    }
  }
  if (
    x_field == cloud.fields.cend() || y_field == cloud.fields.cend() ||
    z_field == cloud.fields.cend() || time_field == cloud.fields.cend()) {
    return scan;
  }

  const auto read_float = [&cloud](
                            const sensor_msgs::msg::PointField & field, const std::size_t index) {
    const std::uint8_t * pointer = cloud.data.data() + index * cloud.point_step + field.offset;
    if (field.datatype == sensor_msgs::msg::PointField::FLOAT32) {
      float value = 0.0F;
      std::memcpy(&value, pointer, sizeof(float));
      return static_cast<double>(value);
    }
    if (field.datatype == sensor_msgs::msg::PointField::FLOAT64) {
      double value = 0.0;
      std::memcpy(&value, pointer, sizeof(double));
      return value;
    }
    return std::numeric_limits<double>::quiet_NaN();
  };

  const std::size_t count = static_cast<std::size_t>(cloud.width) * cloud.height;
  scan.points.reserve(count);
  scan.timestamps.reserve(count);
  double max_abs_time = 0.0;
  for (std::size_t index = 0; index < count; ++index) {
    const double x = read_float(*x_field, index);
    const double y = read_float(*y_field, index);
    const double z = read_float(*z_field, index);
    const double time = read_float(*time_field, index);
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(time)) {
      continue;
    }
    scan.points.emplace_back(x, y, z);
    scan.timestamps.push_back(time);
    max_abs_time = std::max(max_abs_time, std::abs(time));
  }
  if (max_abs_time < 1000.0) {
    for (double & time : scan.timestamps) {
      time += scan.stamp;
    }
  }
  return scan;
}

void BackOdomNode::publish_odometry(
  const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr & publisher,
  const ProcessorOutput & output, const rclcpp::Time & stamp)
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
  publisher->publish(odometry);
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

void BackOdomNode::publish_local_map(const rclcpp::Time & stamp)
{
  map_pub_->publish(to_pointcloud(stamp, parent_frame_, lidar_matcher_->local_map()));
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
