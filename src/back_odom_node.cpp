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
#include <unordered_set>

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

sensor_msgs::msg::PointField float_field(const std::string & name, const std::uint32_t offset)
{
  sensor_msgs::msg::PointField field;
  field.name = name;
  field.offset = offset;
  field.datatype = sensor_msgs::msg::PointField::FLOAT32;
  field.count = 1;
  return field;
}

std::vector<LocalMapPoint> thin_for_display(const std::vector<LocalMapPoint> & points, const double voxel)
{
  if (voxel <= 0.0 || points.size() < 2) {
    return points;
  }
  struct Key
  {
    std::int64_t x;
    std::int64_t y;
    std::int64_t z;
    bool operator==(const Key & other) const { return x == other.x && y == other.y && z == other.z; }
  };
  struct Hash
  {
    std::size_t operator()(const Key & key) const
    {
      std::size_t value = static_cast<std::size_t>(key.x);
      value ^= static_cast<std::size_t>(key.y) + 0x9e3779b97f4a7c15ULL + (value << 6U) + (value >> 2U);
      value ^= static_cast<std::size_t>(key.z) + 0x9e3779b97f4a7c15ULL + (value << 6U) + (value >> 2U);
      return value;
    }
  };
  std::unordered_set<Key, Hash> occupied;
  occupied.reserve(points.size());
  std::vector<LocalMapPoint> kept;
  kept.reserve(points.size() / 4U);
  const double inverse = 1.0 / voxel;
  for (const LocalMapPoint & point : points) {
    const Key key{
      static_cast<std::int64_t>(std::floor(point.position.x() * inverse)),
      static_cast<std::int64_t>(std::floor(point.position.y() * inverse)),
      static_cast<std::int64_t>(std::floor(point.position.z() * inverse))};
    if (occupied.insert(key).second) {
      kept.push_back(point);
    }
  }
  return kept;
}

sensor_msgs::msg::PointCloud2 to_annotated_pointcloud(
  const rclcpp::Time & stamp, const std::string & frame_id,
  const std::vector<LocalMapPoint> & points)
{
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.stamp = stamp;
  cloud.header.frame_id = frame_id;
  cloud.height = 1;
  cloud.width = static_cast<std::uint32_t>(points.size());
  cloud.is_dense = true;
  cloud.is_bigendian = false;
  cloud.fields = {
    float_field("x", 0),
    float_field("y", 4),
    float_field("z", 8),
    float_field("intensity", 12),
    float_field("scan_id", 16),
    float_field("collective_passes", 20)};
  cloud.point_step = 24;
  cloud.row_step = cloud.point_step * cloud.width;
  cloud.data.resize(static_cast<std::size_t>(cloud.row_step));
  for (std::size_t index = 0; index < points.size(); ++index) {
    const LocalMapPoint & point = points[index];
    const float values[6] = {
      static_cast<float>(point.position.x()), static_cast<float>(point.position.y()),
      static_cast<float>(point.position.z()), point.intensity, point.scan_id,
      point.collective_passes};
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
  lidar_params.max_visible_scans = this->declare_parameter<int>("max_visible_scans", 120);
  lidar_params.backward_match_stride = this->declare_parameter<int>("backward_match_stride", 4);
  lidar_params.refine_window = this->declare_parameter<bool>("refine_window", false);
  lidar_params.refine_min_travel = this->declare_parameter<double>("refine_min_travel", 2.0);
  lidar_params.refine_settle_passes = this->declare_parameter<int>("refine_settle_passes", 2);
  lidar_params.fpfh.enabled = this->declare_parameter<bool>("fpfh_enabled", true);
  lidar_params.fpfh.keypoint_voxel = this->declare_parameter<double>("fpfh_keypoint_voxel", 1.5);
  lidar_params.fpfh.normal_radius = this->declare_parameter<double>("fpfh_normal_radius", 2.0);
  lidar_params.fpfh.fpfh_radius = this->declare_parameter<double>("fpfh_radius", 5.0);
  lidar_params.fpfh.max_keypoints = this->declare_parameter<int>("fpfh_max_keypoints", 1500);
  lidar_params.fpfh.correspondence_distance =
    this->declare_parameter<double>("fpfh_correspondence_distance", 5.0);
  lidar_params.fpfh.min_inliers = this->declare_parameter<int>("fpfh_min_inliers", 20);
  lidar_params.fpfh.omp_threads = this->declare_parameter<int>("fpfh_omp_threads", 0);
  lidar_params.gyro_bias_gain = this->declare_parameter<double>("gyro_bias_gain", 0.1);
  lidar_params.accel_bias_gain = this->declare_parameter<double>("accel_bias_gain", 0.02);
  lidar_params.speed_correction_gain =
    this->declare_parameter<double>("speed_correction_gain", 0.5);
  lidar_params.max_longitudinal_correction =
    this->declare_parameter<double>("max_longitudinal_correction", 0.15);
  lidar_params.max_gyro_bias = this->declare_parameter<double>("max_gyro_bias", 0.05);
  lidar_params.max_accel_bias = this->declare_parameter<double>("max_accel_bias", 1.0);
  lidar_params.limits.max_speed = this->declare_parameter<double>("max_speed", 20.0);
  lidar_params.limits.max_acceleration = this->declare_parameter<double>("max_acceleration", 5.0);
  lidar_params.limits.max_yaw_rate = this->declare_parameter<double>("max_yaw_rate", 1.0);
  lidar_params.limits.max_match_rejects = this->declare_parameter<int>("max_match_rejects", 5);
  lidar_params.limits.min_scale_travel = this->declare_parameter<double>("min_scale_travel", 0.5);
  lidar_params.visual_enabled = this->declare_parameter<bool>("visual_enabled", true);
  const auto visual_odom_topic =
    this->declare_parameter<std::string>("visual_odom_topic", "/visual_odom");

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
  params.max_speed = lidar_params.limits.max_speed;
  params.max_acceleration = lidar_params.limits.max_acceleration;
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
  if (lidar_params.visual_enabled) {
    visual_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      visual_odom_topic, rclcpp::QoS(10),
      [this](const nav_msgs::msg::Odometry::ConstSharedPtr msg) { this->callback_visual(msg); });
  }

  odometry_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("back_odom", rclcpp::QoS(10));
  imu_odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("back_odom/imu", rclcpp::QoS(10));
  path_pub_ = this->create_publisher<nav_msgs::msg::Path>("back_odom/path", rclcpp::QoS(10));
  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "back_odom/markers", rclcpp::QoS(10));
  scan_pose_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "back_odom/scan_poses", rclcpp::QoS(10));
  map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
    "back_odom/map", rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
  health_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticStatus>(
    "back_odom/health", rclcpp::QoS(10));
  lidar_debug_pub_ =
    this->create_publisher<nav_msgs::msg::Odometry>("back_odom/debug/lidar", rclcpp::QoS(10));
  lidar_debug_path_pub_ =
    this->create_publisher<nav_msgs::msg::Path>("back_odom/debug/lidar/path", rclcpp::QoS(10));
  camera_debug_pub_ =
    this->create_publisher<nav_msgs::msg::Odometry>("back_odom/debug/camera", rclcpp::QoS(10));
  camera_debug_path_pub_ =
    this->create_publisher<nav_msgs::msg::Path>("back_odom/debug/camera/path", rclcpp::QoS(10));
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
  if (output.speed_clamped) {
    output = imu_processor_->output_at(sample);
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "IMU speed was clamped to the vehicle limit.");
  }

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
  publish_health(stamp);
  match_pending_scan(false);
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
  if (pending_scan_.has_value()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Matching a lidar scan before IMU reached its last point.");
    match_pending_scan(true);
  }
  pending_scan_ = PendingScan{std::move(scan), stamp, source_frame};
  match_pending_scan(false);
}

void BackOdomNode::match_pending_scan(const bool force)
{
  if (!pending_scan_.has_value() || !aligned_ || !imu_processor_->aligned()) {
    return;
  }
  const double scan_end =
    *std::max_element(pending_scan_->scan.timestamps.cbegin(), pending_scan_->scan.timestamps.cend());
  const double imu_stamp = imu_processor_->latest_stamp();
  if (!force && imu_stamp + 1.0e-3 < scan_end) {
    if (scan_end - imu_stamp > 0.25) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Lidar scan end is %.3f s ahead of the newest IMU. Matching with the IMU pose in hand.",
        scan_end - imu_stamp);
    } else {
      return;
    }
  }
  PendingScan pending = std::move(*pending_scan_);
  pending_scan_.reset();
  LidarScan scan = std::move(pending.scan);
  const rclcpp::Time stamp = pending.header_stamp;
  const ProcessorOutput prior = imu_processor_->output_at(
    ImuSample{scan.stamp, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()});
  const MatchResult matched = lidar_matcher_->on_scan(scan, *imu_processor_);
  RCLCPP_DEBUG(
    this->get_logger(),
    "scan stages ms deskew %.2f voxel %.2f fpfh %.2f align %.2f cost %.2f refine %.2f map %.2f",
    matched.deskew_ms, matched.voxel_ms, matched.fpfh_ms, matched.align_ms, matched.cost_ms,
    matched.refine_ms, matched.map_ms);
  last_match_ = matched;
  has_last_match_ = true;
  publish_debug_odometry(matched, stamp);
  publish_scan_poses(stamp);
  if (!matched.applied) {
    if (lidar_matcher_->has_reference()) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "ICP correction rejected. health %s, source %s, lidar cost %.3f pass %s, "
        "camera cost %.3f pass %s, translation %.3f m, rotation %.3f rad",
        health_name(matched.health), matched.used_visual_guess ? "camera" : "lidar",
        matched.lidar_cost, matched.lidar_passed ? "yes" : "no", matched.camera_cost,
        matched.camera_passed ? "yes" : "no", matched.translation_error, matched.rotation_error);
    }
    publish_health(stamp);
    return;
  }
  if (!matched.first_scan) {
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "ICP correction applied from the %s guess. lidar cost %.3f, camera cost %.3f, "
      "translation %.3f m, rotation %.4f rad, health %s",
      matched.used_visual_guess ? "camera" : "lidar", matched.lidar_cost, matched.camera_cost,
      matched.translation_error, matched.rotation_error, health_name(matched.health));
  }
  if (matched.refined_window) {
    RCLCPP_INFO(
      this->get_logger(),       "Refined %d scan poses, %d already settled. largest shift %.3f m",
      matched.refine_scans, matched.refine_settled, matched.refine_shift);
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
  publish_health(stamp);
  if (matched.first_scan) {
    RCLCPP_INFO(this->get_logger(), "Stored the first lidar scan in the local map.");
  }
}

void BackOdomNode::callback_visual(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
  const std::string camera_frame =
    msg->child_frame_id.empty() ? msg->header.frame_id : msg->child_frame_id;
  if (!camera_frame.empty() && camera_frame != parent_frame_) {
    Sophus::SE3d base_from_camera;
    if (base_from_frame(camera_frame, base_from_camera)) {
      lidar_matcher_->set_body_from_camera(base_from_camera);
    }
  }
  const Eigen::Quaterniond rotation(
    msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
    msg->pose.pose.orientation.z);
  if (rotation.norm() < 1.0e-9) {
    return;
  }
  const Sophus::SE3d world_from_camera(
    Sophus::SO3d(rotation.normalized()),
    Eigen::Vector3d(
      msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z));
  lidar_matcher_->push_visual_pose(rclcpp::Time(msg->header.stamp).seconds(), world_from_camera);
}

void BackOdomNode::publish_health(const rclcpp::Time & stamp)
{
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "back_odom";
  status.hardware_id = "back_odom";
  const LocalizationHealth health = lidar_matcher_->health();
  status.message = health_name(health);
  if (health == LocalizationHealth::Healthy) {
    status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
  } else if (health == LocalizationHealth::Degraded) {
    status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
  } else {
    status.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
  }
  const auto add_value = [&status](const std::string & key, const std::string & value) {
    diagnostic_msgs::msg::KeyValue entry;
    entry.key = key;
    entry.value = value;
    status.values.push_back(entry);
  };
  add_value(
    "visual_scale",
    lidar_matcher_->has_visual_scale() ? std::to_string(lidar_matcher_->visual_scale()) : "unset");
  add_value("scale_frozen", lidar_matcher_->scale_frozen() ? "true" : "false");
  add_value("reject_streak", std::to_string(lidar_matcher_->reject_streak()));
  if (has_last_match_) {
    add_value("source", last_match_.used_visual_guess ? "camera" : "lidar");
    add_value("lidar_cost", last_match_.has_lidar_debug ? std::to_string(last_match_.lidar_cost) : "none");
    add_value("lidar_passed", last_match_.lidar_passed ? "true" : "false");
    add_value(
      "camera_cost", last_match_.has_camera_debug ? std::to_string(last_match_.camera_cost) : "none");
    add_value("camera_passed", last_match_.camera_passed ? "true" : "false");
    add_value("translation_error_m", std::to_string(last_match_.translation_error));
    add_value("rotation_error_rad", std::to_string(last_match_.rotation_error));
  }
  (void)stamp;
  health_pub_->publish(status);
}

void BackOdomNode::publish_debug_odometry(const MatchResult & match, const rclcpp::Time & stamp)
{
  const auto publish_one =
    [this, &stamp](
      const Sophus::SE3d & pose, const std::string & child_frame,
      const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr & odometry_publisher,
      nav_msgs::msg::Path & path,
      const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr & path_publisher) {
      nav_msgs::msg::Odometry odometry;
      odometry.header.stamp = stamp;
      odometry.header.frame_id = parent_frame_;
      odometry.child_frame_id = child_frame;
      odometry.pose.pose.position.x = pose.translation().x();
      odometry.pose.pose.position.y = pose.translation().y();
      odometry.pose.pose.position.z = pose.translation().z();
      set_quaternion(odometry.pose.pose.orientation, pose.so3());
      odometry_publisher->publish(odometry);

      geometry_msgs::msg::PoseStamped stamped;
      stamped.header = odometry.header;
      stamped.pose = odometry.pose.pose;
      path.header = odometry.header;
      path.poses.push_back(stamped);
      if (path.poses.size() > path_max_poses_) {
        path.poses.erase(path.poses.begin());
      }
      path_publisher->publish(path);
    };

  if (match.has_lidar_debug) {
    publish_one(match.lidar_pose, "lidar_odom", lidar_debug_pub_, lidar_debug_path_, lidar_debug_path_pub_);
  }
  if (match.has_camera_debug) {
    publish_one(
      match.camera_pose, "camera_odom", camera_debug_pub_, camera_debug_path_, camera_debug_path_pub_);
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
  const double header = scan.stamp;
  bool near_header = true;
  for (const double time : scan.timestamps) {
    if (std::abs(time - header) > 0.25) {
      near_header = false;
      break;
    }
  }
  if (near_header) {
    return scan;
  }
  if (max_abs_time < 1000.0) {
    for (double & time : scan.timestamps) {
      time += header;
    }
    return scan;
  }
  RCLCPP_WARN_THROTTLE(
    this->get_logger(), *this->get_clock(), 5000,
    "Lidar point times are not on the IMU clock (%.3f s vs header %.3f s). "
    "The sweep is posed at the cloud header, so forward motion during the scan is not removed.",
    scan.timestamps.front(), header);
  for (double & time : scan.timestamps) {
    time = header;
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
    path_.header.stamp = stamp;
    path_.header.frame_id = parent_frame_;
    path_pub_->publish(path_);
  }
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
  const std::vector<LocalMapPoint> shown =
    thin_for_display(lidar_matcher_->annotated_local_map(), 0.5);
  map_pub_->publish(to_annotated_pointcloud(stamp, parent_frame_, shown));
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
    stream << "tracking " << health_name(lidar_matcher_->health())
           << " |f|=" << output.specific_force_body.norm()
           << " |w|=" << output.angular_velocity_body.norm();
  } else {
    stream << "collecting " << output.sample_count << "/" << alignment_sample_count_;
  }
  text.text = stream.str();
  markers.markers.push_back(text);
  marker_pub_->publish(markers);
}

void BackOdomNode::publish_scan_poses(const rclcpp::Time & stamp)
{
  const std::vector<StampedPose> poses = lidar_matcher_->visible_scan_poses();
  visualization_msgs::msg::MarkerArray markers;

  const auto delete_namespace = [&](const std::string & ns) {
    visualization_msgs::msg::Marker clear;
    clear.header.stamp = stamp;
    clear.header.frame_id = parent_frame_;
    clear.ns = ns;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(clear);
  };
  delete_namespace("scan_horizon");
  delete_namespace("scan_pose");
  if (poses.empty()) {
    scan_pose_pub_->publish(markers);
    return;
  }

  visualization_msgs::msg::Marker horizon;
  horizon.header.stamp = stamp;
  horizon.header.frame_id = parent_frame_;
  horizon.ns = "scan_horizon";
  horizon.id = 0;
  horizon.type = visualization_msgs::msg::Marker::LINE_STRIP;
  horizon.action = visualization_msgs::msg::Marker::ADD;
  horizon.pose.orientation.w = 1.0;
  horizon.scale.x = 0.08;
  horizon.color.r = 1.0F;
  horizon.color.g = 0.85F;
  horizon.color.b = 0.2F;
  horizon.color.a = 0.9F;
  horizon.points.reserve(poses.size());
  for (const StampedPose & scan : poses) {
    horizon.points.push_back(to_point(scan.pose.translation()));
  }
  markers.markers.push_back(horizon);

  const std::size_t count = poses.size();
  for (std::size_t index = 0; index < count; ++index) {
    const float age =
      count == 1U ? 1.0F : static_cast<float>(index) / static_cast<float>(count - 1U);
    const Eigen::Vector3d origin = poses[index].pose.translation();
    const Eigen::Vector3d forward = origin + poses[index].pose.so3() * Eigen::Vector3d(3.0, 0.0, 0.0);
    visualization_msgs::msg::Marker arrow = make_arrow(
      stamp, parent_frame_, static_cast<int>(index), origin, forward, age, 0.35F, 1.0F - age);
    arrow.ns = "scan_pose";
    arrow.scale.x = 0.08;
    arrow.scale.y = 0.16;
    arrow.scale.z = 0.22;
    arrow.color.a = 0.45F + 0.55F * age;
    markers.markers.push_back(arrow);
  }

  visualization_msgs::msg::Marker text;
  text.header.stamp = stamp;
  text.header.frame_id = parent_frame_;
  text.ns = "scan_pose";
  text.id = static_cast<int>(count);
  text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  text.action = visualization_msgs::msg::Marker::ADD;
  text.pose.position = to_point(poses.back().pose.translation());
  text.pose.position.z += 1.2;
  text.pose.orientation.w = 1.0;
  text.scale.z = 0.4;
  text.color.r = 1.0F;
  text.color.g = 1.0F;
  text.color.b = 1.0F;
  text.color.a = 1.0F;
  text.text = std::to_string(count) + " scans in view";
  markers.markers.push_back(text);
  scan_pose_pub_->publish(markers);
}

}  // namespace back_odom

RCLCPP_COMPONENTS_REGISTER_NODE(back_odom::BackOdomNode)
