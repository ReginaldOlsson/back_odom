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

#include "back_odom/node/back_odom_node.hpp"

#include "back_odom/imu/imu_alignment.hpp"
#include "back_odom/lidar/device_accel.hpp"

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
#include <chrono>
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
  twist_variance_vx_ = this->declare_parameter<double>("twist_variance_vx", 0.05);
  twist_variance_wz_ = this->declare_parameter<double>("twist_variance_wz", 0.01);
  pose_variance_xy_ = this->declare_parameter<double>("pose_variance_xy", 0.05);
  pose_variance_yaw_ = this->declare_parameter<double>("pose_variance_yaw", 0.01);
  if (
    twist_variance_vx_ <= 0.0 || twist_variance_wz_ <= 0.0 || pose_variance_xy_ <= 0.0 ||
    pose_variance_yaw_ <= 0.0) {
    throw std::invalid_argument("EKF measurement variances must be positive");
  }
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
  const double imu_trajectory_horizon =
    this->declare_parameter<double>("imu_trajectory_horizon", 30.0);
  const int imu_max_trajectory_poses =
    this->declare_parameter<int>("imu_max_trajectory_poses", 20000);
  imu_max_pair_dt_ = this->declare_parameter<double>("imu_max_pair_dt", 0.05);
  imu_wait_timeout_ = this->declare_parameter<double>("imu_wait_timeout", 0.5);
  path_min_dt_ = this->declare_parameter<double>("path_min_dt", 0.1);
  marker_min_dt_ = this->declare_parameter<double>("marker_min_dt", 0.1);
  health_min_dt_ = this->declare_parameter<double>("health_min_dt", 0.1);
  map_publish_period_ = this->declare_parameter<double>("map_publish_period", 1.0);
  min_range_ = this->declare_parameter<double>("min_range", 1.0);
  max_range_ = this->declare_parameter<double>("max_range", 0.0);
  const int path_max_poses = this->declare_parameter<int>("path_max_poses", 1000);
  const int imu_queue_depth = this->declare_parameter<int>("imu_queue_depth", 2000);
  time_field_ = this->declare_parameter<std::string>("time_field", "time");
  LidarMatchParams lidar_params;
  lidar_params.voxel_size = this->declare_parameter<double>("voxel_size", 0.5);
  lidar_params.crop_longitudinal = this->declare_parameter<double>("crop_longitudinal", 75.0);
  lidar_params.crop_lateral = this->declare_parameter<double>("crop_lateral", 50.0);
  lidar_params.max_correspondence_distance =
    this->declare_parameter<double>("max_correspondence_distance", 2.0);
  lidar_params.kernel_scale = this->declare_parameter<double>("kernel_scale", 0.5);
  lidar_params.convergence_criterion =
    this->declare_parameter<double>("convergence_criterion", 1.0e-3);
  lidar_params.max_iterations = this->declare_parameter<int>("max_iterations", 20);
  lidar_params.align_rate_hz = this->declare_parameter<double>("align_rate_hz", 10.0);
  lidar_params.partial_min_iterations = this->declare_parameter<int>("partial_min_iterations", 3);
  lidar_params.partial_max_step = this->declare_parameter<double>("partial_max_step", 0.01);
  lidar_params.max_points_per_voxel = this->declare_parameter<int>("max_points_per_voxel", 20);
  lidar_params.max_visible_scans = this->declare_parameter<int>("max_visible_scans", 120);
  lidar_params.backward_match_stride = this->declare_parameter<int>("backward_match_stride", 4);
  lidar_params.refine_window = this->declare_parameter<bool>("refine_window", false);
  lidar_params.refine_min_travel = this->declare_parameter<double>("refine_min_travel", 2.0);
  lidar_params.refine_settle_passes = this->declare_parameter<int>("refine_settle_passes", 2);
  lidar_params.keyframe_min_translation =
    this->declare_parameter<double>("keyframe_min_translation", 0.5);
  lidar_params.keyframe_min_rotation =
    this->declare_parameter<double>("keyframe_min_rotation", 0.05);
  lidar_params.device_bakeoff_scan = this->declare_parameter<int>("device_bakeoff_scan", 10);
  lidar_params.device_recheck_scans = this->declare_parameter<int>("device_recheck_scans", 200);
  lidar_params.apply_imu_correction =
    this->declare_parameter<bool>("apply_imu_correction", true);
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
  lidar_params.max_ndt_cost = this->declare_parameter<double>("max_ndt_cost", 7.81);
  lidar_params.ndt_cov_regularization =
    this->declare_parameter<double>("ndt_cov_regularization", 1.0e-3);
  lidar_params.min_ndt_correspondences =
    this->declare_parameter<int>("min_ndt_correspondences", 20);
  lidar_params.min_ndt_inlier_fraction =
    this->declare_parameter<double>("min_ndt_inlier_fraction", 0.5);
  lidar_params.min_inlier_ratio = this->declare_parameter<double>("min_inlier_ratio", 0.3);
  if (alignment_sample_count <= 0) {
    throw std::invalid_argument("alignment_sample_count must be positive");
  }
  if (path_max_poses <= 0) {
    throw std::invalid_argument("path_max_poses must be positive");
  }
  if (imu_max_trajectory_poses < 2) {
    throw std::invalid_argument("imu_max_trajectory_poses must be at least 2");
  }
  if (!(imu_trajectory_horizon > 0.0) || !std::isfinite(imu_trajectory_horizon)) {
    throw std::invalid_argument("imu_trajectory_horizon must be positive and finite");
  }
  if (!(imu_max_pair_dt_ > 0.0) || !std::isfinite(imu_max_pair_dt_)) {
    throw std::invalid_argument("imu_max_pair_dt must be positive and finite");
  }
  if (!(imu_wait_timeout_ >= 0.0) || !std::isfinite(imu_wait_timeout_)) {
    throw std::invalid_argument("imu_wait_timeout must be non-negative and finite");
  }
  if (imu_queue_depth < 1) {
    throw std::invalid_argument("imu_queue_depth must be positive");
  }
  if (min_range_ < 0.0 || max_range_ < 0.0) {
    throw std::invalid_argument("min_range and max_range must be non-negative");
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
  params.trajectory_horizon = imu_trajectory_horizon;
  params.max_trajectory_poses = static_cast<std::size_t>(imu_max_trajectory_poses);
  params.max_pair_dt = imu_max_pair_dt_;
  imu_processor_ = std::make_unique<ImuProcessor>(params);
  lidar_matcher_ = std::make_unique<LidarImuMatcher>(lidar_params);

  // Deep IMU queue: a 200 Hz stream must survive a scan callback or TF stall without dropping
  // samples, since a dropped sample becomes an integration hole (dt > max_dt).
  imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
    imu_topic, rclcpp::QoS(rclcpp::KeepLast(static_cast<std::size_t>(imu_queue_depth))).reliable(),
    [this](const sensor_msgs::msg::Imu::ConstSharedPtr msg) { this->callback_imu(msg); });
  // Scans go through a single-slot mailbox; no point queueing more than the newest two.
  pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    pointcloud_topic, rclcpp::QoS(rclcpp::KeepLast(2)).best_effort(),
    [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
      this->callback_pointcloud(msg);
    });

  odometry_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("back_odom", rclcpp::QoS(10));
  imu_odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("back_odom/imu", rclcpp::QoS(10));
  twist_pub_ = this->create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>(
    "/back_odom/twist_with_covariance", rclcpp::QoS(10));
  pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
    "/back_odom/pose_with_covariance", rclcpp::QoS(10));
  initial_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
    "/back_odom/initialpose", rclcpp::QoS(10));
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
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_, this, false);

  if (map_publish_period_ > 0.0) {
    map_timer_ = this->create_wall_timer(
      std::chrono::duration<double>(map_publish_period_), [this]() { this->publish_local_map(); });
  }
  worker_ = std::thread([this]() { this->worker_loop(); });

  RCLCPP_INFO(
    this->get_logger(), "BackOdomNode initialized. imu_left_handed=%s device=%s",
    imu_left_handed_ ? "true" : "false", device_available() ? "cuda" : "cpu");
}

BackOdomNode::~BackOdomNode()
{
  stop_ = true;
  scan_cv_.notify_all();
  imu_cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
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

  ProcessorOutput output;
  LocalizationHealth health = LocalizationHealth::Healthy;
  int reject_streak = 0;
  int timeout_streak = 0;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    output = imu_processor_->process(sample);
    if (output.speed_clamped) {
      lidar_matcher_->notify_speed_limit(*imu_processor_);
      output = imu_processor_->output_at(sample);
    }
    health = lidar_matcher_->health();
    reject_streak = lidar_matcher_->reject_streak();
    timeout_streak = lidar_matcher_->timeout_streak();
  }
  imu_cv_.notify_all();
  if (output.speed_clamped) {
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
    publish_markers(output, health, stamp);
    return;
  }

  publish_odometry(odometry_pub_, output, stamp);
  publish_twist(output, stamp);
  publish_path(output, stamp);
  publish_markers(output, health, stamp);
  if (publish_tf_) {
    publish_tf(output, stamp);
  }
  bool health_due = false;
  {
    std::lock_guard<std::mutex> lock(publish_mutex_);
    health_due = !has_published_health_ || (stamp - last_health_stamp_).seconds() >= health_min_dt_ ||
                 (stamp - last_health_stamp_).seconds() < 0.0;
    if (health_due) {
      last_health_stamp_ = stamp;
      has_published_health_ = true;
    }
  }
  if (health_due) {
    publish_health(health, reject_streak, timeout_streak);
  }
}

void BackOdomNode::callback_pointcloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  if (!aligned_) {
    return;
  }
  if (msg->header.frame_id.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "Point cloud frame_id is empty, so it cannot be looked up in the TF tree.");
    return;
  }
  // Hand the raw message to the worker. Parsing 200k points here would stall the IMU stream.
  std::uint64_t dropped = 0;
  {
    std::lock_guard<std::mutex> lock(scan_mutex_);
    if (pending_cloud_) {
      ++dropped_scans_;
      dropped = dropped_scans_;
    }
    pending_cloud_ = msg;
  }
  scan_cv_.notify_one();
  if (dropped > 0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Scan matcher is behind: replaced an unprocessed scan (%lu dropped so far).",
      static_cast<unsigned long>(dropped));  // NOLINT(runtime/int)
  }
}

void BackOdomNode::worker_loop()
{
  while (!stop_) {
    sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud;
    {
      std::unique_lock<std::mutex> lock(scan_mutex_);
      scan_cv_.wait(lock, [this]() { return stop_ || pending_cloud_ != nullptr; });
      if (stop_) {
        return;
      }
      cloud = std::move(pending_cloud_);
      pending_cloud_.reset();
    }
    try {
      process_cloud(cloud);
    } catch (const std::exception & error) {
      RCLCPP_ERROR(this->get_logger(), "Scan processing failed: %s", error.what());
    }
  }
}

void BackOdomNode::process_cloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg)
{
  LidarScan scan = scan_from_cloud(*msg);
  if (scan.points.empty()) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "Skipping lidar scan. Need xyz points and a timestamp field.");
    return;
  }
  Sophus::SE3d base_from_lidar;
  if (!base_from_frame(msg->header.frame_id, base_from_lidar)) {
    return;
  }
  const rclcpp::Time stamp(msg->header.stamp);
  const double scan_start = *std::min_element(scan.timestamps.cbegin(), scan.timestamps.cend());
  const double scan_end = *std::max_element(scan.timestamps.cbegin(), scan.timestamps.cend());

  // Wait (bounded) until the IMU has integrated past the end of the sweep, then snapshot the
  // integrator so ICP can run without holding up the IMU thread.
  std::optional<ImuProcessor> snapshot;
  const auto wall_deadline = std::chrono::steady_clock::now() +
                             std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                               std::chrono::duration<double>(imu_wait_timeout_));
  {
    std::unique_lock<std::mutex> lock(state_mutex_);
    const auto imu_ready = [&]() {
      if (!imu_processor_->aligned()) {
        return false;
      }
      if (imu_processor_->latest_stamp() + 1.0e-3 < scan_end) {
        return false;
      }
      return std::abs(imu_processor_->closest_pose(scan_end).stamp - scan_end) <= imu_max_pair_dt_;
    };
    const auto newer_scan_waiting = [this]() {
      std::lock_guard<std::mutex> mailbox(scan_mutex_);
      return pending_cloud_ != nullptr;
    };
    while (!stop_ && !imu_ready()) {
      if (std::chrono::steady_clock::now() >= wall_deadline || newer_scan_waiting()) {
        break;
      }
      imu_cv_.wait_for(lock, std::chrono::milliseconds(5));
    }
    if (stop_ || !imu_processor_->aligned()) {
      return;
    }
    const double imu_stamp = imu_processor_->latest_stamp();
    if (imu_stamp + 1.0e-3 < scan_end) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Lidar scan end is %.3f s ahead of the newest IMU. Matching with the IMU pose in hand.",
        scan_end - imu_stamp);
    }
    if (
      !imu_processor_->trajectory().empty() &&
      imu_processor_->trajectory().front().stamp > scan_start + imu_max_pair_dt_) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "IMU buffer no longer covers scan start (buffer front %.3f, scan_start %.3f). "
        "Increase imu_trajectory_horizon.",
        imu_processor_->trajectory().front().stamp, scan_start);
      return;
    }
    lidar_matcher_->set_body_from_lidar(base_from_lidar);
    snapshot.emplace(*imu_processor_);
  }

  ScanAlignment alignment = lidar_matcher_->align_scan(scan, *snapshot);
  snapshot.reset();

  MatchResult matched;
  ProcessorOutput prior;
  ProcessorOutput corrected;
  std::vector<StampedPose> scan_poses;
  LocalizationHealth health = LocalizationHealth::Healthy;
  int reject_streak = 0;
  int timeout_streak = 0;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    prior = imu_processor_->output_at(
      ImuSample{scan.stamp, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()});
    matched = lidar_matcher_->commit_scan(std::move(alignment), *imu_processor_);
    corrected = imu_processor_->output_at(
      ImuSample{scan.stamp, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()});
    scan_poses = lidar_matcher_->visible_scan_poses();
    health = lidar_matcher_->health();
    reject_streak = lidar_matcher_->reject_streak();
    timeout_streak = lidar_matcher_->timeout_streak();
  }
  RCLCPP_DEBUG(
    this->get_logger(),
    "scan stages ms deskew %.2f voxel %.2f align %.2f cost %.2f refine %.2f map %.2f timed_out %s",
    matched.deskew_ms, matched.voxel_ms, matched.align_ms, matched.cost_ms, matched.refine_ms,
    matched.map_ms, matched.align_timed_out ? "yes" : "no");
  {
    std::lock_guard<std::mutex> lock(publish_mutex_);
    last_match_ = matched;
    has_last_match_ = true;
    last_health_stamp_ = stamp;
    has_published_health_ = true;
  }
  publish_debug_odometry(matched, stamp);
  publish_scan_poses(scan_poses, stamp);
  publish_match(matched, prior, corrected, stamp);
  publish_health(health, reject_streak, timeout_streak);
}

void BackOdomNode::publish_match(
  const MatchResult & matched, const ProcessorOutput & prior, const ProcessorOutput & corrected,
  const rclcpp::Time & stamp)
{
  const double total_ms = matched.deskew_ms + matched.voxel_ms + matched.align_ms + matched.cost_ms +
                          matched.refine_ms + matched.map_ms;
  if (matched.skipped) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000, "Scan skipped: no usable points or IMU pose.");
    return;
  }
  if (matched.align_timed_out && !matched.partial_used) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "ICP hit the rate budget after %.0f iterations (deskew %.1f voxel %.1f align %.1f ms, "
      "%.1f ms total). Scan not inserted; health %s.",
      matched.iterations, matched.deskew_ms, matched.voxel_ms, matched.align_ms, total_ms,
      health_name(matched.health));
    return;
  }
  if (!matched.applied) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "ICP correction rejected. health %s, lidar cost %.3f ndt %.3f (inliers %.2f) ratio %.2f "
      "pass %s, translation %.3f m, rotation %.3f rad, degenerate %d",
      health_name(matched.health), matched.lidar_cost,
      std::isfinite(matched.ndt_cost) ? matched.ndt_cost : -1.0,
      std::isfinite(matched.ndt_inlier_fraction) ? matched.ndt_inlier_fraction : -1.0,
      matched.inlier_ratio, matched.lidar_passed ? "yes" : "no", matched.translation_error,
      matched.rotation_error, matched.degenerate_axes);
    return;
  }
  if (matched.first_scan) {
    RCLCPP_INFO(this->get_logger(), "Stored the first lidar scan in the local map.");
  } else {
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "ICP %s. cost %.3f ndt %.3f ratio %.2f iters %.0f%s, translation %.3f m, rotation %.4f rad, "
      "health %s, %s, %.1f ms (align %.1f)",
      matched.saturated ? "saturated" : "applied", matched.lidar_cost,
      std::isfinite(matched.ndt_cost) ? matched.ndt_cost : -1.0, matched.inlier_ratio,
      matched.iterations, matched.partial_used ? " (partial)" : "", matched.translation_error,
      matched.rotation_error, health_name(matched.health),
      matched.inserted_scan ? "inserted" : "no keyframe", total_ms, matched.align_ms);
  }
  if (matched.refined_window) {
    RCLCPP_INFO(
      this->get_logger(), "Refined %d scan poses, %d already settled. largest shift %.3f m",
      matched.refine_scans, matched.refine_settled, matched.refine_shift);
  }
  publish_odometry(imu_odom_pub_, prior, stamp);
  publish_odometry(odometry_pub_, corrected, stamp);
  publish_pose(corrected, stamp);
  publish_path(corrected, stamp);
  if (publish_tf_) {
    publish_tf(corrected, stamp);
  }
}

void BackOdomNode::publish_health(
  const LocalizationHealth health, const int reject_streak, const int timeout_streak)
{
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "back_odom";
  status.hardware_id = "back_odom";
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
  add_value("reject_streak", std::to_string(reject_streak));
  add_value("timeout_streak", std::to_string(timeout_streak));
  {
    std::lock_guard<std::mutex> lock(publish_mutex_);
    if (has_last_match_) {
      const MatchResult & last = last_match_;
      add_value("lidar_cost", last.has_lidar_debug ? std::to_string(last.lidar_cost) : "none");
      add_value(
        "ndt_cost",
        last.has_lidar_debug && std::isfinite(last.ndt_cost) ? std::to_string(last.ndt_cost) : "none");
      add_value(
        "ndt_inlier_fraction",
        std::isfinite(last.ndt_inlier_fraction) ? std::to_string(last.ndt_inlier_fraction) : "none");
      add_value("inlier_ratio", std::to_string(last.inlier_ratio));
      add_value("correspondences", std::to_string(last.correspondences));
      add_value("degenerate_axes", std::to_string(last.degenerate_axes));
      add_value("lidar_passed", last.lidar_passed ? "true" : "false");
      add_value("saturated", last.saturated ? "true" : "false");
      add_value("align_timed_out", last.align_timed_out ? "true" : "false");
      add_value("partial_used", last.partial_used ? "true" : "false");
      add_value("inserted_scan", last.inserted_scan ? "true" : "false");
      add_value("iterations", std::to_string(last.iterations));
      add_value("deskew_ms", std::to_string(last.deskew_ms));
      add_value("voxel_ms", std::to_string(last.voxel_ms));
      add_value("align_ms", std::to_string(last.align_ms));
      add_value("cost_ms", std::to_string(last.cost_ms));
      add_value("map_ms", std::to_string(last.map_ms));
      add_value("translation_error_m", std::to_string(last.translation_error));
      add_value("rotation_error_rad", std::to_string(last.rotation_error));
    }
  }
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
      std::lock_guard<std::mutex> lock(publish_mutex_);
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
}

bool BackOdomNode::base_from_frame(
  const std::string & source_frame, Sophus::SE3d & target_from_source)
{
  {
    std::lock_guard<std::mutex> lock(extrinsics_mutex_);
    const auto found = extrinsics_.find(source_frame);
    if (found != extrinsics_.end()) {
      target_from_source = found->second;
      return true;
    }
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
  std::lock_guard<std::mutex> lock(extrinsics_mutex_);
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
  auto intensity_field = std::find_if(
    cloud.fields.cbegin(), cloud.fields.cend(),
    [](const sensor_msgs::msg::PointField & field) { return field.name == "intensity"; });
  if (intensity_field == cloud.fields.cend()) {
    for (const char * fallback : {"i", "reflectivity"}) {
      intensity_field = std::find_if(
        cloud.fields.cbegin(), cloud.fields.cend(),
        [fallback](const sensor_msgs::msg::PointField & field) { return field.name == fallback; });
      if (intensity_field != cloud.fields.cend()) {
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
  const auto read_intensity = [&cloud](
                                const sensor_msgs::msg::PointField & field, const std::size_t index) {
    const std::uint8_t * pointer = cloud.data.data() + index * cloud.point_step + field.offset;
    switch (field.datatype) {
      case sensor_msgs::msg::PointField::FLOAT32: {
        float value = 0.0F;
        std::memcpy(&value, pointer, sizeof(float));
        return value;
      }
      case sensor_msgs::msg::PointField::FLOAT64: {
        double value = 0.0;
        std::memcpy(&value, pointer, sizeof(double));
        return static_cast<float>(value);
      }
      case sensor_msgs::msg::PointField::UINT8:
        return static_cast<float>(*pointer);
      case sensor_msgs::msg::PointField::UINT16: {
        std::uint16_t value = 0;
        std::memcpy(&value, pointer, sizeof(std::uint16_t));
        return static_cast<float>(value);
      }
      case sensor_msgs::msg::PointField::INT16: {
        std::int16_t value = 0;
        std::memcpy(&value, pointer, sizeof(std::int16_t));
        return static_cast<float>(value);
      }
      case sensor_msgs::msg::PointField::UINT32: {
        std::uint32_t value = 0;
        std::memcpy(&value, pointer, sizeof(std::uint32_t));
        return static_cast<float>(value);
      }
      default:
        return 0.0F;
    }
  };

  const std::size_t count = static_cast<std::size_t>(cloud.width) * cloud.height;
  scan.points.reserve(count);
  scan.timestamps.reserve(count);
  scan.intensities.reserve(count);
  const bool have_intensity = intensity_field != cloud.fields.cend();
  // Returns off the ego vehicle and beyond the crop box never reach the map; drop them here so
  // deskew and voxelisation do not pay for them.
  const double min_range_sq = min_range_ * min_range_;
  const double max_range_sq = max_range_ > 0.0 ? max_range_ * max_range_
                                               : std::numeric_limits<double>::infinity();
  double max_abs_time = 0.0;
  for (std::size_t index = 0; index < count; ++index) {
    const double x = read_float(*x_field, index);
    const double y = read_float(*y_field, index);
    const double z = read_float(*z_field, index);
    const double time = read_float(*time_field, index);
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(time)) {
      continue;
    }
    const double range_sq = x * x + y * y + z * z;
    if (range_sq < min_range_sq || range_sq > max_range_sq) {
      continue;
    }
    scan.points.emplace_back(x, y, z);
    scan.timestamps.push_back(time);
    scan.intensities.push_back(have_intensity ? read_intensity(*intensity_field, index) : 0.0F);
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

void BackOdomNode::publish_twist(const ProcessorOutput & output, const rclcpp::Time & stamp)
{
  geometry_msgs::msg::TwistWithCovarianceStamped twist;
  twist.header.stamp = stamp;
  twist.header.frame_id = child_frame_;
  const Eigen::Vector3d velocity_body = output.orientation.inverse() * output.velocity_world;
  twist.twist.twist.linear.x = velocity_body.x();
  twist.twist.twist.angular.z = output.angular_velocity_body.z();
  twist.twist.covariance[0] = twist_variance_vx_;
  twist.twist.covariance[35] = twist_variance_wz_;
  twist_pub_->publish(twist);
}

void BackOdomNode::publish_pose(const ProcessorOutput & output, const rclcpp::Time & stamp)
{
  geometry_msgs::msg::PoseWithCovarianceStamped pose;
  pose.header.stamp = stamp;
  pose.header.frame_id = parent_frame_;
  pose.pose.pose.position.x = output.position.x();
  pose.pose.pose.position.y = output.position.y();
  pose.pose.pose.position.z = output.position.z();
  set_quaternion(pose.pose.pose.orientation, output.orientation);
  pose.pose.covariance[0] = pose_variance_xy_;
  pose.pose.covariance[7] = pose_variance_xy_;
  pose.pose.covariance[14] = pose_variance_xy_;
  pose.pose.covariance[21] = pose_variance_yaw_;
  pose.pose.covariance[28] = pose_variance_yaw_;
  pose.pose.covariance[35] = pose_variance_yaw_;
  pose_pub_->publish(pose);
  bool send_initial = false;
  {
    std::lock_guard<std::mutex> lock(publish_mutex_);
    send_initial = !initial_pose_sent_;
    initial_pose_sent_ = true;
  }
  if (send_initial) {
    initial_pose_pub_->publish(pose);
  }
}

void BackOdomNode::publish_path(const ProcessorOutput & output, const rclcpp::Time & stamp)
{
  std::lock_guard<std::mutex> lock(publish_mutex_);
  const bool append_pose = !has_published_path_ ||
                           (stamp - last_path_stamp_).seconds() >= path_min_dt_ ||
                           (stamp - last_path_stamp_).seconds() < 0.0;
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

void BackOdomNode::publish_local_map()
{
  // The annotated cloud is a few hundred thousand points; build it only for a live subscriber.
  if (!aligned_ || map_pub_->get_subscription_count() == 0) {
    return;
  }
  std::vector<LocalMapPoint> points;
  double stamp_seconds = 0.0;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!lidar_matcher_->has_reference()) {
      return;
    }
    points = lidar_matcher_->annotated_local_map();
    stamp_seconds = imu_processor_->latest_stamp();
  }
  const std::vector<LocalMapPoint> shown = thin_for_display(points, 0.5);
  const rclcpp::Time stamp(
    static_cast<std::int64_t>(std::llround(stamp_seconds * 1.0e9)), RCL_ROS_TIME);
  map_pub_->publish(to_annotated_pointcloud(stamp, parent_frame_, shown));
}

void BackOdomNode::publish_markers(
  const ProcessorOutput & output, const LocalizationHealth health, const rclcpp::Time & stamp)
{
  {
    std::lock_guard<std::mutex> lock(publish_mutex_);
    const double since = (stamp - last_marker_stamp_).seconds();
    if (has_published_markers_ && since >= 0.0 && since < marker_min_dt_) {
      return;
    }
    last_marker_stamp_ = stamp;
    has_published_markers_ = true;
  }
  if (marker_pub_->get_subscription_count() == 0) {
    return;
  }
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
    stream << "tracking " << health_name(health)
           << " |f|=" << output.specific_force_body.norm()
           << " |w|=" << output.angular_velocity_body.norm();
  } else {
    stream << "collecting " << output.sample_count << "/" << alignment_sample_count_;
  }
  text.text = stream.str();
  markers.markers.push_back(text);
  marker_pub_->publish(markers);
}

void BackOdomNode::publish_scan_poses(
  const std::vector<StampedPose> & poses, const rclcpp::Time & stamp)
{
  if (scan_pose_pub_->get_subscription_count() == 0) {
    return;
  }
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
