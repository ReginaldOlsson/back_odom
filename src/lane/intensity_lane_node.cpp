#include "back_odom/lane/intensity_lane.hpp"

#include <rclcpp_components/register_node_macro.hpp>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2/exceptions.hpp>
#include <tf2/time.hpp>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace back_odom
{
namespace
{

using lane::CellKey;
using lane::CellMap;
using lane::IntensityPca;

struct Aabb
{
  double min_x{0.0};
  double max_x{0.0};
  double min_y{0.0};
  double max_y{0.0};
  double min_z{0.0};
  double max_z{0.0};
};

const sensor_msgs::msg::PointField * find_field(
  const sensor_msgs::msg::PointCloud2 & cloud, const std::string & name)
{
  const auto found = std::find_if(
    cloud.fields.cbegin(), cloud.fields.cend(),
    [&name](const sensor_msgs::msg::PointField & field) { return field.name == name; });
  if (found == cloud.fields.cend() || found->datatype != sensor_msgs::msg::PointField::FLOAT32) {
    return nullptr;
  }
  return &(*found);
}

float read_float(
  const sensor_msgs::msg::PointCloud2 & cloud, const sensor_msgs::msg::PointField & field,
  std::size_t index)
{
  float value = 0.0F;
  std::memcpy(
    &value, cloud.data.data() + index * cloud.point_step + field.offset, sizeof(value));
  return value;
}

void apply_transform(
  const geometry_msgs::msg::Transform & transform, double px, double py, double pz, double & ox,
  double & oy, double & oz)
{
  const double qx = transform.rotation.x;
  const double qy = transform.rotation.y;
  const double qz = transform.rotation.z;
  const double qw = transform.rotation.w;
  const double uvx = qy * pz - qz * py;
  const double uvy = qz * px - qx * pz;
  const double uvz = qx * py - qy * px;
  ox = transform.translation.x + px + 2.0 * (qw * uvx + qy * uvz - qz * uvy);
  oy = transform.translation.y + py + 2.0 * (qw * uvy + qz * uvx - qx * uvz);
  oz = transform.translation.z + pz + 2.0 * (qw * uvz + qx * uvy - qy * uvx);
}

geometry_msgs::msg::Transform invert_transform(const geometry_msgs::msg::Transform & transform)
{
  geometry_msgs::msg::Transform inverse;
  inverse.rotation.x = -transform.rotation.x;
  inverse.rotation.y = -transform.rotation.y;
  inverse.rotation.z = -transform.rotation.z;
  inverse.rotation.w = transform.rotation.w;
  inverse.translation.x = 0.0;
  inverse.translation.y = 0.0;
  inverse.translation.z = 0.0;
  double rotated_x = 0.0;
  double rotated_y = 0.0;
  double rotated_z = 0.0;
  apply_transform(
    inverse, transform.translation.x, transform.translation.y, transform.translation.z, rotated_x,
    rotated_y, rotated_z);
  inverse.translation.x = -rotated_x;
  inverse.translation.y = -rotated_y;
  inverse.translation.z = -rotated_z;
  return inverse;
}

bool in_crop(double x, double y, double z, double longitudinal, double lateral, double z_below)
{
  return std::abs(x) <= longitudinal && std::abs(y) <= lateral && z <= 0.0 && z >= -z_below;
}

Aabb crop_aabb_in_map(
  const geometry_msgs::msg::Transform & base_to_map, double longitudinal, double lateral,
  double z_below)
{
  Aabb box;
  box.min_x = box.min_y = box.min_z = std::numeric_limits<double>::infinity();
  box.max_x = box.max_y = box.max_z = -std::numeric_limits<double>::infinity();
  const double xs[2] = {-longitudinal, longitudinal};
  const double ys[2] = {-lateral, lateral};
  const double zs[2] = {-z_below, 0.0};
  for (double x : xs) {
    for (double y : ys) {
      for (double z : zs) {
        double mx = 0.0;
        double my = 0.0;
        double mz = 0.0;
        apply_transform(base_to_map, x, y, z, mx, my, mz);
        box.min_x = std::min(box.min_x, mx);
        box.max_x = std::max(box.max_x, mx);
        box.min_y = std::min(box.min_y, my);
        box.max_y = std::max(box.max_y, my);
        box.min_z = std::min(box.min_z, mz);
        box.max_z = std::max(box.max_z, mz);
      }
    }
  }
  return box;
}

bool outside_aabb(double x, double y, double z, const Aabb & box)
{
  return x < box.min_x || x > box.max_x || y < box.min_y || y > box.max_y || z < box.min_z ||
         z > box.max_z;
}

/// Distinct, deterministic color per integer class id (golden-ratio hue walk).
std_msgs::msg::ColorRGBA label_color(int id)
{
  const double hue = std::fmod(static_cast<double>(id) * 0.618033988749895, 1.0);
  const double saturation = 0.8;
  const double value = 0.95;
  const double sector = hue * 6.0;
  const int segment = static_cast<int>(std::floor(sector)) % 6;
  const double fraction = sector - std::floor(sector);
  const double p = value * (1.0 - saturation);
  const double q = value * (1.0 - saturation * fraction);
  const double t = value * (1.0 - saturation * (1.0 - fraction));
  double r = value;
  double g = t;
  double b = p;
  switch (segment) {
    case 1: r = q; g = value; b = p; break;
    case 2: r = p; g = value; b = t; break;
    case 3: r = p; g = q; b = value; break;
    case 4: r = t; g = p; b = value; break;
    case 5: r = value; g = p; b = q; break;
    default: break;
  }
  std_msgs::msg::ColorRGBA color;
  color.r = static_cast<float>(r);
  color.g = static_cast<float>(g);
  color.b = static_cast<float>(b);
  color.a = 0.6F;
  return color;
}

std_msgs::msg::ColorRGBA intensity_color(double mean, double stddev, double low, double high)
{
  const double span = std::max(high - low, 1e-6);
  const double unit = std::clamp((mean - low) / span, 0.0, 1.0);
  std_msgs::msg::ColorRGBA color;
  color.r = static_cast<float>(unit);
  color.g = static_cast<float>(unit);
  color.b = static_cast<float>(1.0 - unit);
  const double spread = std::clamp(stddev / std::max(std::abs(high - low), 1.0), 0.0, 1.0);
  color.a = static_cast<float>(1.0 - 0.85 * spread);
  return color;
}

visualization_msgs::msg::Marker cube_list(
  const std_msgs::msg::Header & header, const std::string & ns, int id, double voxel_size,
  double voxel_height)
{
  visualization_msgs::msg::Marker marker;
  marker.header = header;
  marker.ns = ns;
  marker.id = id;
  marker.type = visualization_msgs::msg::Marker::CUBE_LIST;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = voxel_size;
  marker.scale.y = voxel_size;
  marker.scale.z = voxel_height;
  marker.color.a = 1.0F;
  return marker;
}

std_msgs::msg::ColorRGBA axis_color(int axis)
{
  std_msgs::msg::ColorRGBA color;
  color.a = 1.0F;
  if (axis == 0) {
    color.r = 1.0F;
  } else if (axis == 1) {
    color.g = 1.0F;
  } else {
    color.b = 1.0F;
  }
  return color;
}

geometry_msgs::msg::Point to_point(double x, double y, double z)
{
  geometry_msgs::msg::Point point;
  point.x = x;
  point.y = y;
  point.z = z;
  return point;
}

geometry_msgs::msg::Quaternion quaternion_from_rotation(const Eigen::Matrix3d & rotation)
{
  geometry_msgs::msg::Quaternion quaternion;
  const double trace = rotation.trace();
  if (trace > 0.0) {
    const double scale = 0.5 / std::sqrt(trace + 1.0);
    quaternion.w = 0.25 / scale;
    quaternion.x = (rotation(2, 1) - rotation(1, 2)) * scale;
    quaternion.y = (rotation(0, 2) - rotation(2, 0)) * scale;
    quaternion.z = (rotation(1, 0) - rotation(0, 1)) * scale;
  } else if (rotation(0, 0) > rotation(1, 1) && rotation(0, 0) > rotation(2, 2)) {
    const double scale = 2.0 * std::sqrt(1.0 + rotation(0, 0) - rotation(1, 1) - rotation(2, 2));
    quaternion.w = (rotation(2, 1) - rotation(1, 2)) / scale;
    quaternion.x = 0.25 * scale;
    quaternion.y = (rotation(0, 1) + rotation(1, 0)) / scale;
    quaternion.z = (rotation(0, 2) + rotation(2, 0)) / scale;
  } else if (rotation(1, 1) > rotation(2, 2)) {
    const double scale = 2.0 * std::sqrt(1.0 + rotation(1, 1) - rotation(0, 0) - rotation(2, 2));
    quaternion.w = (rotation(0, 2) - rotation(2, 0)) / scale;
    quaternion.x = (rotation(0, 1) + rotation(1, 0)) / scale;
    quaternion.y = 0.25 * scale;
    quaternion.z = (rotation(1, 2) + rotation(2, 1)) / scale;
  } else {
    const double scale = 2.0 * std::sqrt(1.0 + rotation(2, 2) - rotation(0, 0) - rotation(1, 1));
    quaternion.w = (rotation(1, 0) - rotation(0, 1)) / scale;
    quaternion.x = (rotation(0, 2) + rotation(2, 0)) / scale;
    quaternion.y = (rotation(1, 2) + rotation(2, 1)) / scale;
    quaternion.z = 0.25 * scale;
  }
  return quaternion;
}

void append_pca_axis(
  visualization_msgs::msg::Marker & lines, const Eigen::Vector3d & center,
  const Eigen::Vector3d & direction, double radius, int color_index)
{
  if (!(radius > 1e-4)) {
    return;
  }
  const std_msgs::msg::ColorRGBA color = axis_color(color_index);
  lines.points.push_back(
    to_point(center.x() - direction.x() * radius, center.y() - direction.y() * radius,
      center.z() - direction.z() * radius));
  lines.colors.push_back(color);
  lines.points.push_back(
    to_point(center.x() + direction.x() * radius, center.y() + direction.y() * radius,
      center.z() + direction.z() * radius));
  lines.colors.push_back(color);
}

visualization_msgs::msg::Marker covariance_ellipsoid(
  const std_msgs::msg::Header & header, int id, const IntensityPca & pca, double voxel_size,
  double voxel_height)
{
  visualization_msgs::msg::Marker sphere;
  sphere.header = header;
  sphere.ns = "covariance";
  sphere.id = id;
  sphere.type = visualization_msgs::msg::Marker::SPHERE;
  sphere.action = visualization_msgs::msg::Marker::ADD;
  sphere.pose.position.x = pca.center.x();
  sphere.pose.position.y = pca.center.y();
  sphere.pose.position.z = pca.center.z();
  sphere.pose.orientation = quaternion_from_rotation(pca.axes);
  const double diameter_x = std::min(2.0 * pca.radii.x(), voxel_size);
  const double diameter_y = std::min(2.0 * pca.radii.y(), voxel_size);
  const double diameter_z = std::min(2.0 * pca.radii.z(), voxel_height);
  sphere.scale.x = std::max(diameter_x, 1e-3);
  sphere.scale.y = std::max(diameter_y, 1e-3);
  sphere.scale.z = std::max(diameter_z, 1e-3);
  sphere.color.r = 0.2F;
  sphere.color.g = 0.7F;
  sphere.color.b = 1.0F;
  sphere.color.a = 0.25F;
  return sphere;
}

}  // namespace

IntensityLaneNode::IntensityLaneNode(const rclcpp::NodeOptions & options)
: Node("intensity_lane_node", options)
{
  const auto pointcloud_topic =
    this->declare_parameter<std::string>("pointcloud_topic", "/boreas/pointcloud");
  map_frame_ = this->declare_parameter<std::string>("map_frame", "map");
  base_frame_ = this->declare_parameter<std::string>("base_frame", "base_link");
  const auto marker_topic =
    this->declare_parameter<std::string>("marker_topic", "back_odom/intensity_grid");
  voxel_size_ = this->declare_parameter<double>("voxel_size", 0.25);
  voxel_height_ = this->declare_parameter<double>("voxel_height", 0.5);
  longitudinal_distance_ = this->declare_parameter<double>("longitudinal_distance", 50.0);
  lateral_distance_ = this->declare_parameter<double>("lateral_distance", 25.0);
  z_below_ = this->declare_parameter<double>("z_below", 5.0);
  min_points_ = this->declare_parameter<int>("min_points", 3);
  intensity_low_ = this->declare_parameter<double>("intensity_low", 0.0);
  intensity_high_ = this->declare_parameter<double>("intensity_high", 80.0);
  publish_ellipsoids_ = this->declare_parameter<bool>("publish_ellipsoids", false);
  accumulate_input_ = this->declare_parameter<bool>("accumulate_input", true);

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
  cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    pointcloud_topic, rclcpp::QoS(rclcpp::KeepLast(2)).reliable(),
    [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) { this->callback_cloud(msg); });
  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    marker_topic, rclcpp::QoS(rclcpp::KeepLast(1)));
  const auto pca_topic =
    this->declare_parameter<std::string>("pca_topic", "back_odom/pca_axes");
  pca_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    pca_topic, rclcpp::QoS(rclcpp::KeepLast(1)));
  const auto accumulated_topic =
    this->declare_parameter<std::string>("accumulated_topic", "back_odom/accumulated_cloud");
  accumulated_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
    accumulated_topic, rclcpp::QoS(rclcpp::KeepLast(1)));

  lane::SegmentationParams & seg = segmentation_params_;
  seg.bins = this->declare_parameter<int>("segmentation.bins", seg.bins);
  seg.curvature_max =
    this->declare_parameter<double>("segmentation.curvature_max", seg.curvature_max);
  seg.cv_max = this->declare_parameter<double>("segmentation.cv_max", seg.cv_max);
  seg.laplacian_max =
    this->declare_parameter<double>("segmentation.laplacian_max", seg.laplacian_max);
  seg.peak_min_votes =
    this->declare_parameter<double>("segmentation.peak_min_votes", seg.peak_min_votes);
  seg.peak_min_separation =
    this->declare_parameter<int>("segmentation.peak_min_separation", seg.peak_min_separation);
  seg.max_classes = this->declare_parameter<int>("segmentation.max_classes", seg.max_classes);
  seg.feature_sigma =
    this->declare_parameter<double>("segmentation.feature_sigma", seg.feature_sigma);
  seg.propagation_iterations = this->declare_parameter<int>(
    "segmentation.propagation_iterations", seg.propagation_iterations);
  const auto segments_topic =
    this->declare_parameter<std::string>("segments_topic", "back_odom/segments");
  segments_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    segments_topic, rclcpp::QoS(rclcpp::KeepLast(1)));
  const auto segmented_cloud_topic =
    this->declare_parameter<std::string>("segmented_cloud_topic", "back_odom/segmented_cloud");
  segmented_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
    segmented_cloud_topic, rclcpp::QoS(rclcpp::KeepLast(1)));
}

void IntensityLaneNode::callback_cloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  if (msg->header.frame_id.empty()) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Point cloud has no frame_id.");
    return;
  }

  geometry_msgs::msg::TransformStamped sensor_to_map;
  geometry_msgs::msg::TransformStamped sensor_to_base;
  geometry_msgs::msg::TransformStamped map_to_base;
  try {
    sensor_to_map = tf_buffer_->lookupTransform(
      map_frame_, msg->header.frame_id, msg->header.stamp, tf2::durationFromSec(0.2));
    sensor_to_base = tf_buffer_->lookupTransform(
      base_frame_, msg->header.frame_id, msg->header.stamp, tf2::durationFromSec(0.2));
    map_to_base = tf_buffer_->lookupTransform(
      base_frame_, map_frame_, msg->header.stamp, tf2::durationFromSec(0.2));
  } catch (const tf2::TransformException & exception) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000, "TF for scan accumulation is not ready: %s",
      exception.what());
    return;
  }

  const auto * x_field = find_field(*msg, "x");
  const auto * y_field = find_field(*msg, "y");
  const auto * z_field = find_field(*msg, "z");
  const auto * intensity_field = find_field(*msg, "intensity");
  if (x_field == nullptr || y_field == nullptr || z_field == nullptr || intensity_field == nullptr ||
      msg->point_step == 0)
  {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "Scan is missing float x/y/z/intensity fields.");
    return;
  }

  const bool publish_grid = marker_pub_->get_subscription_count() > 0;
  const bool publish_pca = pca_pub_->get_subscription_count() > 0;
  const bool publish_cloud = accumulated_pub_->get_subscription_count() > 0;
  const bool publish_segments = segments_pub_->get_subscription_count() > 0 ||
                                segmented_cloud_pub_->get_subscription_count() > 0;
  const bool need_cells = publish_grid || publish_pca || publish_segments;
  const double inverse_voxel = 1.0 / voxel_size_;
  const double inverse_height = 1.0 / voxel_height_;
  CellMap cells;
  base_points_.clear();
  if (!accumulate_input_) {
    // The input is already a map; appending it every frame would only duplicate points.
    accumulated_.clear();
  }
  const auto started = std::chrono::steady_clock::now();

  auto bin_point = [this, &cells, inverse_voxel, inverse_height, publish_segments](
                     double vx, double vy, double vz, float intensity) {
    const CellKey key = lane::add_to_cell(cells, vx, vy, vz, intensity, inverse_voxel, inverse_height);
    if (publish_segments) {
      BasePoint base;
      base.x = static_cast<float>(vx);
      base.y = static_cast<float>(vy);
      base.z = static_cast<float>(vz);
      base.intensity = intensity;
      base.key = key;
      base_points_.push_back(base);
    }
  };

  const Aabb map_crop = crop_aabb_in_map(
    invert_transform(map_to_base.transform), longitudinal_distance_, lateral_distance_, z_below_);
  std::size_t kept = 0;
  for (const MapPoint & point : accumulated_) {
    if (outside_aabb(point.x, point.y, point.z, map_crop)) {
      continue;
    }
    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;
    apply_transform(map_to_base.transform, point.x, point.y, point.z, vx, vy, vz);
    if (!in_crop(vx, vy, vz, longitudinal_distance_, lateral_distance_, z_below_)) {
      continue;
    }
    accumulated_[kept] = point;
    ++kept;
    if (need_cells) {
      bin_point(vx, vy, vz, point.intensity);
    }
  }
  accumulated_.resize(kept);

  const std::size_t count = static_cast<std::size_t>(msg->width) * msg->height;
  accumulated_.reserve(accumulated_.size() + count);
  for (std::size_t index = 0; index < count; ++index) {
    const double sx = read_float(*msg, *x_field, index);
    const double sy = read_float(*msg, *y_field, index);
    const double sz = read_float(*msg, *z_field, index);
    const float intensity = read_float(*msg, *intensity_field, index);
    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;
    apply_transform(sensor_to_base.transform, sx, sy, sz, vx, vy, vz);
    if (!in_crop(vx, vy, vz, longitudinal_distance_, lateral_distance_, z_below_)) {
      continue;
    }
    MapPoint point;
    apply_transform(sensor_to_map.transform, sx, sy, sz, point.x, point.y, point.z);
    point.intensity = intensity;
    accumulated_.push_back(point);
    if (need_cells) {
      bin_point(vx, vy, vz, intensity);
    }
  }

  std_msgs::msg::Header header = msg->header;
  header.frame_id = base_frame_;

  if (publish_cloud) {
    sensor_msgs::msg::PointCloud2 debug_cloud;
    debug_cloud.header.stamp = msg->header.stamp;
    debug_cloud.header.frame_id = map_frame_;
    debug_cloud.height = 1;
    debug_cloud.width = static_cast<std::uint32_t>(accumulated_.size());
    debug_cloud.is_dense = true;
    debug_cloud.is_bigendian = false;
    sensor_msgs::msg::PointField field;
    field.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field.count = 1;
    const char * names[4] = {"x", "y", "z", "intensity"};
    debug_cloud.fields.reserve(4);
    for (int index = 0; index < 4; ++index) {
      field.name = names[index];
      field.offset = static_cast<std::uint32_t>(index * 4);
      debug_cloud.fields.push_back(field);
    }
    debug_cloud.point_step = 16;
    debug_cloud.row_step = debug_cloud.point_step * debug_cloud.width;
    debug_cloud.data.resize(static_cast<std::size_t>(debug_cloud.row_step));
    for (std::size_t index = 0; index < accumulated_.size(); ++index) {
      const MapPoint & point = accumulated_[index];
      const float values[4] = {
        static_cast<float>(point.x), static_cast<float>(point.y), static_cast<float>(point.z),
        point.intensity};
      std::memcpy(debug_cloud.data.data() + index * debug_cloud.point_step, values, sizeof(values));
    }
    accumulated_pub_->publish(debug_cloud);
  }

  if (!need_cells) {
    return;
  }

  visualization_msgs::msg::Marker clear;
  clear.header = header;
  clear.action = visualization_msgs::msg::Marker::DELETEALL;

  auto intensity_marker = cube_list(header, "intensity", 0, voxel_size_, voxel_height_);
  intensity_marker.points.reserve(cells.size());
  intensity_marker.colors.reserve(cells.size());
  visualization_msgs::msg::Marker axes;
  axes.header = header;
  axes.ns = "pca";
  axes.id = 0;
  axes.type = visualization_msgs::msg::Marker::LINE_LIST;
  axes.action = visualization_msgs::msg::Marker::ADD;
  axes.pose.orientation.w = 1.0;
  axes.scale.x = 0.03;
  axes.color.a = 1.0F;
  visualization_msgs::msg::MarkerArray pca_markers;
  pca_markers.markers.push_back(clear);
  int ellipsoid_id = 1;
  for (const auto & item : cells) {
    if (item.second.count < min_points_) {
      continue;
    }
    if (publish_grid) {
      geometry_msgs::msg::Point center;
      center.x = (static_cast<double>(item.first.x) + 0.5) * voxel_size_;
      center.y = (static_cast<double>(item.first.y) + 0.5) * voxel_size_;
      center.z = (static_cast<double>(item.first.z) + 0.5) * voxel_height_;
      const double mean_intensity =
        item.second.sum_intensity / static_cast<double>(item.second.count);
      const double variance =
        item.second.sum_intensity_sq / static_cast<double>(item.second.count) -
        mean_intensity * mean_intensity;
      intensity_marker.points.push_back(center);
      intensity_marker.colors.push_back(intensity_color(
        mean_intensity, std::sqrt(std::max(variance, 0.0)), intensity_low_, intensity_high_));
    }
    if (!publish_pca) {
      continue;
    }
    const IntensityPca pca = lane::fit_intensity_pca(item.second, min_points_);
    if (!pca.valid) {
      continue;
    }
    if (publish_ellipsoids_) {
      pca_markers.markers.push_back(
        covariance_ellipsoid(header, ellipsoid_id, pca, voxel_size_, voxel_height_));
      ++ellipsoid_id;
    }
    append_pca_axis(
      axes, pca.center, pca.axes.col(0), std::min(pca.radii.x(), 0.5 * voxel_size_), 0);
    append_pca_axis(
      axes, pca.center, pca.axes.col(1), std::min(pca.radii.y(), 0.5 * voxel_size_), 1);
    append_pca_axis(
      axes, pca.center, pca.axes.col(2), std::min(pca.radii.z(), 0.5 * voxel_height_), 2);
  }
  if (publish_grid) {
    visualization_msgs::msg::MarkerArray markers;
    markers.markers.push_back(clear);
    if (!intensity_marker.points.empty()) {
      markers.markers.push_back(std::move(intensity_marker));
    }
    marker_pub_->publish(markers);
  }
  if (publish_pca) {
    if (!axes.points.empty()) {
      pca_markers.markers.push_back(std::move(axes));
    }
    pca_pub_->publish(pca_markers);
  }

  if (publish_segments) {
    lane::CellGrid grid = lane::flatten(cells);
    lane::compute_local_features(grid, min_points_);
    lane::compute_neighbor_features(grid, voxel_size_, voxel_height_);
    const std::size_t valid_cells = static_cast<std::size_t>(std::count_if(
      grid.features.begin(), grid.features.end(),
      [](const lane::CellFeatures & feature) { return feature.valid; }));
    if (valid_cells == 0 && !grid.keys.empty()) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "No cell reached min_points=%d (%zu cells, %zu points). The input is too sparse for "
        "per-cell statistics; feed raw scans or enlarge voxel_size.",
        min_points_, grid.keys.size(), accumulated_.size());
    }
    const lane::SegmentationResult result =
      lane::segment_cells(grid, segmentation_params_, previous_peaks_);
    previous_peaks_ = result.peaks;
    publish_segmentation(header, grid, result);
    const double elapsed_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
        .count();
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "segmentation: %zu points, %zu cells, %zu valid, %zu classes, %.1f ms",
      accumulated_.size(), grid.keys.size(), valid_cells, result.peaks.size(), elapsed_ms);
  }
}

void IntensityLaneNode::publish_segmentation(
  const std_msgs::msg::Header & header, const lane::CellGrid & grid,
  const lane::SegmentationResult & result)
{
  if (segments_pub_->get_subscription_count() > 0) {
    visualization_msgs::msg::Marker clear;
    clear.header = header;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    auto cubes = cube_list(header, "segments", 0, voxel_size_, voxel_height_);
    cubes.points.reserve(grid.size());
    cubes.colors.reserve(grid.size());
    for (std::size_t i = 0; i < grid.size(); ++i) {
      const int label = result.label[i];
      if (label <= 0) {
        continue;
      }
      const CellKey & key = grid.keys[i];
      cubes.points.push_back(to_point(
        (static_cast<double>(key.x) + 0.5) * voxel_size_,
        (static_cast<double>(key.y) + 0.5) * voxel_size_,
        (static_cast<double>(key.z) + 0.5) * voxel_height_));
      cubes.colors.push_back(label_color(label));
    }
    visualization_msgs::msg::MarkerArray markers;
    markers.markers.push_back(clear);
    if (!cubes.points.empty()) {
      markers.markers.push_back(std::move(cubes));
    }
    segments_pub_->publish(markers);
  }

  if (segmented_cloud_pub_->get_subscription_count() > 0) {
    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header = header;
    cloud.height = 1;
    cloud.width = static_cast<std::uint32_t>(base_points_.size());
    cloud.is_dense = true;
    cloud.is_bigendian = false;
    const char * float_names[4] = {"x", "y", "z", "intensity"};
    cloud.fields.reserve(6);
    sensor_msgs::msg::PointField field;
    field.count = 1;
    field.datatype = sensor_msgs::msg::PointField::FLOAT32;
    for (int index = 0; index < 4; ++index) {
      field.name = float_names[index];
      field.offset = static_cast<std::uint32_t>(index * 4);
      cloud.fields.push_back(field);
    }
    field.datatype = sensor_msgs::msg::PointField::UINT32;
    field.name = "label";
    field.offset = 16;
    cloud.fields.push_back(field);
    field.name = "segment";
    field.offset = 20;
    cloud.fields.push_back(field);
    cloud.point_step = 24;
    cloud.row_step = cloud.point_step * cloud.width;
    cloud.data.resize(static_cast<std::size_t>(cloud.row_step));
    for (std::size_t index = 0; index < base_points_.size(); ++index) {
      const BasePoint & point = base_points_[index];
      const int cell = grid.find(point.key);
      const std::uint32_t label =
        cell >= 0 ? static_cast<std::uint32_t>(std::max(result.label[cell], 0)) : 0U;
      const std::uint32_t segment =
        cell >= 0 ? static_cast<std::uint32_t>(std::max(result.segment[cell], 0)) : 0U;
      std::uint8_t * destination = cloud.data.data() + index * cloud.point_step;
      const float values[4] = {point.x, point.y, point.z, point.intensity};
      std::memcpy(destination, values, sizeof(values));
      std::memcpy(destination + 16, &label, sizeof(label));
      std::memcpy(destination + 20, &segment, sizeof(segment));
    }
    segmented_cloud_pub_->publish(cloud);
  }
}

}  // namespace back_odom

RCLCPP_COMPONENTS_REGISTER_NODE(back_odom::IntensityLaneNode)
