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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <utility>
#include <vector>

namespace back_odom
{
namespace
{

struct CellKey
{
  std::int64_t x{0};
  std::int64_t y{0};
  std::int64_t z{0};

  bool operator==(const CellKey & other) const
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct CellKeyHash
{
  std::size_t operator()(const CellKey & key) const
  {
    return (static_cast<std::size_t>(key.x) * 1315423911U) ^
           (static_cast<std::size_t>(key.y) * 2654435761U) ^
           (static_cast<std::size_t>(key.z) * 97531U);
  }
};

struct CellStats
{
  double sum_intensity{0.0};
  double sum_intensity_sq{0.0};
  double sum_weight{0.0};
  double sum_wx{0.0};
  double sum_wy{0.0};
  double sum_wz{0.0};
  double sum_wxx{0.0};
  double sum_wyy{0.0};
  double sum_wzz{0.0};
  double sum_wxy{0.0};
  double sum_wxz{0.0};
  double sum_wyz{0.0};
  int count{0};
};

struct IntensityPca
{
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d axes{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d radii{Eigen::Vector3d::Zero()};
  bool valid{false};
};

struct CellDistribution
{
  double mean_x{0.0};
  double mean_y{0.0};
  double mean_z{0.0};
  double mean_intensity{0.0};
  double stddev{0.0};
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
  const double tw = qx * px + qy * py + qz * pz;
  const double txr = qw * px + qy * pz - qz * py;
  const double tyr = qw * py + qz * px - qx * pz;
  const double tzr = qw * pz + qx * py - qy * px;
  ox = transform.translation.x + (txr * qw - tw * qx - tyr * qz + tzr * qy);
  oy = transform.translation.y + (tyr * qw - tw * qy - tzr * qx + txr * qz);
  oz = transform.translation.z + (tzr * qw - tw * qz - txr * qy + tyr * qx);
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

/// Intensity-weighted covariance of positions about the voxel center. Columns of `axes` are the
/// principal components (red = largest, green = middle, blue = smallest).
IntensityPca fit_intensity_pca(
  const CellStats & cell, double center_x, double center_y, double center_z, int min_points)
{
  IntensityPca pca;
  pca.center = Eigen::Vector3d(center_x, center_y, center_z);
  if (cell.count < min_points || !(cell.sum_weight > 1e-6)) {
    return pca;
  }
  const double inverse = 1.0 / cell.sum_weight;
  const double cx = center_x;
  const double cy = center_y;
  const double cz = center_z;
  Eigen::Matrix3d covariance;
  covariance << (cell.sum_wxx - 2.0 * cx * cell.sum_wx + cx * cx * cell.sum_weight) * inverse,
    (cell.sum_wxy - cx * cell.sum_wy - cy * cell.sum_wx + cx * cy * cell.sum_weight) * inverse,
    (cell.sum_wxz - cx * cell.sum_wz - cz * cell.sum_wx + cx * cz * cell.sum_weight) * inverse,
    (cell.sum_wxy - cx * cell.sum_wy - cy * cell.sum_wx + cx * cy * cell.sum_weight) * inverse,
    (cell.sum_wyy - 2.0 * cy * cell.sum_wy + cy * cy * cell.sum_weight) * inverse,
    (cell.sum_wyz - cy * cell.sum_wz - cz * cell.sum_wy + cy * cz * cell.sum_weight) * inverse,
    (cell.sum_wxz - cx * cell.sum_wz - cz * cell.sum_wx + cx * cz * cell.sum_weight) * inverse,
    (cell.sum_wyz - cy * cell.sum_wz - cz * cell.sum_wy + cy * cz * cell.sum_weight) * inverse,
    (cell.sum_wzz - 2.0 * cz * cell.sum_wz + cz * cz * cell.sum_weight) * inverse;
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  if (solver.info() != Eigen::Success) {
    return pca;
  }
  const Eigen::Vector3d values = solver.eigenvalues();
  if (!(values.maxCoeff() > 1e-8)) {
    return pca;
  }
  pca.axes.col(0) = solver.eigenvectors().col(2);
  pca.axes.col(1) = solver.eigenvectors().col(1);
  pca.axes.col(2) = solver.eigenvectors().col(0);
  if (pca.axes.determinant() < 0.0) {
    pca.axes.col(2) *= -1.0;
  }
  pca.radii = Eigen::Vector3d(
    std::sqrt(std::max(values[2], 0.0)), std::sqrt(std::max(values[1], 0.0)),
    std::sqrt(std::max(values[0], 0.0)));
  pca.valid = true;
  return pca;
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
  min_points_ = this->declare_parameter<int>("min_points", 8);
  intensity_low_ = this->declare_parameter<double>("intensity_low", 0.0);
  intensity_high_ = this->declare_parameter<double>("intensity_high", 80.0);

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
}

void IntensityLaneNode::callback_cloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  if (msg->header.frame_id.empty()) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Point cloud has no frame_id.");
    return;
  }

  geometry_msgs::msg::TransformStamped sensor_to_map;
  geometry_msgs::msg::TransformStamped map_to_base;
  try {
    sensor_to_map = tf_buffer_->lookupTransform(
      map_frame_, msg->header.frame_id, msg->header.stamp, tf2::durationFromSec(0.2));
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

  struct VehiclePoint
  {
    double x{0.0};
    double y{0.0};
    double z{0.0};
    float intensity{0.0F};
  };
  const std::size_t count = static_cast<std::size_t>(msg->width) * msg->height;
  accumulated_.reserve(accumulated_.size() + count);
  for (std::size_t index = 0; index < count; ++index) {
    MapPoint point;
    apply_transform(
      sensor_to_map.transform, read_float(*msg, *x_field, index), read_float(*msg, *y_field, index),
      read_float(*msg, *z_field, index), point.x, point.y, point.z);
    point.intensity = read_float(*msg, *intensity_field, index);
    accumulated_.push_back(point);
  }

  std::vector<MapPoint> kept;
  std::vector<VehiclePoint> vehicle_points;
  kept.reserve(accumulated_.size());
  vehicle_points.reserve(accumulated_.size());
  for (const MapPoint & point : accumulated_) {
    VehiclePoint vehicle;
    apply_transform(map_to_base.transform, point.x, point.y, point.z, vehicle.x, vehicle.y, vehicle.z);
    vehicle.intensity = point.intensity;
    if (std::abs(vehicle.x) > longitudinal_distance_ || std::abs(vehicle.y) > lateral_distance_ ||
        vehicle.z > 0.0 || vehicle.z < -z_below_)
    {
      continue;
    }
    kept.push_back(point);
    vehicle_points.push_back(vehicle);
  }
  accumulated_.swap(kept);

  std_msgs::msg::Header header = msg->header;
  header.frame_id = base_frame_;

  if (accumulated_pub_->get_subscription_count() > 0) {
    sensor_msgs::msg::PointCloud2 debug_cloud;
    debug_cloud.header = header;
    debug_cloud.height = 1;
    debug_cloud.width = static_cast<std::uint32_t>(vehicle_points.size());
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
    for (std::size_t index = 0; index < vehicle_points.size(); ++index) {
      const VehiclePoint & point = vehicle_points[index];
      const float values[4] = {
        static_cast<float>(point.x), static_cast<float>(point.y), static_cast<float>(point.z),
        point.intensity};
      std::memcpy(debug_cloud.data.data() + index * debug_cloud.point_step, values, sizeof(values));
    }
    accumulated_pub_->publish(debug_cloud);
  }

  const double inverse_voxel = 1.0 / voxel_size_;
  const double inverse_height = 1.0 / voxel_height_;
  std::unordered_map<CellKey, CellStats, CellKeyHash> cells;
  for (const VehiclePoint & point : vehicle_points) {
    const CellKey key{
      static_cast<std::int64_t>(std::floor(point.x * inverse_voxel)),
      static_cast<std::int64_t>(std::floor(point.y * inverse_voxel)),
      static_cast<std::int64_t>(std::floor(point.z * inverse_height))};
    CellStats & cell = cells[key];
    const double intensity = point.intensity;
    cell.sum_intensity += intensity;
    cell.sum_intensity_sq += intensity * intensity;
    const double weight = std::max(intensity, 0.0);
    cell.sum_weight += weight;
    cell.sum_wx += weight * point.x;
    cell.sum_wy += weight * point.y;
    cell.sum_wz += weight * point.z;
    cell.sum_wxx += weight * point.x * point.x;
    cell.sum_wyy += weight * point.y * point.y;
    cell.sum_wzz += weight * point.z * point.z;
    cell.sum_wxy += weight * point.x * point.y;
    cell.sum_wxz += weight * point.x * point.z;
    cell.sum_wyz += weight * point.y * point.z;
    ++cell.count;
  }

  std::vector<CellDistribution> distributions;
  distributions.reserve(cells.size());
  for (const auto & item : cells) {
    if (item.second.count < min_points_) {
      continue;
    }
    CellDistribution distribution;
    distribution.mean_x = (static_cast<double>(item.first.x) + 0.5) * voxel_size_;
    distribution.mean_y = (static_cast<double>(item.first.y) + 0.5) * voxel_size_;
    distribution.mean_z = (static_cast<double>(item.first.z) + 0.5) * voxel_height_;
    distribution.mean_intensity =
      item.second.sum_intensity / static_cast<double>(item.second.count);
    const double variance =
      item.second.sum_intensity_sq / static_cast<double>(item.second.count) -
      distribution.mean_intensity * distribution.mean_intensity;
    distribution.stddev = std::sqrt(std::max(variance, 0.0));
    distributions.push_back(distribution);
  }

  visualization_msgs::msg::Marker clear;
  clear.header = header;
  clear.action = visualization_msgs::msg::Marker::DELETEALL;

  auto intensity_marker = cube_list(header, "intensity", 0, voxel_size_, voxel_height_);
  intensity_marker.points.reserve(distributions.size());
  intensity_marker.colors.reserve(distributions.size());

  for (const CellDistribution & distribution : distributions) {
    geometry_msgs::msg::Point center;
    center.x = distribution.mean_x;
    center.y = distribution.mean_y;
    center.z = distribution.mean_z;
    intensity_marker.points.push_back(center);
    intensity_marker.colors.push_back(intensity_color(
      distribution.mean_intensity, distribution.stddev, intensity_low_, intensity_high_));
  }

  visualization_msgs::msg::MarkerArray markers;
  markers.markers.push_back(clear);
  if (!intensity_marker.points.empty()) {
    markers.markers.push_back(std::move(intensity_marker));
  }
  marker_pub_->publish(markers);

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
    const double center_x = (static_cast<double>(item.first.x) + 0.5) * voxel_size_;
    const double center_y = (static_cast<double>(item.first.y) + 0.5) * voxel_size_;
    const double center_z = (static_cast<double>(item.first.z) + 0.5) * voxel_height_;
    const IntensityPca pca =
      fit_intensity_pca(item.second, center_x, center_y, center_z, min_points_);
    if (!pca.valid) {
      continue;
    }
    pca_markers.markers.push_back(
      covariance_ellipsoid(header, ellipsoid_id, pca, voxel_size_, voxel_height_));
    ++ellipsoid_id;
    append_pca_axis(
      axes, pca.center, pca.axes.col(0), std::min(pca.radii.x(), 0.5 * voxel_size_), 0);
    append_pca_axis(
      axes, pca.center, pca.axes.col(1), std::min(pca.radii.y(), 0.5 * voxel_size_), 1);
    append_pca_axis(
      axes, pca.center, pca.axes.col(2), std::min(pca.radii.z(), 0.5 * voxel_height_), 2);
  }
  if (!axes.points.empty()) {
    pca_markers.markers.push_back(std::move(axes));
  }
  pca_pub_->publish(pca_markers);
}

}  // namespace back_odom

RCLCPP_COMPONENTS_REGISTER_NODE(back_odom::IntensityLaneNode)
