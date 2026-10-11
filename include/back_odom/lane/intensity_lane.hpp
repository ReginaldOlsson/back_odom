#ifndef BACK_ODOM__LANE__INTENSITY_LANE_HPP_
#define BACK_ODOM__LANE__INTENSITY_LANE_HPP_

#include "back_odom/lane/cell_features.hpp"
#include "back_odom/lane/cell_segmentation.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <memory>
#include <string>
#include <vector>

namespace back_odom
{

/// Accumulates lidar scans into the map with TF, then estimates intensity on a grid in base_link.
class IntensityLaneNode : public rclcpp::Node
{
public:
  explicit IntensityLaneNode(const rclcpp::NodeOptions & options);

private:
  struct MapPoint
  {
    double x{0.0};
    double y{0.0};
    double z{0.0};
    float intensity{0.0F};
  };

  /// Point in base_link for the current frame, with the cell it was binned into.
  struct BasePoint
  {
    float x{0.0F};
    float y{0.0F};
    float z{0.0F};
    float intensity{0.0F};
    lane::CellKey key;
  };

  void callback_cloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
  void publish_segmentation(
    const std_msgs::msg::Header & header, const lane::CellGrid & grid,
    const lane::SegmentationResult & result);

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pca_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr accumulated_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr segments_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr segmented_cloud_pub_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

  std::vector<MapPoint> accumulated_;
  std::vector<BasePoint> base_points_;
  std::vector<lane::Peak> previous_peaks_;
  lane::SegmentationParams segmentation_params_;
  std::string map_frame_{"map"};
  std::string base_frame_{"base_link"};

  double voxel_size_{0.5};
  double voxel_height_{1.0};
  double longitudinal_distance_{50.0};
  double lateral_distance_{25.0};
  double z_below_{5.0};
  int min_points_{8};
  double intensity_low_{0.0};
  double intensity_high_{80.0};
  bool publish_ellipsoids_{false};
  /// False when the input is already an accumulated map: replace the buffer instead of appending.
  bool accumulate_input_{true};
};

}  // namespace back_odom

#endif  // BACK_ODOM__LANE__INTENSITY_LANE_HPP_
