#!/usr/bin/env python3
"""Convert livox_ros_driver2/CustomMsg into PointCloud2 for back_odom.

CustomMsg.header.stamp is the end of the sweep and matches the IMU clock.
offset_time is nanoseconds from the start of that sweep (timebase is a device
clock, not Unix time). Point time is therefore seconds before the header, so
back_odom's relative-time path places scan_end on the header. Intensity is the
Livox reflectivity.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2

from livox_ros_driver2.msg import CustomMsg


class LivoxCustomToCloud(Node):
    def __init__(self) -> None:
        super().__init__("livox_custom_to_cloud")
        input_topic = self.declare_parameter("input_topic", "/livox/lidar").value
        output_topic = self.declare_parameter("output_topic", "/livox/points").value
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self._pub = self.create_publisher(PointCloud2, output_topic, qos)
        self.create_subscription(CustomMsg, input_topic, self._on_cloud, qos)
        self.get_logger().info(f"Converting {input_topic} -> {output_topic}")

    def _on_cloud(self, msg: CustomMsg) -> None:
        count = int(msg.point_num)
        if count <= 0:
            return
        end_ns = max(point.offset_time for point in msg.points[:count])
        rows = [
            (
                point.x,
                point.y,
                point.z,
                (float(point.offset_time) - float(end_ns)) * 1.0e-9,
                float(point.reflectivity),
            )
            for point in msg.points[:count]
        ]
        cloud = point_cloud2.create_cloud(
            msg.header,
            [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="time", offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name="intensity", offset=16, datatype=PointField.FLOAT32, count=1),
            ],
            rows,
        )
        self._pub.publish(cloud)


def main() -> None:
    rclpy.init()
    node = LivoxCustomToCloud()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
