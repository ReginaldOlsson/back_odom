#!/usr/bin/env python3
# Copyright 2026 huzaifa.osal
#
# Record nav_msgs/Odometry topics to CSV while a bag plays.

"""Dump odometry topics to stamp,x,y,z,qx,qy,qz,qw CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy


class OdomDumper(Node):
    def __init__(self, topics: list[str], out_dir: Path) -> None:
        super().__init__("dump_odom_csv")
        self._out_dir = out_dir
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._writers: dict[str, csv.writer] = {}
        self._files = []
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,
        )
        for topic in topics:
            safe = topic.strip("/").replace("/", "_") or "odom"
            path = self._out_dir / f"{safe}.csv"
            handle = path.open("w", newline="", encoding="utf-8")
            self._files.append(handle)
            writer = csv.writer(handle)
            writer.writerow(["stamp", "x", "y", "z", "qx", "qy", "qz", "qw"])
            self._writers[topic] = writer
            self.create_subscription(
                Odometry,
                topic,
                lambda msg, t=topic: self._on_odom(t, msg),
                qos,
            )
            self.get_logger().info(f"recording {topic} -> {path}")

    def _on_odom(self, topic: str, msg: Odometry) -> None:
        stamp = float(msg.header.stamp.sec) + 1.0e-9 * float(msg.header.stamp.nanosec)
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._writers[topic].writerow([stamp, p.x, p.y, p.z, q.x, q.y, q.z, q.w])

    def destroy_node(self) -> bool:
        for handle in self._files:
            handle.close()
        return super().destroy_node()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topics",
        nargs="+",
        default=["/ekf_odom", "/back_odom"],
        help="Odometry topics to record",
    )
    parser.add_argument("--out", type=Path, default=Path("/tmp/boreas_odom_csv"))
    args = parser.parse_args()

    rclpy.init()
    node = OdomDumper(args.topics, args.out)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
