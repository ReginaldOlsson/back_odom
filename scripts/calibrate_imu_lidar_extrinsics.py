#!/usr/bin/env python3
# Copyright 2026 huzaifa.osal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Roll and pitch of base_link <- lidar from a stationary cloud.

Fit the ground plane in a stationary aggregate (the first scans, before the
vehicle moves). The rotation that takes that normal to +Z is the mount roll
and pitch. Yaw is the nominal mount yaw plus a residual measured from an ATE
trace; both yaws are applied about the leveled up axis.

Print the radians into the lidar joint of sensor_kit_calibration.yaml
(base_link <- lidar). This script does not edit that file.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def upward_normal(normal: np.ndarray) -> np.ndarray:
    """Unit normal with a positive Z component."""
    vector = np.asarray(normal, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(vector))
    if length < 1.0e-12:
        raise ValueError("ground normal has zero length")
    vector = vector / length
    if vector[2] < 0.0:
        vector = -vector
    return vector


def rotation_that_levels_normal(normal_lidar: np.ndarray) -> np.ndarray:
    """Rotation taking an upward lidar-frame normal onto +Z. Identity if already level."""
    normal = upward_normal(normal_lidar)
    z_axis = np.array([0.0, 0.0, 1.0])
    cross = np.cross(normal, z_axis)
    sine = float(np.linalg.norm(cross))
    if sine < 1.0e-12:
        return np.eye(3)
    cosine = float(np.dot(normal, z_axis))
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + skew + (skew @ skew) * ((1.0 - cosine) / (sine * sine))


def _rot_z(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def compose_base_from_lidar(
    normal_lidar: np.ndarray, nominal_yaw_deg: float, yaw_residual_deg: float
) -> np.ndarray:
    """base_link <- lidar. Level in the lidar frame, then yaw about the up axis."""
    level = rotation_that_levels_normal(normal_lidar)
    return _rot_z(yaw_residual_deg) @ _rot_z(nominal_yaw_deg) @ level


def euler_xyz(rotation: np.ndarray) -> tuple[float, float, float]:
    """Roll, pitch, yaw in radians. Same decomposition as intrinsic xyz / ROS rpy."""
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    roll = math.atan2(matrix[2, 1], matrix[2, 2])
    pitch = math.atan2(-matrix[2, 0], math.hypot(matrix[2, 1], matrix[2, 2]))
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    return roll, pitch, yaw


def fit_ground_normal(pcd_path: Path, radius_m: float = 15.0, max_z: float = -1.0) -> np.ndarray:
    """RANSAC ground normal in the lidar frame, forced to point up."""
    try:
        import open3d as o3d
    except ImportError as error:
        raise SystemExit("open3d is required to fit a cloud: pip install open3d") from error

    cloud = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(cloud.points)
    if points.size == 0:
        raise SystemExit(f"no points in {pcd_path}")
    radii = np.linalg.norm(points[:, :2], axis=1)
    near = points[(radii < radius_m) & (points[:, 2] < max_z)]
    if near.shape[0] < 3:
        raise SystemExit(
            f"need at least 3 ground points inside {radius_m:.0f} m and z < {max_z}, found {near.shape[0]}"
        )
    ground = o3d.geometry.PointCloud()
    ground.points = o3d.utility.Vector3dVector(near)
    plane_model, _inliers = ground.segment_plane(
        distance_threshold=0.05, ransac_n=3, num_iterations=1000
    )
    return upward_normal(np.asarray(plane_model[:3], dtype=np.float64))


def format_extrinsic(
    normal_lidar: np.ndarray, nominal_yaw_deg: float, yaw_residual_deg: float
) -> str:
    rotation = compose_base_from_lidar(normal_lidar, nominal_yaw_deg, yaw_residual_deg)
    roll, pitch, yaw = euler_xyz(rotation)
    names = ("roll", "pitch", "yaw")
    radians = (roll, pitch, yaw)
    lines = [f"ground_normal_lidar:  {normal_lidar[0]:.6f} {normal_lidar[1]:.6f} {normal_lidar[2]:.6f}"]
    lines.append("base_from_lidar rpy for sensor_kit_calibration.yaml lidar:")
    for name, angle in zip(names, radians, strict=True):
        lines.append(f"{name + '_deg':<12} {math.degrees(angle):.4f}")
        lines.append(f"{name + '_rad':<12} {angle:.6f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pcd", type=Path, help="Stationary aggregate cloud in the lidar frame")
    parser.add_argument("--nominal-yaw-deg", type=float, default=-45.0)
    parser.add_argument(
        "--yaw-residual-deg",
        type=float,
        default=1.853,
        help="ATE yaw residual added about the leveled up axis",
    )
    parser.add_argument("--radius-m", type=float, default=15.0)
    parser.add_argument("--max-z", type=float, default=-1.0)
    args = parser.parse_args(argv)
    normal = fit_ground_normal(args.pcd, radius_m=args.radius_m, max_z=args.max_z)
    print(format_extrinsic(normal, args.nominal_yaw_deg, args.yaw_residual_deg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
