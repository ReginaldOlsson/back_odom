#!/usr/bin/env python3
# Copyright 2026 huzaifa.osal

"""Synthetic checks for trajectory_metrics (no ROS bags)."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from trajectory_metrics import (  # noqa: E402
    Trajectory,
    Trajectory3,
    apply_se2,
    apply_se3,
    ate_metrics,
    ate_split,
    classify_error_source,
    contiguous_error_spans,
    detect_time_origin,
    evaluate_pair,
    interpolate_trajectory3,
    kitti_devkit,
    kitti_metrics,
    lidar_rotation_to_base,
    load_boreas_lidar_poses_se3,
    load_ri_ekf_trace,
    rotation_from_rpy,
    rpy_from_rotation,
    section_threshold,
    umeyama_se2,
    umeyama_se3,
    wrap_angle,
)


def _circle_traj(n: int = 400, radius: float = 50.0, dt: float = 0.1) -> Trajectory:
    stamps = np.arange(n, dtype=np.float64) * dt
    angles = stamps * (2.0 * math.pi / (n * dt))  # one full lap over the sequence duration
    # Constant speed around a circle: yaw is tangent.
    xy = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    yaw = angles + 0.5 * math.pi
    return Trajectory(stamps, xy, yaw)


class TrajectoryMetricsTest(unittest.TestCase):
    def test_umeyama_recovers_rigid_se2(self) -> None:
        gt = _circle_traj()
        # Rotate 30 deg, translate (10, -5).
        yaw = math.radians(30.0)
        c, s = math.cos(yaw), math.sin(yaw)
        transform = np.array([[c, -s, 10.0], [s, c, -5.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        # Estimate is GT mapped into another frame: est = T^{-1} * gt, so Umeyama recovers T.
        inv = np.linalg.inv(transform)
        estimate = apply_se2(inv, gt)

        recovered = umeyama_se2(estimate, gt)
        aligned = apply_se2(recovered, estimate)
        ate = ate_metrics(aligned, gt)
        self.assertLess(ate.rmse_xy_m, 1.0e-6)
        self.assertLess(ate.rmse_yaw_deg, 1.0e-4)

        report = evaluate_pair("synthetic", estimate, gt, max_dt_s=0.05)
        self.assertLess(report.ate_umeyama.rmse_xy_m, 1.0e-6)
        self.assertGreater(report.ate_umeyama.num_poses, 100)

        for seg in report.kitti:
            if seg.num_segments == 0:
                continue
            self.assertLess(seg.translation_percent, 1.0e-3)
            self.assertLess(seg.rotation_deg_per_100m, 1.0e-3)

    def test_kitti_zero_on_identical_paths(self) -> None:
        gt = _circle_traj()
        segments = kitti_metrics(gt, gt)
        for seg in segments:
            if seg.num_segments == 0:
                continue
            self.assertAlmostEqual(seg.translation_percent, 0.0, places=9)
            self.assertAlmostEqual(seg.rotation_deg_per_100m, 0.0, places=9)

    def test_rpy_roundtrip_matches_ekf_convention(self) -> None:
        roll = np.array([0.2, -0.4])
        pitch = np.array([-0.3, 0.15])
        yaw = np.array([1.4, -2.7])
        recovered = rpy_from_rotation(rotation_from_rpy(roll, pitch, yaw))
        np.testing.assert_allclose(recovered[:, 0], roll, atol=1e-12)
        np.testing.assert_allclose(recovered[:, 1], pitch, atol=1e-12)
        np.testing.assert_allclose(recovered[:, 2], yaw, atol=1e-12)

    def test_lidar_mount_yaw_matches_base_link(self) -> None:
        heading = np.array([0.619])
        rotation = lidar_rotation_to_base(heading, np.array([0.0]), np.array([0.0]), -math.pi / 4.0)
        yaw = float(rpy_from_rotation(rotation)[0, 2])
        expected = float(wrap_angle(-heading[0] + math.pi / 4.0))
        self.assertAlmostEqual(yaw, expected, places=8)

    def test_umeyama_se3_and_kitti_ignore_a_rigid_frame(self) -> None:
        ground_truth = _helix()
        yaw, pitch, roll = 0.4, -0.2, 0.1
        rotation = rotation_from_rpy(np.array([roll]), np.array([pitch]), np.array([yaw]))[0]
        transform = np.eye(4)
        transform[0:3, 0:3] = rotation
        transform[0:3, 3] = np.array([3.0, -4.0, 2.0])
        estimate = apply_se3(np.linalg.inv(transform), ground_truth)

        aligned = apply_se3(umeyama_se3(estimate, ground_truth), estimate)
        ate = ate_split(aligned, ground_truth)
        self.assertLess(ate.rmse_xyz_m, 1.0e-6)
        self.assertLess(ate.rmse_yaw_deg, 1.0e-4)

        devkit = kitti_devkit(estimate, ground_truth, space="se3")
        self.assertGreater(devkit.num_segments, 0)
        self.assertLess(abs(devkit.pooled_translation_percent), 1.0e-6)
        self.assertLess(abs(devkit.pooled_rotation_deg_per_100m), 1.0e-6)

        identical = kitti_devkit(ground_truth, ground_truth, space="se2")
        self.assertGreater(identical.num_segments, 0)
        self.assertAlmostEqual(identical.pooled_translation_percent, 0.0, places=8)

    def test_position_bump_becomes_one_section(self) -> None:
        distance = np.linspace(0.0, 300.0, 301)
        error = np.zeros_like(distance)
        error[(distance >= 100.0) & (distance <= 140.0)] = 8.0
        threshold = section_threshold(error, floor_m=5.0, percentile=90.0)
        self.assertGreaterEqual(threshold, 5.0)
        self.assertLessEqual(threshold, 8.0)
        spans = contiguous_error_spans(error, distance, threshold, min_length_m=20.0)
        self.assertEqual(len(spans), 1)
        self.assertGreaterEqual(distance[spans[0].start], 100.0 - 1.0)
        self.assertLessEqual(distance[spans[0].end], 140.0 + 1.0)
        self.assertEqual(classify_error_source(20.0, 18.0, 0.4, 5.0), "back_odom_led")
        self.assertEqual(classify_error_source(1.0, 20.0, 15.0, 5.0), "ekf_diverged")
        self.assertEqual(classify_error_source(20.0, 20.0, 12.0, 5.0), "both")

    def test_trace_loader_and_time_origin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            applanix = root / "applanix"
            applanix.mkdir()
            (applanix / "imu.csv").write_text(
                "GPSTime,angvel_z\n1606417090.0,0.0\n", encoding="utf-8"
            )
            (applanix / "gps_post_process.csv").write_text(
                "GPSTime,easting\n1606417090.0,0.0\n", encoding="utf-8"
            )
            lidar = applanix / "lidar_poses.csv"
            lidar.write_text(
                "GPSTime,easting,northing,altitude,vel_east,vel_north,vel_up,"
                "roll,pitch,heading,angvel_z,angvel_y,angvel_x\n"
                "1606417100000000,1.0,2.0,3.0,0,0,0,0,0,0.619,0,0,0\n"
                "1606417101000000,4.0,5.0,6.0,0,0,0,0,0,0.619,0,0,0\n",
                encoding="utf-8",
            )
            poses = load_boreas_lidar_poses_se3(lidar, -math.pi / 4.0)
            self.assertEqual(len(poses), 2)
            self.assertAlmostEqual(float(poses.stamps[0]), 1606417100.0, places=3)
            expected_yaw = float(wrap_angle(-0.619 + math.pi / 4.0))
            self.assertAlmostEqual(float(rpy_from_rotation(poses.rotation)[0, 2]), expected_yaw, places=8)

            header = (
                "stamp,ekf_x,ekf_y,ekf_z,ekf_roll,ekf_pitch,ekf_yaw,"
                "ekf_vx,ekf_vy,ekf_vz,ekf_body_vx,ekf_body_vy,ekf_body_vz,"
                "bg_x,bg_y,bg_z,ba_x,ba_y,ba_z,imu_ax,imu_ay,imu_az,imu_wx,imu_wy,imu_wz,"
                "back_odom_x,back_odom_y,back_odom_z,back_odom_roll,back_odom_pitch,back_odom_yaw,"
                "back_odom_age,err_x,err_y,err_z,err_yaw,pose_applied,pose_dropped_late,"
                "pose_dropped_no_buffer"
            )
            rows = [
                "10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
                "nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,0,0",
                "10.1,1,0,0,0,0,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
                "1,2,3,0,0,0.1,0.05,0.2,0,0,0,1,0,0",
                "10.2,1,0,0,0,0,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
                "1,2,3,0,0,0.1,0.15,0.2,0,0,0,1,0,0",
                "10.3,2,0,0,0,0,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
                "4,5,6,0,0,0.2,0.02,0,0,0,0,2,0,0",
            ]
            trace_path = root / "trace.csv"
            trace_path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
            trace = load_ri_ekf_trace(trace_path)
            self.assertEqual(len(trace.ekf), 4)
            self.assertEqual(len(trace.back_odom), 2)
            self.assertAlmostEqual(float(trace.back_odom.stamps[0]), 10.05, places=8)
            self.assertAlmostEqual(float(trace.back_odom.xyz[0, 0]), 1.0, places=8)
            self.assertAlmostEqual(float(trace.back_odom.stamps[1]), 10.28, places=8)

            origin, source = detect_time_origin(root, float(trace.ekf.stamps[0]), float(poses.stamps[0]))
            self.assertEqual(source, "imu_gnss_lidar")
            self.assertAlmostEqual(origin, 1606417090.0, places=3)
            fallback, fallback_source = detect_time_origin(root, 5.0, float(poses.stamps[0]))
            self.assertEqual(fallback_source, "lidar_minus_trace")
            self.assertAlmostEqual(fallback, 1606417095.0, places=3)

    def test_interpolate_midpoint(self) -> None:
        rotation = rotation_from_rpy(np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, math.pi / 2]))
        source = Trajectory3(
            np.array([0.0, 1.0]),
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            rotation,
        )
        target_rot = rotation_from_rpy(np.zeros(1), np.zeros(1), np.zeros(1))
        target = Trajectory3(np.array([0.5]), np.zeros((1, 3)), target_rot)
        estimate, ground_truth, _src, gt_idx = interpolate_trajectory3(source, target, max_dt_s=0.6)
        self.assertEqual(len(estimate), 1)
        self.assertEqual(int(gt_idx[0]), 0)
        np.testing.assert_allclose(estimate.xyz[0], np.array([1.0, 0.0, 0.0]), atol=1e-9)
        self.assertAlmostEqual(float(rpy_from_rotation(estimate.rotation)[0, 2]), math.pi / 4.0, places=6)
        self.assertEqual(len(ground_truth), 1)


def _helix() -> Trajectory3:
    parameter = np.linspace(0.0, 6.0 * math.pi, 600)
    xyz = np.column_stack((30.0 * np.cos(parameter), 18.0 * np.sin(parameter), 2.0 * parameter))
    yaw = np.arctan2(18.0 * np.cos(parameter), -30.0 * np.sin(parameter))
    roll = 0.1 * np.sin(0.5 * parameter)
    pitch = 0.05 * np.cos(0.3 * parameter)
    return Trajectory3(np.arange(len(parameter), dtype=np.float64) * 0.1, xyz, rotation_from_rpy(roll, pitch, yaw))


if __name__ == "__main__":
    unittest.main()
