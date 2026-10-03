#!/usr/bin/env python3
# Copyright 2026 huzaifa.osal

"""Synthetic checks for trajectory_metrics (no ROS bags)."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from trajectory_metrics import (  # noqa: E402
    Trajectory,
    ate_metrics,
    evaluate_pair,
    kitti_metrics,
    umeyama_se2,
    apply_se2,
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


if __name__ == "__main__":
    unittest.main()
