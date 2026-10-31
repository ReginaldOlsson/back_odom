#!/usr/bin/env python3
# Copyright 2026 huzaifa.osal
#
# Yaw of base_link -> lidar from a lidar-frame scan-matcher trajectory.

"""Estimate the lidar mount yaw from motion heading versus reported yaw.

The input poses must be the lidar frame: the lidar origin and the lidar
orientation, as KISS-ICP reports them. For each step the motion yaw is
atan2 of the position change, and the reported yaw is the scan matcher's
orientation. The distance-weighted gap over the whole moving path is the
yaw of the static transform base_link -> lidar.

A trajectory that is already expressed in base_link (for example
/back_odom/debug/lidar after the extrinsic was applied) estimates a residual
near zero, not the mount angle.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from trajectory_metrics import (  # noqa: E402
    LidarYawCalibration,
    Trajectory,
    estimate_base_to_lidar_yaw,
    load_odom_csv,
    wrap_angle,
)


def _load_bag_topic(bag_path: Path, topic: str) -> Trajectory:
    from eval_trajectory import _load_estimate_bag

    loaded = _load_estimate_bag(bag_path, [topic])
    trajectory = loaded[topic]
    if len(trajectory) == 0:
        raise SystemExit(f"no messages on {topic} in {bag_path}")
    return trajectory


def _format_report(result: LidarYawCalibration, nominal_deg: float | None) -> str:
    rows = [
        ("yaw_rad", _fmt(result.yaw_rad)),
        ("yaw_deg", _fmt(math.degrees(result.yaw_rad))),
        ("circular_std_deg", _fmt(math.degrees(result.circular_std_rad))),
        ("samples_used", str(result.num_used)),
        ("samples_short", str(result.num_short)),
        ("samples_reverse", str(result.num_reverse)),
        ("straight_yaw_deg", _fmt(math.degrees(result.straight_yaw_rad))),
    ]
    if nominal_deg is not None and math.isfinite(result.yaw_rad):
        delta = float(wrap_angle(result.yaw_rad - math.radians(nominal_deg)))
        rows.append(("nominal_deg", f"{nominal_deg:.4f}"))
        rows.append(("delta_from_nominal_deg", _fmt(math.degrees(delta))))
    width = max(len(name) for name, _ in rows)
    return "\n".join(f"{name:<{width}}:  {value}" for name, value in rows)


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.6f}"


def _write_residuals(path: Path, result: LidarYawCalibration) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stamp", "step_m", "yaw_motion", "yaw_reported", "residual_rad"])
        for stamp, step, motion, reported, residual in zip(
            result.stamp,
            result.step_m,
            result.yaw_motion,
            result.yaw_reported,
            result.residual_rad,
            strict=True,
        ):
            writer.writerow([stamp, step, motion, reported, residual])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Odometry CSV (stamp,x,y,z,qx,qy,qz,qw) of the lidar-frame trajectory",
    )
    parser.add_argument(
        "--bag",
        type=Path,
        default=None,
        help="rosbag2 directory or .mcap/.db3 with the lidar-frame odometry topic",
    )
    parser.add_argument("--topic", default=None, help="Odometry topic to read from --bag")
    parser.add_argument(
        "--nominal-deg",
        type=float,
        default=None,
        help="Current base_link to lidar yaw [deg]. Prints the delta from the estimate.",
    )
    parser.add_argument(
        "--residuals-csv",
        type=Path,
        default=None,
        help="Optional per-step CSV: stamp, step length, motion yaw, reported yaw, residual",
    )
    parser.add_argument(
        "--min-step-m",
        type=float,
        default=0.2,
        help="Ignore steps shorter than this. Stopped samples make atan2 noise.",
    )
    args = parser.parse_args(argv)

    if (args.csv is None) == (args.bag is None):
        raise SystemExit("provide exactly one of --csv or --bag")
    if args.bag is not None and not args.topic:
        raise SystemExit("--bag requires --topic")

    trajectory = load_odom_csv(args.csv) if args.csv is not None else _load_bag_topic(args.bag, args.topic)
    result = estimate_base_to_lidar_yaw(trajectory, min_step_m=args.min_step_m)
    print(_format_report(result, args.nominal_deg))
    if args.residuals_csv is not None:
        _write_residuals(args.residuals_csv, result)
        print(f"wrote {args.residuals_csv}")
    return 0 if result.num_used > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
