#!/usr/bin/env python3
# Copyright 2026 huzaifa.osal
#
# KITTI SE(2)/SE(3) evaluation of an ri_ekf trace against Boreas lidar ground truth.

"""Align RI-EKF and back_odom to Boreas ground truth and export KITTI error sections."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from trajectory_metrics import (  # noqa: E402
    DEFAULT_LIDAR_TO_BASE_YAW,
    AteSplit,
    KittiDevkit,
    KittiWindow,
    RiEkfTrace,
    RpeSplit,
    Trajectory3,
    apply_planar,
    apply_se3,
    associate_trajectory3,
    ate_split,
    body_frame_error,
    classify_error_source,
    contiguous_error_spans,
    cumulative_distance,
    detect_time_origin,
    first_pose_alignment_se3,
    interpolate_trajectory3,
    kitti_devkit,
    load_boreas_lidar_poses_se3,
    load_ri_ekf_trace,
    rpe_split,
    rpy_from_rotation,
    section_threshold,
    shift_time,
    umeyama_se2,
    umeyama_se3,
    wrap_angle,
    as_se2,
)

POSE_FIELDS = [
    "estimator",
    "stamp_gps",
    "stamp_bag",
    "distance_m",
    "gt_x",
    "gt_y",
    "gt_z",
    "gt_roll",
    "gt_pitch",
    "gt_yaw",
    "est_x",
    "est_y",
    "est_z",
    "est_roll",
    "est_pitch",
    "est_yaw",
    "se2_x",
    "se2_y",
    "se2_yaw",
    "se3_x",
    "se3_y",
    "se3_z",
    "se3_yaw",
    "err_horiz_se2_m",
    "err_yaw_se2_deg",
    "err_horiz_se3_m",
    "err_vert_raw_m",
    "err_xyz_se3_m",
    "err_long_m",
    "err_lat_m",
    "err_vert_m",
    "err_roll_deg",
    "err_pitch_deg",
    "err_yaw_deg",
    "err_x",
    "err_y",
    "err_z",
    "err_yaw_filter",
    "bg_x",
    "bg_y",
    "bg_z",
    "ba_x",
    "ba_y",
    "ba_z",
    "imu_ax",
    "imu_ay",
    "imu_az",
    "imu_wx",
    "imu_wy",
    "imu_wz",
    "body_vx",
    "body_vy",
    "body_vz",
    "back_odom_age",
    "pose_applied",
]

SECTION_FIELDS = [
    "kind",
    "estimator",
    "space",
    "tag",
    "stamp_bag_start",
    "stamp_bag_end",
    "stamp_gps_start",
    "stamp_gps_end",
    "distance_start_m",
    "distance_end_m",
    "horiz_rmse_m",
    "horiz_max_m",
    "vert_rmse_m",
    "yaw_rmse_deg",
    "back_odom_horiz_mean_m",
    "ekf_horiz_mean_m",
    "residual_xy_mean_m",
    "filter_yaw_rmse_deg",
    "gyro_bias_delta",
    "accel_bias_delta",
    "back_odom_age_p95_s",
    "speed_mean_mps",
    "kitti_length_m",
    "kitti_translation_percent",
    "kitti_rotation_deg_per_100m",
    "threshold_m",
]

SEGMENT_FIELDS = [
    "estimator",
    "space",
    "length_m",
    "stamp_bag_start",
    "stamp_bag_end",
    "stamp_gps_start",
    "stamp_gps_end",
    "distance_start_m",
    "distance_end_m",
    "translation_m",
    "translation_percent",
    "rotation_deg",
    "rotation_deg_per_100m",
]


@dataclass
class Compared:
    name: str
    estimate: Trajectory3
    ground_truth: Trajectory3
    gt_index: np.ndarray
    source_index: np.ndarray
    aligned_se2: Trajectory3
    aligned_se3: Trajectory3
    ate_raw: AteSplit
    ate_first: AteSplit
    ate_se2: AteSplit
    ate_se3: AteSplit
    rpe: RpeSplit
    kitti_se2: KittiDevkit
    kitti_se3: KittiDevkit
    umeyama_se2: np.ndarray
    umeyama_se3: np.ndarray
    horiz_se2: np.ndarray
    yaw_se2_deg: np.ndarray
    vert_raw: np.ndarray
    body_se3: np.ndarray
    distance_m: np.ndarray


def _finite(value: float) -> float | None:
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        return float(value)
    return None


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return _finite(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    return value


def _csv_num(value: float) -> float | str:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return ""
    return float(value)


def _fmt(value: float, digits: int = 3) -> str:
    if not isinstance(value, (float, int, np.floating, np.integer)) or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def _ate_dict(ate: AteSplit) -> dict:
    return {
        "rmse_horiz_m": ate.rmse_horiz_m,
        "mean_horiz_m": ate.mean_horiz_m,
        "median_horiz_m": ate.median_horiz_m,
        "max_horiz_m": ate.max_horiz_m,
        "rmse_vert_m": ate.rmse_vert_m,
        "mean_vert_m": ate.mean_vert_m,
        "rmse_xyz_m": ate.rmse_xyz_m,
        "rmse_yaw_deg": ate.rmse_yaw_deg,
        "mean_yaw_deg": ate.mean_yaw_deg,
        "rmse_roll_deg": ate.rmse_roll_deg,
        "rmse_pitch_deg": ate.rmse_pitch_deg,
        "num_poses": ate.num_poses,
    }


def _kitti_dict(devkit: KittiDevkit) -> dict:
    return {
        "space": devkit.space,
        "pooled_translation_percent": devkit.pooled_translation_percent,
        "pooled_rotation_deg_per_100m": devkit.pooled_rotation_deg_per_100m,
        "num_segments": devkit.num_segments,
        "lengths": [
            {
                "length_m": seg.length_m,
                "translation_percent": seg.translation_percent,
                "rotation_deg_per_100m": seg.rotation_deg_per_100m,
                "num_segments": seg.num_segments,
            }
            for seg in devkit.lengths
        ],
    }


def _rigid_summary(transform: np.ndarray) -> dict:
    if transform.shape == (3, 3):
        yaw = math.degrees(math.atan2(transform[1, 0], transform[0, 0]))
        translation = transform[0:2, 2]
        return {
            "translation_m": [float(translation[0]), float(translation[1])],
            "translation_norm_m": float(np.linalg.norm(translation)),
            "yaw_deg": yaw,
        }
    rpy = rpy_from_rotation(transform[None, 0:3, 0:3])[0]
    translation = transform[0:3, 3]
    trace = float(transform[0, 0] + transform[1, 1] + transform[2, 2])
    angle = math.degrees(math.acos(min(1.0, max(-1.0, 0.5 * (trace - 1.0)))))
    return {
        "translation_m": [float(v) for v in translation],
        "translation_norm_m": float(np.linalg.norm(translation)),
        "rpy_deg": [float(np.degrees(v)) for v in rpy],
        "rotation_deg": angle,
    }


def compare_trajectory(
    name: str,
    estimate: Trajectory3,
    ground_truth: Trajectory3,
    gt_index: np.ndarray,
    source_index: np.ndarray,
    route_distance: np.ndarray,
) -> Compared | None:
    if len(estimate) < 2:
        return None
    planar = umeyama_se2(as_se2(estimate), as_se2(ground_truth))
    spatial = umeyama_se3(estimate, ground_truth)
    first = first_pose_alignment_se3(estimate, ground_truth)
    aligned_se2 = apply_planar(planar, estimate)
    aligned_se3 = apply_se3(spatial, estimate)
    delta_xy = aligned_se2.xyz[:, :2] - ground_truth.xyz[:, :2]
    yaw_err = np.degrees(
        np.asarray(
            wrap_angle(
                rpy_from_rotation(aligned_se2.rotation)[:, 2]
                - rpy_from_rotation(ground_truth.rotation)[:, 2]
            ),
            dtype=np.float64,
        )
    )
    return Compared(
        name=name,
        estimate=estimate,
        ground_truth=ground_truth,
        gt_index=np.asarray(gt_index, dtype=np.int64),
        source_index=np.asarray(source_index, dtype=np.int64),
        aligned_se2=aligned_se2,
        aligned_se3=aligned_se3,
        ate_raw=ate_split(estimate, ground_truth),
        ate_first=ate_split(apply_se3(first, estimate), ground_truth),
        ate_se2=ate_split(aligned_se2, ground_truth),
        ate_se3=ate_split(aligned_se3, ground_truth),
        rpe=rpe_split(estimate, ground_truth, delta_s=1.0),
        kitti_se2=kitti_devkit(estimate, ground_truth, space="se2"),
        kitti_se3=kitti_devkit(estimate, ground_truth, space="se3"),
        umeyama_se2=planar,
        umeyama_se3=spatial,
        horiz_se2=np.hypot(delta_xy[:, 0], delta_xy[:, 1]),
        yaw_se2_deg=yaw_err,
        vert_raw=estimate.xyz[:, 2] - ground_truth.xyz[:, 2],
        body_se3=body_frame_error(aligned_se3.xyz, ground_truth),
        distance_m=route_distance[np.asarray(gt_index, dtype=np.int64)],
    )


def _signal_rows(trace: RiEkfTrace, source_index: np.ndarray) -> dict[str, np.ndarray]:
    idx = np.asarray(source_index, dtype=np.int64)
    signals = trace.signals
    return {
        "err_x": signals.err_xyz[idx, 0],
        "err_y": signals.err_xyz[idx, 1],
        "err_z": signals.err_xyz[idx, 2],
        "err_yaw_filter": signals.err_yaw[idx],
        "bg_x": signals.gyro_bias[idx, 0],
        "bg_y": signals.gyro_bias[idx, 1],
        "bg_z": signals.gyro_bias[idx, 2],
        "ba_x": signals.accel_bias[idx, 0],
        "ba_y": signals.accel_bias[idx, 1],
        "ba_z": signals.accel_bias[idx, 2],
        "imu_ax": signals.imu_accel[idx, 0],
        "imu_ay": signals.imu_accel[idx, 1],
        "imu_az": signals.imu_accel[idx, 2],
        "imu_wx": signals.imu_gyro[idx, 0],
        "imu_wy": signals.imu_gyro[idx, 1],
        "imu_wz": signals.imu_gyro[idx, 2],
        "body_vx": signals.body_velocity[idx, 0],
        "body_vy": signals.body_velocity[idx, 1],
        "body_vz": signals.body_velocity[idx, 2],
        "back_odom_age": signals.back_odom_age[idx],
        "pose_applied": signals.pose_applied[idx],
    }


def write_per_pose(path: Path, compared: list[Compared], trace: RiEkfTrace, origin_s: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POSE_FIELDS)
        writer.writeheader()
        for item in compared:
            gt_rpy = rpy_from_rotation(item.ground_truth.rotation)
            est_rpy = rpy_from_rotation(item.estimate.rotation)
            se2_yaw = rpy_from_rotation(item.aligned_se2.rotation)[:, 2]
            se3_rpy = rpy_from_rotation(item.aligned_se3.rotation)
            raw_rpy_err = np.degrees(
                np.column_stack(
                    (
                        np.asarray(wrap_angle(est_rpy[:, 0] - gt_rpy[:, 0]), dtype=np.float64),
                        np.asarray(wrap_angle(est_rpy[:, 1] - gt_rpy[:, 1]), dtype=np.float64),
                        np.asarray(wrap_angle(est_rpy[:, 2] - gt_rpy[:, 2]), dtype=np.float64),
                    )
                )
            )
            se3_delta = item.aligned_se3.xyz - item.ground_truth.xyz
            signals = _signal_rows(trace, item.source_index) if item.name == "ri_ekf" else None
            for i in range(len(item.estimate)):
                row = {
                    "estimator": item.name,
                    "stamp_gps": item.estimate.stamps[i],
                    "stamp_bag": item.estimate.stamps[i] - origin_s,
                    "distance_m": item.distance_m[i],
                    "gt_x": item.ground_truth.xyz[i, 0],
                    "gt_y": item.ground_truth.xyz[i, 1],
                    "gt_z": item.ground_truth.xyz[i, 2],
                    "gt_roll": gt_rpy[i, 0],
                    "gt_pitch": gt_rpy[i, 1],
                    "gt_yaw": gt_rpy[i, 2],
                    "est_x": item.estimate.xyz[i, 0],
                    "est_y": item.estimate.xyz[i, 1],
                    "est_z": item.estimate.xyz[i, 2],
                    "est_roll": est_rpy[i, 0],
                    "est_pitch": est_rpy[i, 1],
                    "est_yaw": est_rpy[i, 2],
                    "se2_x": item.aligned_se2.xyz[i, 0],
                    "se2_y": item.aligned_se2.xyz[i, 1],
                    "se2_yaw": se2_yaw[i],
                    "se3_x": item.aligned_se3.xyz[i, 0],
                    "se3_y": item.aligned_se3.xyz[i, 1],
                    "se3_z": item.aligned_se3.xyz[i, 2],
                    "se3_yaw": se3_rpy[i, 2],
                    "err_horiz_se2_m": item.horiz_se2[i],
                    "err_yaw_se2_deg": item.yaw_se2_deg[i],
                    "err_horiz_se3_m": float(np.hypot(se3_delta[i, 0], se3_delta[i, 1])),
                    "err_vert_raw_m": item.vert_raw[i],
                    "err_xyz_se3_m": float(np.linalg.norm(se3_delta[i])),
                    "err_long_m": item.body_se3[i, 0],
                    "err_lat_m": item.body_se3[i, 1],
                    "err_vert_m": item.body_se3[i, 2],
                    "err_roll_deg": raw_rpy_err[i, 0],
                    "err_pitch_deg": raw_rpy_err[i, 1],
                    "err_yaw_deg": raw_rpy_err[i, 2],
                }
                if signals is None:
                    for key in POSE_FIELDS:
                        row.setdefault(key, math.nan)
                else:
                    for key, values in signals.items():
                        row[key] = values[i]
                writer.writerow({key: _csv_num(row[key]) if key != "estimator" else row[key] for key in POSE_FIELDS})


def _rmse(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return math.nan
    return float(np.sqrt(np.mean(finite**2)))


def _mean_horiz(item: Compared, t0: float, t1: float) -> float:
    mask = (item.estimate.stamps >= t0) & (item.estimate.stamps <= t1)
    if not np.any(mask):
        return math.nan
    return float(np.mean(item.horiz_se2[mask]))


def _filter_stats(item: Compared, trace: RiEkfTrace, t0: float, t1: float) -> dict[str, float]:
    empty = {
        "residual_xy_mean_m": math.nan,
        "filter_yaw_rmse_deg": math.nan,
        "gyro_bias_delta": math.nan,
        "accel_bias_delta": math.nan,
        "back_odom_age_p95_s": math.nan,
        "speed_mean_mps": math.nan,
    }
    if item.name != "ri_ekf":
        return empty
    mask = (item.estimate.stamps >= t0) & (item.estimate.stamps <= t1)
    idx = item.source_index[mask]
    if len(idx) == 0:
        return empty
    err = trace.signals.err_xyz[idx]
    yaw = trace.signals.err_yaw[idx]
    yaw = yaw[np.isfinite(yaw)]
    gyro = trace.signals.gyro_bias[idx]
    accel = trace.signals.accel_bias[idx]
    age = trace.signals.back_odom_age[idx]
    speed = np.hypot(trace.signals.body_velocity[idx, 0], trace.signals.body_velocity[idx, 1])
    gyro_delta = math.nan
    accel_delta = math.nan
    finite_gyro = np.isfinite(gyro).all(axis=1)
    if int(np.count_nonzero(finite_gyro)) >= 2:
        used = gyro[finite_gyro]
        gyro_delta = float(np.linalg.norm(used[-1] - used[0]))
    finite_accel = np.isfinite(accel).all(axis=1)
    if int(np.count_nonzero(finite_accel)) >= 2:
        used = accel[finite_accel]
        accel_delta = float(np.linalg.norm(used[-1] - used[0]))
    return {
        "residual_xy_mean_m": float(np.nanmean(np.hypot(err[:, 0], err[:, 1]))),
        "filter_yaw_rmse_deg": float(np.degrees(np.sqrt(np.mean(yaw**2)))) if len(yaw) else math.nan,
        "gyro_bias_delta": gyro_delta,
        "accel_bias_delta": accel_delta,
        "back_odom_age_p95_s": float(np.nanpercentile(age, 95)),
        "speed_mean_mps": float(np.nanmean(speed)),
    }


def _section_row(
    kind: str,
    space: str,
    owner: Compared,
    start: int,
    end: int,
    by_name: dict[str, Compared],
    trace: RiEkfTrace,
    origin_s: float,
    threshold: float,
    kitti: KittiWindow | None = None,
) -> dict:
    t0 = float(owner.estimate.stamps[start])
    t1 = float(owner.estimate.stamps[end])
    sl = slice(start, end + 1)
    ekf = by_name.get("ri_ekf")
    back_odom = by_name.get("back_odom")
    ekf_horiz = _mean_horiz(ekf, t0, t1) if ekf is not None else math.nan
    back_horiz = _mean_horiz(back_odom, t0, t1) if back_odom is not None else math.nan
    stats_source = ekf if ekf is not None else owner
    stats = _filter_stats(stats_source, trace, t0, t1)
    row = {
        "kind": kind,
        "estimator": owner.name,
        "space": space,
        "tag": classify_error_source(
            back_horiz, ekf_horiz, stats["residual_xy_mean_m"], threshold
        ),
        "stamp_bag_start": t0 - origin_s,
        "stamp_bag_end": t1 - origin_s,
        "stamp_gps_start": t0,
        "stamp_gps_end": t1,
        "distance_start_m": float(owner.distance_m[start]),
        "distance_end_m": float(owner.distance_m[end]),
        "horiz_rmse_m": _rmse(owner.horiz_se2[sl]),
        "horiz_max_m": float(np.nanmax(owner.horiz_se2[sl])),
        "vert_rmse_m": _rmse(owner.vert_raw[sl]),
        "yaw_rmse_deg": _rmse(owner.yaw_se2_deg[sl]),
        "back_odom_horiz_mean_m": back_horiz,
        "ekf_horiz_mean_m": ekf_horiz,
        "threshold_m": threshold,
        "kitti_length_m": math.nan if kitti is None else kitti.length_m,
        "kitti_translation_percent": math.nan if kitti is None else kitti.translation_percent,
        "kitti_rotation_deg_per_100m": math.nan if kitti is None else kitti.rotation_deg_per_100m,
    }
    row.update(stats)
    return row


def _overlaps(left: KittiWindow, right: KittiWindow) -> bool:
    return not (left.last < right.first or right.last < left.first)


def _nonoverlapping(windows: list[KittiWindow], top_k: int) -> list[KittiWindow]:
    """Worst 100 m windows whose pose ranges do not overlap."""
    ranked = sorted(
        (window for window in windows if window.length_m == 100.0),
        key=lambda window: window.translation_percent,
        reverse=True,
    )
    chosen: list[KittiWindow] = []
    for window in ranked:
        if len(chosen) >= top_k:
            break
        if any(_overlaps(window, kept) for kept in chosen):
            continue
        chosen.append(window)
    return chosen


def build_sections(
    compared: list[Compared],
    trace: RiEkfTrace,
    origin_s: float,
    floor_m: float,
    percentile: float,
    min_length_m: float,
    top_k: int,
) -> tuple[list[dict], dict[str, float]]:
    by_name = {item.name: item for item in compared}
    rows: list[dict] = []
    thresholds: dict[str, float] = {}
    for item in compared:
        threshold = section_threshold(item.horiz_se2, floor_m=floor_m, percentile=percentile)
        thresholds[item.name] = threshold
        for span in contiguous_error_spans(item.horiz_se2, item.distance_m, threshold, min_length_m):
            rows.append(
                _section_row(
                    "ate_run",
                    "se2",
                    item,
                    span.start,
                    span.end,
                    by_name,
                    trace,
                    origin_s,
                    threshold,
                )
            )
        for space, devkit in (("se2", item.kitti_se2), ("se3", item.kitti_se3)):
            for window in _nonoverlapping(devkit.windows, top_k):
                rows.append(
                    _section_row(
                        "kitti_100m",
                        space,
                        item,
                        window.first,
                        window.last,
                        by_name,
                        trace,
                        origin_s,
                        threshold,
                        kitti=window,
                    )
                )
    return rows, thresholds


def write_sections(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SECTION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: row[key] if key in ("kind", "estimator", "space", "tag") else _csv_num(row[key])
                    for key in SECTION_FIELDS
                }
            )


def write_kitti_segments(
    path: Path, compared: list[Compared], origin_s: float
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEGMENT_FIELDS)
        writer.writeheader()
        for item in compared:
            for space, devkit in (("se2", item.kitti_se2), ("se3", item.kitti_se3)):
                for window in devkit.windows:
                    writer.writerow(
                        {
                            "estimator": item.name,
                            "space": space,
                            "length_m": window.length_m,
                            "stamp_bag_start": _csv_num(window.stamp_start - origin_s),
                            "stamp_bag_end": _csv_num(window.stamp_end - origin_s),
                            "stamp_gps_start": _csv_num(window.stamp_start),
                            "stamp_gps_end": _csv_num(window.stamp_end),
                            "distance_start_m": _csv_num(item.distance_m[window.first]),
                            "distance_end_m": _csv_num(item.distance_m[window.last]),
                            "translation_m": _csv_num(window.translation_m),
                            "translation_percent": _csv_num(window.translation_percent),
                            "rotation_deg": _csv_num(window.rotation_deg),
                            "rotation_deg_per_100m": _csv_num(window.rotation_deg_per_100m),
                        }
                    )


def _ate_lines(title: str, ate: AteSplit) -> list[str]:
    return [
        f"{title}",
        (
            f"  horiz rmse {_fmt(ate.rmse_horiz_m)} m  mean {_fmt(ate.mean_horiz_m)}  "
            f"median {_fmt(ate.median_horiz_m)}  max {_fmt(ate.max_horiz_m)}"
        ),
        (
            f"  vert rmse {_fmt(ate.rmse_vert_m)} m  mean {_fmt(ate.mean_vert_m)}  "
            f"xyz rmse {_fmt(ate.rmse_xyz_m)} m"
        ),
        (
            f"  yaw rmse {_fmt(ate.rmse_yaw_deg)} deg  mean {_fmt(ate.mean_yaw_deg)}  "
            f"roll rmse {_fmt(ate.rmse_roll_deg)}  pitch rmse {_fmt(ate.rmse_pitch_deg)}"
        ),
    ]


def _kitti_lines(title: str, devkit: KittiDevkit) -> list[str]:
    lines = [
        (
            f"{title}  pooled t {_fmt(devkit.pooled_translation_percent)} %  "
            f"r {_fmt(devkit.pooled_rotation_deg_per_100m)} deg/100m  "
            f"n {devkit.num_segments}"
        )
    ]
    for seg in devkit.lengths:
        lines.append(
            f"  {seg.length_m:6.0f} m  t {_fmt(seg.translation_percent, 4):>10} %  "
            f"r {_fmt(seg.rotation_deg_per_100m, 4):>10} deg/100m  n {seg.num_segments}"
        )
    return lines


def format_trace_report(
    compared: list[Compared],
    sections: list[dict],
    thresholds: dict[str, float],
    origin_s: float,
    origin_source: str,
    lidar_to_base_yaw: float,
    route_length_m: float,
) -> str:
    lines = [
        "Boreas KITTI comparison",
        f"time origin {origin_s:.6f} s ({origin_source})",
        f"lidar-to-base yaw {lidar_to_base_yaw:.6f} rad ({math.degrees(lidar_to_base_yaw):.2f} deg)",
        f"GT path length {_fmt(route_length_m, 1)} m",
        "",
        "Tags: back_odom_led = odometry is far from GT and the EKF follows it;",
        "      ekf_diverged = back_odom stays close while the EKF does not;",
        "      both = each is far from GT and they disagree with each other.",
        "ATE runs are contiguous stretches of SE(2) Umeyama horizontal error.",
        "KITTI rows are the worst non-overlapping 100 m segments.",
        "Relative KITTI error does not use the Umeyama alignment.",
    ]
    for item in compared:
        duration = float(item.estimate.stamps[-1] - item.estimate.stamps[0])
        lines.extend(
            [
                "",
                f"=== {item.name} ===",
                f"poses {len(item.estimate)}  duration {_fmt(duration, 1)} s  "
                f"section threshold {_fmt(thresholds.get(item.name, math.nan))} m",
                (
                    f"mount yaw residual mean {_fmt(item.ate_raw.mean_yaw_deg)} deg  "
                    f"rmse {_fmt(item.ate_raw.rmse_yaw_deg)} deg"
                ),
                (
                    f"altitude raw vert rmse {_fmt(item.ate_raw.rmse_vert_m)} m  "
                    f"mean {_fmt(item.ate_raw.mean_vert_m)}  "
                    f"est z [{_fmt(float(np.min(item.estimate.xyz[:, 2])), 2)}, "
                    f"{_fmt(float(np.max(item.estimate.xyz[:, 2])), 2)}]  "
                    f"gt z [{_fmt(float(np.min(item.ground_truth.xyz[:, 2])), 2)}, "
                    f"{_fmt(float(np.max(item.ground_truth.xyz[:, 2])), 2)}]"
                ),
            ]
        )
        se2 = _rigid_summary(item.umeyama_se2)
        se3 = _rigid_summary(item.umeyama_se3)
        lines.append(
            f"Umeyama SE(2) |t| {_fmt(se2['translation_norm_m'])} m  yaw {_fmt(se2['yaw_deg'])} deg"
        )
        lines.append(
            f"Umeyama SE(3) |t| {_fmt(se3['translation_norm_m'])} m  "
            f"angle {_fmt(se3['rotation_deg'])} deg"
        )
        lines.extend(_ate_lines("ATE raw (same ENU frame, mount applied)", item.ate_raw))
        lines.extend(_ate_lines("ATE first pose SE(3)", item.ate_first))
        lines.extend(_ate_lines("ATE Umeyama SE(2) (z is not aligned)", item.ate_se2))
        lines.extend(_ate_lines("ATE Umeyama SE(3)", item.ate_se3))
        rpe = item.rpe
        lines.append(
            f"RPE {rpe.delta_s:.0f}s  se2 trans rmse {_fmt(rpe.se2_translation_rmse_m)} m "
            f"({_fmt(rpe.se2_translation_percent)} %)  yaw {_fmt(rpe.se2_yaw_rmse_deg)} deg  "
            f"se3 trans rmse {_fmt(rpe.se3_translation_rmse_m)} m  "
            f"rot {_fmt(rpe.se3_rotation_rmse_deg)} deg  pairs {rpe.num_pairs}"
        )
        lines.extend(_kitti_lines("KITTI SE(2)", item.kitti_se2))
        lines.extend(_kitti_lines("KITTI SE(3)", item.kitti_se3))

    ate_rows = [row for row in sections if row["kind"] == "ate_run"]
    kitti_rows = [row for row in sections if row["kind"] == "kitti_100m"]
    lines.extend(["", "Worst ATE sections"])
    if not ate_rows:
        lines.append("  none above the threshold")
    for row in sorted(ate_rows, key=lambda item: item["horiz_rmse_m"], reverse=True)[:5]:
        lines.append(
            f"  {row['estimator']:10}  bag {_fmt(row['stamp_bag_start'], 1)}-"
            f"{_fmt(row['stamp_bag_end'], 1)} s  "
            f"dist {_fmt(row['distance_start_m'], 0)}-{_fmt(row['distance_end_m'], 0)} m  "
            f"horiz rmse {_fmt(row['horiz_rmse_m'])} m  vert {_fmt(row['vert_rmse_m'])} m  "
            f"tag {row['tag']}"
        )
    lines.extend(["", "Worst KITTI 100 m sections"])
    if not kitti_rows:
        lines.append("  none")
    ranked = sorted(kitti_rows, key=lambda item: item["kitti_translation_percent"], reverse=True)[:8]
    for row in ranked:
        lines.append(
            f"  {row['estimator']:10} {row['space']:3}  bag {_fmt(row['stamp_bag_start'], 1)}-"
            f"{_fmt(row['stamp_bag_end'], 1)} s  "
            f"dist {_fmt(row['distance_start_m'], 0)}-{_fmt(row['distance_end_m'], 0)} m  "
            f"t {_fmt(row['kitti_translation_percent'])} %  "
            f"r {_fmt(row['kitti_rotation_deg_per_100m'])} deg/100m  tag {row['tag']}"
        )
    return "\n".join(lines) + "\n"


def _metrics_payload(
    compared: list[Compared],
    thresholds: dict[str, float],
    origin_s: float,
    origin_source: str,
    lidar_to_base_yaw: float,
    route_length_m: float,
    trace_path: Path,
    sequence_dir: Path,
) -> dict:
    estimators = []
    for item in compared:
        estimators.append(
            {
                "name": item.name,
                "num_poses": len(item.estimate),
                "duration_s": float(item.estimate.stamps[-1] - item.estimate.stamps[0]),
                "section_threshold_m": thresholds.get(item.name),
                "umeyama_se2": _rigid_summary(item.umeyama_se2),
                "umeyama_se3": _rigid_summary(item.umeyama_se3),
                "ate_raw": _ate_dict(item.ate_raw),
                "ate_first_pose": _ate_dict(item.ate_first),
                "ate_umeyama_se2": _ate_dict(item.ate_se2),
                "ate_umeyama_se3": _ate_dict(item.ate_se3),
                "rpe_1s": {
                    "delta_s": item.rpe.delta_s,
                    "se2_translation_rmse_m": item.rpe.se2_translation_rmse_m,
                    "se2_translation_percent": item.rpe.se2_translation_percent,
                    "se2_yaw_rmse_deg": item.rpe.se2_yaw_rmse_deg,
                    "se3_translation_rmse_m": item.rpe.se3_translation_rmse_m,
                    "se3_rotation_rmse_deg": item.rpe.se3_rotation_rmse_deg,
                    "num_pairs": item.rpe.num_pairs,
                },
                "kitti_se2": _kitti_dict(item.kitti_se2),
                "kitti_se3": _kitti_dict(item.kitti_se3),
            }
        )
    return {
        "trace": str(trace_path),
        "sequence": str(sequence_dir),
        "time_origin_s": origin_s,
        "time_origin_source": origin_source,
        "lidar_to_base_yaw_rad": lidar_to_base_yaw,
        "route_length_m": route_length_m,
        "estimators": estimators,
    }


def _write_plots(out_dir: Path, ground_truth: Trajectory3, compared: list[Compared], sections: list[dict]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[str] = []
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(ground_truth.xyz[:, 0], ground_truth.xyz[:, 1], color="black", linewidth=1.2, label="GT")
    for item in compared:
        ax.plot(
            item.aligned_se2.xyz[:, 0],
            item.aligned_se2.xyz[:, 1],
            linewidth=1.0,
            label=f"{item.name} SE(2)",
        )
    ate_sections = [row for row in sections if row["kind"] == "ate_run"]
    for row in ate_sections:
        owner = next(item for item in compared if item.name == row["estimator"])
        mask = (owner.distance_m >= row["distance_start_m"]) & (owner.distance_m <= row["distance_end_m"])
        ax.plot(
            owner.ground_truth.xyz[mask, 0],
            owner.ground_truth.xyz[mask, 1],
            color="crimson",
            linewidth=2.0,
            solid_capstyle="round",
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("easting [m]")
    ax.set_ylabel("northing [m]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Umeyama SE(2) alignment")
    fig.tight_layout()
    path = out_dir / "trajectories.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for item in compared:
        axes[0].plot(item.distance_m, item.horiz_se2, linewidth=0.8, label=item.name)
        axes[1].plot(item.distance_m, item.vert_raw, linewidth=0.8, label=item.name)
    for row in ate_sections:
        for axis in axes:
            axis.axvspan(row["distance_start_m"], row["distance_end_m"], color="crimson", alpha=0.12)
    axes[0].set_ylabel("SE(2) horizontal error [m]")
    axes[1].set_ylabel("raw vertical error [m]")
    axes[1].set_xlabel("distance along GT [m]")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    path = out_dir / "error_vs_distance.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    lengths = [seg.length_m for seg in compared[0].kitti_se2.lengths] if compared else []
    series = []
    for item in compared:
        series.append((f"{item.name} SE2", [seg.translation_percent for seg in item.kitti_se2.lengths],
                       [seg.rotation_deg_per_100m for seg in item.kitti_se2.lengths]))
        series.append((f"{item.name} SE3", [seg.translation_percent for seg in item.kitti_se3.lengths],
                       [seg.rotation_deg_per_100m for seg in item.kitti_se3.lengths]))
    x = np.arange(len(lengths), dtype=np.float64)
    width = 0.18
    for index, (label, trans, rot) in enumerate(series):
        offset = (index - 0.5 * (len(series) - 1)) * width
        axes[0].bar(x + offset, [0.0 if not math.isfinite(v) else v for v in trans], width=width, label=label)
        axes[1].bar(x + offset, [0.0 if not math.isfinite(v) else v for v in rot], width=width, label=label)
    for axis, ylabel in ((axes[0], "translation [%]"), (axes[1], "rotation [deg/100 m]")):
        axis.set_xticks(x, [str(int(length)) for length in lengths])
        axis.set_xlabel("subsequence length [m]")
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "kitti_bars.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True, help="ri_ekf_localizer trace CSV")
    parser.add_argument("--sequence", type=Path, required=True, help="Boreas sequence directory")
    parser.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="lidar_poses.csv (default: SEQUENCE/applanix/lidar_poses.csv)",
    )
    parser.add_argument(
        "--time-origin",
        type=float,
        default=None,
        help="GPS seconds added to trace stamps. Default: detect from the sequence.",
    )
    parser.add_argument(
        "--lidar-to-base-yaw",
        type=float,
        default=DEFAULT_LIDAR_TO_BASE_YAW,
        help="base_link to lidar yaw [rad]. Default is -pi/4.",
    )
    parser.add_argument("--max-dt", type=float, default=0.05, help="Association gate [s]")
    parser.add_argument("--out", type=Path, default=None, help="Output directory")
    parser.add_argument("--section-floor-m", type=float, default=5.0)
    parser.add_argument("--section-percentile", type=float, default=90.0)
    parser.add_argument("--section-min-length-m", type=float, default=20.0)
    parser.add_argument("--top-k", type=int, default=15, help="Worst non-overlapping 100 m KITTI segments")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    gt_path = args.gt if args.gt is not None else args.sequence / "applanix" / "lidar_poses.csv"
    out_dir = args.out if args.out is not None else Path("/tmp") / f"boreas_eval_{args.sequence.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = load_boreas_lidar_poses_se3(gt_path, args.lidar_to_base_yaw)
    trace = load_ri_ekf_trace(args.trace)
    if args.time_origin is None:
        if len(trace.ekf) == 0 or len(ground_truth) == 0:
            raise SystemExit("trace or ground truth is empty")
        origin_s, origin_source = detect_time_origin(
            args.sequence, float(trace.ekf.stamps[0]), float(ground_truth.stamps[0])
        )
    else:
        origin_s, origin_source = float(args.time_origin), "cli"

    ekf = shift_time(trace.ekf, origin_s)
    back_odom = shift_time(trace.back_odom, origin_s)
    route_distance = cumulative_distance(ground_truth.xyz, planar=False)

    ekf_est, ekf_gt, ekf_src, ekf_gt_index = interpolate_trajectory3(ekf, ground_truth, args.max_dt)
    bo_est, bo_gt, bo_src, bo_gt_index = associate_trajectory3(back_odom, ground_truth, args.max_dt)

    compared = [
        item
        for item in (
            compare_trajectory("ri_ekf", ekf_est, ekf_gt, ekf_gt_index, ekf_src, route_distance),
            compare_trajectory("back_odom", bo_est, bo_gt, bo_gt_index, bo_src, route_distance),
        )
        if item is not None
    ]
    if not compared:
        raise SystemExit("no poses associated; check --time-origin and --max-dt")

    sections, thresholds = build_sections(
        compared,
        trace,
        origin_s,
        args.section_floor_m,
        args.section_percentile,
        args.section_min_length_m,
        args.top_k,
    )
    report = format_trace_report(
        compared,
        sections,
        thresholds,
        origin_s,
        origin_source,
        args.lidar_to_base_yaw,
        float(route_distance[-1]) if len(route_distance) else math.nan,
    )
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    payload = _metrics_payload(
        compared,
        thresholds,
        origin_s,
        origin_source,
        args.lidar_to_base_yaw,
        float(route_distance[-1]) if len(route_distance) else math.nan,
        args.trace,
        args.sequence,
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8"
    )
    write_per_pose(out_dir / "per_pose.csv", compared, trace, origin_s)
    write_kitti_segments(out_dir / "kitti_segments.csv", compared, origin_s)
    write_sections(out_dir / "error_sections.csv", sections)
    print(report, end="")
    print(f"wrote {out_dir / 'metrics.json'}")
    print(f"wrote {out_dir / 'per_pose.csv'}")
    print(f"wrote {out_dir / 'kitti_segments.csv'}")
    print(f"wrote {out_dir / 'error_sections.csv'}")
    if not args.no_plot:
        try:
            for plot_path in _write_plots(out_dir, ground_truth, compared, sections):
                print(f"wrote {plot_path}")
        except Exception as exc:  # matplotlib is optional at runtime
            print(f"plot skipped: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
