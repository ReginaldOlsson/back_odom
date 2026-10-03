#!/usr/bin/env python3
# Copyright 2026 huzaifa.osal
#
# Trajectory metrics for Boreas-style odometry evaluation (numpy only).

"""ATE, RPE, and KITTI-style relative pose errors on SE(2) trajectories."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass
class Trajectory:
    """Poses as time [s], XY [m], yaw [rad]."""

    stamps: np.ndarray  # (N,)
    xy: np.ndarray  # (N, 2)
    yaw: np.ndarray  # (N,)

    def __post_init__(self) -> None:
        self.stamps = np.asarray(self.stamps, dtype=np.float64).reshape(-1)
        self.xy = np.asarray(self.xy, dtype=np.float64).reshape(-1, 2)
        self.yaw = np.asarray(self.yaw, dtype=np.float64).reshape(-1)
        if not (len(self.stamps) == len(self.xy) == len(self.yaw)):
            raise ValueError("Trajectory arrays must share the same length")

    def __len__(self) -> int:
        return int(self.stamps.shape[0])

    def path_length(self) -> float:
        if len(self) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.xy, axis=0), axis=1).sum())


@dataclass
class AteMetrics:
    rmse_xy_m: float
    mean_xy_m: float
    median_xy_m: float
    max_xy_m: float
    rmse_yaw_deg: float
    num_poses: int


@dataclass
class RpeMetrics:
    delta_s: float
    translation_rmse_m: float
    translation_percent: float
    yaw_rmse_deg: float
    yaw_deg_per_m: float
    num_pairs: int


@dataclass
class KittiSegment:
    length_m: float
    translation_percent: float
    rotation_deg_per_100m: float
    num_segments: int


@dataclass
class EvalReport:
    name: str
    association_hz: float
    ate_umeyama: AteMetrics
    ate_first_pose: AteMetrics
    rpe_1s: RpeMetrics
    kitti: list[KittiSegment]


def wrap_angle(yaw: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(yaw) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def load_boreas_lidar_poses(path: Path | str) -> Trajectory:
    """Load Boreas applanix/lidar_poses.csv (GPSTime in microseconds)."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    stamps = np.asarray(data["GPSTime"], dtype=np.float64)
    # Some files store integer microseconds, others float seconds.
    if np.nanmedian(stamps) > 1.0e12:
        stamps = stamps * 1.0e-6
    xy = np.column_stack(
        (np.asarray(data["easting"], dtype=np.float64), np.asarray(data["northing"], dtype=np.float64))
    )
    yaw = np.asarray(data["heading"], dtype=np.float64)
    order = np.argsort(stamps)
    return Trajectory(stamps[order], xy[order], yaw[order])


def load_odom_csv(path: Path | str) -> Trajectory:
    """Load stamp,x,y,z,qx,qy,qz,qw CSV (header required)."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    stamps = np.asarray(data["stamp"], dtype=np.float64)
    xy = np.column_stack((np.asarray(data["x"], dtype=np.float64), np.asarray(data["y"], dtype=np.float64)))
    yaws = np.array(
        [
            yaw_from_quat(float(qx), float(qy), float(qz), float(qw))
            for qx, qy, qz, qw in zip(data["qx"], data["qy"], data["qz"], data["qw"], strict=True)
        ],
        dtype=np.float64,
    )
    order = np.argsort(stamps)
    return Trajectory(stamps[order], xy[order], yaws[order])


def associate(
    estimate: Trajectory, ground_truth: Trajectory, max_dt_s: float = 0.05
) -> tuple[Trajectory, Trajectory]:
    """Nearest-neighbor association of each estimate pose to GT within max_dt_s."""
    if len(estimate) == 0 or len(ground_truth) == 0:
        return Trajectory([], np.zeros((0, 2)), []), Trajectory([], np.zeros((0, 2)), [])

    gt_stamps = ground_truth.stamps
    est_idx: list[int] = []
    gt_idx: list[int] = []
    for i, t in enumerate(estimate.stamps):
        j = int(np.searchsorted(gt_stamps, t))
        candidates = []
        if j < len(gt_stamps):
            candidates.append(j)
        if j > 0:
            candidates.append(j - 1)
        best = min(candidates, key=lambda k: abs(gt_stamps[k] - t))
        if abs(gt_stamps[best] - t) <= max_dt_s:
            est_idx.append(i)
            gt_idx.append(best)

    if not est_idx:
        return Trajectory([], np.zeros((0, 2)), []), Trajectory([], np.zeros((0, 2)), [])

    e = np.asarray(est_idx, dtype=np.int64)
    g = np.asarray(gt_idx, dtype=np.int64)
    return (
        Trajectory(estimate.stamps[e], estimate.xy[e], estimate.yaw[e]),
        Trajectory(ground_truth.stamps[g], ground_truth.xy[g], ground_truth.yaw[g]),
    )


def se2_from_xy_yaw(xy: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Return (N, 3, 3) SE(2) matrices."""
    c = np.cos(yaw)
    s = np.sin(yaw)
    t = np.zeros((len(yaw), 3, 3), dtype=np.float64)
    t[:, 0, 0] = c
    t[:, 0, 1] = -s
    t[:, 1, 0] = s
    t[:, 1, 1] = c
    t[:, 0, 2] = xy[:, 0]
    t[:, 1, 2] = xy[:, 1]
    t[:, 2, 2] = 1.0
    return t


def apply_se2(transform: np.ndarray, traj: Trajectory) -> Trajectory:
    """Left-multiply every pose by a 3x3 SE(2) transform."""
    poses = se2_from_xy_yaw(traj.xy, traj.yaw)
    aligned = transform @ poses
    xy = aligned[:, 0:2, 2]
    yaw = wrap_angle(np.arctan2(aligned[:, 1, 0], aligned[:, 0, 0]))
    return Trajectory(traj.stamps.copy(), xy, yaw)


def first_pose_alignment(estimate: Trajectory, ground_truth: Trajectory) -> np.ndarray:
    """SE(2) that maps the first estimate pose onto the first GT pose."""
    if len(estimate) == 0:
        return np.eye(3)
    e0 = se2_from_xy_yaw(estimate.xy[:1], estimate.yaw[:1])[0]
    g0 = se2_from_xy_yaw(ground_truth.xy[:1], ground_truth.yaw[:1])[0]
    return g0 @ np.linalg.inv(e0)


def umeyama_se2(estimate: Trajectory, ground_truth: Trajectory) -> np.ndarray:
    """Rigid SE(2) Umeyama (no scale) aligning estimate XY to GT XY, then yaw from R."""
    if len(estimate) < 2:
        return first_pose_alignment(estimate, ground_truth)

    src = estimate.xy.T  # 2xN
    dst = ground_truth.xy.T
    mu_src = src.mean(axis=1, keepdims=True)
    mu_dst = dst.mean(axis=1, keepdims=True)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = dst_c @ src_c.T / float(src.shape[1])
    u, _, vt = np.linalg.svd(cov)
    s = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        s[1, 1] = -1.0
    r = u @ s @ vt
    t = mu_dst.reshape(2) - r @ mu_src.reshape(2)
    transform = np.eye(3)
    transform[0:2, 0:2] = r
    transform[0:2, 2] = t
    return transform


def ate_metrics(estimate: Trajectory, ground_truth: Trajectory) -> AteMetrics:
    if len(estimate) == 0:
        return AteMetrics(math.nan, math.nan, math.nan, math.nan, math.nan, 0)
    err_xy = np.linalg.norm(estimate.xy - ground_truth.xy, axis=1)
    err_yaw = wrap_angle(estimate.yaw - ground_truth.yaw)
    return AteMetrics(
        rmse_xy_m=float(np.sqrt(np.mean(err_xy**2))),
        mean_xy_m=float(np.mean(err_xy)),
        median_xy_m=float(np.median(err_xy)),
        max_xy_m=float(np.max(err_xy)),
        rmse_yaw_deg=float(np.degrees(np.sqrt(np.mean(err_yaw**2)))),
        num_poses=len(estimate),
    )


def rpe_metrics(estimate: Trajectory, ground_truth: Trajectory, delta_s: float = 1.0) -> RpeMetrics:
    if len(estimate) < 2:
        return RpeMetrics(delta_s, math.nan, math.nan, math.nan, math.nan, 0)

    e_poses = se2_from_xy_yaw(estimate.xy, estimate.yaw)
    g_poses = se2_from_xy_yaw(ground_truth.xy, ground_truth.yaw)
    trans_errs: list[float] = []
    yaw_errs: list[float] = []
    path_lens: list[float] = []

    j = 0
    for i in range(len(estimate)):
        t_target = estimate.stamps[i] + delta_s
        while j + 1 < len(estimate) and estimate.stamps[j] < t_target:
            j += 1
        if j <= i or abs(estimate.stamps[j] - t_target) > 0.5 * delta_s:
            continue
        e_rel = np.linalg.inv(e_poses[i]) @ e_poses[j]
        g_rel = np.linalg.inv(g_poses[i]) @ g_poses[j]
        err = np.linalg.inv(g_rel) @ e_rel
        trans = float(np.linalg.norm(err[0:2, 2]))
        yaw = abs(float(wrap_angle(math.atan2(err[1, 0], err[0, 0]))))
        seg = float(np.linalg.norm(ground_truth.xy[j] - ground_truth.xy[i]))
        if seg < 1.0e-3:
            continue
        trans_errs.append(trans)
        yaw_errs.append(yaw)
        path_lens.append(seg)

    if not trans_errs:
        return RpeMetrics(delta_s, math.nan, math.nan, math.nan, math.nan, 0)

    trans_arr = np.asarray(trans_errs)
    yaw_arr = np.asarray(yaw_errs)
    lens = np.asarray(path_lens)
    trans_rmse = float(np.sqrt(np.mean(trans_arr**2)))
    yaw_rmse = float(np.degrees(np.sqrt(np.mean(yaw_arr**2))))
    return RpeMetrics(
        delta_s=delta_s,
        translation_rmse_m=trans_rmse,
        translation_percent=float(100.0 * np.mean(trans_arr / lens)),
        yaw_rmse_deg=yaw_rmse,
        yaw_deg_per_m=float(np.mean(np.degrees(yaw_arr) / lens)),
        num_pairs=len(trans_errs),
    )


def kitti_metrics(
    estimate: Trajectory,
    ground_truth: Trajectory,
    lengths_m: Sequence[float] | None = None,
) -> list[KittiSegment]:
    """KITTI-style average relative errors over fixed path lengths."""
    if lengths_m is None:
        lengths_m = (100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0)
    if len(estimate) < 2:
        return [
            KittiSegment(length_m=float(L), translation_percent=math.nan, rotation_deg_per_100m=math.nan, num_segments=0)
            for L in lengths_m
        ]

    e_poses = se2_from_xy_yaw(estimate.xy, estimate.yaw)
    g_poses = se2_from_xy_yaw(ground_truth.xy, ground_truth.yaw)
    step = np.linalg.norm(np.diff(ground_truth.xy, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(step)))

    results: list[KittiSegment] = []
    for length in lengths_m:
        trans_pct: list[float] = []
        rot_per_100: list[float] = []
        j = 0
        for i in range(len(estimate) - 1):
            target = cum[i] + length
            if target > cum[-1]:
                break
            while j < len(cum) - 1 and cum[j] < target:
                j += 1
            if j <= i:
                continue
            # Prefer the index whose cumulative distance is closest to the target length.
            if j > 0 and abs(cum[j - 1] - target) < abs(cum[j] - target):
                j_use = j - 1
            else:
                j_use = j
            if j_use <= i:
                continue
            traveled = cum[j_use] - cum[i]
            if traveled < 0.5 * length:
                continue
            e_rel = np.linalg.inv(e_poses[i]) @ e_poses[j_use]
            g_rel = np.linalg.inv(g_poses[i]) @ g_poses[j_use]
            err = np.linalg.inv(g_rel) @ e_rel
            t_err = float(np.linalg.norm(err[0:2, 2]))
            r_err = abs(float(wrap_angle(math.atan2(err[1, 0], err[0, 0]))))
            trans_pct.append(100.0 * t_err / traveled)
            rot_per_100.append(math.degrees(r_err) * 100.0 / traveled)
        if not trans_pct:
            results.append(
                KittiSegment(
                    length_m=float(length),
                    translation_percent=math.nan,
                    rotation_deg_per_100m=math.nan,
                    num_segments=0,
                )
            )
        else:
            results.append(
                KittiSegment(
                    length_m=float(length),
                    translation_percent=float(np.mean(trans_pct)),
                    rotation_deg_per_100m=float(np.mean(rot_per_100)),
                    num_segments=len(trans_pct),
                )
            )
    return results


def evaluate_pair(name: str, estimate: Trajectory, ground_truth: Trajectory, max_dt_s: float = 0.05) -> EvalReport:
    est_a, gt_a = associate(estimate, ground_truth, max_dt_s=max_dt_s)
    if len(est_a) < 2:
        empty_ate = AteMetrics(math.nan, math.nan, math.nan, math.nan, math.nan, len(est_a))
        empty_rpe = RpeMetrics(1.0, math.nan, math.nan, math.nan, math.nan, 0)
        return EvalReport(
            name=name,
            association_hz=0.0,
            ate_umeyama=empty_ate,
            ate_first_pose=empty_ate,
            rpe_1s=empty_rpe,
            kitti=kitti_metrics(est_a, gt_a),
        )

    duration = float(est_a.stamps[-1] - est_a.stamps[0])
    hz = float(len(est_a) / duration) if duration > 1.0e-6 else 0.0

    first = apply_se2(first_pose_alignment(est_a, gt_a), est_a)
    ume = apply_se2(umeyama_se2(est_a, gt_a), est_a)
    return EvalReport(
        name=name,
        association_hz=hz,
        ate_umeyama=ate_metrics(ume, gt_a),
        ate_first_pose=ate_metrics(first, gt_a),
        rpe_1s=rpe_metrics(ume, gt_a, delta_s=1.0),
        kitti=kitti_metrics(ume, gt_a),
    )


def report_to_dict(report: EvalReport) -> dict:
    payload = asdict(report)
    return payload


def write_metrics_json(path: Path | str, reports: Iterable[EvalReport]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([report_to_dict(r) for r in reports], indent=2) + "\n", encoding="utf-8")


def format_report(report: EvalReport) -> str:
    lines = [
        f"=== {report.name} ===",
        f"poses={report.ate_umeyama.num_poses}  assoc_hz={report.association_hz:.2f}",
        (
            "ATE Umeyama SE(2): "
            f"rmse_xy={report.ate_umeyama.rmse_xy_m:.4f} m  "
            f"mean={report.ate_umeyama.mean_xy_m:.4f}  "
            f"median={report.ate_umeyama.median_xy_m:.4f}  "
            f"max={report.ate_umeyama.max_xy_m:.4f}  "
            f"rmse_yaw={report.ate_umeyama.rmse_yaw_deg:.4f} deg"
        ),
        (
            "ATE first-pose:    "
            f"rmse_xy={report.ate_first_pose.rmse_xy_m:.4f} m  "
            f"rmse_yaw={report.ate_first_pose.rmse_yaw_deg:.4f} deg"
        ),
        (
            "RPE 1s:            "
            f"trans_rmse={report.rpe_1s.translation_rmse_m:.4f} m  "
            f"trans%={report.rpe_1s.translation_percent:.4f}  "
            f"yaw_rmse={report.rpe_1s.yaw_rmse_deg:.4f} deg  "
            f"pairs={report.rpe_1s.num_pairs}"
        ),
        "KITTI-style (Umeyama-aligned):",
    ]
    for seg in report.kitti:
        lines.append(
            f"  {seg.length_m:6.0f} m  "
            f"t_err={seg.translation_percent:8.4f} %  "
            f"r_err={seg.rotation_deg_per_100m:8.4f} deg/100m  "
            f"n={seg.num_segments}"
        )
    return "\n".join(lines)
