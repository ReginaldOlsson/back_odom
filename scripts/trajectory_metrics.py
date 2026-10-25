#!/usr/bin/env python3
# Copyright 2026 huzaifa.osal
#
# Trajectory metrics for Boreas-style odometry evaluation (numpy only).

"""ATE, RPE, and KITTI relative pose errors on SE(2) and SE(3) trajectories."""

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


# --- SE(3) evaluation against Boreas lidar ground truth -------------------------

# sensor_kit_base_link -> lidar yaw. T_enu_base = T_enu_lidar * Rz(-this).
DEFAULT_LIDAR_TO_BASE_YAW = -math.pi / 4.0
KITTI_LENGTHS_M: tuple[float, ...] = (100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0)
KITTI_STEP = 10


@dataclass
class Trajectory3:
    """Poses as time [s], position [m], and body-to-world rotation."""

    stamps: np.ndarray  # (N,)
    xyz: np.ndarray  # (N, 3)
    rotation: np.ndarray  # (N, 3, 3)

    def __post_init__(self) -> None:
        self.stamps = np.asarray(self.stamps, dtype=np.float64).reshape(-1)
        self.xyz = np.asarray(self.xyz, dtype=np.float64).reshape(-1, 3)
        self.rotation = np.asarray(self.rotation, dtype=np.float64).reshape(-1, 3, 3)
        if not (len(self.stamps) == len(self.xyz) == len(self.rotation)):
            raise ValueError("Trajectory3 arrays must share the same length")

    def __len__(self) -> int:
        return int(self.stamps.shape[0])

    def path_length(self, planar: bool = False) -> float:
        if len(self) < 2:
            return 0.0
        xyz = self.xyz[:, :2] if planar else self.xyz
        return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


@dataclass
class TraceSignals:
    """RI-EKF debug columns, one row per EKF sample in RiEkfTrace.ekf."""

    err_xyz: np.ndarray  # (N, 3) EKF position minus back_odom, world frame
    err_yaw: np.ndarray  # (N,)
    gyro_bias: np.ndarray  # (N, 3)
    accel_bias: np.ndarray  # (N, 3)
    imu_accel: np.ndarray  # (N, 3)
    imu_gyro: np.ndarray  # (N, 3)
    body_velocity: np.ndarray  # (N, 3)
    back_odom_age: np.ndarray  # (N,)
    pose_applied: np.ndarray  # (N,)


@dataclass
class RiEkfTrace:
    ekf: Trajectory3
    back_odom: Trajectory3
    signals: TraceSignals


@dataclass
class AteSplit:
    rmse_horiz_m: float
    mean_horiz_m: float
    median_horiz_m: float
    max_horiz_m: float
    rmse_vert_m: float
    mean_vert_m: float
    rmse_xyz_m: float
    rmse_yaw_deg: float
    mean_yaw_deg: float
    rmse_roll_deg: float
    rmse_pitch_deg: float
    num_poses: int


@dataclass
class RpeSplit:
    delta_s: float
    se2_translation_rmse_m: float
    se2_translation_percent: float
    se2_yaw_rmse_deg: float
    se3_translation_rmse_m: float
    se3_rotation_rmse_deg: float
    num_pairs: int


@dataclass
class KittiWindow:
    length_m: float
    first: int
    last: int
    stamp_start: float
    stamp_end: float
    distance_start: float
    distance_end: float
    translation_m: float
    translation_percent: float
    rotation_deg: float
    rotation_deg_per_100m: float


@dataclass
class KittiDevkit:
    space: str
    lengths: list[KittiSegment]
    windows: list[KittiWindow]
    pooled_translation_percent: float
    pooled_rotation_deg_per_100m: float
    num_segments: int


@dataclass
class ErrorSpan:
    """Inclusive index range of a contiguous error run."""

    start: int
    end: int


def rotation_from_rpy(
    roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray
) -> np.ndarray:
    """ROS ZYX rotation, matching ri_ekf roll_pitch_yaw / rotation_from_rpy."""
    roll = np.asarray(roll, dtype=np.float64).reshape(-1)
    pitch = np.asarray(pitch, dtype=np.float64).reshape(-1)
    yaw = np.asarray(yaw, dtype=np.float64).reshape(-1)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rot = np.zeros((len(yaw), 3, 3), dtype=np.float64)
    rot[:, 0, 0] = cy * cp
    rot[:, 0, 1] = cy * sp * sr - sy * cr
    rot[:, 0, 2] = cy * sp * cr + sy * sr
    rot[:, 1, 0] = sy * cp
    rot[:, 1, 1] = sy * sp * sr + cy * cr
    rot[:, 1, 2] = sy * sp * cr - cy * sr
    rot[:, 2, 0] = -sp
    rot[:, 2, 1] = cp * sr
    rot[:, 2, 2] = cp * cr
    return rot


def rpy_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Inverse of rotation_from_rpy. Returns (N, 3) roll, pitch, yaw."""
    rotation = np.asarray(rotation, dtype=np.float64).reshape(-1, 3, 3)
    roll = np.arctan2(rotation[:, 2, 1], rotation[:, 2, 2])
    horizontal = np.hypot(rotation[:, 0, 0], rotation[:, 1, 0])
    pitch = np.arctan2(-rotation[:, 2, 0], horizontal)
    yaw = np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
    return np.column_stack((roll, pitch, yaw))


def barfoot_rotation(heading: np.ndarray, pitch: np.ndarray, roll: np.ndarray) -> np.ndarray:
    """Boreas / Barfoot R = R_roll @ R_pitch @ R_yaw, as in boreas_cpp."""
    heading = np.asarray(heading, dtype=np.float64).reshape(-1)
    pitch = np.asarray(pitch, dtype=np.float64).reshape(-1)
    roll = np.asarray(roll, dtype=np.float64).reshape(-1)
    ch, sh = np.cos(heading), np.sin(heading)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    yaw_r = np.zeros((len(heading), 3, 3), dtype=np.float64)
    pitch_r = np.zeros_like(yaw_r)
    roll_r = np.zeros_like(yaw_r)
    yaw_r[:, 0, 0] = ch
    yaw_r[:, 0, 1] = sh
    yaw_r[:, 1, 0] = -sh
    yaw_r[:, 1, 1] = ch
    yaw_r[:, 2, 2] = 1.0
    pitch_r[:, 0, 0] = cp
    pitch_r[:, 0, 2] = -sp
    pitch_r[:, 1, 1] = 1.0
    pitch_r[:, 2, 0] = sp
    pitch_r[:, 2, 2] = cp
    roll_r[:, 0, 0] = 1.0
    roll_r[:, 1, 1] = cr
    roll_r[:, 1, 2] = sr
    roll_r[:, 2, 1] = -sr
    roll_r[:, 2, 2] = cr
    return roll_r @ pitch_r @ yaw_r


def lidar_rotation_to_base(
    heading: np.ndarray,
    pitch: np.ndarray,
    roll: np.ndarray,
    lidar_to_base_yaw: float = DEFAULT_LIDAR_TO_BASE_YAW,
) -> np.ndarray:
    """Lidar attitude in ENU, rotated into base_link.

    ``lidar_to_base_yaw`` is the static TF yaw of the lidar in base_link.
    T_enu_base = T_enu_lidar * inv(T_base_lidar) = T_enu_lidar * Rz(-yaw).
    """
    lidar = barfoot_rotation(heading, pitch, roll)
    yaw = -float(lidar_to_base_yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return lidar @ rz


def shift_time(traj: Trajectory3, origin_s: float) -> Trajectory3:
    return Trajectory3(traj.stamps + float(origin_s), traj.xyz, traj.rotation)


def as_se2(traj: Trajectory3) -> Trajectory:
    yaw = rpy_from_rotation(traj.rotation)[:, 2] if len(traj) else np.zeros(0)
    return Trajectory(traj.stamps, traj.xyz[:, :2], yaw)


def _empty_ate() -> AteSplit:
    return AteSplit(*(math.nan,) * 11, 0)


def ate_split(estimate: Trajectory3, ground_truth: Trajectory3) -> AteSplit:
    if len(estimate) == 0 or len(estimate) != len(ground_truth):
        return _empty_ate()
    delta = estimate.xyz - ground_truth.xyz
    horiz = np.hypot(delta[:, 0], delta[:, 1])
    vert = delta[:, 2]
    est_rpy = rpy_from_rotation(estimate.rotation)
    gt_rpy = rpy_from_rotation(ground_truth.rotation)
    droll = np.asarray(wrap_angle(est_rpy[:, 0] - gt_rpy[:, 0]), dtype=np.float64)
    dpitch = np.asarray(wrap_angle(est_rpy[:, 1] - gt_rpy[:, 1]), dtype=np.float64)
    dyaw = np.asarray(wrap_angle(est_rpy[:, 2] - gt_rpy[:, 2]), dtype=np.float64)
    return AteSplit(
        rmse_horiz_m=float(np.sqrt(np.mean(horiz**2))),
        mean_horiz_m=float(np.mean(horiz)),
        median_horiz_m=float(np.median(horiz)),
        max_horiz_m=float(np.max(horiz)),
        rmse_vert_m=float(np.sqrt(np.mean(vert**2))),
        mean_vert_m=float(np.mean(vert)),
        rmse_xyz_m=float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))),
        rmse_yaw_deg=float(np.degrees(np.sqrt(np.mean(dyaw**2)))),
        mean_yaw_deg=float(np.degrees(np.mean(dyaw))),
        rmse_roll_deg=float(np.degrees(np.sqrt(np.mean(droll**2)))),
        rmse_pitch_deg=float(np.degrees(np.sqrt(np.mean(dpitch**2)))),
        num_poses=len(estimate),
    )


def apply_se3(transform: np.ndarray, traj: Trajectory3) -> Trajectory3:
    """Left-multiply every pose by a 4x4 rigid transform."""
    rotation = transform[0:3, 0:3]
    translation = transform[0:3, 3]
    xyz = traj.xyz @ rotation.T + translation
    aligned = np.einsum("ij,njk->nik", rotation, traj.rotation)
    return Trajectory3(traj.stamps.copy(), xyz, aligned)


def apply_planar(transform: np.ndarray, traj: Trajectory3) -> Trajectory3:
    """Embed a 3x3 SE(2) alignment: yaw about world z, xy translation, z unchanged."""
    rotation_z = np.eye(3)
    rotation_z[0:2, 0:2] = transform[0:2, 0:2]
    xyz = traj.xyz.copy()
    xy1 = np.column_stack((traj.xyz[:, 0], traj.xyz[:, 1], np.ones(len(traj))))
    xyz[:, 0:2] = (transform @ xy1.T).T[:, 0:2]
    aligned = np.einsum("ij,njk->nik", rotation_z, traj.rotation)
    return Trajectory3(traj.stamps.copy(), xyz, aligned)


def first_pose_alignment_se3(estimate: Trajectory3, ground_truth: Trajectory3) -> np.ndarray:
    if len(estimate) == 0:
        return np.eye(4)
    transform = np.eye(4)
    rotation = ground_truth.rotation[0] @ estimate.rotation[0].T
    transform[0:3, 0:3] = rotation
    transform[0:3, 3] = ground_truth.xyz[0] - rotation @ estimate.xyz[0]
    return transform


def umeyama_se3(estimate: Trajectory3, ground_truth: Trajectory3) -> np.ndarray:
    """Rigid SE(3) Umeyama (no scale) mapping estimate positions onto GT."""
    if len(estimate) < 2:
        return first_pose_alignment_se3(estimate, ground_truth)
    src = estimate.xyz.T
    dst = ground_truth.xyz.T
    mu_src = src.mean(axis=1, keepdims=True)
    mu_dst = dst.mean(axis=1, keepdims=True)
    cov = (dst - mu_dst) @ (src - mu_src).T / float(src.shape[1])
    u, _, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        sign[2, 2] = -1.0
    rotation = u @ sign @ vt
    translation = mu_dst.reshape(3) - rotation @ mu_src.reshape(3)
    transform = np.eye(4)
    transform[0:3, 0:3] = rotation
    transform[0:3, 3] = translation
    return transform


def _so3_log(rotation: np.ndarray) -> np.ndarray:
    trace = rotation[:, 0, 0] + rotation[:, 1, 1] + rotation[:, 2, 2]
    angle = np.arccos(np.clip(0.5 * (trace - 1.0), -1.0, 1.0))
    vee = np.column_stack(
        (
            rotation[:, 2, 1] - rotation[:, 1, 2],
            rotation[:, 0, 2] - rotation[:, 2, 0],
            rotation[:, 1, 0] - rotation[:, 0, 1],
        )
    )
    scale = np.full(len(rotation), 0.5, dtype=np.float64)
    regular = angle >= 1.0e-8
    sine = np.sin(angle[regular])
    scale[regular] = np.where(np.abs(sine) < 1.0e-8, 0.5, 0.5 * angle[regular] / sine)
    return vee * scale[:, None]


def _so3_exp(omega: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(omega, axis=1)
    small = theta < 1.0e-8
    a = np.ones(len(theta), dtype=np.float64)
    b = np.full(len(theta), 0.5, dtype=np.float64)
    a[~small] = np.sin(theta[~small]) / theta[~small]
    b[~small] = (1.0 - np.cos(theta[~small])) / np.square(theta[~small])
    x, y, z = omega[:, 0], omega[:, 1], omega[:, 2]
    skew = np.zeros((len(omega), 3, 3), dtype=np.float64)
    skew[:, 0, 1] = -z
    skew[:, 0, 2] = y
    skew[:, 1, 0] = z
    skew[:, 1, 2] = -x
    skew[:, 2, 0] = -y
    skew[:, 2, 1] = x
    skew_sq = skew @ skew
    eye = np.broadcast_to(np.eye(3), skew.shape)
    return eye + a[:, None, None] * skew + b[:, None, None] * skew_sq


def _slerp(rotation_0: np.ndarray, rotation_1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    relative = np.einsum("nji,njk->nik", rotation_0, rotation_1)
    increment = _so3_exp(_so3_log(relative) * alpha.reshape(-1, 1))
    return rotation_0 @ increment


def cumulative_distance(xyz: np.ndarray, planar: bool = False) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    if planar:
        xyz = xyz[:, :2]
    if len(xyz) == 0:
        return np.zeros(0, dtype=np.float64)
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return np.concatenate((np.zeros(1), np.cumsum(step)))


def _take(traj: Trajectory3, index: np.ndarray) -> Trajectory3:
    if len(index) == 0:
        return Trajectory3([], np.zeros((0, 3)), np.zeros((0, 3, 3)))
    return Trajectory3(traj.stamps[index], traj.xyz[index], traj.rotation[index])


def associate_trajectory3(
    estimate: Trajectory3, ground_truth: Trajectory3, max_dt_s: float = 0.05
) -> tuple[Trajectory3, Trajectory3, np.ndarray, np.ndarray]:
    """Nearest GT pose for each estimate pose, one estimate per GT sample."""
    empty = (
        Trajectory3([], np.zeros((0, 3)), np.zeros((0, 3, 3))),
        Trajectory3([], np.zeros((0, 3)), np.zeros((0, 3, 3))),
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
    )
    if len(estimate) == 0 or len(ground_truth) == 0:
        return empty
    gt_stamps = ground_truth.stamps
    chosen: dict[int, tuple[float, int]] = {}
    for i, stamp in enumerate(estimate.stamps):
        j = int(np.searchsorted(gt_stamps, stamp))
        candidates = [k for k in (j, j - 1) if 0 <= k < len(gt_stamps)]
        if not candidates:
            continue
        best = min(candidates, key=lambda k: abs(gt_stamps[k] - stamp))
        dt = abs(float(gt_stamps[best] - stamp))
        if dt > max_dt_s:
            continue
        previous = chosen.get(best)
        if previous is None or dt < previous[0]:
            chosen[best] = (dt, i)
    if not chosen:
        return empty
    gt_idx = np.array(sorted(chosen), dtype=np.int64)
    est_idx = np.array([chosen[int(k)][1] for k in gt_idx], dtype=np.int64)
    return _take(estimate, est_idx), _take(ground_truth, gt_idx), est_idx, gt_idx


def interpolate_trajectory3(
    estimate: Trajectory3, ground_truth: Trajectory3, max_dt_s: float = 0.05
) -> tuple[Trajectory3, Trajectory3, np.ndarray, np.ndarray]:
    """Interpolate the estimate onto each GT stamp bracketed within max_dt_s."""
    empty_est = Trajectory3([], np.zeros((0, 3)), np.zeros((0, 3, 3)))
    empty_idx = np.zeros(0, dtype=np.int64)
    if len(estimate) == 0 or len(ground_truth) == 0:
        return empty_est, empty_est, empty_idx, empty_idx

    src = estimate.stamps
    tgt = ground_truth.stamps
    right = np.searchsorted(src, tgt, side="left")
    left = right - 1
    n = len(src)
    valid_left = left >= 0
    valid_right = right < n
    left_c = np.clip(left, 0, n - 1)
    right_c = np.clip(right, 0, n - 1)
    dt_left = np.abs(tgt - src[left_c])
    dt_right = np.abs(tgt - src[right_c])
    dt_left = np.where(valid_left, dt_left, np.inf)
    dt_right = np.where(valid_right, dt_right, np.inf)
    span = src[right_c] - src[left_c]
    inside = valid_left & valid_right & (src[left_c] <= tgt) & (tgt <= src[right_c])
    can_interp = inside & (dt_left <= max_dt_s) & (dt_right <= max_dt_s) & (span > 1.0e-9)
    nearest_is_left = dt_left <= dt_right
    nearest_dt = np.minimum(dt_left, dt_right)
    use = can_interp | (nearest_dt <= max_dt_s)
    gt_idx = np.flatnonzero(use)
    if len(gt_idx) == 0:
        return empty_est, empty_est, empty_idx, empty_idx

    alpha = np.zeros(len(gt_idx), dtype=np.float64)
    interp = can_interp[gt_idx]
    alpha[interp] = (tgt[gt_idx][interp] - src[left_c[gt_idx]][interp]) / span[gt_idx][interp]
    nearest_left = nearest_is_left[gt_idx]
    # Nearest-only samples copy one endpoint. Interpolated samples blend both.
    alpha = np.where(interp, alpha, np.where(nearest_left, 0.0, 1.0))
    i0 = left_c[gt_idx]
    i1 = right_c[gt_idx]
    # A sample that only has a left neighbor still indexes right_c == n-1. Force both
    # ends onto the nearest valid sample when we are not interpolating.
    i0 = np.where(interp | nearest_left, i0, i1)
    i1 = np.where(interp | ~nearest_left, i1, i0)
    xyz = (1.0 - alpha)[:, None] * estimate.xyz[i0] + alpha[:, None] * estimate.xyz[i1]
    rotation = _slerp(estimate.rotation[i0], estimate.rotation[i1], alpha)
    nearest = np.where(dt_left[gt_idx] <= dt_right[gt_idx], i0, i1)
    stamps = tgt[gt_idx]
    return (
        Trajectory3(stamps, xyz, rotation),
        _take(ground_truth, gt_idx),
        nearest.astype(np.int64),
        gt_idx.astype(np.int64),
    )


def _se3_matrices(traj: Trajectory3) -> np.ndarray:
    poses = np.zeros((len(traj), 4, 4), dtype=np.float64)
    poses[:, 0:3, 0:3] = traj.rotation
    poses[:, 0:3, 3] = traj.xyz
    poses[:, 3, 3] = 1.0
    return poses


def _invert_se3(poses: np.ndarray) -> np.ndarray:
    rotation_t = np.transpose(poses[:, 0:3, 0:3], (0, 2, 1))
    out = np.zeros_like(poses)
    out[:, 0:3, 0:3] = rotation_t
    out[:, 0:3, 3] = -np.einsum("nij,nj->ni", rotation_t, poses[:, 0:3, 3])
    out[:, 3, 3] = 1.0
    return out


def _rotation_angle(rotation: np.ndarray) -> float:
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    return float(math.acos(min(1.0, max(-1.0, 0.5 * (trace - 1.0)))))


def _segment_pairs(distance: np.ndarray, lengths_m: Sequence[float], step: int) -> list[tuple[int, int, float]]:
    pairs: list[tuple[int, int, float]] = []
    n = len(distance)
    for first in range(0, n, step):
        base = float(distance[first])
        for length in lengths_m:
            last = int(np.searchsorted(distance, base + float(length), side="right"))
            if last >= n:
                continue
            pairs.append((first, last, float(length)))
    return pairs


def _empty_devkit(space: str, lengths_m: Sequence[float]) -> KittiDevkit:
    lengths = [
        KittiSegment(float(length), math.nan, math.nan, 0) for length in lengths_m
    ]
    return KittiDevkit(space, lengths, [], math.nan, math.nan, 0)


def kitti_devkit(
    estimate: Trajectory3,
    ground_truth: Trajectory3,
    space: str = "se3",
    lengths_m: Sequence[float] | None = None,
    step: int = KITTI_STEP,
    distance_m: np.ndarray | None = None,
) -> KittiDevkit:
    """KITTI odometry segments: every ``step`` poses, lengths 100–800 m.

    Error is inv(estimate_delta) * gt_delta. Translation is divided by the nominal
    length. ``distance_m`` overrides the cumulative distance used to place segment
    endpoints (same length as the trajectories); the relative pose still uses the
    associated poses.
    """
    if lengths_m is None:
        lengths_m = KITTI_LENGTHS_M
    if space not in ("se2", "se3"):
        raise ValueError(f"space must be se2 or se3, got {space!r}")
    if len(estimate) != len(ground_truth) or len(estimate) < 2:
        return _empty_devkit(space, lengths_m)

    if distance_m is None:
        distance = cumulative_distance(ground_truth.xyz, planar=(space == "se2"))
    else:
        distance = np.asarray(distance_m, dtype=np.float64).reshape(-1)
        if len(distance) != len(ground_truth):
            raise ValueError("distance_m must have one entry per associated pose")
    if space == "se2":
        est_poses = se2_from_xy_yaw(estimate.xyz[:, :2], as_se2(estimate).yaw)
        gt_poses = se2_from_xy_yaw(ground_truth.xyz[:, :2], as_se2(ground_truth).yaw)
    else:
        est_poses = _se3_matrices(estimate)
        gt_poses = _se3_matrices(ground_truth)
    est_inv = np.linalg.inv(est_poses) if space == "se2" else _invert_se3(est_poses)
    gt_inv = np.linalg.inv(gt_poses) if space == "se2" else _invert_se3(gt_poses)

    windows: list[KittiWindow] = []
    by_length: dict[float, list[tuple[float, float]]] = {float(length): [] for length in lengths_m}
    trans_sum = 0.0
    rot_sum = 0.0
    for first, last, length in _segment_pairs(distance, lengths_m, step):
        est_delta = est_inv[first] @ est_poses[last]
        gt_delta = gt_inv[first] @ gt_poses[last]
        error = np.linalg.inv(est_delta) @ gt_delta
        if space == "se2":
            translation = float(np.linalg.norm(error[0:2, 2]))
            rotation = abs(float(wrap_angle(math.atan2(error[1, 0], error[0, 0]))))
        else:
            translation = float(np.linalg.norm(error[0:3, 3]))
            rotation = _rotation_angle(error[0:3, 0:3])
        trans_frac = translation / length
        rot_frac = rotation / length
        trans_sum += trans_frac
        rot_sum += rot_frac
        by_length[length].append((trans_frac, rot_frac))
        windows.append(
            KittiWindow(
                length_m=length,
                first=first,
                last=last,
                stamp_start=float(ground_truth.stamps[first]),
                stamp_end=float(ground_truth.stamps[last]),
                distance_start=float(distance[first]),
                distance_end=float(distance[last]),
                translation_m=translation,
                translation_percent=100.0 * trans_frac,
                rotation_deg=float(np.degrees(rotation)),
                rotation_deg_per_100m=float(np.degrees(rot_frac) * 100.0),
            )
        )

    summaries: list[KittiSegment] = []
    for length in lengths_m:
        samples = by_length[float(length)]
        if not samples:
            summaries.append(KittiSegment(float(length), math.nan, math.nan, 0))
            continue
        trans = np.array([item[0] for item in samples])
        rot = np.array([item[1] for item in samples])
        summaries.append(
            KittiSegment(
                length_m=float(length),
                translation_percent=float(100.0 * np.mean(trans)),
                rotation_deg_per_100m=float(np.degrees(np.mean(rot)) * 100.0),
                num_segments=len(samples),
            )
        )
    count = len(windows)
    if count == 0:
        return KittiDevkit(space, summaries, [], math.nan, math.nan, 0)
    return KittiDevkit(
        space=space,
        lengths=summaries,
        windows=windows,
        pooled_translation_percent=float(100.0 * trans_sum / count),
        pooled_rotation_deg_per_100m=float(np.degrees(rot_sum / count) * 100.0),
        num_segments=count,
    )


def rpe_split(estimate: Trajectory3, ground_truth: Trajectory3, delta_s: float = 1.0) -> RpeSplit:
    if len(estimate) < 2 or len(estimate) != len(ground_truth):
        return RpeSplit(delta_s, math.nan, math.nan, math.nan, math.nan, math.nan, 0)
    est2 = se2_from_xy_yaw(estimate.xyz[:, :2], as_se2(estimate).yaw)
    gt2 = se2_from_xy_yaw(ground_truth.xyz[:, :2], as_se2(ground_truth).yaw)
    est3 = _se3_matrices(estimate)
    gt3 = _se3_matrices(ground_truth)
    se2_t: list[float] = []
    se2_y: list[float] = []
    se2_len: list[float] = []
    se3_t: list[float] = []
    se3_r: list[float] = []
    j = 0
    for i in range(len(estimate)):
        target = estimate.stamps[i] + delta_s
        while j + 1 < len(estimate) and estimate.stamps[j] < target:
            j += 1
        if j <= i or abs(estimate.stamps[j] - target) > 0.5 * delta_s:
            continue
        # Devkit order: inv(estimate_delta) * gt_delta.
        err2 = np.linalg.inv(np.linalg.inv(est2[i]) @ est2[j]) @ (np.linalg.inv(gt2[i]) @ gt2[j])
        err3 = np.linalg.inv(np.linalg.inv(est3[i]) @ est3[j]) @ (np.linalg.inv(gt3[i]) @ gt3[j])
        chord2 = float(np.linalg.norm(ground_truth.xyz[j, :2] - ground_truth.xyz[i, :2]))
        if chord2 < 1.0e-3:
            continue
        se2_t.append(float(np.linalg.norm(err2[0:2, 2])))
        se2_y.append(abs(float(wrap_angle(math.atan2(err2[1, 0], err2[0, 0])))))
        se2_len.append(chord2)
        se3_t.append(float(np.linalg.norm(err3[0:3, 3])))
        se3_r.append(_rotation_angle(err3[0:3, 0:3]))
    if not se2_t:
        return RpeSplit(delta_s, math.nan, math.nan, math.nan, math.nan, math.nan, 0)
    trans = np.asarray(se2_t)
    yaw = np.asarray(se2_y)
    lens = np.asarray(se2_len)
    rot3 = np.asarray(se3_r)
    return RpeSplit(
        delta_s=delta_s,
        se2_translation_rmse_m=float(np.sqrt(np.mean(trans**2))),
        se2_translation_percent=float(100.0 * np.mean(trans / lens)),
        se2_yaw_rmse_deg=float(np.degrees(np.sqrt(np.mean(yaw**2)))),
        se3_translation_rmse_m=float(np.sqrt(np.mean(np.asarray(se3_t) ** 2))),
        se3_rotation_rmse_deg=float(np.degrees(np.sqrt(np.mean(rot3**2)))),
        num_pairs=len(se2_t),
    )


def body_frame_error(estimate_xyz: np.ndarray, ground_truth: Trajectory3) -> np.ndarray:
    """Position residual of an aligned estimate, in the ground-truth body frame."""
    delta = np.asarray(estimate_xyz, dtype=np.float64) - ground_truth.xyz
    return np.einsum("nji,nj->ni", ground_truth.rotation, delta)


def section_threshold(error: np.ndarray, floor_m: float = 5.0, percentile: float = 90.0) -> float:
    error = np.asarray(error, dtype=np.float64)
    finite = error[np.isfinite(error)]
    if len(finite) == 0:
        return float(floor_m)
    return float(max(floor_m, np.percentile(finite, percentile)))


def contiguous_error_spans(
    error: np.ndarray, distance: np.ndarray, threshold: float, min_length_m: float
) -> list[ErrorSpan]:
    """Runs where ``error`` stays at or above ``threshold`` for at least ``min_length_m``."""
    error = np.asarray(error, dtype=np.float64).reshape(-1)
    distance = np.asarray(distance, dtype=np.float64).reshape(-1)
    if len(error) != len(distance):
        raise ValueError("error and distance must have the same length")
    above = np.isfinite(error) & (error >= threshold)
    spans: list[ErrorSpan] = []
    i = 0
    n = len(error)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and above[j + 1]:
            j += 1
        if float(distance[j] - distance[i]) >= min_length_m:
            spans.append(ErrorSpan(i, j))
        i = j + 1
    return spans


def classify_error_source(
    back_odom_horiz_m: float,
    ekf_horiz_m: float,
    residual_xy_m: float,
    far_m: float,
) -> str:
    """Blame a window using GT error of each trajectory and the EKF-vs-odom residual."""
    if not all(math.isfinite(v) for v in (back_odom_horiz_m, ekf_horiz_m, residual_xy_m, far_m)):
        return "unknown"
    back_odom_far = back_odom_horiz_m >= far_m
    ekf_far = ekf_horiz_m >= far_m
    scale = max(back_odom_horiz_m, ekf_horiz_m, 0.0)
    residual_small = residual_xy_m < max(1.0, 0.35 * scale)
    if back_odom_far and residual_small:
        return "back_odom_led"
    if ekf_far and not back_odom_far:
        return "ekf_diverged"
    if back_odom_far and ekf_far:
        return "both"
    if ekf_far:
        return "ekf_diverged"
    if back_odom_far:
        return "back_odom_led"
    return "within_threshold"


def _first_csv_float(path: Path, column: str) -> float:
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split(",")
        if column not in header:
            raise ValueError(f"{path} has no column {column!r}")
        index = header.index(column)
        values = handle.readline().rstrip("\n").split(",")
    return float(values[index])


def _stamp_seconds(raw: float) -> float:
    return raw * 1.0e-6 if raw > 1.0e12 else raw


def detect_time_origin(
    sequence_dir: Path | str, trace_stamp0: float, lidar_stamp0: float
) -> tuple[float, str]:
    """Bag time origin in GPS seconds.

    Matches boreas_cpp: the minimum of the first IMU, GNSS, and lidar timestamps.
    The origin is accepted when ``trace_stamp0 + origin`` lands on the first lidar
    pose. Otherwise the camera stream is included, then the offset that pins the
    first trace sample to the first lidar pose.
    """
    sequence_dir = Path(sequence_dir)
    applanix = sequence_dir / "applanix"
    candidates = [float(lidar_stamp0)]
    imu_path = applanix / "imu.csv"
    gps_path = applanix / "gps_post_process.csv"
    if imu_path.exists():
        candidates.append(_stamp_seconds(_first_csv_float(imu_path, "GPSTime")))
    if gps_path.exists():
        candidates.append(_stamp_seconds(_first_csv_float(gps_path, "GPSTime")))
    origin = min(candidates)

    def aligned(value: float) -> bool:
        return abs((float(lidar_stamp0) - value) - float(trace_stamp0)) < 1.0e-3

    if aligned(origin):
        return origin, "imu_gnss_lidar"
    camera_path = applanix / "camera_times.txt"
    if camera_path.exists():
        with camera_path.open(encoding="utf-8") as handle:
            token = handle.readline().strip().split(".", 1)[0]
        if token:
            origin = min(origin, _stamp_seconds(float(token)))
            if aligned(origin):
                return origin, "with_camera"
    return float(lidar_stamp0) - float(trace_stamp0), "lidar_minus_trace"


def load_boreas_lidar_poses_se3(
    path: Path | str, lidar_to_base_yaw: float = DEFAULT_LIDAR_TO_BASE_YAW
) -> Trajectory3:
    """Boreas lidar_poses.csv as base_link poses in the local ENU frame."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64, encoding="utf-8")
    stamps = np.asarray(data["GPSTime"], dtype=np.float64)
    if np.nanmedian(stamps) > 1.0e12:
        stamps = stamps * 1.0e-6
    xyz = np.column_stack(
        (
            np.asarray(data["easting"], dtype=np.float64),
            np.asarray(data["northing"], dtype=np.float64),
            np.asarray(data["altitude"], dtype=np.float64),
        )
    )
    rotation = lidar_rotation_to_base(
        data["heading"], data["pitch"], data["roll"], lidar_to_base_yaw
    )
    order = np.argsort(stamps)
    return Trajectory3(stamps[order], xyz[order], rotation[order])


def _rotation_or_empty(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    if len(roll) == 0:
        return np.zeros((0, 3, 3), dtype=np.float64)
    return rotation_from_rpy(roll, pitch, yaw)


def load_ri_ekf_trace(path: Path | str) -> RiEkfTrace:
    """Load an ri_ekf_localizer trace CSV.

    EKF poses keep the sample stamp. back_odom keeps one pose per ``pose_applied``
    id, stamped at ``stamp - back_odom_age`` (the pose time, not the hold time).
    """
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64, encoding="utf-8")
    stamps = np.asarray(data["stamp"], dtype=np.float64)
    ekf_xyz = np.column_stack((data["ekf_x"], data["ekf_y"], data["ekf_z"]))
    finite_ekf = np.isfinite(ekf_xyz).all(axis=1) & np.isfinite(stamps)
    ekf_index = np.flatnonzero(finite_ekf)
    ekf_index = ekf_index[np.argsort(stamps[ekf_index], kind="mergesort")]
    ekf = Trajectory3(
        stamps[ekf_index],
        ekf_xyz[ekf_index],
        _rotation_or_empty(
            data["ekf_roll"][ekf_index],
            data["ekf_pitch"][ekf_index],
            data["ekf_yaw"][ekf_index],
        ),
    )
    signals = TraceSignals(
        err_xyz=np.column_stack(
            (data["err_x"][ekf_index], data["err_y"][ekf_index], data["err_z"][ekf_index])
        ),
        err_yaw=np.asarray(data["err_yaw"][ekf_index], dtype=np.float64),
        gyro_bias=np.column_stack(
            (data["bg_x"][ekf_index], data["bg_y"][ekf_index], data["bg_z"][ekf_index])
        ),
        accel_bias=np.column_stack(
            (data["ba_x"][ekf_index], data["ba_y"][ekf_index], data["ba_z"][ekf_index])
        ),
        imu_accel=np.column_stack(
            (data["imu_ax"][ekf_index], data["imu_ay"][ekf_index], data["imu_az"][ekf_index])
        ),
        imu_gyro=np.column_stack(
            (data["imu_wx"][ekf_index], data["imu_wy"][ekf_index], data["imu_wz"][ekf_index])
        ),
        body_velocity=np.column_stack(
            (
                data["ekf_body_vx"][ekf_index],
                data["ekf_body_vy"][ekf_index],
                data["ekf_body_vz"][ekf_index],
            )
        ),
        back_odom_age=np.asarray(data["back_odom_age"][ekf_index], dtype=np.float64),
        pose_applied=np.asarray(data["pose_applied"][ekf_index], dtype=np.float64),
    )

    odom_xyz = np.column_stack((data["back_odom_x"], data["back_odom_y"], data["back_odom_z"]))
    odom_finite = (
        np.isfinite(odom_xyz).all(axis=1)
        & np.isfinite(data["back_odom_age"])
        & np.isfinite(data["pose_applied"])
        & np.isfinite(stamps)
    )
    odom_rows = np.flatnonzero(odom_finite)
    if len(odom_rows) == 0:
        back_odom = Trajectory3([], np.zeros((0, 3)), np.zeros((0, 3, 3)))
    else:
        ids = np.rint(data["pose_applied"][odom_rows]).astype(np.int64)
        _, first_local = np.unique(ids, return_index=True)
        chosen = odom_rows[np.sort(first_local)]
        pose_stamps = stamps[chosen] - data["back_odom_age"][chosen]
        order = np.argsort(pose_stamps)
        chosen = chosen[order]
        back_odom = Trajectory3(
            pose_stamps[order],
            odom_xyz[chosen],
            _rotation_or_empty(
                data["back_odom_roll"][chosen],
                data["back_odom_pitch"][chosen],
                data["back_odom_yaw"][chosen],
            ),
        )
    return RiEkfTrace(ekf, back_odom, signals)
