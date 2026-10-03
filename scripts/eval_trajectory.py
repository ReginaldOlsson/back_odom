#!/usr/bin/env python3
# Copyright 2026 huzaifa.osal
#
# Offline Boreas trajectory evaluation: lidar_poses.csv vs recorded odometry.

"""Compare estimate trajectories to Boreas lidar ground truth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from trajectory_metrics import (  # noqa: E402
    Trajectory,
    apply_se2,
    associate,
    evaluate_pair,
    format_report,
    load_boreas_lidar_poses,
    load_odom_csv,
    umeyama_se2,
    write_metrics_json,
    yaw_from_quat,
)


def _load_estimate_bag(bag_path: Path, topics: list[str]) -> dict[str, Trajectory]:
    from rclpy.serialization import deserialize_message
    from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
    from rosidl_runtime_py.utilities import get_message

    bag_path = bag_path.resolve()
    if bag_path.is_dir():
        uri = str(bag_path)
        meta = bag_path / "metadata.yaml"
        storage_id = "mcap"
        if meta.exists():
            text = meta.read_text(encoding="utf-8")
            if "sqlite3" in text:
                storage_id = "sqlite3"
            elif "mcap" in text:
                storage_id = "mcap"
    elif bag_path.suffix == ".mcap":
        uri = str(bag_path)
        storage_id = "mcap"
    elif bag_path.suffix == ".db3":
        uri = str(bag_path)
        storage_id = "sqlite3"
    else:
        raise FileNotFoundError(f"unsupported bag path: {bag_path}")

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=uri, storage_id=storage_id),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    wanted = set(topics)
    missing = [t for t in topics if t not in topic_types]
    if missing:
        raise FileNotFoundError(f"topics not in bag: {missing}; available={sorted(topic_types)}")

    type_map = {name: get_message(topic_types[name]) for name in wanted}
    buckets: dict[str, dict[str, list]] = {
        name: {"stamps": [], "xy": [], "yaw": []} for name in wanted
    }

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in wanted:
            continue
        msg = deserialize_message(data, type_map[topic])
        stamp = float(msg.header.stamp.sec) + 1.0e-9 * float(msg.header.stamp.nanosec)
        buckets[topic]["stamps"].append(stamp)
        buckets[topic]["xy"].append([msg.pose.pose.position.x, msg.pose.pose.position.y])
        q = msg.pose.pose.orientation
        buckets[topic]["yaw"].append(yaw_from_quat(q.x, q.y, q.z, q.w))

    out: dict[str, Trajectory] = {}
    for name, payload in buckets.items():
        if not payload["stamps"]:
            out[name] = Trajectory([], np.zeros((0, 2)), [])
            continue
        stamps = np.asarray(payload["stamps"], dtype=np.float64)
        xy = np.asarray(payload["xy"], dtype=np.float64)
        yaw = np.asarray(payload["yaw"], dtype=np.float64)
        order = np.argsort(stamps)
        out[name] = Trajectory(stamps[order], xy[order], yaw[order])
    return out


def _maybe_plot(
    out_path: Path,
    ground_truth: "Trajectory",
    estimates: dict[str, "Trajectory"],
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(ground_truth.xy[:, 0], ground_truth.xy[:, 1], "k-", linewidth=1.5, label="GT lidar")
    for name, est in estimates.items():
        est_a, gt_a = associate(est, ground_truth, max_dt_s=0.05)
        if len(est_a) < 2:
            continue
        aligned = apply_se2(umeyama_se2(est_a, gt_a), est_a)
        ax.plot(aligned.xy[:, 0], aligned.xy[:, 1], "-", linewidth=1.0, label=f"{name} (Umeyama)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("easting / x [m]")
    ax.set_ylabel("northing / y [m]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt",
        type=Path,
        required=True,
        help="Boreas applanix/lidar_poses.csv",
    )
    parser.add_argument(
        "--estimate-bag",
        type=Path,
        default=None,
        help="rosbag2 directory or .mcap/.db3 with odometry topics",
    )
    parser.add_argument(
        "--estimate-csv",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Named odometry CSV (stamp,x,y,z,qx,qy,qz,qw). Repeatable.",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=["/ekf_odom", "/back_odom"],
        help="Odometry topics to read from the estimate bag",
    )
    parser.add_argument("--max-dt", type=float, default=0.05, help="Association gate [s]")
    parser.add_argument("--out", type=Path, default=Path("/tmp/boreas_eval"), help="Output directory")
    parser.add_argument("--plot", type=Path, default=None, help="Optional XY plot path")
    args = parser.parse_args(argv)

    gt = load_boreas_lidar_poses(args.gt)
    estimates: dict[str, Trajectory] = {}

    if args.estimate_bag is not None:
        estimates.update(_load_estimate_bag(args.estimate_bag, list(args.topics)))

    for item in args.estimate_csv:
        if "=" not in item:
            raise SystemExit(f"--estimate-csv expects NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        estimates[name] = load_odom_csv(path)

    if not estimates:
        raise SystemExit("provide --estimate-bag and/or --estimate-csv NAME=PATH")

    args.out.mkdir(parents=True, exist_ok=True)
    reports = []
    for name, est in estimates.items():
        report = evaluate_pair(name, est, gt, max_dt_s=args.max_dt)
        reports.append(report)
        print(format_report(report))
        print()

    metrics_path = args.out / "metrics.json"
    write_metrics_json(metrics_path, reports)
    print(f"wrote {metrics_path}")

    plot_path = args.plot if args.plot is not None else args.out / "trajectories.png"
    try:
        _maybe_plot(plot_path, gt, estimates)
        print(f"wrote {plot_path}")
    except Exception as exc:  # matplotlib optional at runtime
        print(f"plot skipped: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
