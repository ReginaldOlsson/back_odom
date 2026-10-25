# Benchmark against Boreas ground truth

Offline evaluation of `/ekf_odom` and `/back_odom` against
`applanix/lidar_poses.csv` from a Boreas sequence.

## Ground truth

Use the lidar pose file from the sequence folder, for example:

`/home/mosal/Downloads/boreas-2020-11-26-13-58/applanix/lidar_poses.csv`

Timestamps are epoch microseconds. Positions are easting/northing/altitude in the
same ENU/`map` frame that `boreas_cpp` publishes. `base_link` is at the lidar
origin, so the position needs no lever arm. The lidar yaw in `base_link` is
`-π/4` (`sensor_kit_calibration.yaml`). Attitude comparison has to rotate the
Barfoot lidar pose by that mount; `eval_trace_kitti.py` does this. The older
`eval_trajectory.py` path still treats the `heading` column as yaw.

## Record an estimate bag

```bash
source /opt/ros/jazzy/setup.bash
source /home/mosal/autoware/install/setup.bash

# terminal 1
ros2 launch back_odom back_odom_ekf.launch.xml use_sim_time:=true

# terminal 2
ros2 bag record -o estimate_run /ekf_odom /back_odom /back_odom/pose_with_covariance /tf

# terminal 3
ros2 bag play /path/to/boreas_bag --clock
```

Stop recording after the sequence ends.

## Evaluate

```bash
ros2 run back_odom eval_trajectory.py \
  --gt /home/mosal/Downloads/boreas-2020-11-26-13-58/applanix/lidar_poses.csv \
  --estimate-bag ./estimate_run \
  --topics /ekf_odom /back_odom \
  --out /tmp/boreas_eval
```

Outputs:

- `/tmp/boreas_eval/metrics.json` — ATE (Umeyama SE(2) and first-pose), RPE at 1 s, KITTI-style relative errors at 100–800 m
- `/tmp/boreas_eval/trajectories.png` — GT vs aligned estimate XY paths

## CSV fallback

If you prefer CSV dumps while the bag plays:

```bash
ros2 run back_odom dump_odom_csv.py --out /tmp/odom_csv --topics /ekf_odom /back_odom
```

Then:

```bash
ros2 run back_odom eval_trajectory.py \
  --gt /home/mosal/Downloads/boreas-2020-11-26-13-58/applanix/lidar_poses.csv \
  --estimate-csv ekf=/tmp/odom_csv/ekf_odom.csv \
  --estimate-csv back_odom=/tmp/odom_csv/back_odom.csv \
  --out /tmp/boreas_eval
```

## RI-EKF trace (KITTI SE(2) and SE(3))

`ri_ekf_localizer` can write both trajectories, plus the filter residual, into one CSV (`trace_csv_path` in `ekf_back_odom.param.yaml`). Stamps in that file are bag-relative. The evaluator adds the Boreas time origin (the earliest IMU/GNSS/lidar time, the same origin `boreas_cpp` uses) and checks that the first trace sample lands on the first lidar pose.

back_odom rows are held at the EKF rate. The score uses one pose per `pose_applied`, stamped at `stamp - back_odom_age`. The EKF pose is interpolated onto each lidar time.

```bash
python3 scripts/eval_trace_kitti.py \
  --trace /tmp/ri_ekf_trace.csv \
  --sequence /home/mosal/Downloads/boreas-2020-11-26-13-58 \
  --out /tmp/boreas_eval_2020-11-26-13-58
```

`--lidar-to-base-yaw` defaults to `-π/4`. Rigid Kabsch–Umeyama (no scale) aligns each estimate onto the ground truth for ATE and the plots. KITTI relative error is the devkit average over 100–800 m, every 10 poses, in SE(2) and SE(3), and does not depend on that alignment.

Outputs:

- `metrics.json`, `report.txt` — ATE (raw, first pose, Umeyama SE(2), Umeyama SE(3)), 1 s RPE, KITTI per length and the pooled average
- `per_pose.csv` — aligned error, GT-body longitudinal/lateral/vertical, and the trace internals on RI-EKF rows
- `kitti_segments.csv` — every KITTI window
- `error_sections.csv` — long stretches where the SE(2) horizontal error stays high, plus the worst non-overlapping 100 m KITTI segments, tagged `back_odom_led`, `ekf_diverged`, or `both`
- `trajectories.png`, `error_vs_distance.png`, `kitti_bars.png`
