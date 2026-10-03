# Benchmark against Boreas ground truth

Offline evaluation of `/ekf_odom` and `/back_odom` against
`applanix/lidar_poses.csv` from a Boreas sequence.

## Ground truth

Use the lidar pose file from the sequence folder, for example:

`/home/mosal/Downloads/boreas-2020-11-26-13-58/applanix/lidar_poses.csv`

Timestamps are epoch microseconds. Positions are easting/northing/altitude in the
same ENU/`map` frame that `boreas_cpp` publishes. For the Boreas sensor kit,
`base_link` is at the lidar origin, so these poses match EKF and back_odom
`map → base_link` without an extra extrinsic.

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
