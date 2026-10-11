// Copyright 2026 huzaifa.osal
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef BACK_ODOM__TRAJECTORY_SYNC_HPP_
#define BACK_ODOM__TRAJECTORY_SYNC_HPP_

#include <sophus/se3.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <utility>
#include <vector>

namespace back_odom
{

struct AssociatedPose
{
  int32_t sec{0};
  uint32_t nanosec{0};
  Sophus::SE3d ground_truth{};
  Sophus::SE3d estimate{};
};

/// Buffers the estimate and emits pairs at each ground-truth stamp the buffer can interpolate.
class TrajectorySync
{
public:
  std::vector<AssociatedPose> push_estimate(const double stamp, const Sophus::SE3d & pose)
  {
    push_ordered(estimate_, stamp, pose);
    return drain();
  }

  std::vector<AssociatedPose> push_ground_truth(
    const double stamp, const int32_t sec, const uint32_t nanosec, const Sophus::SE3d & pose)
  {
    if (!pending_.empty() && stamp + 1.0e-9 < pending_.back().stamp) {
      return {};
    }
    pending_.push_back(Pending{stamp, sec, nanosec, pose});
    return drain();
  }

  void set_max_dt(const double max_dt) { max_dt_ = max_dt; }

private:
  struct TimedPose
  {
    double stamp{0.0};
    Sophus::SE3d pose{};
  };

  struct Pending
  {
    double stamp{0.0};
    int32_t sec{0};
    uint32_t nanosec{0};
    Sophus::SE3d pose{};
  };

  static void push_ordered(std::vector<TimedPose> & poses, const double stamp, const Sophus::SE3d & pose)
  {
    if (!poses.empty() && stamp + 1.0e-9 < poses.back().stamp) {
      return;
    }
    if (!poses.empty() && std::abs(stamp - poses.back().stamp) < 1.0e-9) {
      poses.back().pose = pose;
      return;
    }
    poses.push_back(TimedPose{stamp, pose});
  }

  bool interpolate(const double stamp, Sophus::SE3d & pose) const
  {
    if (estimate_.empty() || !(max_dt_ >= 0.0)) {
      return false;
    }
    const auto after = std::lower_bound(
      estimate_.cbegin(), estimate_.cend(), stamp,
      [](const TimedPose & sample, const double value) { return sample.stamp < value; });
    if (after == estimate_.cbegin()) {
      if (std::abs(estimate_.front().stamp - stamp) > max_dt_) {
        return false;
      }
      pose = estimate_.front().pose;
      return true;
    }
    if (after == estimate_.cend()) {
      if (std::abs(estimate_.back().stamp - stamp) > max_dt_) {
        return false;
      }
      pose = estimate_.back().pose;
      return true;
    }
    const TimedPose & before = *std::prev(after);
    const double closest = std::min(stamp - before.stamp, after->stamp - stamp);
    if (closest > max_dt_) {
      return false;
    }
    const double interval = after->stamp - before.stamp;
    const double alpha = interval > 1.0e-9 ? (stamp - before.stamp) / interval : 0.0;
    const Sophus::SE3d relative = before.pose.inverse() * after->pose;
    pose = before.pose * Sophus::SE3d::exp(alpha * relative.log());
    return true;
  }

  std::vector<AssociatedPose> drain()
  {
    std::vector<AssociatedPose> pairs;
    if (estimate_.empty()) {
      return pairs;
    }
    while (!pending_.empty()) {
      const Pending & ground_truth = pending_.front();
      if (ground_truth.stamp < estimate_.front().stamp - max_dt_) {
        pending_.pop_front();
        continue;
      }
      if (ground_truth.stamp > estimate_.back().stamp + 1.0e-9) {
        break;
      }
      Sophus::SE3d estimate;
      if (!interpolate(ground_truth.stamp, estimate)) {
        pending_.pop_front();
        continue;
      }
      pairs.push_back(AssociatedPose{
        ground_truth.sec, ground_truth.nanosec, ground_truth.pose, estimate});
      pending_.pop_front();
    }
    return pairs;
  }

  std::vector<TimedPose> estimate_;
  std::deque<Pending> pending_;
  double max_dt_{0.05};
};

}  // namespace back_odom

#endif  // BACK_ODOM__TRAJECTORY_SYNC_HPP_
