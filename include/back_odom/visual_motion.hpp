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

#ifndef BACK_ODOM__VISUAL_MOTION_HPP_
#define BACK_ODOM__VISUAL_MOTION_HPP_

#include "back_odom/kinematic_limits.hpp"

#include <sophus/se3.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace back_odom
{

struct MatchCandidate
{
  Sophus::SE3d pose{};
  double cost{std::numeric_limits<double>::infinity()};
  double iterations{0.0};
  bool passes{false};
  bool present{false};
  bool saturated{false};
};

struct RecoveryGuess
{
  Sophus::SE3d last_healthy{};
  Sophus::SE3d visual_body{};
  bool has_visual{false};
  Sophus::SO3d tail_rotation{};
  double tail_dt{0.0};
  double visual_dt{0.0};
  double held_speed{0.0};
  double max_speed{20.0};
};

struct ScaleEstimator
{
  std::vector<double> samples;
  double scale{1.0};
  bool has_scale{false};
  bool frozen{false};
  bool has_anchor{false};
  Eigen::Vector3d anchor_lidar{Eigen::Vector3d::Zero()};
  Eigen::Vector3d anchor_visual{Eigen::Vector3d::Zero()};

  void observe(
    const Eigen::Vector3d & lidar, const Eigen::Vector3d & visual, const double min_travel)
  {
    if (frozen) {
      return;
    }
    if (!has_anchor) {
      anchor_lidar = lidar;
      anchor_visual = visual;
      has_anchor = true;
      return;
    }
    const double lidar_distance = (lidar - anchor_lidar).norm();
    const double visual_distance = (visual - anchor_visual).norm();
    if (lidar_distance < min_travel || visual_distance < 1.0e-4) {
      return;
    }
    samples.push_back(lidar_distance / visual_distance);
    if (samples.size() > 11U) {
      samples.erase(samples.begin());
    }
    std::vector<double> ordered = samples;
    std::sort(ordered.begin(), ordered.end());
    const std::size_t mid = ordered.size() / 2U;
    scale = ordered.size() % 2U == 0U ? 0.5 * (ordered[mid - 1U] + ordered[mid]) : ordered[mid];
    has_scale = true;
    anchor_lidar = lidar;
    anchor_visual = visual;
  }

  void freeze() { frozen = true; }

  void resume() { frozen = false; }
};

/// Camera relative motion, scaled, expressed as a vehicle-frame increment.
[[nodiscard]] inline Sophus::SE3d scaled_body_motion(
  const Sophus::SE3d & world_from_camera_0, const Sophus::SE3d & world_from_camera_1,
  const Sophus::SE3d & body_from_camera, const double scale)
{
  Sophus::SE3d camera_relative = world_from_camera_0.inverse() * world_from_camera_1;
  camera_relative.translation() *= scale;
  return body_from_camera * camera_relative * body_from_camera.inverse();
}

[[nodiscard]] inline double signed_visual_speed(
  const Sophus::SE3d & visual_body, const double visual_dt, const double max_speed)
{
  if (!(visual_dt >= 1.0e-3)) {
    return 0.0;
  }
  const double signed_speed = std::copysign(visual_body.translation().norm() / visual_dt, visual_body.translation().x());
  const double limit = std::max(0.0, max_speed);
  return std::clamp(signed_speed, -limit, limit);
}

/// Last healthy pose, plus scaled visual motion, plus a short IMU tail at the visual speed.
[[nodiscard]] inline Sophus::SE3d make_recovery_guess(const RecoveryGuess & guess)
{
  const double speed = guess.has_visual
                         ? signed_visual_speed(guess.visual_body, guess.visual_dt, guess.max_speed)
                         : std::clamp(guess.held_speed, -std::max(0.0, guess.max_speed), std::max(0.0, guess.max_speed));
  const Sophus::SE3d motion = guess.has_visual ? guess.visual_body : Sophus::SE3d();
  const double tail_dt = std::max(0.0, guess.tail_dt);
  const Sophus::SE3d tail(guess.tail_rotation, Eigen::Vector3d(speed * tail_dt, 0.0, 0.0));
  return guess.last_healthy * motion * tail;
}

/// Lower plane residual wins. A candidate that fails its gate cannot beat one that passes.
[[nodiscard]] inline const MatchCandidate * select_match_candidate(
  const MatchCandidate & imu, const MatchCandidate * visual)
{
  const MatchCandidate * best = nullptr;
  if (imu.present && imu.passes) {
    best = &imu;
  }
  if (visual != nullptr && visual->present && visual->passes) {
    if (best == nullptr || visual->cost < best->cost) {
      best = visual;
    }
  }
  return best;
}

}  // namespace back_odom

#endif  // BACK_ODOM__VISUAL_MOTION_HPP_
