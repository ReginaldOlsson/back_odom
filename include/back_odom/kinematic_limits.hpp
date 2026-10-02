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

#ifndef BACK_ODOM__KINEMATIC_LIMITS_HPP_
#define BACK_ODOM__KINEMATIC_LIMITS_HPP_

#include <sophus/se3.hpp>

#include <algorithm>
#include <cmath>

namespace back_odom
{

struct VehicleLimits
{
  double max_speed{20.0};
  double max_acceleration{5.0};
  double max_yaw_rate{1.0};
  int max_match_rejects{5};
};

enum class LocalizationHealth { Healthy, Degraded, Diverged };

enum class HealthEvent
{
  Accepted,
  Saturated,
  Rejected,
  KinematicViolation,
  SpeedClamped
};

struct HealthState
{
  LocalizationHealth health{LocalizationHealth::Healthy};
  int reject_streak{0};
  bool update_bias{true};
  bool restore_speed{false};
};

/// Body-frame speed with lateral motion removed and each axis inside [-max_speed, max_speed].
[[nodiscard]] inline Eigen::Vector3d clamp_body_velocity(
  const Eigen::Vector3d & velocity_body, const double max_speed)
{
  Eigen::Vector3d clamped = velocity_body;
  clamped.y() = 0.0;
  const double limit = std::max(0.0, max_speed);
  clamped.x() = std::clamp(clamped.x(), -limit, limit);
  clamped.z() = std::clamp(clamped.z(), -limit, limit);
  return clamped;
}

/// Limits the change along the vehicle axes, then clamps the result to max_speed.
[[nodiscard]] inline Eigen::Vector3d limit_body_velocity(
  const Eigen::Vector3d & velocity_before, const Eigen::Vector3d & velocity_after, const double dt,
  const VehicleLimits & limits)
{
  const Eigen::Vector3d before = clamp_body_velocity(velocity_before, limits.max_speed);
  Eigen::Vector3d after = clamp_body_velocity(velocity_after, limits.max_speed);
  if (dt > 1.0e-6) {
    const double max_delta = std::max(0.0, limits.max_acceleration) * dt;
    after.x() = std::clamp(after.x(), before.x() - max_delta, before.x() + max_delta);
    after.z() = std::clamp(after.z(), before.z() - max_delta, before.z() + max_delta);
  }
  return clamp_body_velocity(after, limits.max_speed);
}

[[nodiscard]] inline bool clamp_world_velocity(
  const Sophus::SO3d & orientation, Eigen::Vector3d & velocity_world, const double max_speed)
{
  const Eigen::Vector3d world = orientation * clamp_body_velocity(orientation.inverse() * velocity_world, max_speed);
  const bool changed = (world - velocity_world).norm() > 1.0e-9;
  velocity_world = world;
  return changed;
}

/// Reject a committed vehicle step that is faster or spinning harder than the vehicle allows.
/// A very small dt skips the rate check so a same-stamp pose fix still uses the displacement gate.
[[nodiscard]] inline bool step_within_vehicle_limits(
  const Sophus::SE3d & body_delta, const double dt, const double /*previous_speed*/,
  const VehicleLimits & limits)
{
  if (!body_delta.translation().allFinite() || !body_delta.so3().log().allFinite()) {
    return false;
  }
  if (!(dt >= 1.0e-3)) {
    return true;
  }
  const double speed = body_delta.translation().norm() / dt;
  const double yaw_rate = std::abs(body_delta.so3().log().z()) / dt;
  return speed <= limits.max_speed && yaw_rate <= limits.max_yaw_rate;
}

[[nodiscard]] inline HealthState advance_localization_health(
  const HealthState & state, const HealthEvent event, const int max_rejects)
{
  HealthState next = state;
  next.restore_speed = false;
  next.update_bias = true;
  const int reject_limit = std::max(1, max_rejects);

  if (event == HealthEvent::KinematicViolation || event == HealthEvent::SpeedClamped) {
    next.health = LocalizationHealth::Diverged;
    next.update_bias = false;
    next.restore_speed = true;
    next.reject_streak = state.reject_streak + 1;
    return next;
  }
  if (event == HealthEvent::Rejected) {
    next.reject_streak = state.reject_streak + 1;
    next.update_bias = false;
    if (next.reject_streak >= reject_limit || state.health == LocalizationHealth::Diverged) {
      next.health = LocalizationHealth::Diverged;
      next.restore_speed = true;
    } else {
      next.health = LocalizationHealth::Degraded;
    }
    return next;
  }
  if (event == HealthEvent::Saturated) {
    next.update_bias = false;
    if (state.health == LocalizationHealth::Diverged) {
      next.health = LocalizationHealth::Diverged;
      next.restore_speed = true;
      return next;
    }
    next.health = LocalizationHealth::Degraded;
    return next;
  }

  next.health = LocalizationHealth::Healthy;
  next.reject_streak = 0;
  next.update_bias = true;
  next.restore_speed = false;
  return next;
}

[[nodiscard]] inline const char * health_name(const LocalizationHealth health)
{
  switch (health) {
    case LocalizationHealth::Healthy:
      return "healthy";
    case LocalizationHealth::Degraded:
      return "degraded";
    case LocalizationHealth::Diverged:
      return "diverged";
  }
  return "healthy";
}

}  // namespace back_odom

#endif  // BACK_ODOM__KINEMATIC_LIMITS_HPP_
