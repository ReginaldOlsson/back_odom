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

#include "back_odom/imu/imu_alignment.hpp"
#include "back_odom/imu/imu_dead_reckoning.hpp"
#include "back_odom/imu/imu_processor.hpp"

#include <Eigen/Geometry>

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace back_odom
{
namespace
{

ImuSample make_sample(
  const double stamp, const Eigen::Vector3d & linear_acceleration,
  const Eigen::Vector3d & angular_velocity)
{
  ImuSample sample;
  sample.stamp = stamp;
  sample.linear_acceleration = linear_acceleration;
  sample.angular_velocity = angular_velocity;
  return sample;
}

}  // namespace

TEST(ImuAlignment, level_gravity_keeps_body_z_parallel_to_map_z)
{
  const Sophus::SO3d orientation = compute_initial_alignment({Eigen::Vector3d(0.0, 0.0, -9.81)});
  EXPECT_TRUE(orientation.matrix().isApprox(Eigen::Matrix3d::Identity(), 1e-9));
  EXPECT_TRUE((orientation * Eigen::Vector3d::UnitZ()).isApprox(Eigen::Vector3d::UnitZ(), 1e-9));
  EXPECT_TRUE((orientation * Eigen::Vector3d(0.0, 0.0, -1.0))
                .isApprox(Eigen::Vector3d(0.0, 0.0, -1.0), 1e-9));
}

TEST(ImuAlignment, gravity_along_body_y_points_down_in_the_map)
{
  std::vector<Eigen::Vector3d> accelerations(10, Eigen::Vector3d(0.0, -9.81, 0.0));
  const Sophus::SO3d orientation = compute_initial_alignment(accelerations);
  const Eigen::Vector3d world_up = orientation * Eigen::Vector3d::UnitY();
  EXPECT_TRUE(world_up.isApprox(Eigen::Vector3d::UnitZ(), 1e-9));
}

TEST(ImuAlignment, empty_acceleration_throws)
{
  EXPECT_THROW(compute_initial_alignment({}), std::invalid_argument);
}

TEST(ImuAlignment, mean_angular_velocity_matches_constant_gyro)
{
  std::vector<ImuSample> samples;
  const Eigen::Vector3d gyro(0.01, -0.02, 0.03);
  for (int i = 0; i < 5; ++i) {
    samples.push_back(make_sample(0.01 * i, Eigen::Vector3d(0.0, 0.0, 9.81), gyro));
  }
  EXPECT_TRUE(mean_angular_velocity(samples).isApprox(gyro, 1e-12));
}

TEST(ImuAlignment, moving_gyro_is_not_stationary)
{
  std::vector<ImuSample> samples;
  samples.push_back(
    make_sample(0.0, Eigen::Vector3d(0.0, 0.0, 9.81), Eigen::Vector3d(1.0, 0.0, 0.0)));
  EXPECT_FALSE(is_stationary(samples, 0.05, 0.5));
}

TEST(ImuDeadReckoning, stationary_sample_keeps_velocity_near_zero)
{
  ImuDeadReckoning dead_reckoning(9.81);
  dead_reckoning.set_initial_orientation(Sophus::SO3d());
  dead_reckoning.set_gyro_bias(Eigen::Vector3d::Zero());
  dead_reckoning.integrate(Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, -9.81), 0.01);
  EXPECT_NEAR(dead_reckoning.velocity().norm(), 0.0, 1e-9);
  EXPECT_NEAR(dead_reckoning.position().norm(), 0.0, 1e-9);
  EXPECT_NEAR(dead_reckoning.linear_acceleration_world().z(), 0.0, 1e-9);
}

TEST(ImuDeadReckoning, accel_bias_is_removed_before_gravity)
{
  ImuDeadReckoning dead_reckoning(9.81);
  dead_reckoning.set_initial_orientation(Sophus::SO3d());
  dead_reckoning.set_accel_bias(Eigen::Vector3d(0.0, 0.0, 0.2));
  dead_reckoning.integrate(Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, -9.61), 0.1);
  EXPECT_NEAR(dead_reckoning.velocity().norm(), 0.0, 1e-9);
  EXPECT_NEAR(dead_reckoning.linear_acceleration_world().z(), 0.0, 1e-9);
}

TEST(ImuDeadReckoning, yaw_carries_forward_velocity_with_the_heading)
{
  ImuDeadReckoning dead_reckoning(9.81);
  dead_reckoning.set_initial_orientation(Sophus::SO3d());
  dead_reckoning.reset_state(Sophus::SE3d(), Eigen::Vector3d(10.0, 0.0, 0.0));
  constexpr double k_half_pi = 1.5707963267948966;
  dead_reckoning.integrate(
    Eigen::Vector3d(0.0, 0.0, k_half_pi), Eigen::Vector3d(0.0, 0.0, -9.81), 1.0);
  EXPECT_NEAR(dead_reckoning.velocity().x(), 0.0, 1e-6);
  EXPECT_NEAR(dead_reckoning.velocity().y(), 10.0, 1e-6);
  EXPECT_NEAR(dead_reckoning.velocity().z(), 0.0, 1e-6);
  EXPECT_NEAR(dead_reckoning.orientation().log().z(), k_half_pi, 1e-6);
}

TEST(ImuDeadReckoning, yaw_rate_integrates_about_z)
{
  ImuDeadReckoning dead_reckoning(9.81);
  dead_reckoning.set_initial_orientation(Sophus::SO3d());
  dead_reckoning.integrate(Eigen::Vector3d(0.0, 0.0, 0.1), Eigen::Vector3d(0.0, 0.0, -9.81), 0.1);
  const Eigen::Matrix3d expected =
    Eigen::AngleAxisd(0.01, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  EXPECT_TRUE(dead_reckoning.orientation().matrix().isApprox(expected, 1e-8));
}

TEST(ImuProcessor, positive_body_correction_adds_forward_and_yaw_about_the_vehicle)
{
  ProcessorParams params;
  params.alignment_sample_count = 5;
  params.gravity = 9.81;
  params.max_dt = 0.1;
  ImuProcessor processor(params);
  ProcessorOutput output;
  for (int i = 0; i < 5; ++i) {
    output = processor.process(
      make_sample(0.01 * i, Eigen::Vector3d(0.0, 0.0, -9.81), Eigen::Vector3d::Zero()));
  }
  ASSERT_TRUE(output.aligned);
  const double stamp = processor.latest_stamp();
  processor.reset_state(
    Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(40.0, 0.0, 0.0)), Eigen::Vector3d::Zero(), stamp);

  const Sophus::SE3d body_delta(Sophus::SO3d::rotZ(0.1), Eigen::Vector3d(1.0, 0.0, 0.0));
  processor.apply_lidar_correction(body_delta, Eigen::Vector3d(1.0, 0.0, 0.0), stamp, stamp);
  EXPECT_NEAR(processor.pose().translation().x(), 41.0, 1e-9);
  EXPECT_NEAR(processor.pose().translation().y(), 0.0, 1e-9);
  EXPECT_NEAR(processor.pose().so3().log().z(), 0.1, 1e-9);
}

TEST(ImuProcessor, negative_body_correction_moves_backward)
{
  ProcessorParams params;
  params.alignment_sample_count = 5;
  params.gravity = 9.81;
  params.max_dt = 0.1;
  ImuProcessor processor(params);
  ProcessorOutput output;
  for (int i = 0; i < 5; ++i) {
    output = processor.process(
      make_sample(0.01 * i, Eigen::Vector3d(0.0, 0.0, -9.81), Eigen::Vector3d::Zero()));
  }
  ASSERT_TRUE(output.aligned);
  const double stamp = processor.latest_stamp();
  processor.reset_state(
    Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(10.0, 0.0, 0.0)), Eigen::Vector3d(2.0, 0.0, 0.0),
    stamp);

  const Sophus::SE3d body_delta(Sophus::SO3d(), Eigen::Vector3d(-0.5, 0.0, 0.0));
  processor.apply_lidar_correction(body_delta, Eigen::Vector3d(1.5, 0.0, 0.0), stamp, stamp);
  EXPECT_NEAR(processor.pose().translation().x(), 9.5, 1e-9);
  EXPECT_NEAR(processor.pose().translation().y(), 0.0, 1e-9);
  EXPECT_NEAR(processor.velocity().x(), 1.5, 1e-9);
}

TEST(ImuProcessor, static_window_then_forward_accel)
{
  ProcessorParams params;
  params.alignment_sample_count = 20;
  params.stationary_gyro_thresh = 0.05;
  params.stationary_accel_dev_thresh = 0.5;
  params.gravity = 9.81;
  params.max_dt = 0.1;
  ImuProcessor processor(params);

  ProcessorOutput output;
  const Eigen::Vector3d gyro_bias(0.01, 0.0, 0.0);
  for (int i = 0; i < 20; ++i) {
    output = processor.process(make_sample(0.01 * i, Eigen::Vector3d(0.0, 0.0, -9.81), gyro_bias));
  }
  EXPECT_TRUE(output.aligned);
  EXPECT_TRUE(output.orientation.matrix().isApprox(Eigen::Matrix3d::Identity(), 1e-8));
  EXPECT_NEAR(output.velocity_world.norm(), 0.0, 1e-9);

  output = processor.process(make_sample(0.19, Eigen::Vector3d(0.0, 0.0, -9.81), gyro_bias));
  EXPECT_TRUE(output.orientation.matrix().isApprox(Eigen::Matrix3d::Identity(), 1e-6));

  output = processor.process(make_sample(0.29, Eigen::Vector3d(1.0, 0.0, -9.81), gyro_bias));
  EXPECT_NEAR(output.velocity_world.x(), 0.1, 1e-6);
  EXPECT_NEAR(output.velocity_world.y(), 0.0, 1e-6);
  EXPECT_NEAR(output.velocity_world.z(), 0.0, 1e-6);
  EXPECT_NEAR(output.position.x(), 0.01, 1e-6);
}

TEST(ImuFrame, left_handed_sample_follows_base_forward_after_mount_yaw)
{
  constexpr double k_pi = 3.14159265358979323846;
  const Sophus::SE3d base_from_imu(Sophus::SO3d::rotZ(k_pi), Eigen::Vector3d::Zero());
  // x-back, y-left, z-up: forward acceleration is negative x, gravity is negative z.
  const ImuSample raw =
    make_sample(1.0, Eigen::Vector3d(-1.5, 0.4, -9.81), Eigen::Vector3d(0.2, -0.3, 0.5));
  const ImuSample body = transform_imu_sample(
    to_right_handed_imu(raw), base_from_imu, Eigen::Vector3d::Zero(), 0.0, false);

  EXPECT_NEAR(body.linear_acceleration.x(), 1.5, 1e-9);
  EXPECT_NEAR(body.linear_acceleration.y(), 0.4, 1e-9);
  EXPECT_NEAR(body.linear_acceleration.z(), -9.81, 1e-9);
  EXPECT_NEAR(body.angular_velocity.x(), 0.2, 1e-9);
  EXPECT_NEAR(body.angular_velocity.y(), 0.3, 1e-9);
  EXPECT_NEAR(body.angular_velocity.z(), -0.5, 1e-9);
}

TEST(ImuFrame, rotated_mount_keeps_up_and_turns_forward)
{
  constexpr double k_pi = 3.14159265358979323846;
  const Sophus::SE3d base_from_imu(Sophus::SO3d::rotZ(k_pi), Eigen::Vector3d(0.0, 0.0, -0.25));
  const ImuSample level =
    make_sample(1.0, Eigen::Vector3d(0.0, 0.0, 9.81), Eigen::Vector3d::Zero());
  const ImuSample body =
    transform_imu_sample(level, base_from_imu, Eigen::Vector3d::Zero(), 0.0, false);

  EXPECT_NEAR(body.linear_acceleration.x(), 0.0, 1e-9);
  EXPECT_NEAR(body.linear_acceleration.y(), 0.0, 1e-9);
  EXPECT_NEAR(body.linear_acceleration.z(), 9.81, 1e-9);
  EXPECT_GT((base_from_imu.so3() * Eigen::Vector3d::UnitZ()).z(), 0.0);
  EXPECT_LT((base_from_imu.so3() * Eigen::Vector3d::UnitX()).x(), 0.0);
}

TEST(ImuFrame, lever_arm_adds_centripetal_acceleration)
{
  const Sophus::SE3d base_from_imu(Sophus::SO3d(), Eigen::Vector3d(0.0, 0.0, -0.25));
  const ImuSample spinning =
    make_sample(0.1, Eigen::Vector3d(0.0, 0.0, 9.81), Eigen::Vector3d(1.0, 0.0, 0.0));
  const ImuSample body =
    transform_imu_sample(spinning, base_from_imu, Eigen::Vector3d(1.0, 0.0, 0.0), 0.0, true);
  EXPECT_NEAR(body.linear_acceleration.z(), 9.81 - 0.25, 1e-9);
}

TEST(ImuProcessor, full_window_of_motion_stays_unaligned)
{
  ProcessorParams params;
  params.alignment_sample_count = 5;
  ImuProcessor processor(params);
  ProcessorOutput output;
  for (int i = 0; i < 5; ++i) {
    output = processor.process(
      make_sample(0.01 * i, Eigen::Vector3d(0.0, 0.0, 9.81), Eigen::Vector3d(1.0, 0.0, 0.0)));
  }
  EXPECT_FALSE(output.aligned);
  EXPECT_EQ(output.phase, ProcessorPhase::Collecting);
}

}  // namespace back_odom
