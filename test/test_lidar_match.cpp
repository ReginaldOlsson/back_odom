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

#include "back_odom/imu_dead_reckoning.hpp"
#include "back_odom/imu_processor.hpp"
#include "back_odom/kinematic_limits.hpp"
#include "back_odom/lidar_imu_matcher.hpp"
#include "back_odom/lidar_preprocess.hpp"
#include "back_odom/scan_window.hpp"
#include "back_odom/match_candidate.hpp"

#include <kiss_icp_cpp/core/Registration.hpp>
#include <kiss_icp_cpp/core/VoxelHashMap.hpp>
#include <sophus/se3.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace back_odom
{

TEST(LidarPreprocess, crop_keeps_points_inside_the_box)
{
  const std::vector<Eigen::Vector3d> points = {
    {10.0, 0.0, 5.0}, {40.0, 0.0, 0.0}, {0.0, 30.0, -20.0}, {-37.5, 25.0, 100.0}};
  const std::vector<Eigen::Vector3d> cropped = crop_lidar_box(points, 37.5, 25.0);
  ASSERT_EQ(cropped.size(), 2U);
  EXPECT_NEAR(cropped[0].z(), 5.0, 1e-9);
  EXPECT_NEAR(cropped[1].z(), 100.0, 1e-9);
}

TEST(LidarPreprocess, deskew_moves_the_scan_start_into_the_end_frame)
{
  std::vector<StampedPose> trajectory(2);
  trajectory[0].stamp = 0.0;
  trajectory[0].pose = Sophus::SE3d();
  trajectory[1].stamp = 1.0;
  trajectory[1].pose = Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(1.0, 0.0, 0.0));

  const std::vector<Eigen::Vector3d> deskewed = deskew_to_scan_end(
    {Eigen::Vector3d::Zero(), Eigen::Vector3d(2.0, 0.0, 0.0)}, {0.0, 1.0}, trajectory);
  ASSERT_EQ(deskewed.size(), 2U);
  EXPECT_NEAR(deskewed[0].x(), -1.0, 1e-6);
  EXPECT_NEAR(deskewed[0].y(), 0.0, 1e-6);
  EXPECT_NEAR(deskewed[0].z(), 0.0, 1e-6);
  EXPECT_NEAR(deskewed[1].x(), 2.0, 1e-6);
}

TEST(LidarPreprocess, lidar_yaw_is_removed_before_the_imu_motion)
{
  std::vector<StampedPose> trajectory(2);
  trajectory[0].stamp = 0.0;
  trajectory[0].pose = Sophus::SE3d();
  trajectory[1].stamp = 1.0;
  trajectory[1].pose = Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(1.0, 0.0, 0.0));
  constexpr double k_pi = 3.14159265358979323846;
  const Sophus::SE3d body_from_lidar(Sophus::SO3d::rotZ(-k_pi / 4.0), Eigen::Vector3d::Zero());

  const std::vector<Eigen::Vector3d> deskewed = deskew_to_scan_end(
    {Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0)}, {0.0, 1.0}, trajectory,
    body_from_lidar);
  ASSERT_EQ(deskewed.size(), 2U);
  EXPECT_NEAR(deskewed[0].x(), std::sqrt(0.5) - 1.0, 1e-6);
  EXPECT_NEAR(deskewed[0].y(), -std::sqrt(0.5), 1e-6);
  EXPECT_NEAR(deskewed[0].z(), 0.0, 1e-6);
  EXPECT_NEAR(deskewed[1].x(), std::sqrt(0.5), 1e-6);
  EXPECT_NEAR(deskewed[1].y(), -std::sqrt(0.5), 1e-6);
}

TEST(ImuDeadReckoning, reset_state_replaces_pose_and_velocity)
{
  ImuDeadReckoning dead_reckoning(9.81);
  const Sophus::SE3d pose(Sophus::SO3d(), Eigen::Vector3d(2.0, 3.0, 4.0));
  dead_reckoning.reset_state(pose, Eigen::Vector3d(0.5, 0.0, 0.0));
  EXPECT_TRUE(dead_reckoning.position().isApprox(Eigen::Vector3d(2.0, 3.0, 4.0)));
  EXPECT_NEAR(dead_reckoning.velocity().x(), 0.5, 1e-12);
}

TEST(PlaneAlignment, shifted_guess_on_a_corner_is_corrected)
{
  kiss_icp::VoxelHashMap map(0.5, 1000.0, 20);
  std::vector<Eigen::Vector3d> cloud;
  for (int i = 0; i <= 16; ++i) {
    for (int j = 0; j <= 16; ++j) {
      const double a = 0.25 * static_cast<double>(i);
      const double b = 0.25 * static_cast<double>(j);
      cloud.emplace_back(a, b, 0.0);
      cloud.emplace_back(0.0, a, b);
      cloud.emplace_back(a, 0.0, b);
    }
  }
  map.Update(cloud, Sophus::SE3d());

  kiss_icp::Registration registration(40, 1.0e-4, 1);
  const Sophus::SE3d guess(Sophus::SO3d(), Eigen::Vector3d(0.2, 0.15, 0.1));
  double iterations = 0.0;
  const kiss_icp::PlaneAlignResult result =
    registration.AlignPointsToPlane(cloud, map, guess, 1.5, 0.5, iterations);
  EXPECT_NEAR(result.pose.translation().x(), 0.0, 0.05);
  EXPECT_NEAR(result.pose.translation().y(), 0.0, 0.05);
  EXPECT_NEAR(result.pose.translation().z(), 0.0, 0.05);
  EXPECT_GT(iterations, 0.0);
}

TEST(PlaneAlignment, flat_ground_keeps_the_lateral_imu_prior)
{
  kiss_icp::VoxelHashMap map(0.5, 1000.0, 20);
  std::vector<Eigen::Vector3d> cloud;
  for (int ix = -8; ix <= 8; ++ix) {
    for (int iy = -8; iy <= 8; ++iy) {
      cloud.emplace_back(0.5 * ix, 0.5 * iy, 0.0);
    }
  }
  map.Update(cloud, Sophus::SE3d());

  kiss_icp::Registration registration(40, 1.0e-4, 1);
  const Sophus::SE3d guess(Sophus::SO3d(), Eigen::Vector3d(0.4, 0.0, 0.15));
  double iterations = 0.0;
  const kiss_icp::PlaneAlignResult result =
    registration.AlignPointsToPlane(cloud, map, guess, 1.5, 0.5, iterations);
  EXPECT_NEAR(result.pose.translation().x(), 0.4, 0.05);
  EXPECT_NEAR(result.pose.translation().y(), 0.0, 0.05);
  EXPECT_NEAR(result.pose.translation().z(), 0.0, 0.05);
}

LidarScan corner_scan(const double stamp)
{
  LidarScan scan;
  scan.stamp = stamp;
  for (int i = 0; i <= 16; ++i) {
    for (int j = 0; j <= 16; ++j) {
      const double a = 0.25 * static_cast<double>(i);
      const double b = 0.25 * static_cast<double>(j);
      scan.points.emplace_back(a, b, 0.0);
      scan.points.emplace_back(0.0, a, b);
      scan.points.emplace_back(a, 0.0, b);
      scan.timestamps.insert(scan.timestamps.end(), 3, stamp);
    }
  }
  return scan;
}

ImuProcessor aligned_imu()
{
  ProcessorParams params;
  params.alignment_sample_count = 5;
  ImuProcessor imu(params);
  ProcessorOutput output;
  for (int i = 0; i < 5; ++i) {
    output =
      imu.process(ImuSample{0.01 * i, Eigen::Vector3d(0.0, 0.0, -9.81), Eigen::Vector3d::Zero()});
  }
  if (!output.aligned) {
    throw std::runtime_error("stationary window did not align");
  }
  return imu;
}

LidarMatchParams match_params()
{
  LidarMatchParams params;
  params.voxel_size = 0.25;
  params.backward_match_stride = 3;
  params.max_correspondence_distance = 2.0;
  return params;
}

TEST(BackwardMatch, drifted_imu_is_corrected_when_the_next_scan_is_aligned)
{
  ImuProcessor imu = aligned_imu();
  ASSERT_TRUE(imu.aligned());
  LidarImuMatcher matcher(match_params());
  const LidarScan scan = corner_scan(imu.latest_stamp());
  ASSERT_TRUE(matcher.on_scan(scan, imu).first_scan);

  imu.reset_state(
    Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.3, 0.0, 0.0)), Eigen::Vector3d::Zero(),
    imu.latest_stamp());
  const MatchResult matched = matcher.on_scan(scan, imu);
  ASSERT_TRUE(matched.applied);
  EXPECT_NEAR(imu.pose().translation().x(), 0.0, 0.15);
  EXPECT_NEAR(imu.pose().translation().y(), 0.0, 0.15);
  EXPECT_NEAR(imu.pose().translation().z(), 0.0, 0.15);
}

TEST(BackwardMatch, imu_motion_between_scans_is_kept)
{
  ImuProcessor imu = aligned_imu();
  ASSERT_TRUE(imu.aligned());
  LidarImuMatcher matcher(match_params());
  const LidarScan scan = corner_scan(imu.latest_stamp());
  ASSERT_TRUE(matcher.on_scan(scan, imu).first_scan);

  const Sophus::SE3d moved(Sophus::SO3d(), Eigen::Vector3d(1.0, 0.0, 0.0));
  imu.reset_state(moved, Eigen::Vector3d::Zero(), imu.latest_stamp());
  MatchResult backward;
  for (int step = 0; step < 3; ++step) {
    backward = matcher.on_imu(imu);
  }
  EXPECT_NEAR(imu.pose().translation().x(), 1.0, 0.05);
  EXPECT_LT(backward.correction.translation().norm(), 0.05);
}

TEST(BackwardMatch, older_scan_does_not_rewind_a_later_imu_pose)
{
  ImuProcessor imu = aligned_imu();
  ASSERT_TRUE(imu.aligned());
  LidarImuMatcher matcher(match_params());
  const LidarScan scan = corner_scan(imu.latest_stamp());
  ASSERT_TRUE(matcher.on_scan(scan, imu).first_scan);

  const double later = imu.latest_stamp() + 1.0;
  const ProcessorOutput stepped =
    imu.process(ImuSample{later, Eigen::Vector3d(0.0, 0.0, -9.81), Eigen::Vector3d::Zero()});
  ASSERT_TRUE(stepped.aligned);
  imu.reset_state(
    Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(1.0, 0.0, 0.0)), Eigen::Vector3d::Zero(), later);

  const MatchResult matched = matcher.on_scan(scan, imu);
  ASSERT_TRUE(matched.applied);
  EXPECT_NEAR(imu.pose().translation().x(), 1.0, 0.15);
}

TEST(InertialCorrection, curve_keeps_the_tangent_instead_of_the_chord)
{
  constexpr double yaw = 0.2;
  constexpr double speed = 10.0;
  constexpr double dt = 1.0;
  const double radius = speed / yaw;
  const Eigen::Vector3d position(radius * std::sin(yaw), radius * (1.0 - std::cos(yaw)), 0.0);
  const Sophus::SE3d corrected(Sophus::SO3d::rotZ(yaw), position);
  const Eigen::Vector3d tangent = speed * Eigen::Vector3d(std::cos(yaw), std::sin(yaw), 0.0);

  const InertialCorrection correction = inertial_correction_from_match(
    Sophus::SE3d(), Sophus::SO3d(), corrected, tangent, Eigen::Vector3d::Zero(), dt, 0.1, 0.02,
    0.5);

  const Eigen::Vector3d forward = corrected.so3() * Eigen::Vector3d::UnitX();
  const double imu_forward = tangent.dot(forward);
  const double lidar_forward = position.dot(forward) / dt;
  const double blended = 0.5 * imu_forward + 0.5 * lidar_forward;
  const Eigen::Vector3d expected = blended * forward;
  EXPECT_NEAR(correction.velocity_world.x(), expected.x(), 1e-6);
  EXPECT_NEAR(correction.velocity_world.y(), expected.y(), 1e-6);
  EXPECT_GT(correction.velocity_world.y(), 1.5);
  EXPECT_NEAR(correction.accel_bias_delta.y(), 0.0, 1e-12);
  EXPECT_GT(correction.accel_bias_delta.x(), 0.0);
  EXPECT_NEAR(correction.gyro_bias_delta.norm(), 0.0, 1e-12);
}

TEST(InertialCorrection, off_heading_velocity_is_pulled_onto_the_matched_forward_axis)
{
  constexpr double k_pi = 3.14159265358979323846;
  constexpr double speed = 10.0;
  constexpr double angle = k_pi / 6.0;
  const Sophus::SE3d corrected(Sophus::SO3d(), Eigen::Vector3d(10.0, 0.0, 0.0));
  const Eigen::Vector3d velocity(speed * std::cos(angle), speed * std::sin(angle), 0.0);
  const InertialCorrection correction = inertial_correction_from_match(
    Sophus::SE3d(), Sophus::SO3d(), corrected, velocity, Eigen::Vector3d::Zero(), 1.0, 0.1, 0.02,
    0.5);
  const double blended = 0.5 * velocity.x() + 0.5 * 10.0;
  EXPECT_NEAR(correction.velocity_world.x(), blended, 1e-9);
  EXPECT_NEAR(correction.velocity_world.y(), 0.0, 1e-12);
  EXPECT_NEAR(correction.velocity_world.z(), 0.0, 1e-12);
  EXPECT_NEAR(correction.accel_bias_delta.y(), 0.0, 1e-12);
}

TEST(InertialCorrection, lateral_displacement_does_not_enter_velocity_or_accel_bias)
{
  const Sophus::SE3d corrected(Sophus::SO3d(), Eigen::Vector3d(0.0, 1.0, 0.0));
  const InertialCorrection correction = inertial_correction_from_match(
    Sophus::SE3d(), Sophus::SO3d(), corrected, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
    1.0, 0.1, 0.02, 0.5);
  EXPECT_NEAR(correction.velocity_world.norm(), 0.0, 1e-12);
  EXPECT_NEAR(correction.accel_bias_delta.y(), 0.0, 1e-12);
}

TEST(InertialCorrection, extra_positive_yaw_increases_gyro_bias)
{
  const Sophus::SE3d vehicle_delta(Sophus::SO3d::rotZ(-0.002), Eigen::Vector3d::Zero());
  const InertialCorrection correction = inertial_correction_from_match(
    vehicle_delta, Sophus::SO3d(), Sophus::SE3d(), Eigen::Vector3d(1.0, 0.0, 0.0),
    Eigen::Vector3d::Zero(), 0.2, 0.1, 0.02, 0.5);
  EXPECT_NEAR(correction.gyro_bias_delta.z(), 0.001, 1e-9);
  EXPECT_NEAR(correction.gyro_bias_delta.x(), 0.0, 1e-12);
  EXPECT_NEAR(correction.gyro_bias_delta.y(), 0.0, 1e-12);
}

TEST(LidarMatch, yaw_correction_far_from_the_origin_updates_gyro_bias)
{
  ImuProcessor imu = aligned_imu();
  LidarImuMatcher matcher(match_params());
  imu.reset_state(
    Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(40.0, 0.0, 0.0)), Eigen::Vector3d::Zero(),
    imu.latest_stamp());
  ASSERT_TRUE(matcher.on_scan(corner_scan(imu.latest_stamp()), imu).first_scan);

  const double later = imu.latest_stamp() + 0.2;
  const ProcessorOutput stepped =
    imu.process(ImuSample{later, Eigen::Vector3d(0.0, 0.0, -9.81), Eigen::Vector3d::Zero()});
  ASSERT_TRUE(stepped.aligned);
  constexpr double k_yaw = 0.05;
  imu.reset_state(
    Sophus::SE3d(Sophus::SO3d::rotZ(k_yaw), Eigen::Vector3d(40.0, 0.0, 0.0)),
    Eigen::Vector3d::Zero(), later);

  const MatchResult matched = matcher.on_scan(corner_scan(later), imu);
  ASSERT_TRUE(matched.applied);
  EXPECT_GT(matched.correction.translation().norm(), 1.0);
  EXPECT_LT(matched.translation_error, 0.5);
  EXPECT_NEAR(imu.pose().translation().x(), 40.0, 0.25);
  EXPECT_NEAR(imu.pose().translation().y(), 0.0, 0.25);
  EXPECT_LT(imu.pose().so3().log().norm(), 0.03);
  EXPECT_GT(imu.gyro_bias().z(), 0.0);
  EXPECT_NEAR(imu.accel_bias().y(), 0.0, 1e-9);
}

TEST(ScanWindow, shifted_corner_is_pulled_back_to_the_frozen_map)
{
  const LidarScan scan = corner_scan(0.0);
  kiss_icp::VoxelHashMap frozen(0.25, 1.0e6, 20);
  frozen.Update(scan.points, Sophus::SE3d());
  std::vector<Sophus::SE3d> poses{Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.3, 0.0, 0.0))};

  optimize_scan_window(
    poses, {scan.points}, {scan.points}, {}, frozen, 2.0, 0.5, 15, 1.0e-4, 0.0, 0.0, 0.25, 20);

  EXPECT_NEAR(poses[0].translation().x(), 0.0, 0.15);
  EXPECT_NEAR(poses[0].translation().y(), 0.0, 0.05);
  EXPECT_NEAR(poses[0].translation().z(), 0.0, 0.05);
}

TEST(ScanWindow, imu_edge_pulls_an_empty_scan_toward_its_neighbor)
{
  const LidarScan scan = corner_scan(0.0);
  kiss_icp::VoxelHashMap frozen(0.25, 1.0e6, 20);
  frozen.Update(scan.points, Sophus::SE3d());
  std::vector<Sophus::SE3d> poses{
    Sophus::SE3d(), Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.4, 0.0, 0.0))};
  const std::vector<Eigen::Vector3d> empty;

  optimize_scan_window(
    poses, {scan.points, empty}, {scan.points, empty}, {Sophus::SE3d()}, frozen, 2.0, 0.5, 15,
    1.0e-4, 0.0, 0.0, 0.25, 20);

  EXPECT_NEAR(poses[0].translation().x(), 0.0, 0.15);
  EXPECT_NEAR(poses[1].translation().x(), 0.0, 0.15);
  EXPECT_LT(poses[1].translation().x(), 0.2);
}

TEST(VehicleLimits, fast_translation_or_yaw_is_rejected)
{
  const VehicleLimits limits;
  const Sophus::SE3d fast(Sophus::SO3d(), Eigen::Vector3d(30.0, 0.0, 0.0));
  EXPECT_FALSE(step_within_vehicle_limits(fast, 1.0, 0.0, limits));
  const Sophus::SE3d spin(Sophus::SO3d::rotZ(2.0), Eigen::Vector3d::Zero());
  EXPECT_FALSE(step_within_vehicle_limits(spin, 0.5, 0.0, limits));
  const Sophus::SE3d mild(Sophus::SO3d(), Eigen::Vector3d(1.0, 0.0, 0.0));
  EXPECT_TRUE(step_within_vehicle_limits(mild, 0.1, 0.0, limits));
}

TEST(VehicleLimits, fast_scan_step_does_not_change_bias)
{
  ImuProcessor imu = aligned_imu();
  LidarImuMatcher matcher(match_params());
  ASSERT_TRUE(matcher.on_scan(corner_scan(imu.latest_stamp()), imu).first_scan);
  const double later = imu.latest_stamp() + 0.5;
  imu.reset_state(
    Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(30.0, 0.0, 0.0)), Eigen::Vector3d::Zero(), later);
  const MatchResult matched = matcher.on_scan(corner_scan(later), imu);
  EXPECT_FALSE(matched.applied);
  EXPECT_NEAR(imu.gyro_bias().norm(), 0.0, 1.0e-9);
  EXPECT_NEAR(imu.accel_bias().norm(), 0.0, 1.0e-9);
}

TEST(LocalizationHealth, rejected_scans_keep_matching_and_do_not_rewrite_speed)
{
  ImuProcessor imu = aligned_imu();
  LidarImuMatcher matcher(match_params());
  ASSERT_TRUE(matcher.on_scan(corner_scan(imu.latest_stamp()), imu).first_scan);
  for (int step = 0; step < 5; ++step) {
    const double stamp = imu.latest_stamp() + 0.5;
    imu.reset_state(
      Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(1.2, 0.0, 0.0)), Eigen::Vector3d(40.0, 0.0, 0.0),
      stamp);
    const MatchResult matched = matcher.on_scan(corner_scan(stamp), imu);
    EXPECT_FALSE(matched.applied);
  }
  EXPECT_EQ(matcher.health(), LocalizationHealth::Healthy);
  EXPECT_NEAR(imu.velocity().x(), 40.0, 1.0e-6);
}

TEST(ImuIntegration, body_speed_stays_inside_the_vehicle_limit)
{
  ProcessorParams params;
  params.alignment_sample_count = 5;
  params.max_speed = 20.0;
  params.max_acceleration = 5.0;
  ImuProcessor imu(params);
  for (int i = 0; i < 5; ++i) {
    const ProcessorOutput aligned =
      imu.process(ImuSample{0.01 * i, Eigen::Vector3d(0.0, 0.0, -9.81), Eigen::Vector3d::Zero()});
    if (i + 1 == 5) {
      ASSERT_TRUE(aligned.aligned);
    }
  }
  ASSERT_TRUE(imu.aligned());
  imu.reset_state(imu.pose(), Eigen::Vector3d(100.0, 0.0, 0.0), imu.latest_stamp());
  const ProcessorOutput output = imu.process(
    ImuSample{imu.latest_stamp() + 0.05, Eigen::Vector3d(0.0, 0.0, -9.81), Eigen::Vector3d::Zero()});
  EXPECT_TRUE(output.speed_clamped);
  EXPECT_LE(std::abs((imu.pose().so3().inverse() * imu.velocity()).x()), 20.0);
}



TEST(ScanPoses, pose_outside_the_crop_is_not_part_of_the_visible_cloud)
{
  const Sophus::SE3d latest;
  const Sophus::SE3d inside(Sophus::SO3d(), Eigen::Vector3d(-10.0, 2.0, 0.0));
  const Sophus::SE3d outside(Sophus::SO3d(), Eigen::Vector3d(-60.0, 0.0, 0.0));
  EXPECT_TRUE(scan_pose_inside_visible_cloud(latest, inside, 50.0, 50.0));
  EXPECT_FALSE(scan_pose_inside_visible_cloud(latest, outside, 50.0, 50.0));
}

TEST(ScanPoses, each_accepted_scan_keeps_its_lidar_pose)
{
  ImuProcessor imu = aligned_imu();
  LidarImuMatcher matcher(match_params());
  ASSERT_TRUE(matcher.on_scan(corner_scan(imu.latest_stamp()), imu).first_scan);
  const std::size_t first_points = matcher.visible_scan_point_count();
  ASSERT_GT(first_points, 0U);
  imu.reset_state(
    Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.3, 0.0, 0.0)), Eigen::Vector3d::Zero(),
    imu.latest_stamp());
  ASSERT_TRUE(matcher.on_scan(corner_scan(imu.latest_stamp()), imu).applied);
  const std::vector<StampedPose> poses = matcher.visible_scan_poses();
  ASSERT_EQ(poses.size(), 2U);
  EXPECT_GT(matcher.visible_scan_point_count(), first_points);
  EXPECT_NEAR(poses.front().pose.translation().x(), 0.0, 0.15);
  EXPECT_NEAR(poses.back().pose.translation().x(), 0.0, 0.15);
}

TEST(ScanPoses, map_points_carry_scan_id_and_collective_pass_count)
{
  ImuProcessor imu = aligned_imu();
  LidarImuMatcher matcher(match_params());
  ASSERT_TRUE(matcher.on_scan(corner_scan(imu.latest_stamp()), imu).first_scan);
  const std::vector<LocalMapPoint> first = matcher.annotated_local_map();
  ASSERT_FALSE(first.empty());
  EXPECT_FLOAT_EQ(first.front().scan_id, 0.0F);
  EXPECT_FLOAT_EQ(first.front().collective_passes, 0.0F);
  EXPECT_FLOAT_EQ(first.front().intensity, 30.0F);
  imu.reset_state(
    Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.3, 0.0, 0.0)), Eigen::Vector3d::Zero(),
    imu.latest_stamp());
  ASSERT_TRUE(matcher.on_scan(corner_scan(imu.latest_stamp()), imu).applied);
  bool saw_first = false;
  bool saw_second = false;
  for (const LocalMapPoint & point : matcher.annotated_local_map()) {
    saw_first = saw_first || point.scan_id == 0.0F;
    saw_second = saw_second || point.scan_id == 1.0F;
    EXPECT_FLOAT_EQ(point.collective_passes, 0.0F);
    EXPECT_FLOAT_EQ(point.intensity, 30.0F);
  }
  EXPECT_TRUE(saw_first);
  EXPECT_TRUE(saw_second);
}

TEST(ScanPoses, the_scan_list_drops_the_oldest_past_the_cap)
{
  ImuProcessor imu = aligned_imu();
  LidarMatchParams params = match_params();
  params.max_visible_scans = 2;
  LidarImuMatcher matcher(params);
  ASSERT_TRUE(matcher.on_scan(corner_scan(imu.latest_stamp()), imu).first_scan);
  for (int step = 0; step < 2; ++step) {
    imu.reset_state(
      Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.3, 0.0, 0.0)), Eigen::Vector3d::Zero(),
      imu.latest_stamp());
    ASSERT_TRUE(matcher.on_scan(corner_scan(imu.latest_stamp()), imu).applied);
  }
  EXPECT_EQ(matcher.visible_scan_poses().size(), 2U);
}

TEST(ScanHorizon, older_scan_moves_toward_the_pinned_newest_pose)
{
  const std::vector<Eigen::Vector3d> cloud = corner_scan(0.0).points;
  std::vector<Sophus::SE3d> poses{
    Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.04, 0.0, 0.0)), Sophus::SE3d()};
  const std::vector<Sophus::SE3d> priors = poses;
  const std::vector<Sophus::SE3d> relatives{Sophus::SE3d(), Sophus::SE3d()};
  const std::vector<const std::vector<Eigen::Vector3d> *> points{&cloud, &cloud};
  ScanHorizonRefine params;
  params.voxel_size = 0.5;
  params.outer_iterations = 3;
  params.inner_iterations = 4;
  params.max_query_points = 2000;
  refine_scan_horizon(poses, points, priors, relatives, params, {});
  EXPECT_NEAR(poses[1].translation().norm(), 0.0, 1.0e-12);
  EXPECT_LT(poses[0].translation().x(), 0.035);
  EXPECT_GT(poses[0].translation().x(), 0.0);
  EXPECT_NEAR(poses[0].translation().y(), 0.0, 0.02);
  EXPECT_NEAR(poses[0].translation().z(), 0.0, 0.02);
}

TEST(ScanHorizon, settled_scan_stays_put_while_its_points_remain_a_reference)
{
  const std::vector<Eigen::Vector3d> cloud = corner_scan(0.0).points;
  const Sophus::SE3d shifted(Sophus::SO3d(), Eigen::Vector3d(0.04, 0.0, 0.0));
  std::vector<Sophus::SE3d> poses{shifted, Sophus::SE3d()};
  const std::vector<Sophus::SE3d> priors = poses;
  const std::vector<Sophus::SE3d> relatives{Sophus::SE3d(), Sophus::SE3d()};
  const std::vector<const std::vector<Eigen::Vector3d> *> points{&cloud, &cloud};
  ScanHorizonRefine params;
  params.voxel_size = 0.5;
  params.outer_iterations = 3;
  params.inner_iterations = 4;
  refine_scan_horizon(poses, points, priors, relatives, params, {0, 0});
  EXPECT_NEAR(poses[0].translation().x(), 0.04, 1.0e-12);
  EXPECT_NEAR(poses[1].translation().norm(), 0.0, 1.0e-12);
}

TEST(ScanHorizon, current_pose_takes_the_error_against_the_settled_scan)
{
  const std::vector<Eigen::Vector3d> cloud = corner_scan(0.0).points;
  std::vector<Sophus::SE3d> poses{
    Sophus::SE3d(), Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.04, 0.0, 0.0))};
  const std::vector<Sophus::SE3d> priors = poses;
  const std::vector<Sophus::SE3d> relatives{Sophus::SE3d(), Sophus::SE3d()};
  const std::vector<const std::vector<Eigen::Vector3d> *> points{&cloud, &cloud};
  ScanHorizonRefine params;
  params.voxel_size = 0.5;
  params.outer_iterations = 3;
  params.inner_iterations = 4;
  params.max_query_points = 2000;
  refine_scan_horizon(poses, points, priors, relatives, params, {0, 1});
  EXPECT_NEAR(poses[0].translation().norm(), 0.0, 1.0e-12);
  EXPECT_LT(poses[1].translation().x(), 0.02);
  EXPECT_GT(poses[1].translation().x(), -0.01);
}

TEST(LidarMatch, forward_slide_is_limited_and_lateral_shift_is_kept)
{
  const Sophus::SO3d yaw = Sophus::SO3d::rotZ(1.5707963267948966);
  const Sophus::SE3d predicted(yaw, Eigen::Vector3d(4.0, -2.0, 0.5));
  const Sophus::SE3d aligned =
    predicted * Sophus::SE3d(Sophus::SO3d::rotZ(0.04), Eigen::Vector3d(0.8, 0.3, -0.05));
  const Sophus::SE3d limited = limit_longitudinal_correction(predicted, aligned, 0.15);
  const Sophus::SE3d body = predicted.inverse() * limited;
  EXPECT_NEAR(body.translation().x(), 0.15, 1.0e-9);
  EXPECT_NEAR(body.translation().y(), 0.3, 1.0e-9);
  EXPECT_NEAR(body.translation().z(), -0.05, 1.0e-9);
  EXPECT_NEAR(body.so3().log().norm(), 0.04, 1.0e-9);
}


LidarScan street_scan(const double stamp)
{
  LidarScan scan;
  scan.stamp = stamp;
  scan.points.reserve(90000);
  scan.timestamps.reserve(90000);
  const auto add = [&](const double x, const double y, const double z) {
    scan.points.emplace_back(x, y, z);
    scan.timestamps.push_back(stamp);
  };
  for (double x = -15.0; x <= 25.0; x += 0.15) {
    for (double y = -12.0; y <= 12.0; y += 0.15) {
      add(x, y, 0.0);
    }
  }
  for (double x = -15.0; x <= 25.0; x += 0.2) {
    for (double z = 0.2; z <= 3.0; z += 0.2) {
      add(x, -8.0, z);
      add(x, 8.0, z);
    }
  }
  return scan;
}

TEST(LidarMatch, stage_times_without_coarse_guess)
{
  LidarMatchParams params;
  params.voxel_size = 0.5;
  params.crop_longitudinal = 50.0;
  params.crop_lateral = 50.0;
  params.max_iterations = 50;
  params.max_points_per_voxel = 7;
  params.refine_window = true;
  params.refine_min_travel = 2.0;
  params.convergence_criterion = 1.0e-4;
  LidarImuMatcher matcher(params);
  ImuProcessor imu = aligned_imu();
  const LidarScan scan = street_scan(imu.latest_stamp());
  std::vector<MatchResult> results;
  results.reserve(8);
  for (int step = 0; step < 8; ++step) {
    const double stamp = imu.latest_stamp() + 0.1;
    LidarScan moved = scan;
    moved.stamp = stamp;
    std::fill(moved.timestamps.begin(), moved.timestamps.end(), stamp);
    imu.reset_state(
      Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(1.5 * step, 0.0, 0.0)), Eigen::Vector3d(10.0, 0.0, 0.0),
      stamp);
    const auto started = std::chrono::steady_clock::now();
    const MatchResult matched = matcher.on_scan(moved, imu);
    const double wall_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    std::cout << "scan " << step << " wall " << wall_ms << " deskew " << matched.deskew_ms << " voxel "
              << matched.voxel_ms << " align " << matched.align_ms << " cost " << matched.cost_ms
              << " refine " << matched.refine_ms << " map " << matched.map_ms << " iters "
              << matched.iterations << " applied " << matched.applied << " points "
              << moved.points.size() << std::endl;
    results.push_back(matched);
  }
  const auto median = [](std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
  };
  std::vector<double> deskew;
  std::vector<double> voxel;
  std::vector<double> align;
  std::vector<double> refine;
  std::vector<double> map;
  for (const MatchResult & result : results) {
    deskew.push_back(result.deskew_ms);
    voxel.push_back(result.voxel_ms);
    align.push_back(result.align_ms);
    refine.push_back(result.refine_ms);
    map.push_back(result.map_ms);
  }
  std::cout << "median ms deskew " << median(deskew) << " voxel " << median(voxel) << " align "
            << median(align) << " refine " << median(refine) << " map " << median(map) << std::endl;
}

}  // namespace back_odom
