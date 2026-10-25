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

#include "back_odom/imu/imu_dead_reckoning.hpp"
#include "back_odom/imu/imu_processor.hpp"
#include "back_odom/imu/kinematic_limits.hpp"
#include "back_odom/lidar/lidar_imu_matcher.hpp"
#include "back_odom/lidar/lidar_preprocess.hpp"
#include "back_odom/lidar/match_candidate.hpp"
#include "back_odom/lidar/plane_class.hpp"
#include "back_odom/lidar/ndt_health.hpp"
#include "back_odom/lidar/plane_icp.hpp"
#include "back_odom/lidar/scan_window.hpp"

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

TEST(LidarPreprocess, ground_plane_on_a_flat_cloud_points_up)
{
  std::vector<Eigen::Vector3d> cloud;
  for (int ix = 0; ix < 20; ++ix) {
    for (int iy = 0; iy < 20; ++iy) {
      cloud.emplace_back(0.3 * ix, 0.3 * iy, 0.0);
    }
  }
  const GroundPlaneFit fit = fit_ground_plane(cloud, 0.15, 32, 1);
  EXPECT_TRUE(fit.valid);
  EXPECT_GT(fit.inlier_ratio, 0.9);
  EXPECT_NEAR(fit.normal.z(), 1.0, 1e-6);
}

TEST(LidarPreprocess, pose_buffer_covers_a_dense_sweep_and_rejects_a_hole)
{
  std::vector<StampedPose> dense(5);
  for (std::size_t index = 0; index < dense.size(); ++index) {
    dense[index].stamp = 0.01 * static_cast<double>(index);
  }
  EXPECT_TRUE(pose_buffer_covers(dense, 0.0, 0.04, 0.05));

  std::vector<StampedPose> hole = dense;
  hole[2].stamp = 0.2;
  hole[3].stamp = 0.21;
  hole[4].stamp = 0.22;
  EXPECT_FALSE(pose_buffer_covers(hole, 0.0, 0.22, 0.05));
}

TEST(LidarPreprocess, deskew_uses_the_relative_pose_when_the_buffer_covers_the_sweep)
{
  std::vector<StampedPose> trajectory(101);
  for (std::size_t index = 0; index < trajectory.size(); ++index) {
    trajectory[index].stamp = 0.01 * static_cast<double>(index);
    trajectory[index].pose =
      Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(trajectory[index].stamp, 0.0, 0.0));
  }
  ASSERT_TRUE(pose_buffer_covers(trajectory, 0.0, 1.0, 0.05));
  const std::vector<Eigen::Vector3d> deskewed = deskew_to_scan_end(
    {Eigen::Vector3d::Zero(), Eigen::Vector3d(1.0, 0.0, 0.0)}, {0.0, 1.0}, trajectory);
  ASSERT_EQ(deskewed.size(), 2U);
  EXPECT_NEAR(deskewed[0].x(), -1.0, 1e-6);
  EXPECT_NEAR(deskewed[1].x(), 1.0, 1e-6);
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
      scan.intensities.insert(scan.intensities.end(), 3, 42.0F + static_cast<float>(i + j));
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
  params.apply_imu_correction = true;
  // The synthetic scans sit at the same place; insert every accepted scan so the horizon
  // tests can count them.
  params.keyframe_min_translation = 0.0;
  params.keyframe_min_rotation = 0.0;
  // Keep the bake-off out of the unit tests: deterministic CPU path.
  params.device_bakeoff_scan = 0;
  return params;
}

TEST(BodyTwistPrior, constant_forward_speed_integrates_along_body_x)
{
  const std::vector<BodyTwistSample> samples = {
    {0.0, Eigen::Vector3d(10.0, 0.0, 0.0), Eigen::Vector3d::Zero()},
    {0.05, Eigen::Vector3d(10.0, 0.0, 0.0), Eigen::Vector3d::Zero()},
    {0.1, Eigen::Vector3d(10.0, 0.0, 0.0), Eigen::Vector3d::Zero()},
  };
  const std::optional<Sophus::SE3d> delta = integrate_body_twist(samples, 0.0, 0.1, 0.05);
  ASSERT_TRUE(delta.has_value());
  EXPECT_NEAR(delta->translation().x(), 1.0, 1.0e-6);
  EXPECT_NEAR(delta->translation().y(), 0.0, 1.0e-6);
  EXPECT_NEAR(delta->so3().log().norm(), 0.0, 1.0e-6);
}

TEST(BodyTwistPrior, constant_yaw_rate_is_a_body_rotation)
{
  const std::vector<BodyTwistSample> samples = {
    {0.0, Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, 1.0)},
    {0.1, Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, 1.0)},
  };
  const std::optional<Sophus::SE3d> delta = integrate_body_twist(samples, 0.0, 0.1, 0.1);
  ASSERT_TRUE(delta.has_value());
  EXPECT_NEAR(delta->so3().log().z(), 0.1, 1.0e-6);
  EXPECT_NEAR(delta->translation().norm(), 0.0, 1.0e-6);
}

TEST(BodyTwistPrior, a_hole_in_the_stream_refuses_the_prior)
{
  const std::vector<BodyTwistSample> samples = {
    {0.0, Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d::Zero()},
    {0.2, Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d::Zero()},
  };
  EXPECT_FALSE(integrate_body_twist(samples, 0.0, 0.2, 0.05).has_value());
}

TEST(BodyTwistPrior, one_sample_matches_the_screw_exponential)
{
  BodyTwistSample sample;
  sample.stamp = 1.0;
  sample.linear = Eigen::Vector3d(5.0, 0.2, 0.0);
  sample.angular = Eigen::Vector3d(0.0, 0.0, 0.4);
  Sophus::SE3d::Tangent twist;
  twist << sample.linear, sample.angular;
  const Sophus::SE3d expected = Sophus::SE3d::exp(twist * 0.1);
  const std::optional<Sophus::SE3d> delta = integrate_body_twist({sample}, 1.0, 1.1, 0.1);
  ASSERT_TRUE(delta.has_value());
  EXPECT_NEAR((expected.inverse() * *delta).log().norm(), 0.0, 1.0e-8);
}

TEST(BodyTwistPrior, integrated_twist_replaces_the_imu_seed)
{
  ImuProcessor imu = aligned_imu();
  LidarMatchParams params = match_params();
  params.max_iterations = 1;
  params.align_rate_hz = 100.0;
  LidarImuMatcher matcher(params);
  const double t0 = imu.latest_stamp();
  const MatchResult first = matcher.on_scan(corner_scan(t0), imu);
  ASSERT_TRUE(first.first_scan);

  const double t1 = t0 + 0.1;
  for (int step = 1; step <= 5; ++step) {
    const ProcessorOutput output = imu.process(
      ImuSample{t0 + 0.02 * step, Eigen::Vector3d(0.0, 0.0, -9.81), Eigen::Vector3d::Zero()});
    ASSERT_TRUE(output.aligned);
  }
  std::vector<BodyTwistSample> twists;
  for (int step = 0; step <= 5; ++step) {
    BodyTwistSample sample;
    sample.stamp = t0 + 0.02 * step;
    sample.linear = Eigen::Vector3d(10.0, 0.0, 0.0);
    twists.push_back(sample);
  }

  const ScanAlignment with_twist = matcher.align_scan(corner_scan(t1), imu, twists);
  ASSERT_TRUE(with_twist.ready);
  EXPECT_TRUE(with_twist.result.ekf_twist_prior);
  const Sophus::SE3d ekf_motion = first.corrected_pose.inverse() * with_twist.predicted;
  EXPECT_NEAR(ekf_motion.translation().x(), 1.0, 1.0e-6);
  EXPECT_NEAR(ekf_motion.translation().y(), 0.0, 1.0e-6);

  const ScanAlignment imu_only = matcher.align_scan(corner_scan(t1), imu);
  ASSERT_TRUE(imu_only.ready);
  EXPECT_FALSE(imu_only.result.ekf_twist_prior);
  const Sophus::SE3d imu_motion = first.corrected_pose.inverse() * imu_only.predicted;
  EXPECT_LT(imu_motion.translation().norm(), 0.2);
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

TEST(BackwardMatch, fifteen_centimetre_offset_is_applied)
{
  ImuProcessor imu = aligned_imu();
  LidarImuMatcher matcher(match_params());
  const LidarScan scan = corner_scan(imu.latest_stamp());
  ASSERT_TRUE(matcher.on_scan(scan, imu).first_scan);
  imu.reset_state(
    Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.15, 0.0, 0.0)), Eigen::Vector3d::Zero(),
    imu.latest_stamp());
  const MatchResult matched = matcher.on_scan(scan, imu);
  ASSERT_TRUE(matched.applied);
  EXPECT_NEAR(imu.pose().translation().x(), 0.0, 0.08);
}

TEST(PlaneIcp, correspondence_distance_switches_after_coarse_iterations)
{
  PlaneIcpParams params;
  params.max_correspondence_distance = 2.0;
  params.coarse_correspondence_distance = 4.0;
  params.coarse_iterations = 6;
  EXPECT_DOUBLE_EQ(correspondence_distance_for_iteration(params, 0), 4.0);
  EXPECT_DOUBLE_EQ(correspondence_distance_for_iteration(params, 5), 4.0);
  EXPECT_DOUBLE_EQ(correspondence_distance_for_iteration(params, 6), 2.0);
  EXPECT_DOUBLE_EQ(correspondence_distance_for_iteration(params, 20), 2.0);
}

TEST(BackwardMatch, align_rate_budget_skips_the_scan_and_degrades_health)
{
  ImuProcessor imu = aligned_imu();
  ASSERT_TRUE(imu.aligned());
  LidarMatchParams params = match_params();
  // Near-zero ICP budget: the second scan must abandon before any iteration.
  params.align_rate_hz = 1.0e12;
  LidarImuMatcher matcher(params);
  const LidarScan scan = corner_scan(imu.latest_stamp());
  ASSERT_TRUE(matcher.on_scan(scan, imu).first_scan);
  const std::size_t points_before = matcher.visible_scan_point_count();

  const Sophus::SE3d drifted(Sophus::SO3d(), Eigen::Vector3d(0.3, 0.0, 0.0));
  imu.reset_state(drifted, Eigen::Vector3d::Zero(), imu.latest_stamp());
  const MatchResult matched = matcher.on_scan(scan, imu);
  // A scan whose pose was never verified must not be written into the map or move the IMU.
  EXPECT_FALSE(matched.applied);
  EXPECT_FALSE(matched.inserted_scan);
  EXPECT_TRUE(matched.align_timed_out);
  EXPECT_FALSE(matched.partial_used);
  EXPECT_FALSE(matched.lidar_passed);
  EXPECT_EQ(matcher.visible_scan_poses().size(), 1U);
  EXPECT_EQ(matcher.visible_scan_point_count(), points_before);
  EXPECT_NEAR(imu.pose().translation().x(), 0.3, 1.0e-6);
  EXPECT_EQ(matcher.health(), LocalizationHealth::Degraded);
  EXPECT_EQ(matcher.timeout_streak(), 1);
  EXPECT_EQ(matcher.reject_streak(), 0);
}

TEST(BackwardMatch, rejected_icp_keeps_the_imu_seed_as_the_lidar_pose)
{
  ImuProcessor imu = aligned_imu();
  LidarImuMatcher matcher(match_params());
  ASSERT_TRUE(matcher.on_scan(corner_scan(imu.latest_stamp()), imu).first_scan);
  const double later = imu.latest_stamp() + 0.5;
  imu.reset_state(
    Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(1.2, 0.0, 0.0)), Eigen::Vector3d(2.0, 0.0, 0.0),
    later);
  const MatchResult matched = matcher.on_scan(corner_scan(later), imu);
  EXPECT_FALSE(matched.applied);
  EXPECT_FALSE(matched.inserted_scan);
  // The failed basin is not published: lidar debug stays on the IMU-propagated prior.
  EXPECT_GT(matched.lidar_pose.translation().x(), 0.8);
  EXPECT_NEAR(matched.lidar_pose.translation().x(), imu.pose().translation().x(), 0.25);
}

TEST(BackwardMatch, stationary_scans_are_matched_but_not_inserted)
{
  ImuProcessor imu = aligned_imu();
  LidarMatchParams params = match_params();
  params.keyframe_min_translation = 0.5;
  params.keyframe_min_rotation = 0.05;
  LidarImuMatcher matcher(params);
  const LidarScan scan = corner_scan(imu.latest_stamp());
  ASSERT_TRUE(matcher.on_scan(scan, imu).first_scan);
  const std::size_t points_before = matcher.visible_scan_point_count();
  imu.reset_state(
    Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.1, 0.0, 0.0)), Eigen::Vector3d::Zero(),
    imu.latest_stamp());
  const MatchResult matched = matcher.on_scan(scan, imu);
  ASSERT_TRUE(matched.applied);
  EXPECT_FALSE(matched.inserted_scan);
  EXPECT_EQ(matcher.visible_scan_point_count(), points_before);
  // The correction still reaches the IMU.
  EXPECT_NEAR(imu.pose().translation().x(), 0.0, 0.1);
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

TEST(InertialCorrection, curve_keeps_the_screw_instead_of_the_chord)
{
  constexpr double yaw = 0.2;
  constexpr double speed = 10.0;
  constexpr double dt = 1.0;
  const double radius = speed / yaw;
  const Eigen::Vector3d position(radius * std::sin(yaw), radius * (1.0 - std::cos(yaw)), 0.0);
  const Sophus::SE3d corrected(Sophus::SO3d::rotZ(yaw), position);
  const Eigen::Vector3d tangent = speed * Eigen::Vector3d(std::cos(yaw), std::sin(yaw), 0.0);

  const InertialCorrection correction = inertial_correction_from_match(
    Sophus::SE3d(), Sophus::SO3d(), corrected, tangent, Sophus::SE3d(), dt, 0.1, 0.02, 0.5);

  const Eigen::Vector3d screw = corrected.log().head<3>() / dt;
  const Eigen::Vector3d naive = corrected.so3().inverse() * position / dt;
  const Eigen::Vector3d body = corrected.so3().inverse() * correction.velocity_world;
  EXPECT_NEAR(body.x(), 0.5 * speed + 0.5 * screw.x(), 1e-6);
  EXPECT_NEAR(body.y(), screw.y(), 1e-6);
  EXPECT_NEAR(screw.x(), speed, 1e-6);
  EXPECT_LT(std::abs(body.y()), 1e-6);
  EXPECT_GT(std::abs(naive.y()), 0.5);
  EXPECT_NEAR(correction.accel_bias_delta.y(), 0.0, 1e-12);
  EXPECT_NEAR(correction.gyro_bias_delta.norm(), 0.0, 1e-12);
}

TEST(InertialCorrection, screw_recovers_the_body_velocity_of_a_turning_step)
{
  const Eigen::Vector3d velocity(9.5, 0.12, -0.05);
  constexpr double dt = 0.1;
  Eigen::Matrix<double, 6, 1> tangent;
  tangent << 9.5 * dt, 0.12 * dt, -0.05 * dt, 0.02, -0.03, 0.18;
  const Sophus::SE3d step = Sophus::SE3d::exp(tangent);
  const Eigen::Vector3d recovered = step.log().head<3>() / dt;
  const Eigen::Vector3d naive = step.so3().inverse() * step.translation() / dt;
  EXPECT_NEAR(recovered.x(), velocity.x(), 1e-9);
  EXPECT_NEAR(recovered.y(), velocity.y(), 1e-9);
  EXPECT_NEAR(recovered.z(), velocity.z(), 1e-9);
  EXPECT_GT(std::abs(naive.y() - velocity.y()), 0.5);
}

TEST(InertialCorrection, off_heading_velocity_is_pulled_onto_the_matched_forward_axis)
{
  constexpr double k_pi = 3.14159265358979323846;
  constexpr double speed = 10.0;
  constexpr double angle = k_pi / 6.0;
  const Sophus::SE3d corrected(Sophus::SO3d(), Eigen::Vector3d(10.0, 0.0, 0.0));
  const Eigen::Vector3d velocity(speed * std::cos(angle), speed * std::sin(angle), 0.0);
  const InertialCorrection correction = inertial_correction_from_match(
    Sophus::SE3d(), Sophus::SO3d(), corrected, velocity, Sophus::SE3d(), 1.0, 0.1, 0.02, 0.5);
  const double blended = 0.5 * velocity.x() + 0.5 * 10.0;
  EXPECT_NEAR(correction.velocity_world.x(), blended, 1e-9);
  EXPECT_NEAR(correction.velocity_world.y(), 0.0, 1e-12);
  EXPECT_NEAR(correction.velocity_world.z(), 0.0, 1e-12);
  EXPECT_NEAR(correction.accel_bias_delta.y(), 0.0, 1e-12);
}

TEST(InertialCorrection, lateral_displacement_sets_lateral_velocity_but_not_accel_bias)
{
  const Sophus::SE3d corrected(Sophus::SO3d(), Eigen::Vector3d(0.0, 1.0, 0.0));
  const InertialCorrection correction = inertial_correction_from_match(
    Sophus::SE3d(), Sophus::SO3d(), corrected, Eigen::Vector3d::Zero(), Sophus::SE3d(), 1.0, 0.1,
    0.02, 0.5);
  EXPECT_NEAR(correction.velocity_world.x(), 0.0, 1e-12);
  EXPECT_NEAR(correction.velocity_world.y(), 1.0, 1e-9);
  EXPECT_NEAR(correction.accel_bias_delta.y(), 0.0, 1e-12);
}

TEST(InertialCorrection, extra_positive_yaw_increases_gyro_bias)
{
  const Sophus::SE3d vehicle_delta(Sophus::SO3d::rotZ(-0.002), Eigen::Vector3d::Zero());
  const InertialCorrection correction = inertial_correction_from_match(
    vehicle_delta, Sophus::SO3d(), Sophus::SE3d(), Eigen::Vector3d(1.0, 0.0, 0.0), Sophus::SE3d(),
    0.2, 0.1, 0.02, 0.5);
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

TEST(LocalizationHealth, rejected_scans_degrade_then_diverge)
{
  ImuProcessor imu = aligned_imu();
  LidarMatchParams params = match_params();
  params.limits.max_match_rejects = 3;
  LidarImuMatcher matcher(params);
  ASSERT_TRUE(matcher.on_scan(corner_scan(imu.latest_stamp()), imu).first_scan);
  for (int step = 0; step < 2; ++step) {
    const double stamp = imu.latest_stamp() + 0.5;
    imu.reset_state(
      Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(1.2, 0.0, 0.0)), Eigen::Vector3d(40.0, 0.0, 0.0),
      stamp);
    const MatchResult matched = matcher.on_scan(corner_scan(stamp), imu);
    EXPECT_FALSE(matched.applied);
    EXPECT_EQ(matcher.health(), LocalizationHealth::Degraded);
  }
  const double stamp = imu.latest_stamp() + 0.5;
  imu.reset_state(
    Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(1.2, 0.0, 0.0)), Eigen::Vector3d(40.0, 0.0, 0.0),
    stamp);
  ASSERT_FALSE(matcher.on_scan(corner_scan(stamp), imu).applied);
  EXPECT_EQ(matcher.health(), LocalizationHealth::Diverged);
  // Matcher stays available for further scans after divergence.
  EXPECT_TRUE(matcher.has_reference());
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
  EXPECT_GE(first.front().intensity, 42.0F);
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
    EXPECT_GE(point.intensity, 42.0F);
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
  params.max_points_per_voxel = 20;
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

namespace
{

kiss_icp::VoxelHashMap make_planar_ndt_map(const double voxel_size)
{
  // Keep every sample inside voxel (0,0,0) and past kiss's per-voxel spacing.
  kiss_icp::VoxelHashMap map(voxel_size, 1.0e6, 20);
  std::vector<Eigen::Vector3d> points;
  points.reserve(20);
  for (int x = 0; x < 5; ++x) {
    for (int y = 0; y < 4; ++y) {
      points.emplace_back(
        0.05 + 0.09 * static_cast<double>(x), 0.05 + 0.09 * static_cast<double>(y),
        0.001 * static_cast<double>((x + y) % 3));
    }
  }
  map.AddPoints(points);
  return map;
}

}  // namespace

TEST(NdtHealth, inlier_pose_has_low_mahalanobis_cost)
{
  constexpr double voxel_size = 0.5;
  const kiss_icp::VoxelHashMap map = make_planar_ndt_map(voxel_size);
  NdtVoxelMap ndt;
  ndt.cov_regularization = 1.0e-3;
  ndt.min_points = 5;
  ndt.rebuild_from(map);
  ASSERT_FALSE(ndt.voxels.empty());

  std::vector<Eigen::Vector3d> query;
  for (const auto & item : map.map_) {
    for (const Eigen::Vector3d & point : item.second) {
      query.push_back(point);
    }
  }
  const double cost = ndt_mean_mahalanobis(query, Sophus::SE3d{}, ndt, 10);
  ASSERT_TRUE(std::isfinite(cost)) << "query=" << query.size() << " ndt=" << ndt.voxels.size();
  EXPECT_LT(cost, 7.81);
}

TEST(NdtHealth, offset_points_raise_mahalanobis_above_threshold)
{
  constexpr double voxel_size = 0.5;
  const kiss_icp::VoxelHashMap map = make_planar_ndt_map(voxel_size);
  NdtVoxelMap ndt;
  ndt.cov_regularization = 1.0e-3;
  ndt.min_points = 5;
  ndt.rebuild_from(map);

  std::vector<Eigen::Vector3d> query;
  for (const auto & item : map.map_) {
    for (const Eigen::Vector3d & point : item.second) {
      query.push_back(point);
    }
  }
  const double inlier_cost = ndt_mean_mahalanobis(query, Sophus::SE3d{}, ndt, 10);
  ASSERT_TRUE(std::isfinite(inlier_cost));

  // Stay inside the same 0.5 m voxel but sit far from the fitted mean (z is thin).
  std::vector<Eigen::Vector3d> offset_query = query;
  for (Eigen::Vector3d & point : offset_query) {
    point += Eigen::Vector3d(0.0, 0.0, 0.2);
  }
  const double offset_cost = ndt_mean_mahalanobis(offset_query, Sophus::SE3d{}, ndt, 10);
  ASSERT_TRUE(std::isfinite(offset_cost));
  EXPECT_GT(offset_cost, inlier_cost);
  EXPECT_GT(offset_cost, 7.81);
}

TEST(NdtHealth, sparse_map_skips_gate_with_nan)
{
  NdtVoxelMap ndt;
  ndt.voxel_size = 0.5;
  const std::vector<Eigen::Vector3d> query = {{0.0, 0.0, 0.0}, {0.1, 0.0, 0.0}};
  const double cost = ndt_mean_mahalanobis(query, Sophus::SE3d{}, ndt, 20);
  EXPECT_TRUE(std::isnan(cost));
}

TEST(NdtHealth, inlier_fraction_separates_a_half_offset_scan)
{
  constexpr double voxel_size = 0.5;
  const kiss_icp::VoxelHashMap map = make_planar_ndt_map(voxel_size);
  SurfaceVoxelMap surface;
  surface.voxel_size = voxel_size;
  surface.rebuild_from(map);
  std::vector<Eigen::Vector3d> query;
  for (const auto & item : map.map_) {
    for (const Eigen::Vector3d & point : item.second) {
      query.push_back(point);
    }
  }
  const NdtScore clean = ndt_score(query, Sophus::SE3d{}, surface, 7.81, 10);
  ASSERT_TRUE(std::isfinite(clean.inlier_fraction));
  EXPECT_GT(clean.inlier_fraction, 0.9);
  // Lift half the points off the plane: the mean may still pass but the fraction must not.
  for (std::size_t index = 0; index < query.size(); index += 2) {
    query[index] += Eigen::Vector3d(0.0, 0.0, 0.2);
  }
  const NdtScore half = ndt_score(query, Sophus::SE3d{}, surface, 7.81, 10);
  ASSERT_TRUE(std::isfinite(half.inlier_fraction));
  EXPECT_LT(half.inlier_fraction, 0.6);
  EXPECT_GT(half.inlier_fraction, 0.3);
}

TEST(SurfaceVoxelMap, incremental_update_matches_a_full_rebuild)
{
  constexpr double voxel_size = 0.5;
  kiss_icp::VoxelHashMap map(voxel_size, 1.0e6, 20);
  const std::vector<Eigen::Vector3d> first = corner_scan(0.0).points;
  map.AddPoints(first);
  SurfaceVoxelMap incremental;
  incremental.voxel_size = voxel_size;
  incremental.rebuild_from(map);
  const std::size_t voxels_before = incremental.voxels.size();
  ASSERT_GT(voxels_before, 0U);

  // Second scan shifted so it touches new voxels and adds points to existing ones.
  std::vector<Eigen::Vector3d> second;
  std::vector<kiss_icp::Voxel> touched;
  for (const Eigen::Vector3d & point : first) {
    const Eigen::Vector3d moved = point + Eigen::Vector3d(0.6, 0.35, 0.0);
    second.push_back(moved);
    touched.push_back(kiss_icp::PointToVoxel(moved, voxel_size));
  }
  map.AddPoints(second);
  incremental.update_voxels(map, touched);

  SurfaceVoxelMap full;
  full.voxel_size = voxel_size;
  full.rebuild_from(map);
  ASSERT_EQ(incremental.voxels.size(), full.voxels.size());
  EXPECT_GT(incremental.voxels.size(), voxels_before);
  for (const auto & item : full.voxels) {
    const SurfaceVoxel * entry = incremental.find(item.first);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->point_count, item.second.point_count);
    EXPECT_EQ(entry->planar, item.second.planar);
    EXPECT_EQ(entry->own_fit, item.second.own_fit);
    EXPECT_LT((entry->mean - item.second.mean).norm(), 1.0e-9);
    if (item.second.planar) {
      EXPECT_GT(std::abs(entry->normal.dot(item.second.normal)), 0.999);
    }
  }

  // Culling the map must leave no stale voxels behind after prune_missing.
  const Sophus::SE3d far_away(Sophus::SO3d(), Eigen::Vector3d(500.0, 0.0, 0.0));
  map.RemovePointsOutsideBox(far_away.inverse(), 10.0, 10.0);
  incremental.prune_missing(map);
  EXPECT_TRUE(map.Empty());
  EXPECT_TRUE(incremental.voxels.empty());
}

TEST(PlaneClass, ground_loses_planar_votes_and_the_axle_prior_holds_longitudinal_slip)
{
  Eigen::Matrix<double, 6, 6> ground = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 6> wall = Eigen::Matrix<double, 6, 6>::Zero();
  ground(0, 0) = 40.0;
  ground(2, 2) = 10.0;
  ground(3, 3) = 4.0;
  ground(4, 4) = 4.0;
  wall(0, 0) = 8.0;
  wall(1, 1) = 8.0;
  wall(5, 5) = 2.0;
  const Eigen::Matrix<double, 6, 6> other = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> ground_gradient = Eigen::Matrix<double, 6, 1>::Zero();
  ground_gradient(0) = 4.0;
  const Eigen::Matrix<double, 6, 1> wall_gradient = Eigen::Matrix<double, 6, 1>::Zero();
  const Eigen::Matrix<double, 6, 1> other_gradient = Eigen::Matrix<double, 6, 1>::Zero();

  CurveConstraint turning;
  turning.yaw_rate = 0.2;
  turning.axle_in_body = Eigen::Vector3d(-0.65, 0.0, -1.80);
  turning.longitudinal_sigma = 0.08;
  turning.xi_from_seed(0) = 0.2;

  Eigen::Matrix<double, 6, 6> hessian;
  Eigen::Matrix<double, 6, 1> gradient;
  compose_class_system(
    ground, wall, other, ground_gradient, wall_gradient, other_gradient, turning, hessian,
    gradient);

  const double prior = hessian(0, 0) - wall(0, 0);
  EXPECT_GT(prior, 0.0);
  EXPECT_NEAR(hessian(0, 4), turning.axle_in_body.z() * prior, 1e-6);
  EXPECT_GT(hessian(2, 2), 0.0);
  EXPECT_GT(gradient(0), 0.0);

  CurveConstraint masked = turning;
  masked.longitudinal_sigma = 0.0;
  Eigen::Matrix<double, 6, 6> masked_hessian;
  Eigen::Matrix<double, 6, 1> masked_gradient;
  compose_class_system(
    ground, wall, other, ground_gradient, wall_gradient, other_gradient, masked, masked_hessian,
    masked_gradient);
  EXPECT_NEAR(masked_hessian(0, 0), wall(0, 0), 1e-9);
  EXPECT_NEAR(masked_hessian(2, 2), ground(2, 2), 1e-9);
  EXPECT_NEAR(masked_gradient(0), 0.0, 1e-9);

  CurveConstraint straight;
  Eigen::Matrix<double, 6, 6> straight_hessian;
  Eigen::Matrix<double, 6, 1> straight_gradient;
  compose_class_system(
    ground, wall, other, ground_gradient, wall_gradient, other_gradient, straight,
    straight_hessian, straight_gradient);
  EXPECT_GT(straight_hessian(0, 0), wall(0, 0));
}

TEST(PlaneClass, ground_scale_matches_wall_weight)
{
  EXPECT_DOUBLE_EQ(ground_balance_scale(0.0, 4.0), 1.0);
  EXPECT_DOUBLE_EQ(ground_balance_scale(2.0, 8.0), 4.0);
  EXPECT_DOUBLE_EQ(ground_balance_scale(1.0, 100.0), 20.0);
  EXPECT_EQ(classify_plane_normal(0.99), PlaneClass::Ground);
  EXPECT_EQ(classify_plane_normal(0.1), PlaneClass::Wall);
}

TEST(PlaneIcp, wall_heavy_pitched_ground_recovers_pitch)
{
  constexpr double voxel_size = 0.5;
  kiss_icp::VoxelHashMap map(voxel_size, 1.0e6, 20);
  std::vector<Eigen::Vector3d> world;
  for (int ix = -6; ix <= 6; ++ix) {
    for (int iy = -6; iy <= 6; ++iy) {
      world.emplace_back(0.5 * ix, 0.5 * iy, 0.0);
    }
  }
  for (int iz = 0; iz <= 8; ++iz) {
    for (int iy = -6; iy <= 6; ++iy) {
      const double y = 0.5 * iy;
      const double z = 0.4 * iz;
      world.emplace_back(-4.0, y, z);
      world.emplace_back(4.0, y, z);
      world.emplace_back(y, -4.0, z);
      world.emplace_back(y, 4.0, z);
    }
  }
  map.AddPoints(world);
  SurfaceVoxelMap surface;
  surface.voxel_size = voxel_size;
  surface.rebuild_from(map);

  const Sophus::SO3d pitch = Sophus::SO3d::rotY(0.02);
  std::vector<Eigen::Vector3d> source;
  source.reserve(world.size());
  for (const Eigen::Vector3d & point : world) {
    source.push_back(pitch.inverse() * point);
  }
  PlaneIcpParams params;
  params.max_iterations = 40;
  params.convergence_criterion = 1.0e-4;
  const PlaneIcpResult result =
    align_to_surface(source, map, surface, Sophus::SE3d(), params);
  EXPECT_NEAR(result.pose.so3().log().y(), 0.02, 0.01);
  EXPECT_LT(std::abs(result.pose.so3().log().x()), 0.01);
  EXPECT_LT(std::abs(result.pose.so3().log().z()), 0.01);
}

TEST(PlaneIcp, align_to_surface_recovers_a_shifted_corner)
{
  constexpr double voxel_size = 0.25;
  kiss_icp::VoxelHashMap map(voxel_size, 1.0e6, 20);
  const std::vector<Eigen::Vector3d> points = corner_scan(0.0).points;
  map.AddPoints(points);
  SurfaceVoxelMap surface;
  surface.voxel_size = voxel_size;
  surface.rebuild_from(map);
  const Sophus::SE3d guess(
    Sophus::SO3d::exp(Eigen::Vector3d(0.0, 0.0, 0.02)), Eigen::Vector3d(0.15, -0.1, 0.05));
  PlaneIcpParams params;
  params.max_iterations = 30;
  params.convergence_criterion = 1.0e-4;
  const PlaneIcpResult result = align_to_surface(points, map, surface, guess, params);
  EXPECT_FALSE(result.timed_out);
  EXPECT_GT(result.correspondences, static_cast<int>(points.size() / 2));
  EXPECT_LT(result.pose.translation().norm(), 0.03);
  EXPECT_LT(result.pose.so3().log().norm(), 0.01);
  EXPECT_LT(result.robust_cost, 0.05);
  // correction * guess == pose
  EXPECT_LT(((result.correction * guess).inverse() * result.pose).log().norm(), 1.0e-9);
}

TEST(PlaneIcp, deadline_returns_the_partial_pose_not_the_seed)
{
  constexpr double voxel_size = 0.25;
  kiss_icp::VoxelHashMap map(voxel_size, 1.0e6, 20);
  const std::vector<Eigen::Vector3d> points = corner_scan(0.0).points;
  map.AddPoints(points);
  SurfaceVoxelMap surface;
  surface.voxel_size = voxel_size;
  surface.rebuild_from(map);
  const Sophus::SE3d guess(Sophus::SO3d(), Eigen::Vector3d(0.15, 0.0, 0.0));
  PlaneIcpParams params;
  params.max_iterations = 50;
  // A step is never "< 0": the solver can only stop on the cap or the deadline.
  params.convergence_criterion = 0.0;
  // Deadline already passed: zero iterations, pose == seed, flagged.
  const auto now = std::chrono::steady_clock::now();
  const PlaneIcpResult none = align_to_surface(points, map, surface, guess, params, &now);
  EXPECT_TRUE(none.timed_out);
  EXPECT_DOUBLE_EQ(none.iterations, 0.0);
  EXPECT_LT((none.pose.inverse() * guess).log().norm(), 1.0e-12);
  // Generous deadline but an impossible criterion: saturated, not timed out.
  const PlaneIcpResult saturated = align_to_surface(points, map, surface, guess, params);
  EXPECT_FALSE(saturated.timed_out);
  EXPECT_TRUE(saturated.saturated);
  EXPECT_DOUBLE_EQ(saturated.iterations, 50.0);
}

TEST(LocalizationHealth, timeouts_degrade_without_counting_as_rejects)
{
  HealthState state;
  state = advance_localization_health(state, HealthEvent::TimedOut, 3);
  EXPECT_EQ(state.health, LocalizationHealth::Degraded);
  EXPECT_EQ(state.timeout_streak, 1);
  EXPECT_EQ(state.reject_streak, 0);
  EXPECT_FALSE(state.update_bias);
  state = advance_localization_health(state, HealthEvent::TimedOut, 3);
  EXPECT_EQ(state.timeout_streak, 2);
  EXPECT_EQ(state.health, LocalizationHealth::Degraded);
  // One accepted scan clears the streak and recovers.
  state = advance_localization_health(state, HealthEvent::Accepted, 3);
  EXPECT_EQ(state.timeout_streak, 0);
  EXPECT_EQ(state.health, LocalizationHealth::Healthy);
  EXPECT_TRUE(state.update_bias);
  // Saturated is accepted but must not feed the biases.
  state = advance_localization_health(state, HealthEvent::Saturated, 3);
  EXPECT_FALSE(state.update_bias);
}

TEST(SweepDeskew, sixty_four_buckets_stay_within_a_centimetre_at_forty_metres)
{
  Eigen::Matrix<double, 6, 1> xi;
  xi << 0.15, -0.08, 0.01, 0.002, -0.003, 1.5 * 3.14159265358979323846 / 180.0;
  const Eigen::Vector3d point(35.0, 20.0, 2.0);
  double max_error = 0.0;
  for (int sample = 0; sample < 1000; ++sample) {
    const double tau = static_cast<double>(sample) / 999.0;
    std::vector<Eigen::Vector3d> deskewed;
    apply_sweep_residual_deskew(
      {point}, {static_cast<float>(tau)}, xi, deskewed);
    const Eigen::Vector3d exact = Sophus::SE3d::exp((tau - 1.0) * xi) * point;
    max_error = std::max(max_error, (deskewed[0] - exact).norm());
  }
  EXPECT_GT(max_error, 0.001);
  EXPECT_LT(max_error, 0.010);
  std::vector<Eigen::Vector3d> at_end;
  apply_sweep_residual_deskew({point}, {1.0F}, xi, at_end);
  EXPECT_LT((at_end[0] - point).norm(), 1.0e-9);
}

TEST(PlaneClass, one_sided_wall_loosens_the_weak_planar_direction)
{
  Eigen::Matrix<double, 6, 6> wall = Eigen::Matrix<double, 6, 6>::Zero();
  wall(0, 0) = 1.0e6;
  wall(1, 1) = 1.0e-8;
  wall(5, 5) = 1.0e6 * 225.0;
  const Eigen::Matrix3d straight = adaptive_planar_covariance(wall, 0.08, 0.9, 0.0);
  EXPECT_LT(straight(0, 0), straight(1, 1));
  EXPECT_GT(straight(1, 1), 0.5);
  EXPECT_GE(straight(2, 2), 0.015 * 0.015 - 1.0e-12);
  EXPECT_LE(straight(2, 2), 0.12 * 0.12 + 1.0e-12);

  // A moderate eigenvalue sits inside the variance clamps, so a turn can still inflate it.
  Eigen::Matrix<double, 6, 6> moderate = Eigen::Matrix<double, 6, 6>::Zero();
  moderate(0, 0) = 0.18;
  moderate(1, 1) = 0.18;
  moderate(5, 5) = 0.18 * 225.0;
  const Eigen::Matrix3d cruising = adaptive_planar_covariance(moderate, 0.08, 0.9, 0.0);
  const Eigen::Matrix3d turning = adaptive_planar_covariance(moderate, 0.08, 0.5, 0.15);
  EXPECT_GT(turning.trace(), cruising.trace());

  const std::array<double, 36> pose =
    pose_covariance_with_planar(straight, 1.5707963267948966, 0.64, 0.04);
  EXPECT_NEAR(pose[0], straight(1, 1), 1.0e-9);
  EXPECT_NEAR(pose[7], straight(0, 0), 1.0e-9);
  EXPECT_DOUBLE_EQ(pose[14], 0.64);
  EXPECT_DOUBLE_EQ(pose[21], 0.04);
  EXPECT_DOUBLE_EQ(pose[28], 0.04);
}

}  // namespace back_odom
