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

#ifndef BACK_ODOM__IMU_PROCESSOR_HPP_
#define BACK_ODOM__IMU_PROCESSOR_HPP_

#include "back_odom/imu_dead_reckoning.hpp"
#include "back_odom/imu_types.hpp"

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <optional>
#include <vector>

namespace back_odom
{

class ImuProcessor
{
public:
  explicit ImuProcessor(const ProcessorParams & params);

  [[nodiscard]] ProcessorOutput process(const ImuSample & sample);
  [[nodiscard]] ProcessorOutput output_at(const ImuSample & sample) const;
  [[nodiscard]] bool aligned() const;
  [[nodiscard]] Sophus::SE3d pose() const;
  [[nodiscard]] Eigen::Vector3d velocity() const;
  [[nodiscard]] double latest_stamp() const;
  [[nodiscard]] const std::vector<StampedPose> & trajectory() const;
  [[nodiscard]] Sophus::SE3d interpolate_pose(double stamp) const;
  void reset_state(const Sophus::SE3d & pose, const Eigen::Vector3d & velocity_world, double stamp);

private:
  void record_pose(double stamp);
  [[nodiscard]] ProcessorOutput process_collecting(const ImuSample & sample);
  [[nodiscard]] ProcessorOutput process_tracking(const ImuSample & sample);
  [[nodiscard]] ProcessorOutput make_tracking_output(const ImuSample & sample) const;

  ProcessorParams params_;
  std::vector<ImuSample> window_;
  std::optional<ImuDeadReckoning> dead_reckoning_;
  std::vector<StampedPose> trajectory_;
  double previous_stamp_{0.0};
  bool has_previous_stamp_{false};
};

}  // namespace back_odom

#endif  // BACK_ODOM__IMU_PROCESSOR_HPP_
