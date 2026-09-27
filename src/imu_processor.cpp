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

#include "back_odom/imu_processor.hpp"

#include "back_odom/imu_alignment.hpp"

#include <stdexcept>

namespace back_odom
{

ImuProcessor::ImuProcessor(const ProcessorParams & params) : params_(params)
{
  if (params_.alignment_sample_count == 0) {
    throw std::invalid_argument("alignment_sample_count must be positive");
  }
  if (params_.gravity <= 0.0) {
    throw std::invalid_argument("gravity must be positive");
  }
  if (params_.max_dt <= 0.0) {
    throw std::invalid_argument("max_dt must be positive");
  }
}

ProcessorOutput ImuProcessor::process(const ImuSample & sample)
{
  if (!dead_reckoning_) {
    return process_collecting(sample);
  }
  return process_tracking(sample);
}

ProcessorOutput ImuProcessor::process_collecting(const ImuSample & sample)
{
  window_.push_back(sample);
  if (window_.size() > params_.alignment_sample_count) {
    window_.erase(window_.begin());
  }

  ProcessorOutput output;
  output.phase = ProcessorPhase::Collecting;
  output.aligned = false;
  output.sample_count = window_.size();
  output.specific_force_body = sample.linear_acceleration;
  output.angular_velocity_body = sample.angular_velocity;
  output.gravity = params_.gravity;

  if (
    window_.size() != params_.alignment_sample_count ||
    !is_stationary(window_, params_.stationary_gyro_thresh, params_.stationary_accel_dev_thresh)) {
    return output;
  }

  const AlignmentResult alignment = align_stationary_window(window_, params_.gravity);
  dead_reckoning_.emplace(alignment.gravity);
  dead_reckoning_->set_initial_orientation(alignment.orientation);
  dead_reckoning_->set_gyro_bias(alignment.gyro_bias);
  window_.clear();
  has_previous_stamp_ = true;
  previous_stamp_ = sample.stamp;

  output = make_tracking_output(sample);
  output.sample_count = params_.alignment_sample_count;
  return output;
}

ProcessorOutput ImuProcessor::process_tracking(const ImuSample & sample)
{
  const double dt = sample.stamp - previous_stamp_;
  if (has_previous_stamp_ && dt > 0.0 && dt <= params_.max_dt) {
    dead_reckoning_->integrate(sample.angular_velocity, sample.linear_acceleration, dt);
  }
  previous_stamp_ = sample.stamp;
  has_previous_stamp_ = true;

  ProcessorOutput output = make_tracking_output(sample);
  output.sample_count = params_.alignment_sample_count;
  return output;
}

ProcessorOutput ImuProcessor::make_tracking_output(const ImuSample & sample) const
{
  ProcessorOutput output;
  output.phase = ProcessorPhase::Tracking;
  output.aligned = true;
  output.orientation = dead_reckoning_->orientation();
  output.position = dead_reckoning_->position();
  output.velocity_world = dead_reckoning_->velocity();
  output.linear_acceleration_world = dead_reckoning_->linear_acceleration_world();
  output.specific_force_body = sample.linear_acceleration;
  output.angular_velocity_body = sample.angular_velocity - dead_reckoning_->gyro_bias();
  output.gravity = dead_reckoning_->gravity();
  return output;
}

}  // namespace back_odom
