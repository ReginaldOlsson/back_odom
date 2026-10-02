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

#ifndef BACK_ODOM__BUMP_MAP_HPP_
#define BACK_ODOM__BUMP_MAP_HPP_

#include <sophus/se3.hpp>

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace back_odom
{

/// Covariance plane per voxel, plus a depth image on that plane.
/// Pixel value is the signed distance off the plane, so a curb or a corner
/// stays visible inside the voxel instead of being flattened to one normal.
struct BumpMapConfig
{
  double voxel_size{0.5};
  double pixel_size{0.05};
  double normal_tolerance_deg{3.0};
  double huber_delta{0.1};
  /// Same gate kiss-icp uses: a point farther than this from the stored surface is not a correspondence.
  double max_correspondence_distance{2.0};
  /// Only points this close to the stored depth are written into the image.
  /// A car or truck that still falls inside the correspondence gate must not be averaged into the wall.
  double fuse_distance{0.15};
  int max_iterations{15};
  int min_points{4};
  /// Inliers required before a voxel built on an earlier scan may pull the pose.
  int min_confirm_points{3};
  bool smooth{true};
  bool weighted{true};
};

struct BumpAlignResult
{
  bool accepted{false};
  Sophus::SE3d pose{};
  double iterations{0.0};
  double mean_cost{0.0};
  int correspondences{0};
  int skipped{0};
};

class BumpMap
{
public:
  explicit BumpMap(const BumpMapConfig & config);

  void clear();

  /// World-frame points. `sensor_origin` is only the range weight.
  /// Points that miss an existing surface are not written into it. A brand-new
  /// cluster is stored, but it cannot constrain a pose until a later scan still sees it.
  void integrate(
    const std::vector<Eigen::Vector3d> & world_points, const Eigen::Vector3d & sensor_origin);

  void cull_outside_box(
    const Sophus::SE3d & world_from_body, double half_longitudinal, double half_lateral);

  [[nodiscard]] std::size_t usable_count() const;

  /// `confirm_points` is the denser cloud used to decide which stored surfaces are still there.
  /// Null uses `body_points`. Moving objects fail this check and are skipped, the same way
  /// kiss-icp drops a point with no correspondence.
  [[nodiscard]] BumpAlignResult align(
    const std::vector<Eigen::Vector3d> & body_points, const Sophus::SE3d & guess,
    const std::vector<Eigen::Vector3d> * confirm_points = nullptr);

private:
  struct Image
  {
    int width{0};
    int height{0};
    std::vector<float> depth;
    std::vector<float> smoothed;
    std::vector<float> weight;
  };

  struct Voxel
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Matrix3d outer{Eigen::Matrix3d::Zero()};
    Eigen::Vector3d sum{Eigen::Vector3d::Zero()};
    int count{0};
    bool planar{false};
    bool usable{false};
    int born_generation{0};
    int last_touch{0};
    Eigen::Matrix3d R_w_o{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d mean{Eigen::Vector3d::Zero()};
    Eigen::Matrix3d R_c_w{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d t_c_w{Eigen::Vector3d::Zero()};
    Image image;
  };

  enum class NormalUpdate
  {
    NotPlanar,
    Unchanged,
    Changed
  };

  [[nodiscard]] bool usable_surface(const Voxel & voxel) const;
  [[nodiscard]] std::uint64_t key_of(const Eigen::Vector3d & point) const;
  [[nodiscard]] const Voxel * find_surface(const Eigen::Vector3d & point) const;
  [[nodiscard]] bool sample(
    const Image & image, double x, double y, bool with_gradient, double & value, double & d_i_dx,
    double & d_i_dy) const;
  [[nodiscard]] bool residual_at(
    const Voxel & voxel, const Eigen::Vector3d & world_point, double & residual) const;
  void confirm_visible(
    const std::vector<Eigen::Vector3d> & body_points, const Sophus::SE3d & guess);
  void recount();
  [[nodiscard]] NormalUpdate update_normal(Voxel & voxel, bool allow_frame_change);
  void update_image(Voxel & voxel, const std::vector<Eigen::Vector4d> & points, bool rebuild_frame);
  void smooth(Voxel & voxel);

  struct Accumulator
  {
    int count{0};
    int skipped{0};
    double cost{0.0};
    double abs_sum{0.0};
    Eigen::Matrix<double, 6, 6> hessian{Eigen::Matrix<double, 6, 6>::Zero()};
    Eigen::Matrix<double, 6, 1> gradient{Eigen::Matrix<double, 6, 1>::Zero()};
  };

  void linearize(
    const std::vector<Eigen::Vector3d> & body_points, const Sophus::SE3d & pose, bool with_jacobian,
    Accumulator & accumulator) const;

  BumpMapConfig config_;
  double inv_voxel_{2.0};
  double inv_pixel_{20.0};
  double normal_tolerance_rad_{0.0};
  double max_correspondence_distance_sq_{4.0};
  Eigen::Matrix3f gauss_kernel_{Eigen::Matrix3f::Ones()};
  int generation_{0};
  std::size_t usable_count_{0};
  std::unordered_map<std::uint64_t, Voxel> voxels_;
};

}  // namespace back_odom

#endif  // BACK_ODOM__BUMP_MAP_HPP_
