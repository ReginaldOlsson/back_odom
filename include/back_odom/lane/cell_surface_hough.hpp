#ifndef BACK_ODOM__LANE__CELL_SURFACE_HOUGH_HPP_
#define BACK_ODOM__LANE__CELL_SURFACE_HOUGH_HPP_

#include "back_odom/lane/cell_features.hpp"

#include <Eigen/Core>

#include <vector>

namespace back_odom
{
namespace lane
{

/// Smooth-surface detection by Hough voting on local tangent planes.
///
/// Each cell votes for its tangent plane (unit normal N on the Gauss sphere, offset rho = N . p)
/// inside an XY tile. Peaks are tangent-plane hypotheses of the surface. Cells join a hypothesis
/// when their offset agrees with the second-order prediction from their own normal
/// (h = -1/2 (N_i - N_k) . X), and hypotheses in adjacent tiles are stitched when the
/// finite-difference Weingarten map between them is bounded and consistent with their offset.
/// A plane is the special case smooth_curvature_max -> 0.
struct SurfaceHoughParams
{
  /// XY tile edge in meters. A tile is where one tangent plane is assumed to hold.
  double tile_size{6.0};
  /// Cube-map resolution per face edge for binning normals (16 -> about 5.6 degrees).
  int face_bins{16};
  /// Offset (rho) bin width in meters.
  double rho_bin{0.05};
  /// Cells whose normal tilt uncertainty sqrt(lambda3 / lambda2) exceeds this do not vote and
  /// are not assigned (see has_reliable_normal).
  double max_normal_sigma{0.5};
  /// Cells with a larger Weingarten misfit do not vote (Gauss map not continuous there).
  double max_residual{0.3};
  /// Per-cell vote weight cap on the point count.
  double vote_cap{32.0};
  /// Minimum raw vote mass inside the angular/offset ball for a peak to be kept.
  double peak_min_mass{30.0};
  int max_planes_per_tile{4};
  /// Largest normal curvature (1/m) still considered smooth.
  double smooth_curvature_max{0.05};
  /// Base angular tolerance (rad) between a cell normal and a hypothesis normal.
  double angle_tolerance{0.17};
  /// Base offset tolerance (m) between a cell and a hypothesis plane.
  double offset_tolerance{0.05};
  /// Expected per-cell normal noise (rad); the offset tolerance grows by 1/2 * noise * r.
  double normal_noise{0.01};
  int min_cells_per_surface{10};
};

struct PlaneHypothesis
{
  int tile_x{0};
  int tile_y{0};
  Eigen::Vector3d normal{Eigen::Vector3d::UnitZ()};
  double rho{0.0};
  Eigen::Vector3d centroid{Eigen::Vector3d::Zero()};
  double mass{0.0};
  int support{0};
  /// Surface id after stitching (0 = dropped).
  int surface{0};
};

struct SurfaceSummary
{
  int id{0};
  int cells{0};
  Eigen::Vector3d normal{Eigen::Vector3d::UnitZ()};
  /// Median and 90th percentile of |k1| over member cells.
  double k1_median{0.0};
  double k1_p90{0.0};
  /// Median Weingarten residual over member cells with a fit.
  double residual_median{0.0};
  bool smooth{false};
};

struct SurfaceResult
{
  /// Surface id per cell, 0 = unassigned.
  std::vector<int> surface;
  /// Index into hypotheses per cell, -1 = unassigned.
  std::vector<int> hypothesis;
  std::vector<PlaneHypothesis> hypotheses;
  std::vector<SurfaceSummary> surfaces;
};

SurfaceResult detect_smooth_surfaces(const CellGrid & grid, const SurfaceHoughParams & params);

}  // namespace lane
}  // namespace back_odom

#endif  // BACK_ODOM__LANE__CELL_SURFACE_HOUGH_HPP_
