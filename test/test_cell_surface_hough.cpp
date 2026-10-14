#include "back_odom/lane/cell_features.hpp"
#include "back_odom/lane/cell_surface_hough.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <optional>

namespace
{

using back_odom::lane::CellFeatures;
using back_odom::lane::CellGrid;
using back_odom::lane::CellMap;
using back_odom::lane::SurfaceHoughParams;
using back_odom::lane::SurfaceResult;

constexpr double kVoxel = 0.5;
constexpr double kHeight = 0.5;
constexpr double kSpacing = 0.1;
constexpr int kMinPoints = 3;
const Eigen::Vector3d kViewpoint(0.0, 0.0, 1.8);

/// Samples z = height(x, y) on a lattice over [x0, x1) x [y0, y1); nullopt skips the sample.
void add_surface(
  CellMap & cells, double x0, double x1, double y0, double y1,
  const std::function<std::optional<double>(double, double)> & height)
{
  for (double x = x0 + 0.5 * kSpacing; x < x1; x += kSpacing) {
    for (double y = y0 + 0.5 * kSpacing; y < y1; y += kSpacing) {
      const std::optional<double> z = height(x, y);
      if (!z) {
        continue;
      }
      back_odom::lane::add_to_cell(cells, x, y, *z, 20.0F, 1.0 / kVoxel, 1.0 / kHeight);
    }
  }
}

void add_wall(CellMap & cells, double wall_x, double y0, double y1, double z0, double z1)
{
  for (double y = y0 + 0.5 * kSpacing; y < y1; y += kSpacing) {
    for (double z = z0 + 0.5 * kSpacing; z < z1; z += kSpacing) {
      back_odom::lane::add_to_cell(cells, wall_x, y, z, 20.0F, 1.0 / kVoxel, 1.0 / kHeight);
    }
  }
}

CellGrid build_grid(const CellMap & cells)
{
  CellGrid grid = back_odom::lane::flatten(cells);
  back_odom::lane::compute_local_features(grid, kMinPoints, kViewpoint);
  back_odom::lane::compute_neighbor_features(grid, kVoxel, kHeight);
  return grid;
}

int cell_index(const CellGrid & grid, double x, double y, double z)
{
  const int index = grid.find(back_odom::lane::cell_key(x, y, z, 1.0 / kVoxel, 1.0 / kHeight));
  EXPECT_GE(index, 0) << "no cell at " << x << ", " << y << ", " << z;
  return std::max(index, 0);
}

const CellFeatures & feature_at(const CellGrid & grid, double x, double y, double z)
{
  return grid.features[static_cast<std::size_t>(cell_index(grid, x, y, z))];
}

int surface_at(const SurfaceResult & result, const CellGrid & grid, double x, double y, double z)
{
  return result.surface[static_cast<std::size_t>(cell_index(grid, x, y, z))];
}

const back_odom::lane::SurfaceSummary & summary_of(const SurfaceResult & result, int id)
{
  EXPECT_GT(id, 0);
  EXPECT_LE(static_cast<std::size_t>(id), result.surfaces.size());
  return result.surfaces[static_cast<std::size_t>(std::max(id, 1) - 1)];
}

}  // namespace

TEST(CellShapeOperator, CylinderHasOnePrincipalCurvature)
{
  const double radius = 200.0;
  CellMap cells;
  add_surface(cells, 0.0, 30.0, -6.0, 6.0, [radius](double, double y) {
    return std::optional<double>(0.1 - (radius - std::sqrt(radius * radius - y * y)));
  });
  const CellGrid grid = build_grid(cells);
  const CellFeatures & center = feature_at(grid, 15.25, 0.25, 0.1);
  ASSERT_TRUE(center.valid);
  ASSERT_TRUE(center.shape_valid);
  EXPECT_GT(center.normal.z(), 0.99);
  EXPECT_NEAR(std::abs(center.k1), 1.0 / radius, 0.002);
  EXPECT_LT(std::abs(center.k2), 0.001);
  EXPECT_LT(std::abs(center.gaussian_curvature), 1e-4);
  EXPECT_GT(std::abs(center.principal_direction.y()), 0.9);
  ASSERT_GE(center.weingarten_residual, 0.0);
  EXPECT_LT(center.weingarten_residual, 0.02);
  EXPECT_LT(center.shape_disagreement, 0.01);
}

TEST(CellShapeOperator, DomeHasTwoEqualCurvatures)
{
  const double radius = 2.0;
  CellMap cells;
  add_surface(cells, 4.0, 8.0, -2.0, 2.0, [radius](double x, double y) -> std::optional<double> {
    const double dx = x - 6.0;
    const double rr = dx * dx + y * y;
    if (rr > 1.9 * 1.9) {
      return std::nullopt;
    }
    return 0.1 + std::sqrt(radius * radius - rr) - radius;
  });
  const CellGrid grid = build_grid(cells);
  const CellFeatures & top = feature_at(grid, 6.25, 0.25, 0.05);
  ASSERT_TRUE(top.valid);
  ASSERT_TRUE(top.shape_valid);
  EXPECT_NEAR(std::abs(top.k1), 1.0 / radius, 0.1);
  EXPECT_NEAR(std::abs(top.k2), 1.0 / radius, 0.1);
  EXPECT_NEAR(top.gaussian_curvature, 1.0 / (radius * radius), 0.1);
}

TEST(SurfaceHough, CylinderRoadIsOneSmoothSurface)
{
  const double radius = 200.0;
  CellMap cells;
  add_surface(cells, 0.0, 30.0, -6.0, 6.0, [radius](double, double y) {
    return std::optional<double>(0.1 - (radius - std::sqrt(radius * radius - y * y)));
  });
  const CellGrid grid = build_grid(cells);
  SurfaceHoughParams params;
  const SurfaceResult result = back_odom::lane::detect_smooth_surfaces(grid, params);

  EXPECT_GE(result.hypotheses.size(), 10U);  // 5 x-tiles times 2 y-tiles
  const int a = surface_at(result, grid, 3.25, 0.25, 0.1);
  const int b = surface_at(result, grid, 15.25, 0.25, 0.1);
  const int c = surface_at(result, grid, 27.25, -4.75, 0.04);
  const int d = surface_at(result, grid, 9.25, 5.25, 0.03);
  EXPECT_GT(a, 0);
  EXPECT_EQ(a, b);
  EXPECT_EQ(a, c);
  EXPECT_EQ(a, d);
  EXPECT_TRUE(summary_of(result, a).smooth);
  EXPECT_LT(summary_of(result, a).k1_median, 0.01);
}

TEST(SurfaceHough, TiltedPlaneIsOneSmoothSurface)
{
  const double slope = std::tan(10.0 * M_PI / 180.0);
  CellMap cells;
  add_surface(cells, 0.0, 20.0, 0.0, 10.0, [slope](double x, double) {
    return std::optional<double>(0.1 + slope * x);
  });
  const CellGrid grid = build_grid(cells);
  SurfaceHoughParams params;
  const SurfaceResult result = back_odom::lane::detect_smooth_surfaces(grid, params);
  const int a = surface_at(result, grid, 2.25, 2.25, 0.1 + slope * 2.25);
  const int b = surface_at(result, grid, 17.25, 7.25, 0.1 + slope * 17.25);
  EXPECT_GT(a, 0);
  EXPECT_EQ(a, b);
  EXPECT_TRUE(summary_of(result, a).smooth);
}

TEST(SurfaceHough, CurbSplitsTwoSurfaces)
{
  CellMap cells;
  add_surface(cells, 0.0, 18.0, -5.0, 5.0, [](double, double y) {
    return std::optional<double>(y < 2.0 ? 0.1 : 0.25);
  });
  const CellGrid grid = build_grid(cells);
  SurfaceHoughParams params;
  const SurfaceResult result = back_odom::lane::detect_smooth_surfaces(grid, params);
  const int low_a = surface_at(result, grid, 9.25, -2.25, 0.1);
  const int low_b = surface_at(result, grid, 9.25, 0.75, 0.1);
  const int high = surface_at(result, grid, 9.25, 3.25, 0.25);
  const int high_far = surface_at(result, grid, 2.25, 4.25, 0.25);
  EXPECT_GT(low_a, 0);
  EXPECT_GT(high, 0);
  EXPECT_EQ(low_a, low_b);
  EXPECT_EQ(high, high_far);
  EXPECT_NE(low_a, high);
}

TEST(SurfaceHough, WallIsSeparateFromGround)
{
  CellMap cells;
  add_surface(cells, 0.0, 12.0, 0.0, 6.0, [](double, double) { return std::optional<double>(0.1); });
  add_wall(cells, 12.05, 0.0, 6.0, 0.0, 3.0);
  const CellGrid grid = build_grid(cells);
  const CellFeatures & wall_cell = feature_at(grid, 12.05, 3.25, 1.75);
  ASSERT_TRUE(wall_cell.valid);
  EXPECT_LT(wall_cell.normal.x(), -0.99);  // oriented toward the viewpoint

  SurfaceHoughParams params;
  const SurfaceResult result = back_odom::lane::detect_smooth_surfaces(grid, params);
  const int ground = surface_at(result, grid, 11.75, 3.25, 0.1);
  const int wall = surface_at(result, grid, 12.05, 3.25, 1.75);
  EXPECT_GT(ground, 0);
  EXPECT_GT(wall, 0);
  EXPECT_NE(ground, wall);
  EXPECT_TRUE(summary_of(result, wall).smooth);
}

TEST(SurfaceHough, ScatterDoesNotVote)
{
  // Filled volume: each 0.5 m cell is isotropic, so tilt uncertainty is too large to vote.
  CellMap cells;
  for (double x = 0.05; x < 6.0; x += 0.15) {
    for (double y = 0.05; y < 6.0; y += 0.15) {
      for (double z = -2.95; z < 0.0; z += 0.15) {
        back_odom::lane::add_to_cell(cells, x, y, z, 20.0F, 1.0 / kVoxel, 1.0 / kHeight);
      }
    }
  }
  const CellGrid grid = build_grid(cells);
  SurfaceHoughParams params;
  params.min_cells_per_surface = 5;
  const SurfaceResult result = back_odom::lane::detect_smooth_surfaces(grid, params);
  EXPECT_TRUE(result.hypotheses.empty());
  EXPECT_TRUE(result.surfaces.empty());
}

TEST(SurfaceHough, DomeIsNotSmooth)
{
  const double radius = 2.0;
  CellMap cells;
  add_surface(cells, 4.0, 8.0, -2.0, 2.0, [radius](double x, double y) -> std::optional<double> {
    const double dx = x - 6.0;
    const double rr = dx * dx + y * y;
    if (rr > 1.9 * 1.9) {
      return std::nullopt;
    }
    return 0.1 + std::sqrt(radius * radius - rr) - radius;
  });
  const CellGrid grid = build_grid(cells);
  SurfaceHoughParams params;
  const SurfaceResult result = back_odom::lane::detect_smooth_surfaces(grid, params);
  const int top = surface_at(result, grid, 6.25, 0.25, 0.05);
  if (top > 0) {
    EXPECT_FALSE(summary_of(result, top).smooth);
  }
  for (const auto & summary : result.surfaces) {
    EXPECT_FALSE(summary.smooth);
  }
}
