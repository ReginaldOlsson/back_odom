#include "back_odom/lane/cell_features.hpp"
#include "back_odom/lane/cell_segmentation.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <set>

namespace
{

using back_odom::lane::CellFeatures;
using back_odom::lane::CellGrid;
using back_odom::lane::CellMap;
using back_odom::lane::SegmentationParams;
using back_odom::lane::SegmentationResult;

constexpr double kVoxel = 0.5;
constexpr double kHeight = 0.5;
constexpr double kSpacing = 0.1;
constexpr int kMinPoints = 3;

/// Samples z = height(x, y), intensity = intensity(x, y) on a lattice over [x0, x1) x [y0, y1).
void add_surface(
  CellMap & cells, double x0, double x1, double y0, double y1,
  const std::function<double(double, double)> & height,
  const std::function<double(double, double)> & intensity)
{
  for (double x = x0 + 0.5 * kSpacing; x < x1; x += kSpacing) {
    for (double y = y0 + 0.5 * kSpacing; y < y1; y += kSpacing) {
      back_odom::lane::add_to_cell(
        cells, x, y, height(x, y), static_cast<float>(intensity(x, y)), 1.0 / kVoxel,
        1.0 / kHeight);
    }
  }
}

/// Vertical wall at x = wall_x spanning y in [y0, y1) and z in [z0, z1).
void add_wall(CellMap & cells, double wall_x, double y0, double y1, double z0, double z1, double value)
{
  for (double y = y0 + 0.5 * kSpacing; y < y1; y += kSpacing) {
    for (double z = z0 + 0.5 * kSpacing; z < z1; z += kSpacing) {
      back_odom::lane::add_to_cell(
        cells, wall_x, y, z, static_cast<float>(value), 1.0 / kVoxel, 1.0 / kHeight);
    }
  }
}

CellGrid build_grid(const CellMap & cells)
{
  CellGrid grid = back_odom::lane::flatten(cells);
  back_odom::lane::compute_local_features(grid, kMinPoints);
  back_odom::lane::compute_neighbor_features(grid, kVoxel, kHeight);
  return grid;
}

const CellFeatures & feature_at(const CellGrid & grid, double x, double y, double z)
{
  const int index = grid.find(back_odom::lane::cell_key(x, y, z, 1.0 / kVoxel, 1.0 / kHeight));
  EXPECT_GE(index, 0) << "no cell at " << x << ", " << y << ", " << z;
  return grid.features[static_cast<std::size_t>(std::max(index, 0))];
}

}  // namespace

TEST(CellFeatures, FlatPlaneIsPlanarAndFlat)
{
  CellMap cells;
  add_surface(
    cells, 0.0, 5.0, 0.0, 5.0, [](double, double) { return 0.1; },
    [](double, double) { return 20.0; });
  const CellGrid grid = build_grid(cells);
  const CellFeatures & center = feature_at(grid, 2.25, 2.25, 0.1);
  ASSERT_TRUE(center.valid);
  EXPECT_GT(center.planarity, 0.9);
  EXPECT_LT(center.verticality, 0.05);
  EXPECT_LT(center.curvature, 1e-3);
  EXPECT_LT(center.intensity_cv, 1e-6);
  EXPECT_LT(center.intensity_laplacian, 1e-6);
}

TEST(CellFeatures, TiltedPlaneHasNoCurvature)
{
  // The LBO of position is intrinsic: a banked road is still flat.
  const double slope = std::tan(10.0 * M_PI / 180.0);
  CellMap cells;
  add_surface(
    cells, 0.0, 6.0, 0.0, 4.0, [slope](double x, double) { return 0.1 + slope * x; },
    [](double, double) { return 20.0; });
  const CellGrid grid = build_grid(cells);
  const CellFeatures & center = feature_at(grid, 3.25, 2.25, 0.1 + slope * 3.25);
  ASSERT_TRUE(center.valid);
  EXPECT_GT(center.planarity, 0.9);
  EXPECT_GT(center.verticality, 0.01);
  EXPECT_LT(center.curvature, 1e-3);
}

TEST(CellFeatures, CurbStepHasCurvatureOnlyAtCrease)
{
  CellMap cells;
  add_surface(
    cells, 0.0, 5.0, 0.0, 4.0, [](double x, double) { return x < 2.5 ? 0.1 : 0.25; },
    [](double, double) { return 20.0; });
  const CellGrid grid = build_grid(cells);
  const CellFeatures & road_side = feature_at(grid, 2.25, 2.25, 0.1);
  const CellFeatures & walk_side = feature_at(grid, 2.75, 2.25, 0.25);
  const CellFeatures & far_road = feature_at(grid, 0.75, 2.25, 0.1);
  const CellFeatures & far_walk = feature_at(grid, 4.25, 2.25, 0.25);
  ASSERT_TRUE(road_side.valid && walk_side.valid && far_road.valid && far_walk.valid);
  EXPECT_GT(road_side.curvature, 0.1);
  EXPECT_GT(walk_side.curvature, 0.1);
  EXPECT_LT(far_road.curvature, 1e-3);
  EXPECT_LT(far_walk.curvature, 1e-3);
}

TEST(CellFeatures, IntensityStepHasLaplacianOnlyAtEdge)
{
  CellMap cells;
  add_surface(
    cells, 0.0, 4.0, 0.0, 5.0, [](double, double) { return 0.1; },
    [](double, double y) { return y < 2.5 ? 20.0 : 60.0; });
  const CellGrid grid = build_grid(cells);
  const CellFeatures & low_side = feature_at(grid, 2.25, 2.25, 0.1);
  const CellFeatures & high_side = feature_at(grid, 2.25, 2.75, 0.1);
  const CellFeatures & far_low = feature_at(grid, 2.25, 0.75, 0.1);
  const CellFeatures & far_high = feature_at(grid, 2.25, 4.25, 0.1);
  ASSERT_TRUE(low_side.valid && high_side.valid && far_low.valid && far_high.valid);
  EXPECT_GT(low_side.intensity_laplacian, 0.2);
  EXPECT_GT(high_side.intensity_laplacian, 0.05);
  EXPECT_LT(far_low.intensity_laplacian, 1e-6);
  EXPECT_LT(far_high.intensity_laplacian, 1e-6);
  EXPECT_LT(low_side.intensity_cv, 1e-6);
  EXPECT_LT(low_side.curvature, 1e-3);
}

TEST(CellSegmentation, SmallRegionStillFormsAClass)
{
  // 3 x 3 m plane: 36 cells x 25 points = 900 raw votes in one bin. A blurred 5D accumulator
  // would show that bin at 900 / 32 < peak_min_votes; thresholding on raw mass must keep it.
  CellMap cells;
  add_surface(
    cells, 0.0, 3.0, 0.0, 3.0, [](double, double) { return 0.1; },
    [](double, double) { return 20.0; });
  const CellGrid grid = build_grid(cells);
  SegmentationParams params;
  const SegmentationResult result = back_odom::lane::segment_cells(grid, params, {});
  ASSERT_EQ(result.peaks.size(), 1U);
  const int index = grid.find(back_odom::lane::cell_key(1.25, 1.25, 0.1, 1.0 / kVoxel, 1.0 / kHeight));
  ASSERT_GE(index, 0);
  EXPECT_EQ(result.label[static_cast<std::size_t>(index)], 1);
}

TEST(CellSegmentation, PlaneAndWallGetTwoStableClasses)
{
  CellMap cells;
  add_surface(
    cells, 0.0, 10.0, 0.0, 10.0, [](double, double) { return 0.1; },
    [](double, double) { return 20.0; });
  // Wall is separated from the plane so neither region contaminates the other's votes.
  add_wall(cells, 12.05, 0.0, 10.0, 0.0, 3.0, 20.0);
  const CellGrid grid = build_grid(cells);

  SegmentationParams params;
  const SegmentationResult first = back_odom::lane::segment_cells(grid, params, {});
  ASSERT_EQ(first.peaks.size(), 2U);

  auto label_at = [&grid](const SegmentationResult & result, double x, double y, double z) {
    const int index = grid.find(back_odom::lane::cell_key(x, y, z, 1.0 / kVoxel, 1.0 / kHeight));
    EXPECT_GE(index, 0);
    return result.label[static_cast<std::size_t>(std::max(index, 0))];
  };

  const int road = label_at(first, 4.25, 4.25, 0.1);
  const int wall = label_at(first, 12.05, 4.25, 1.75);
  EXPECT_GT(road, 0);
  EXPECT_GT(wall, 0);
  EXPECT_NE(road, wall);
  EXPECT_EQ(label_at(first, 6.25, 2.25, 0.1), road);
  EXPECT_EQ(label_at(first, 12.05, 7.25, 2.25), wall);

  std::set<int> ids;
  for (const auto & peak : first.peaks) {
    ids.insert(peak.id);
  }
  EXPECT_EQ(ids, (std::set<int>{1, 2}));

  const SegmentationResult second = back_odom::lane::segment_cells(grid, params, first.peaks);
  ASSERT_EQ(second.peaks.size(), 2U);
  EXPECT_EQ(label_at(second, 4.25, 4.25, 0.1), road);
  EXPECT_EQ(label_at(second, 12.05, 4.25, 1.75), wall);

  // Connected components: the road interior is one segment.
  const int seg_a = second.segment[static_cast<std::size_t>(
    grid.find(back_odom::lane::cell_key(4.25, 4.25, 0.1, 1.0 / kVoxel, 1.0 / kHeight)))];
  const int seg_b = second.segment[static_cast<std::size_t>(
    grid.find(back_odom::lane::cell_key(6.25, 2.25, 0.1, 1.0 / kVoxel, 1.0 / kHeight)))];
  EXPECT_GT(seg_a, 0);
  EXPECT_EQ(seg_a, seg_b);
}
