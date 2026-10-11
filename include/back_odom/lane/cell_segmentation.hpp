#ifndef BACK_ODOM__LANE__CELL_SEGMENTATION_HPP_
#define BACK_ODOM__LANE__CELL_SEGMENTATION_HPP_

#include "back_odom/lane/cell_features.hpp"

#include <Eigen/Core>

#include <vector>

namespace back_odom
{
namespace lane
{

/// Feature axes: planarity, verticality, curvature, intensity CV, intensity Laplacian.
constexpr int kFeatureDim = 5;
using FeatureVector = Eigen::Matrix<double, kFeatureDim, 1>;

struct SegmentationParams
{
  /// Bins per feature axis. The accumulator has bins^kFeatureDim entries.
  int bins{12};
  /// Clamp limits before normalizing each axis to [0, 1].
  double curvature_max{0.5};
  double cv_max{1.0};
  double laplacian_max{1.0};
  /// Minimum raw votes within peak_min_separation bins of a local maximum to form a class.
  /// Votes are min(points, 32) per cell, so 50 is roughly a few dense cells.
  double peak_min_votes{50.0};
  /// Chebyshev distance in bins below which two peaks are merged.
  int peak_min_separation{2};
  int max_classes{8};
  /// Gaussian width in normalized feature units for the cell-graph weights.
  double feature_sigma{0.25};
  int propagation_iterations{3};
};

struct Peak
{
  /// Normalized feature coordinates in [0, 1].
  FeatureVector center{FeatureVector::Zero()};
  double votes{0.0};
  int id{0};
};

struct SegmentationResult
{
  /// Class id per cell, 0 = unlabeled.
  std::vector<int> label;
  /// Connected-component id per cell, 0 = unlabeled.
  std::vector<int> segment;
  std::vector<Peak> peaks;
};

/// Normalized feature vector of a cell, each entry clamped to [0, 1].
FeatureVector feature_vector(const CellFeatures & features, const SegmentationParams & params);

/// Votes cell features into an accumulator, takes its peaks as classes (ids kept stable against
/// previous_peaks), labels cells by nearest peak, smooths labels on the cell graph, and splits
/// each class into connected components.
SegmentationResult segment_cells(
  const CellGrid & grid, const SegmentationParams & params,
  const std::vector<Peak> & previous_peaks);

}  // namespace lane
}  // namespace back_odom

#endif  // BACK_ODOM__LANE__CELL_SEGMENTATION_HPP_
