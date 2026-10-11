#ifndef BACK_ODOM__LANE__CELL_FEATURES_HPP_
#define BACK_ODOM__LANE__CELL_FEATURES_HPP_

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace back_odom
{
namespace lane
{

struct CellKey
{
  std::int64_t x{0};
  std::int64_t y{0};
  std::int64_t z{0};

  bool operator==(const CellKey & other) const
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct CellKeyHash
{
  std::size_t operator()(const CellKey & key) const
  {
    return (static_cast<std::size_t>(key.x) * 1315423911U) ^
           (static_cast<std::size_t>(key.y) * 2654435761U) ^
           (static_cast<std::size_t>(key.z) * 97531U);
  }
};

/// Running sums for one voxel cell. Unweighted sums give the geometry distribution,
/// intensity-weighted sums give the intensity distribution over the cell.
struct CellStats
{
  double sum_intensity{0.0};
  double sum_intensity_sq{0.0};

  double sum_x{0.0};
  double sum_y{0.0};
  double sum_z{0.0};
  double sum_xx{0.0};
  double sum_yy{0.0};
  double sum_zz{0.0};
  double sum_xy{0.0};
  double sum_xz{0.0};
  double sum_yz{0.0};

  double sum_weight{0.0};
  double sum_wx{0.0};
  double sum_wy{0.0};
  double sum_wz{0.0};
  double sum_wxx{0.0};
  double sum_wyy{0.0};
  double sum_wzz{0.0};
  double sum_wxy{0.0};
  double sum_wxz{0.0};
  double sum_wyz{0.0};

  int count{0};
};

using CellMap = std::unordered_map<CellKey, CellStats, CellKeyHash>;

CellKey cell_key(double x, double y, double z, double inverse_voxel, double inverse_height);

/// Adds one point to its cell and returns the cell key.
CellKey add_to_cell(
  CellMap & cells, double x, double y, double z, float intensity, double inverse_voxel,
  double inverse_height);

struct IntensityPca
{
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d axes{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d radii{Eigen::Vector3d::Zero()};
  bool valid{false};
};

/// Intensity-weighted mean is the PCA origin. Covariance is about that mean, so the red axis
/// starts on the principal component and stays parallel to it.
IntensityPca fit_intensity_pca(const CellStats & cell, int min_points);

struct CellFeatures
{
  Eigen::Vector3d mean{Eigen::Vector3d::Zero()};
  /// Columns: e1, e2 (tangent, largest to middle), e3 (normal, smallest).
  Eigen::Matrix3d axes{Eigen::Matrix3d::Identity()};
  /// Descending eigenvalues of the unweighted covariance.
  Eigen::Vector3d lambdas{Eigen::Vector3d::Zero()};
  double mean_intensity{0.0};
  /// Coefficient of variation sigma_I / mean_I. Range falloff cancels inside a cell.
  double intensity_cv{0.0};
  /// (lambda2 - lambda3) / lambda1 in [0, 1].
  double planarity{0.0};
  /// 1 - |e3 . z| in [0, 1]. Zero for a horizontal surface.
  double verticality{0.0};
  /// |H| from a quadric over neighbor means in the tangent frame. Zero if too few neighbors.
  double curvature{0.0};
  /// |Laplace-Beltrami of mean intensity| / mean_I over tangent-plane neighbors.
  double intensity_laplacian{0.0};
  bool valid{false};
};

/// Flattened cells with a key index and a 26-neighbor adjacency in CSR form.
struct CellGrid
{
  std::vector<CellKey> keys;
  std::vector<CellStats> stats;
  std::vector<CellFeatures> features;
  std::unordered_map<CellKey, int, CellKeyHash> index;
  /// neighbor_indices[neighbor_offsets[i] .. neighbor_offsets[i + 1]) are the neighbors of i.
  std::vector<int> neighbor_offsets;
  std::vector<int> neighbor_indices;

  std::size_t size() const { return keys.size(); }
  int find(const CellKey & key) const;
};

CellGrid flatten(const CellMap & cells);

/// Pass A, independent per cell: unweighted mean, covariance, PCA, planarity, verticality, CV.
void compute_local_features(CellGrid & grid, int min_points);

/// Pass B over 26-neighbors: mean curvature and surface Laplacian of intensity.
void compute_neighbor_features(CellGrid & grid, double voxel_size, double voxel_height);

}  // namespace lane
}  // namespace back_odom

#endif  // BACK_ODOM__LANE__CELL_FEATURES_HPP_
