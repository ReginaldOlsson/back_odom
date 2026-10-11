#include "back_odom/lane/cell_features.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>

namespace back_odom
{
namespace lane
{
namespace
{

/// Eigen-decomposition with columns ordered largest to smallest and a right-handed frame.
bool sorted_eigen(
  const Eigen::Matrix3d & covariance, Eigen::Matrix3d & axes, Eigen::Vector3d & values)
{
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  if (solver.info() != Eigen::Success) {
    return false;
  }
  const Eigen::Vector3d ascending = solver.eigenvalues();
  if (!(ascending.maxCoeff() > 1e-10)) {
    return false;
  }
  axes.col(0) = solver.eigenvectors().col(2);
  axes.col(1) = solver.eigenvectors().col(1);
  axes.col(2) = solver.eigenvectors().col(0);
  if (axes.determinant() < 0.0) {
    axes.col(2) *= -1.0;
  }
  values = Eigen::Vector3d(
    std::max(ascending[2], 0.0), std::max(ascending[1], 0.0), std::max(ascending[0], 0.0));
  return true;
}

}  // namespace

CellKey cell_key(double x, double y, double z, double inverse_voxel, double inverse_height)
{
  return CellKey{
    static_cast<std::int64_t>(std::floor(x * inverse_voxel)),
    static_cast<std::int64_t>(std::floor(y * inverse_voxel)),
    static_cast<std::int64_t>(std::floor(z * inverse_height))};
}

CellKey add_to_cell(
  CellMap & cells, double x, double y, double z, float intensity, double inverse_voxel,
  double inverse_height)
{
  const CellKey key = cell_key(x, y, z, inverse_voxel, inverse_height);
  CellStats & cell = cells[key];
  const double value = intensity;
  cell.sum_intensity += value;
  cell.sum_intensity_sq += value * value;

  cell.sum_x += x;
  cell.sum_y += y;
  cell.sum_z += z;
  cell.sum_xx += x * x;
  cell.sum_yy += y * y;
  cell.sum_zz += z * z;
  cell.sum_xy += x * y;
  cell.sum_xz += x * z;
  cell.sum_yz += y * z;

  const double weight = std::max(value, 0.0);
  cell.sum_weight += weight;
  cell.sum_wx += weight * x;
  cell.sum_wy += weight * y;
  cell.sum_wz += weight * z;
  cell.sum_wxx += weight * x * x;
  cell.sum_wyy += weight * y * y;
  cell.sum_wzz += weight * z * z;
  cell.sum_wxy += weight * x * y;
  cell.sum_wxz += weight * x * z;
  cell.sum_wyz += weight * y * z;
  ++cell.count;
  return key;
}

IntensityPca fit_intensity_pca(const CellStats & cell, int min_points)
{
  IntensityPca pca;
  if (cell.count < min_points || !(cell.sum_weight > 1e-6)) {
    return pca;
  }
  const double inverse = 1.0 / cell.sum_weight;
  const double mx = cell.sum_wx * inverse;
  const double my = cell.sum_wy * inverse;
  const double mz = cell.sum_wz * inverse;
  pca.center = Eigen::Vector3d(mx, my, mz);
  Eigen::Matrix3d covariance;
  covariance << cell.sum_wxx * inverse - mx * mx, cell.sum_wxy * inverse - mx * my,
    cell.sum_wxz * inverse - mx * mz, cell.sum_wxy * inverse - mx * my,
    cell.sum_wyy * inverse - my * my, cell.sum_wyz * inverse - my * mz,
    cell.sum_wxz * inverse - mx * mz, cell.sum_wyz * inverse - my * mz,
    cell.sum_wzz * inverse - mz * mz;
  Eigen::Vector3d values;
  if (!sorted_eigen(covariance, pca.axes, values)) {
    return pca;
  }
  pca.radii = values.cwiseSqrt();
  pca.valid = true;
  return pca;
}

int CellGrid::find(const CellKey & key) const
{
  const auto found = index.find(key);
  return found == index.end() ? -1 : found->second;
}

CellGrid flatten(const CellMap & cells)
{
  CellGrid grid;
  grid.keys.reserve(cells.size());
  grid.stats.reserve(cells.size());
  grid.index.reserve(cells.size());
  for (const auto & item : cells) {
    grid.index.emplace(item.first, static_cast<int>(grid.keys.size()));
    grid.keys.push_back(item.first);
    grid.stats.push_back(item.second);
  }
  grid.features.resize(grid.keys.size());

  grid.neighbor_offsets.resize(grid.keys.size() + 1, 0);
  grid.neighbor_indices.reserve(grid.keys.size() * 8);
  for (std::size_t i = 0; i < grid.keys.size(); ++i) {
    const CellKey & key = grid.keys[i];
    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0 && dz == 0) {
            continue;
          }
          const int neighbor = grid.find(CellKey{key.x + dx, key.y + dy, key.z + dz});
          if (neighbor >= 0) {
            grid.neighbor_indices.push_back(neighbor);
          }
        }
      }
    }
    grid.neighbor_offsets[i + 1] = static_cast<int>(grid.neighbor_indices.size());
  }
  return grid;
}

void compute_local_features(CellGrid & grid, int min_points)
{
  const int required = std::max(min_points, 3);
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const CellStats & cell = grid.stats[i];
    CellFeatures & feature = grid.features[i];
    feature = CellFeatures{};
    if (cell.count < required) {
      continue;
    }
    const double inverse = 1.0 / static_cast<double>(cell.count);
    const double mx = cell.sum_x * inverse;
    const double my = cell.sum_y * inverse;
    const double mz = cell.sum_z * inverse;
    feature.mean = Eigen::Vector3d(mx, my, mz);
    Eigen::Matrix3d covariance;
    covariance << cell.sum_xx * inverse - mx * mx, cell.sum_xy * inverse - mx * my,
      cell.sum_xz * inverse - mx * mz, cell.sum_xy * inverse - mx * my,
      cell.sum_yy * inverse - my * my, cell.sum_yz * inverse - my * mz,
      cell.sum_xz * inverse - mx * mz, cell.sum_yz * inverse - my * mz,
      cell.sum_zz * inverse - mz * mz;
    if (!sorted_eigen(covariance, feature.axes, feature.lambdas)) {
      continue;
    }
    feature.planarity =
      std::clamp((feature.lambdas[1] - feature.lambdas[2]) / feature.lambdas[0], 0.0, 1.0);
    feature.verticality = std::clamp(1.0 - std::abs(feature.axes(2, 2)), 0.0, 1.0);

    feature.mean_intensity = cell.sum_intensity * inverse;
    const double variance =
      std::max(cell.sum_intensity_sq * inverse - feature.mean_intensity * feature.mean_intensity, 0.0);
    feature.intensity_cv = std::sqrt(variance) / std::max(std::abs(feature.mean_intensity), 1e-6);
    feature.valid = true;
  }
}

void compute_neighbor_features(CellGrid & grid, double voxel_size, double voxel_height)
{
  const double inverse_sigma_sq = 1.0 / std::max(voxel_size * voxel_size, 1e-12);
  const double max_height = std::max(voxel_height, 1e-6);
  for (std::size_t i = 0; i < grid.size(); ++i) {
    CellFeatures & feature = grid.features[i];
    if (!feature.valid) {
      continue;
    }
    const Eigen::Matrix3d frame = feature.axes.transpose();

    // Quadric h = a u^2 + b uv + c v^2 + d u + e v + f via normal equations.
    Eigen::Matrix<double, 6, 6> normal = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> rhs = Eigen::Matrix<double, 6, 1>::Zero();
    int rows = 0;
    auto add_row = [&normal, &rhs, &rows](double u, double v, double h) {
      Eigen::Matrix<double, 6, 1> row;
      row << u * u, u * v, v * v, u, v, 1.0;
      normal.noalias() += row * row.transpose();
      rhs.noalias() += row * h;
      ++rows;
    };
    add_row(0.0, 0.0, 0.0);

    double laplacian_numerator = 0.0;
    double laplacian_weight = 0.0;

    const int begin = grid.neighbor_offsets[i];
    const int end = grid.neighbor_offsets[i + 1];
    for (int n = begin; n < end; ++n) {
      const CellFeatures & neighbor = grid.features[static_cast<std::size_t>(grid.neighbor_indices[n])];
      if (!neighbor.valid) {
        continue;
      }
      const Eigen::Vector3d local = frame * (neighbor.mean - feature.mean);
      const double u = local.x();
      const double v = local.y();
      const double h = local.z();
      add_row(u, v, h);
      if (std::abs(h) <= max_height) {
        const double weight = std::exp(-(u * u + v * v) * inverse_sigma_sq);
        laplacian_numerator += weight * (neighbor.mean_intensity - feature.mean_intensity);
        laplacian_weight += weight;
      }
    }

    if (rows >= 6) {
      normal.diagonal().array() += 1e-9;
      const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> solver(normal);
      if (solver.info() == Eigen::Success) {
        const Eigen::Matrix<double, 6, 1> coefficients = solver.solve(rhs);
        feature.curvature = std::abs(coefficients[0] + coefficients[2]);
      }
    }
    if (laplacian_weight > 1e-9) {
      feature.intensity_laplacian = std::abs(laplacian_numerator / laplacian_weight) /
                                    std::max(std::abs(feature.mean_intensity), 1e-6);
    }
  }
}

}  // namespace lane
}  // namespace back_odom
