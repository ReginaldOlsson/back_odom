#include "back_odom/lane/cell_surface_hough.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>

namespace back_odom
{
namespace lane
{
namespace
{

struct TileKey
{
  int x{0};
  int y{0};
  bool operator==(const TileKey & other) const { return x == other.x && y == other.y; }
};

struct TileKeyHash
{
  std::size_t operator()(const TileKey & key) const
  {
    return static_cast<std::size_t>(key.x) * 73856093U ^ static_cast<std::size_t>(key.y) * 19349663U;
  }
};

struct BinKey
{
  int direction{0};
  std::int64_t rho{0};
  bool operator==(const BinKey & other) const
  {
    return direction == other.direction && rho == other.rho;
  }
};

struct BinKeyHash
{
  std::size_t operator()(const BinKey & key) const
  {
    return static_cast<std::size_t>(key.direction) * 2654435761U ^
           static_cast<std::size_t>(key.rho) * 97531U;
  }
};

/// Accumulated votes of one (direction, rho) bin, with sums for a sub-bin estimate.
struct Bin
{
  double weight{0.0};
  Eigen::Vector3d normal_sum{Eigen::Vector3d::Zero()};
  double rho_sum{0.0};
};

struct Tile
{
  TileKey key;
  std::vector<int> cells;
  std::unordered_map<BinKey, Bin, BinKeyHash> bins;
};

/// Cube-map bin of a unit vector: 6 faces x face_bins^2, no pole singularity.
int direction_bin(const Eigen::Vector3d & n, int face_bins)
{
  int axis = 0;
  if (std::abs(n.y()) > std::abs(n[axis])) {
    axis = 1;
  }
  if (std::abs(n.z()) > std::abs(n[axis])) {
    axis = 2;
  }
  const int face = 2 * axis + (n[axis] < 0.0 ? 1 : 0);
  const double inverse = 1.0 / std::max(std::abs(n[axis]), 1e-12);
  const double s = n[(axis + 1) % 3] * inverse;
  const double t = n[(axis + 2) % 3] * inverse;
  const int si = std::clamp(
    static_cast<int>(std::floor((s + 1.0) * 0.5 * face_bins)), 0, face_bins - 1);
  const int ti = std::clamp(
    static_cast<int>(std::floor((t + 1.0) * 0.5 * face_bins)), 0, face_bins - 1);
  return (face * face_bins + si) * face_bins + ti;
}

double angle_between(const Eigen::Vector3d & a, const Eigen::Vector3d & b)
{
  return std::acos(std::clamp(a.dot(b), -1.0, 1.0));
}

bool votes(const CellFeatures & feature, const SurfaceHoughParams & params)
{
  if (!has_reliable_normal(feature, params.max_normal_sigma)) {
    return false;
  }
  return feature.weingarten_residual < 0.0 || feature.weingarten_residual <= params.max_residual;
}

/// Greedy peak extraction: highest remaining bin seeds a ball (angle, offset) whose raw mass is
/// the peak score. Bins inside the ball are consumed (non-maximum suppression).
void extract_tile_peaks(
  const Tile & tile, const SurfaceHoughParams & params, std::vector<PlaneHypothesis> & hypotheses)
{
  struct Entry
  {
    Eigen::Vector3d direction;
    double rho;
    const Bin * bin;
    bool consumed;
  };
  std::vector<Entry> entries;
  entries.reserve(tile.bins.size());
  for (const auto & item : tile.bins) {
    if (!(item.second.weight > 0.0)) {
      continue;
    }
    Entry entry;
    entry.direction = item.second.normal_sum.normalized();
    entry.rho = item.second.rho_sum / item.second.weight;
    entry.bin = &item.second;
    entry.consumed = false;
    entries.push_back(entry);
  }
  std::sort(entries.begin(), entries.end(), [](const Entry & a, const Entry & b) {
    return a.bin->weight > b.bin->weight;
  });

  const double offset_ball = std::max(params.offset_tolerance, params.rho_bin);
  int peaks = 0;
  for (Entry & seed : entries) {
    if (peaks >= params.max_planes_per_tile) {
      break;
    }
    if (seed.consumed) {
      continue;
    }
    double mass = 0.0;
    Eigen::Vector3d normal_sum = Eigen::Vector3d::Zero();
    double rho_sum = 0.0;
    for (Entry & other : entries) {
      if (other.consumed) {
        continue;
      }
      if (
        angle_between(other.direction, seed.direction) <= params.angle_tolerance &&
        std::abs(other.rho - seed.rho) <= offset_ball)
      {
        other.consumed = true;
        mass += other.bin->weight;
        normal_sum += other.bin->normal_sum;
        rho_sum += other.bin->rho_sum;
      }
    }
    if (mass < params.peak_min_mass || normal_sum.squaredNorm() < 1e-12) {
      continue;
    }
    PlaneHypothesis hypothesis;
    hypothesis.tile_x = tile.key.x;
    hypothesis.tile_y = tile.key.y;
    hypothesis.normal = normal_sum.normalized();
    hypothesis.rho = rho_sum / mass;
    hypothesis.mass = mass;
    hypotheses.push_back(hypothesis);
    ++peaks;
  }
}

/// Weighted plane fit over the cells of the tile that lie inside the Hough ball.
bool refine_hypothesis(
  const CellGrid & grid, const Tile & tile, const SurfaceHoughParams & params,
  PlaneHypothesis & hypothesis)
{
  const double offset_ball = std::max(params.offset_tolerance, params.rho_bin);
  double total = 0.0;
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  Eigen::Matrix3d second = Eigen::Matrix3d::Zero();
  int support = 0;
  for (int index : tile.cells) {
    const CellFeatures & feature = grid.features[static_cast<std::size_t>(index)];
    if (!votes(feature, params)) {
      continue;
    }
    if (angle_between(feature.normal, hypothesis.normal) > params.angle_tolerance) {
      continue;
    }
    if (std::abs(hypothesis.normal.dot(feature.mean) - hypothesis.rho) > offset_ball) {
      continue;
    }
    const double weight =
      (1.0 - feature.normal_sigma * feature.normal_sigma) *
      std::min(static_cast<double>(grid.stats[static_cast<std::size_t>(index)].count), params.vote_cap);
    total += weight;
    centroid += weight * feature.mean;
    second.noalias() += weight * feature.mean * feature.mean.transpose();
    ++support;
  }
  if (support < 3 || !(total > 0.0)) {
    return false;
  }
  centroid /= total;
  const Eigen::Matrix3d covariance = second / total - centroid * centroid.transpose();
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  hypothesis.centroid = centroid;
  hypothesis.support = support;
  if (solver.info() == Eigen::Success) {
    Eigen::Vector3d normal = solver.eigenvectors().col(0);
    if (normal.dot(hypothesis.normal) < 0.0) {
      normal = -normal;
    }
    // Keep the Hough normal when the support is nearly collinear (plane fit ill-posed).
    const Eigen::Vector3d values = solver.eigenvalues();
    if (values[1] > 1e-6) {
      hypothesis.normal = normal;
    }
  }
  hypothesis.rho = hypothesis.normal.dot(hypothesis.centroid);
  return true;
}

struct UnionFind
{
  std::vector<int> parent;
  explicit UnionFind(std::size_t size) : parent(size)
  {
    std::iota(parent.begin(), parent.end(), 0);
  }
  int find(int a)
  {
    while (parent[static_cast<std::size_t>(a)] != a) {
      parent[static_cast<std::size_t>(a)] = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(a)])];
      a = parent[static_cast<std::size_t>(a)];
    }
    return a;
  }
  void unite(int a, int b)
  {
    a = find(a);
    b = find(b);
    if (a != b) {
      parent[static_cast<std::size_t>(b)] = a;
    }
  }
};

/// Second-order consistency between two tangent planes separated by X: the normal change gives
/// kappa_n and a twist, and the offset must match the osculating paraboloid h = -1/2 dN . X.
bool weingarten_consistent(
  const PlaneHypothesis & k, const PlaneHypothesis & l, const SurfaceHoughParams & params)
{
  const Eigen::Vector3d X = l.centroid - k.centroid;
  const double distance_sq = X.squaredNorm();
  if (distance_sq < 1e-12) {
    return true;
  }
  const double distance = std::sqrt(distance_sq);
  const Eigen::Vector3d dn = l.normal - k.normal;
  const double kappa_n = dn.dot(X) / distance_sq;
  const double twist = (dn - kappa_n * X).norm() / distance;
  if (std::abs(kappa_n) > params.smooth_curvature_max || twist > params.smooth_curvature_max) {
    return false;
  }
  const double predicted = -0.5 * dn.dot(X);
  const double tolerance = params.offset_tolerance + 0.5 * params.normal_noise * distance;
  const double measured_k = k.normal.dot(X);
  const double measured_l = -l.normal.dot(X);
  return std::abs(measured_k - predicted) <= tolerance &&
         std::abs(measured_l - predicted) <= tolerance;
}

double percentile(std::vector<double> & values, double fraction)
{
  if (values.empty()) {
    return 0.0;
  }
  const std::size_t rank = std::min(
    values.size() - 1, static_cast<std::size_t>(std::floor(fraction * (values.size() - 1))));
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(rank), values.end());
  return values[rank];
}

}  // namespace

SurfaceResult detect_smooth_surfaces(const CellGrid & grid, const SurfaceHoughParams & params)
{
  SurfaceResult result;
  result.surface.assign(grid.size(), 0);
  result.hypothesis.assign(grid.size(), -1);
  if (grid.size() == 0 || !(params.tile_size > 0.0)) {
    return result;
  }

  // 1. Tiles and votes.
  const double inverse_tile = 1.0 / params.tile_size;
  const double inverse_rho = 1.0 / std::max(params.rho_bin, 1e-6);
  const int face_bins = std::max(params.face_bins, 2);
  std::unordered_map<TileKey, Tile, TileKeyHash> tiles;
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const CellFeatures & feature = grid.features[i];
    if (!feature.valid) {
      continue;
    }
    const TileKey key{
      static_cast<int>(std::floor(feature.mean.x() * inverse_tile)),
      static_cast<int>(std::floor(feature.mean.y() * inverse_tile))};
    Tile & tile = tiles[key];
    tile.key = key;
    tile.cells.push_back(static_cast<int>(i));
    if (!votes(feature, params)) {
      continue;
    }
    const double rho = feature.normal.dot(feature.mean);
    const double weight = (1.0 - feature.normal_sigma * feature.normal_sigma) *
                          std::min(static_cast<double>(grid.stats[i].count), params.vote_cap);
    Bin & bin = tile.bins[BinKey{
      direction_bin(feature.normal, face_bins),
      static_cast<std::int64_t>(std::floor(rho * inverse_rho))}];
    bin.weight += weight;
    bin.normal_sum += weight * feature.normal;
    bin.rho_sum += weight * rho;
  }

  // 2. Peaks per tile, refined by a weighted plane fit over their supporting cells.
  std::unordered_map<TileKey, std::vector<int>, TileKeyHash> tile_hypotheses;
  for (const auto & item : tiles) {
    std::vector<PlaneHypothesis> peaks;
    extract_tile_peaks(item.second, params, peaks);
    for (PlaneHypothesis & hypothesis : peaks) {
      if (!refine_hypothesis(grid, item.second, params, hypothesis)) {
        continue;
      }
      tile_hypotheses[item.first].push_back(static_cast<int>(result.hypotheses.size()));
      result.hypotheses.push_back(hypothesis);
    }
  }
  if (result.hypotheses.empty()) {
    return result;
  }

  // 3. Assign cells to the best second-order-consistent hypothesis in the 3x3 tile window.
  std::vector<int> cells_per_hypothesis(result.hypotheses.size(), 0);
  for (const auto & item : tiles) {
    for (int index : item.second.cells) {
      const CellFeatures & feature = grid.features[static_cast<std::size_t>(index)];
      if (!has_reliable_normal(feature, params.max_normal_sigma)) {
        continue;
      }
      double best_score = std::numeric_limits<double>::infinity();
      int best = -1;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const auto found = tile_hypotheses.find(TileKey{item.first.x + dx, item.first.y + dy});
          if (found == tile_hypotheses.end()) {
            continue;
          }
          for (int h : found->second) {
            const PlaneHypothesis & hypothesis = result.hypotheses[static_cast<std::size_t>(h)];
            const Eigen::Vector3d X = feature.mean - hypothesis.centroid;
            const double r = X.norm();
            const double angle = angle_between(feature.normal, hypothesis.normal);
            const double angle_tolerance =
              params.angle_tolerance + params.smooth_curvature_max * r;
            if (angle > angle_tolerance) {
              continue;
            }
            const double measured = hypothesis.normal.dot(X);
            const double predicted = -0.5 * (feature.normal - hypothesis.normal).dot(X);
            const double offset_tolerance =
              params.offset_tolerance + 0.5 * params.normal_noise * r;
            const double offset_error = std::abs(measured - predicted);
            if (offset_error > offset_tolerance) {
              continue;
            }
            const double score = (angle / angle_tolerance) * (angle / angle_tolerance) +
                                 (offset_error / offset_tolerance) * (offset_error / offset_tolerance);
            if (score < best_score) {
              best_score = score;
              best = h;
            }
          }
        }
      }
      if (best >= 0) {
        result.hypothesis[static_cast<std::size_t>(index)] = best;
        ++cells_per_hypothesis[static_cast<std::size_t>(best)];
      }
    }
  }

  // 4. Stitch hypotheses of adjacent tiles whose Weingarten map is bounded and consistent.
  UnionFind components(result.hypotheses.size());
  for (const auto & item : tile_hypotheses) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0) {
          continue;
        }
        const auto found = tile_hypotheses.find(TileKey{item.first.x + dx, item.first.y + dy});
        if (found == tile_hypotheses.end()) {
          continue;
        }
        for (int k : item.second) {
          for (int l : found->second) {
            if (k < l && weingarten_consistent(
                           result.hypotheses[static_cast<std::size_t>(k)],
                           result.hypotheses[static_cast<std::size_t>(l)], params))
            {
              components.unite(k, l);
            }
          }
        }
      }
    }
  }

  // 5. Surface ids: compact component roots, drop small surfaces.
  std::unordered_map<int, int> root_cells;
  for (std::size_t h = 0; h < result.hypotheses.size(); ++h) {
    root_cells[components.find(static_cast<int>(h))] += cells_per_hypothesis[h];
  }
  std::unordered_map<int, int> root_id;
  int next_id = 0;
  for (std::size_t h = 0; h < result.hypotheses.size(); ++h) {
    const int root = components.find(static_cast<int>(h));
    auto found = root_id.find(root);
    if (found == root_id.end()) {
      const int id = root_cells[root] >= params.min_cells_per_surface ? ++next_id : 0;
      found = root_id.emplace(root, id).first;
    }
    result.hypotheses[h].surface = found->second;
  }
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const int h = result.hypothesis[i];
    if (h >= 0) {
      result.surface[i] = result.hypotheses[static_cast<std::size_t>(h)].surface;
      if (result.surface[i] == 0) {
        result.hypothesis[i] = -1;
      }
    }
  }

  // 6. Per-surface summaries.
  result.surfaces.resize(static_cast<std::size_t>(next_id));
  std::vector<std::vector<double>> k1_values(static_cast<std::size_t>(next_id) + 1);
  std::vector<std::vector<double>> residuals(static_cast<std::size_t>(next_id) + 1);
  std::vector<Eigen::Vector3d> normal_sums(
    static_cast<std::size_t>(next_id) + 1, Eigen::Vector3d::Zero());
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const int id = result.surface[i];
    if (id <= 0) {
      continue;
    }
    const CellFeatures & feature = grid.features[i];
    SurfaceSummary & summary = result.surfaces[static_cast<std::size_t>(id - 1)];
    summary.id = id;
    ++summary.cells;
    normal_sums[static_cast<std::size_t>(id)] += feature.normal;
    if (feature.shape_valid) {
      k1_values[static_cast<std::size_t>(id)].push_back(std::abs(feature.k1));
    }
    if (feature.weingarten_residual >= 0.0) {
      residuals[static_cast<std::size_t>(id)].push_back(feature.weingarten_residual);
    }
  }
  for (SurfaceSummary & summary : result.surfaces) {
    const std::size_t slot = static_cast<std::size_t>(summary.id);
    if (normal_sums[slot].squaredNorm() > 1e-12) {
      summary.normal = normal_sums[slot].normalized();
    }
    summary.k1_median = percentile(k1_values[slot], 0.5);
    summary.k1_p90 = percentile(k1_values[slot], 0.9);
    summary.residual_median = percentile(residuals[slot], 0.5);
    // Medians: creases at a surface's border (wall foot, curb) are expected and must not flip
    // the verdict for the bulk of the surface.
    summary.smooth = summary.k1_median <= params.smooth_curvature_max &&
                     (residuals[slot].empty() || summary.residual_median <= params.max_residual);
  }
  return result;
}

}  // namespace lane
}  // namespace back_odom
