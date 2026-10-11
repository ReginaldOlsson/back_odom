#include "back_odom/lane/cell_segmentation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

namespace back_odom
{
namespace lane
{
namespace
{

using BinCoords = std::array<int, kFeatureDim>;

constexpr double kVoteCap = 32.0;

/// Dense histogram over the normalized feature cube with separable blur and peak search.
class Accumulator
{
public:
  explicit Accumulator(int bins) : bins_(std::max(bins, 2))
  {
    int stride = 1;
    for (int axis = 0; axis < kFeatureDim; ++axis) {
      strides_[axis] = stride;
      stride *= bins_;
    }
    total_ = stride;
    values_.assign(static_cast<std::size_t>(total_), 0.0);
  }

  int bins() const { return bins_; }
  int total() const { return total_; }
  double value(int index) const { return values_[static_cast<std::size_t>(index)]; }

  BinCoords coords_of(const FeatureVector & features) const
  {
    BinCoords coords;
    for (int axis = 0; axis < kFeatureDim; ++axis) {
      const int bin = static_cast<int>(std::floor(features[axis] * bins_));
      coords[axis] = std::clamp(bin, 0, bins_ - 1);
    }
    return coords;
  }

  int flatten(const BinCoords & coords) const
  {
    int index = 0;
    for (int axis = 0; axis < kFeatureDim; ++axis) {
      index += coords[axis] * strides_[axis];
    }
    return index;
  }

  BinCoords unflatten(int index) const
  {
    BinCoords coords;
    for (int axis = 0; axis < kFeatureDim; ++axis) {
      coords[axis] = (index / strides_[axis]) % bins_;
    }
    return coords;
  }

  void add(const BinCoords & coords, double weight)
  {
    values_[static_cast<std::size_t>(flatten(coords))] += weight;
  }

  /// One 3-tap (0.25, 0.5, 0.25) pass per axis.
  void blur()
  {
    std::vector<double> scratch(values_.size(), 0.0);
    for (int axis = 0; axis < kFeatureDim; ++axis) {
      const int stride = strides_[axis];
      for (int index = 0; index < total_; ++index) {
        const int coordinate = (index / stride) % bins_;
        double sum = 0.5 * values_[static_cast<std::size_t>(index)];
        if (coordinate > 0) {
          sum += 0.25 * values_[static_cast<std::size_t>(index - stride)];
        }
        if (coordinate + 1 < bins_) {
          sum += 0.25 * values_[static_cast<std::size_t>(index + stride)];
        }
        scratch[static_cast<std::size_t>(index)] = sum;
      }
      values_.swap(scratch);
    }
  }

  /// Sum of votes over the Chebyshev ball of the given radius around coords.
  double mass_within(const BinCoords & coords, int radius) const
  {
    const int width = 2 * radius + 1;
    int ball_size = 1;
    for (int axis = 0; axis < kFeatureDim; ++axis) {
      ball_size *= width;
    }
    double mass = 0.0;
    for (int cell = 0; cell < ball_size; ++cell) {
      int remaining = cell;
      int index = 0;
      bool in_bounds = true;
      for (int axis = 0; axis < kFeatureDim; ++axis) {
        const int coordinate = coords[axis] + (remaining % width) - radius;
        remaining /= width;
        if (coordinate < 0 || coordinate >= bins_) {
          in_bounds = false;
          break;
        }
        index += coordinate * strides_[axis];
      }
      if (in_bounds) {
        mass += values_[static_cast<std::size_t>(index)];
      }
    }
    return mass;
  }

  /// True when no in-bounds neighbor over the 3^D stencil has a larger value.
  bool is_local_max(int index) const
  {
    const double center = values_[static_cast<std::size_t>(index)];
    const BinCoords coords = unflatten(index);
    int stencil_size = 1;
    for (int axis = 0; axis < kFeatureDim; ++axis) {
      stencil_size *= 3;
    }
    for (int stencil = 0; stencil < stencil_size; ++stencil) {
      int remaining = stencil;
      int neighbor = index;
      bool in_bounds = true;
      bool is_center = true;
      for (int axis = 0; axis < kFeatureDim; ++axis) {
        const int delta = (remaining % 3) - 1;
        remaining /= 3;
        if (delta == 0) {
          continue;
        }
        is_center = false;
        const int coordinate = coords[axis] + delta;
        if (coordinate < 0 || coordinate >= bins_) {
          in_bounds = false;
          break;
        }
        neighbor += delta * strides_[axis];
      }
      if (is_center || !in_bounds) {
        continue;
      }
      if (values_[static_cast<std::size_t>(neighbor)] > center) {
        return false;
      }
    }
    return true;
  }

private:
  int bins_;
  int total_{0};
  std::array<int, kFeatureDim> strides_{};
  std::vector<double> values_;
};

int chebyshev_distance(const BinCoords & a, const BinCoords & b)
{
  int distance = 0;
  for (int axis = 0; axis < kFeatureDim; ++axis) {
    distance = std::max(distance, std::abs(a[axis] - b[axis]));
  }
  return distance;
}

/// Peaks are located on the blurred accumulator but thresholded on the raw vote mass inside
/// their suppression radius: a separable blur in kFeatureDim dimensions can shrink an isolated
/// bin by 2^kFeatureDim, so the blurred height is not a usable class-size measure.
std::vector<Peak> find_peaks(
  const Accumulator & raw, const Accumulator & blurred, const SegmentationParams & params)
{
  const int radius = std::max(params.peak_min_separation, 1);
  std::vector<std::pair<double, int>> candidates;
  for (int index = 0; index < blurred.total(); ++index) {
    if (!(blurred.value(index) > 0.0) || !blurred.is_local_max(index)) {
      continue;
    }
    const double mass = raw.mass_within(blurred.unflatten(index), radius);
    if (mass >= params.peak_min_votes) {
      candidates.emplace_back(mass, index);
    }
  }
  std::sort(candidates.begin(), candidates.end(), [](const auto & a, const auto & b) {
    return a.first > b.first || (a.first == b.first && a.second < b.second);
  });

  std::vector<Peak> peaks;
  std::vector<BinCoords> kept_coords;
  const double inverse_bins = 1.0 / static_cast<double>(blurred.bins());
  for (const auto & candidate : candidates) {
    if (static_cast<int>(peaks.size()) >= std::max(params.max_classes, 0)) {
      break;
    }
    const BinCoords coords = blurred.unflatten(candidate.second);
    const bool suppressed = std::any_of(
      kept_coords.begin(), kept_coords.end(), [&coords, &params](const BinCoords & kept) {
        return chebyshev_distance(coords, kept) < params.peak_min_separation;
      });
    if (suppressed) {
      continue;
    }
    Peak peak;
    for (int axis = 0; axis < kFeatureDim; ++axis) {
      peak.center[axis] = (static_cast<double>(coords[axis]) + 0.5) * inverse_bins;
    }
    peak.votes = candidate.first;
    peaks.push_back(peak);
    kept_coords.push_back(coords);
  }
  return peaks;
}

/// Reuses the id of the nearest previous peak when one is close enough, otherwise takes the
/// lowest id not used in this frame. Peaks must be sorted by votes descending.
void assign_stable_ids(
  std::vector<Peak> & peaks, const std::vector<Peak> & previous_peaks, double match_distance)
{
  std::vector<bool> previous_used(previous_peaks.size(), false);
  const double match_sq = match_distance * match_distance;
  for (Peak & peak : peaks) {
    peak.id = 0;
    double best_sq = match_sq;
    int best = -1;
    for (std::size_t p = 0; p < previous_peaks.size(); ++p) {
      if (previous_used[p] || previous_peaks[p].id <= 0) {
        continue;
      }
      const double distance_sq = (previous_peaks[p].center - peak.center).squaredNorm();
      if (distance_sq <= best_sq) {
        best_sq = distance_sq;
        best = static_cast<int>(p);
      }
    }
    if (best >= 0) {
      previous_used[static_cast<std::size_t>(best)] = true;
      peak.id = previous_peaks[static_cast<std::size_t>(best)].id;
    }
  }
  std::vector<int> used;
  for (const Peak & peak : peaks) {
    if (peak.id > 0) {
      used.push_back(peak.id);
    }
  }
  for (Peak & peak : peaks) {
    if (peak.id > 0) {
      continue;
    }
    int candidate = 1;
    while (std::find(used.begin(), used.end(), candidate) != used.end()) {
      ++candidate;
    }
    peak.id = candidate;
    used.push_back(candidate);
  }
}

void propagate_labels(
  const CellGrid & grid, const std::vector<FeatureVector> & features, int max_id,
  const SegmentationParams & params, std::vector<int> & labels)
{
  if (params.propagation_iterations <= 0 || max_id <= 0) {
    return;
  }
  const double inverse_sigma_sq =
    1.0 / std::max(params.feature_sigma * params.feature_sigma, 1e-12);
  std::vector<int> next(labels);
  std::vector<double> score(static_cast<std::size_t>(max_id) + 1, 0.0);
  for (int iteration = 0; iteration < params.propagation_iterations; ++iteration) {
    for (std::size_t i = 0; i < grid.size(); ++i) {
      const int own = labels[i];
      if (own <= 0) {
        continue;
      }
      std::fill(score.begin(), score.end(), 0.0);
      score[static_cast<std::size_t>(own)] += 1.0;
      const int begin = grid.neighbor_offsets[i];
      const int end = grid.neighbor_offsets[i + 1];
      for (int n = begin; n < end; ++n) {
        const std::size_t j = static_cast<std::size_t>(grid.neighbor_indices[n]);
        const int other = labels[j];
        if (other <= 0) {
          continue;
        }
        const double distance_sq = (features[i] - features[j]).squaredNorm();
        score[static_cast<std::size_t>(other)] += std::exp(-distance_sq * inverse_sigma_sq);
      }
      int best = own;
      double best_score = score[static_cast<std::size_t>(own)];
      for (int id = 1; id <= max_id; ++id) {
        if (score[static_cast<std::size_t>(id)] > best_score) {
          best_score = score[static_cast<std::size_t>(id)];
          best = id;
        }
      }
      next[i] = best;
    }
    labels.swap(next);
  }
}

void connected_components(
  const CellGrid & grid, const std::vector<int> & labels, std::vector<int> & segments)
{
  segments.assign(grid.size(), 0);
  std::vector<std::size_t> stack;
  int next_segment = 0;
  for (std::size_t seed = 0; seed < grid.size(); ++seed) {
    if (labels[seed] <= 0 || segments[seed] != 0) {
      continue;
    }
    ++next_segment;
    segments[seed] = next_segment;
    stack.clear();
    stack.push_back(seed);
    while (!stack.empty()) {
      const std::size_t i = stack.back();
      stack.pop_back();
      const int begin = grid.neighbor_offsets[i];
      const int end = grid.neighbor_offsets[i + 1];
      for (int n = begin; n < end; ++n) {
        const std::size_t j = static_cast<std::size_t>(grid.neighbor_indices[n]);
        if (segments[j] != 0 || labels[j] != labels[seed]) {
          continue;
        }
        segments[j] = next_segment;
        stack.push_back(j);
      }
    }
  }
}

}  // namespace

FeatureVector feature_vector(const CellFeatures & features, const SegmentationParams & params)
{
  FeatureVector vector;
  vector[0] = std::clamp(features.planarity, 0.0, 1.0);
  vector[1] = std::clamp(features.verticality, 0.0, 1.0);
  vector[2] = std::clamp(features.curvature / std::max(params.curvature_max, 1e-9), 0.0, 1.0);
  vector[3] = std::clamp(features.intensity_cv / std::max(params.cv_max, 1e-9), 0.0, 1.0);
  vector[4] =
    std::clamp(features.intensity_laplacian / std::max(params.laplacian_max, 1e-9), 0.0, 1.0);
  return vector;
}

SegmentationResult segment_cells(
  const CellGrid & grid, const SegmentationParams & params,
  const std::vector<Peak> & previous_peaks)
{
  SegmentationResult result;
  result.label.assign(grid.size(), 0);
  result.segment.assign(grid.size(), 0);

  std::vector<FeatureVector> features(grid.size(), FeatureVector::Zero());
  Accumulator accumulator(params.bins);
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const CellFeatures & cell = grid.features[i];
    if (!cell.valid) {
      continue;
    }
    features[i] = feature_vector(cell, params);
    accumulator.add(
      accumulator.coords_of(features[i]),
      std::min(static_cast<double>(grid.stats[i].count), kVoteCap));
  }
  Accumulator blurred = accumulator;
  blurred.blur();

  result.peaks = find_peaks(accumulator, blurred, params);
  if (result.peaks.empty()) {
    return result;
  }
  const double match_distance =
    static_cast<double>(std::max(params.peak_min_separation, 1)) /
    static_cast<double>(blurred.bins());
  assign_stable_ids(result.peaks, previous_peaks, match_distance);

  int max_id = 0;
  for (const Peak & peak : result.peaks) {
    max_id = std::max(max_id, peak.id);
  }

  for (std::size_t i = 0; i < grid.size(); ++i) {
    if (!grid.features[i].valid) {
      continue;
    }
    double best_sq = std::numeric_limits<double>::infinity();
    int best_id = 0;
    for (const Peak & peak : result.peaks) {
      const double distance_sq = (peak.center - features[i]).squaredNorm();
      if (distance_sq < best_sq) {
        best_sq = distance_sq;
        best_id = peak.id;
      }
    }
    result.label[i] = best_id;
  }

  propagate_labels(grid, features, max_id, params, result.label);
  connected_components(grid, result.label, result.segment);
  return result;
}

}  // namespace lane
}  // namespace back_odom
