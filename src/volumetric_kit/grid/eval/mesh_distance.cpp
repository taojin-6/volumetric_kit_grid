// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/eval/mesh_distance.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

namespace volumetric_kit::recon::eval {
namespace {

Vec3i cell_of(Vec3f p, float cell) { return Vec3i(glm::floor(p / cell)); }

// 21 bits per axis, offset to unsigned: +-2^20 cells, ~10 km at 1 cm cells.
std::uint64_t key(Vec3i c) {
  const auto bits = [](int v) {
    return static_cast<std::uint64_t>(v + (1 << 20)) & 0x1FFFFFu;
  };
  return bits(c.x) | (bits(c.y) << 21) | (bits(c.z) << 42);
}

bool positive_finite(float f) { return std::isfinite(f) && f > 0.0f; }

}  // namespace

Vec3f closest_point_on_triangle(Vec3f p, Vec3f a, Vec3f b, Vec3f c) {
  const Vec3f ab = b - a;
  const Vec3f ac = c - a;
  const Vec3f ap = p - a;
  const float d1 = dot(ab, ap);
  const float d2 = dot(ac, ap);
  if (d1 <= 0.0f && d2 <= 0.0f) return a;
  const Vec3f bp = p - b;
  const float d3 = dot(ab, bp);
  const float d4 = dot(ac, bp);
  if (d3 >= 0.0f && d4 <= d3) return b;
  const float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
    return a + ab * (d1 / (d1 - d3));
  }
  const Vec3f cp = p - c;
  const float d5 = dot(ab, cp);
  const float d6 = dot(ac, cp);
  if (d6 >= 0.0f && d5 <= d6) return c;
  const float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
    return a + ac * (d2 / (d2 - d6));
  }
  const float va = d3 * d6 - d5 * d4;
  if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
    return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
  }
  // Inside the face. A degenerate triangle never gets here: it falls into one
  // of the vertex or edge regions above.
  const float denom = 1.0f / (va + vb + vc);
  return a + ab * (vb * denom) + ac * (vc * denom);
}

Result<MeshDistance> MeshDistance::create(const mesh::Mesh& mesh, float reach) {
  if (!positive_finite(reach)) {
    return Status::invalid_argument(
        "MeshDistance: reach must be finite and positive");
  }
  if (mesh.indices.size() % 3 != 0) {
    return Status::invalid_argument(
        "MeshDistance: the index count is not a multiple of 3");
  }
  for (std::uint32_t i : mesh.indices) {
    if (i >= mesh.vertices.size()) {
      return Status::invalid_argument(
          "MeshDistance: an index is past the vertices (" + std::to_string(i) +
          " of " + std::to_string(mesh.vertices.size()) + ")");
    }
  }
  MeshDistance d;
  d.reach_ = reach;
  d.corners_.reserve(mesh.indices.size());
  for (std::uint32_t i : mesh.indices) {
    d.corners_.push_back(mesh.vertices[i].position);
  }
  for (std::size_t t = 0; t < d.corners_.size(); t += 3) {
    const Vec3f& a = d.corners_[t];
    const Vec3f& b = d.corners_[t + 1];
    const Vec3f& c = d.corners_[t + 2];
    const Vec3i lo = cell_of(glm::min(a, glm::min(b, c)), reach);
    const Vec3i hi = cell_of(glm::max(a, glm::max(b, c)), reach);
    for (int z = lo.z; z <= hi.z; ++z) {
      for (int y = lo.y; y <= hi.y; ++y) {
        for (int x = lo.x; x <= hi.x; ++x) {
          d.cells_[key(Vec3i(x, y, z))].push_back(
              static_cast<std::uint32_t>(t));
        }
      }
    }
  }
  return d;
}

float MeshDistance::distance(Vec3f p) const {
  float best = reach_;
  const Vec3i centre = cell_of(p, reach_);
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const auto it = cells_.find(key(centre + Vec3i(dx, dy, dz)));
        if (it == cells_.end()) {
          continue;
        }
        for (std::uint32_t t : it->second) {
          best = std::min(best, length(p - closest_point_on_triangle(
                                               p, corners_[t], corners_[t + 1],
                                               corners_[t + 2])));
        }
      }
    }
  }
  return best;
}

DistanceStats summarize(std::vector<float> distances, float reach) {
  DistanceStats s;
  s.count = distances.size();
  std::sort(distances.begin(), distances.end());
  const std::size_t within = static_cast<std::size_t>(
      std::lower_bound(distances.begin(), distances.end(), reach) -
      distances.begin());
  s.beyond_reach = distances.size() - within;
  if (within == 0) {
    return s;
  }
  for (std::size_t i = 0; i < within; ++i) {
    s.mean += distances[i];
    s.rms += double(distances[i]) * distances[i];
  }
  s.mean /= double(within);
  s.rms = std::sqrt(s.rms / double(within));
  s.p95 = distances[std::min(within - 1, (within * 95) / 100)];
  s.max = distances[within - 1];
  return s;
}

Result<MeshComparison> compare_meshes(const mesh::Mesh& reference,
                                      const mesh::Mesh& test,
                                      const CompareOptions& options) {
  if (options.stride == 0) {
    return Status::invalid_argument("compare_meshes: stride must be >= 1");
  }
  if (!std::isfinite(options.fscore_threshold) ||
      options.fscore_threshold < 0.0f ||
      options.fscore_threshold > options.reach) {
    return Status::invalid_argument(
        "compare_meshes: the F-score threshold must be in [0, reach]");
  }
  VR_ASSIGN(const MeshDistance to_reference,
            MeshDistance::create(reference, options.reach));
  VR_ASSIGN(const MeshDistance to_test,
            MeshDistance::create(test, options.reach));

  // Every stride-th vertex of `points`, its distance to `surface`; and how
  // many were within the F-score's threshold.
  const float tau = options.fscore_threshold;
  auto measure = [&](const mesh::Mesh& points, const MeshDistance& surface,
                     std::size_t& within_tau) {
    std::vector<float> out;
    out.reserve(points.vertices.size() / options.stride + 1);
    within_tau = 0;
    for (std::size_t i = 0; i < points.vertices.size(); i += options.stride) {
      const float d = surface.distance(points.vertices[i].position);
      within_tau += d < tau ? 1u : 0u;
      out.push_back(d);
    }
    return out;
  };
  std::size_t test_within = 0;
  std::size_t ref_within = 0;
  std::vector<float> acc = measure(test, to_reference, test_within);
  std::vector<float> cov = measure(reference, to_test, ref_within);

  MeshComparison c;
  if (tau > 0.0f) {
    c.fscore.threshold = tau;
    c.fscore.precision = acc.empty() ? 0.0 : double(test_within) / acc.size();
    c.fscore.recall = cov.empty() ? 0.0 : double(ref_within) / cov.size();
    const double sum = c.fscore.precision + c.fscore.recall;
    c.fscore.f =
        sum > 0.0 ? 2.0 * c.fscore.precision * c.fscore.recall / sum : 0.0;
  }
  c.accuracy = summarize(std::move(acc), options.reach);
  c.coverage = summarize(std::move(cov), options.reach);
  return c;
}

}  // namespace volumetric_kit::recon::eval
