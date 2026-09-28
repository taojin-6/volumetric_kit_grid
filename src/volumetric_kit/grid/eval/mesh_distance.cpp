// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/eval/mesh_distance.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>

namespace volumetric_kit::recon::eval {
namespace {

// Cells per reach: a cell is half the reach on a side, so anything within
// reach of a query lies at most kRings cells from its own.
constexpr int kRings = 2;

// Cells are keyed 21 bits per axis, offset to unsigned: [-2^20, 2^20) on each.
// A triangle is filed at most kMaxCell cells from the origin, so a query that
// can see one probes neighbours at most kMaxCell + 2 * kRings out, still
// inside the keys' range, and no two cells share a key.
constexpr float kMaxCell = float((1 << 20) - 1 - 2 * kRings);

std::uint64_t key(Vec3i c) {
  const auto bits = [](int v) {
    return static_cast<std::uint64_t>(v + (1 << 20)) & 0x1FFFFFu;
  };
  return bits(c.x) | (bits(c.y) << 21) | (bits(c.z) << 42);
}

// The cell of `p`, still in floats: the caller range-checks it before it
// becomes an int, which a far or non-finite point would overflow.
Vec3f cell_of(Vec3f p, float cell) { return glm::floor(p / cell); }

// False for a NaN component too.
bool within(Vec3f c, float limit) {
  return std::fabs(c.x) <= limit && std::fabs(c.y) <= limit &&
         std::fabs(c.z) <= limit;
}

bool positive_finite(float f) { return std::isfinite(f) && f > 0.0f; }

// A triangle with no extent, and what an incremental extract retires one to.
bool is_point(Vec3f a, Vec3f b, Vec3f c) { return a == b && b == c; }

Vec3f closest_point_on_segment(Vec3f p, Vec3f a, Vec3f b) {
  const Vec3f ab = b - a;
  const float len2 = dot(ab, ab);
  if (!(len2 > 0.0f)) {
    return a;
  }
  return a + ab * std::clamp(dot(p - a, ab) / len2, 0.0f, 1.0f);
}

// The nearest point of a triangle's three edges, any of which may itself be a
// point.
Vec3f closest_point_on_edges(Vec3f p, Vec3f a, Vec3f b, Vec3f c) {
  Vec3f best = closest_point_on_segment(p, a, b);
  for (const Vec3f q :
       {closest_point_on_segment(p, b, c), closest_point_on_segment(p, c, a)}) {
    if (dot(p - q, p - q) < dot(p - best, p - best)) {
      best = q;
    }
  }
  return best;
}

// Everything MeshDistance::create refuses, in one pass that allocates nothing,
// so compare_meshes can check both meshes before it indexes either.
Status check_surface(const mesh::Mesh& mesh, float reach) {
  if (!positive_finite(reach)) {
    return Status::invalid_argument(
        "MeshDistance: reach must be finite and positive");
  }
  const float cell = reach / float(kRings);
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
  // Each product is below 2^63 once the corners are in range, and the sum is
  // checked as it grows, so it cannot wrap.
  const std::uint64_t budget =
      std::uint64_t{MeshDistance::kMaxCellsPerTriangle} *
      (mesh.indices.size() / 3);
  std::uint64_t cells = 0;
  for (std::size_t t = 0; t < mesh.indices.size(); t += 3) {
    const Vec3f& a = mesh.vertices[mesh.indices[t]].position;
    const Vec3f& b = mesh.vertices[mesh.indices[t + 1]].position;
    const Vec3f& c = mesh.vertices[mesh.indices[t + 2]].position;
    for (const Vec3f& p : {a, b, c}) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
        return Status::invalid_argument(
            "MeshDistance: a triangle corner is not finite");
      }
      if (!within(cell_of(p, cell), kMaxCell)) {
        return Status::invalid_argument(
            "MeshDistance: a triangle corner is more than " +
            std::to_string(std::int64_t(kMaxCell) / kRings) +
            " reaches from the origin");
      }
    }
    if (is_point(a, b, c)) {
      continue;
    }
    const Vec3i span = Vec3i(cell_of(glm::max(a, glm::max(b, c)), cell)) -
                       Vec3i(cell_of(glm::min(a, glm::min(b, c)), cell)) +
                       Vec3i(1);
    cells +=
        std::uint64_t(span.x) * std::uint64_t(span.y) * std::uint64_t(span.z);
    if (cells > budget) {
      return Status::invalid_argument(
          "MeshDistance: the triangles would be filed under more than " +
          std::to_string(MeshDistance::kMaxCellsPerTriangle) +
          " cells each on average; the reach (" + std::to_string(reach) +
          " m) is too small for them");
    }
  }
  return {};
}

Status check_options(const CompareOptions& o) {
  if (!positive_finite(o.reach)) {
    return Status::invalid_argument(
        "CompareOptions: reach must be finite and positive");
  }
  if (o.stride == 0) {
    return Status::invalid_argument("CompareOptions: stride must be >= 1");
  }
  if (!std::isfinite(o.fscore_threshold) || o.fscore_threshold < 0.0f ||
      o.fscore_threshold > o.reach) {
    return Status::invalid_argument(
        "CompareOptions: the F-score threshold must be in [0, reach]");
  }
  return {};
}

// splitmix64's finalizer.
std::uint64_t mix(std::uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

// Whether the vertex at `p` is in the `stride` subsample: a hash of its
// position rather than its place in the array, which follows marching cubes'
// atomics and so changes run to run.
bool sampled(Vec3f p, std::size_t stride) {
  if (stride == 1) {
    return true;
  }
  std::uint32_t x = 0, y = 0, z = 0;
  std::memcpy(&x, &p.x, sizeof x);
  std::memcpy(&y, &p.y, sizeof y);
  std::memcpy(&z, &p.z, sizeof z);
  return mix(((std::uint64_t{x} << 32) | y) ^ mix(z)) % stride == 0;
}

// The points measured from a mesh `check_surface` passed: each vertex a
// surface triangle uses, once however many use it, in the stride's subsample.
std::vector<Vec3f> sample_points(const mesh::Mesh& mesh, std::size_t stride) {
  std::vector<bool> used(mesh.vertices.size(), false);
  for (std::size_t t = 0; t < mesh.indices.size(); t += 3) {
    const std::uint32_t* i = &mesh.indices[t];
    if (!is_point(mesh.vertices[i[0]].position, mesh.vertices[i[1]].position,
                  mesh.vertices[i[2]].position)) {
      used[i[0]] = used[i[1]] = used[i[2]] = true;
    }
  }
  std::vector<Vec3f> points;
  for (std::size_t v = 0; v < used.size(); ++v) {
    if (used[v] && sampled(mesh.vertices[v].position, stride)) {
      points.push_back(mesh.vertices[v].position);
    }
  }
  return points;
}

// TODO(eval): thread this -- room0 at 1 cm takes ~2 s at stride 4 on one core.
std::vector<float> distances_to(const std::vector<Vec3f>& points,
                                const MeshDistance& surface) {
  std::vector<float> out;
  out.reserve(points.size());
  for (const Vec3f& p : points) {
    out.push_back(surface.distance(p));
  }
  return out;
}

MeshComparison combine(std::vector<float> acc, std::vector<float> cov,
                       const CompareOptions& o) {
  MeshComparison c;
  const float tau = o.fscore_threshold;
  if (tau > 0.0f) {
    const auto fraction_within = [tau](const std::vector<float>& d) {
      const auto n =
          std::count_if(d.begin(), d.end(), [tau](float x) { return x < tau; });
      return d.empty() ? 0.0 : double(n) / double(d.size());
    };
    c.fscore.threshold = tau;
    c.fscore.precision = fraction_within(acc);
    c.fscore.recall = fraction_within(cov);
    const double sum = c.fscore.precision + c.fscore.recall;
    c.fscore.f =
        sum > 0.0 ? 2.0 * c.fscore.precision * c.fscore.recall / sum : 0.0;
  }
  c.accuracy = summarize(std::move(acc), o.reach);
  c.coverage = summarize(std::move(cov), o.reach);
  return c;
}

}  // namespace

Vec3f closest_point_on_triangle(Vec3f p, Vec3f a, Vec3f b, Vec3f c) {
  // The projection onto the face when it lands inside, else the nearest edge
  // -- not Ericson's region test, whose divisions now and then lose the
  // answer on a thin triangle: about one query in 300 000 on 1 cm slivers at
  // room coordinates, by up to 90 um. The face is trusted only while the sine
  // of the angle at `a` is above 1e-4, a thousand times the float rounding in
  // the cross product -- a margin, not a measured need; a thinner triangle is
  // measured by its edges, which lie within its height of its face.
  const Vec3f ab = b - a;
  const Vec3f bc = c - b;
  const Vec3f ca = a - c;
  const Vec3f n = cross(ab, c - a);
  const float n2 = dot(n, n);
  if (n2 > 1e-8f * dot(ab, ab) * dot(ca, ca)) {
    const Vec3f q = p - n * (dot(p - a, n) / n2);
    if (dot(cross(ab, q - a), n) >= 0.0f && dot(cross(bc, q - b), n) >= 0.0f &&
        dot(cross(ca, q - c), n) >= 0.0f) {
      return q;
    }
  }
  return closest_point_on_edges(p, a, b, c);
}

Result<MeshDistance> MeshDistance::create(const mesh::Mesh& mesh, float reach) {
  VR_TRY(check_surface(mesh, reach));
  MeshDistance d;
  d.reach_ = reach;
  d.cell_ = reach / float(kRings);
  d.corners_.reserve(mesh.indices.size());
  for (std::size_t i = 0; i < mesh.indices.size(); i += 3) {
    const Vec3f& a = mesh.vertices[mesh.indices[i]].position;
    const Vec3f& b = mesh.vertices[mesh.indices[i + 1]].position;
    const Vec3f& c = mesh.vertices[mesh.indices[i + 2]].position;
    if (is_point(a, b, c)) {
      continue;
    }
    const auto t = static_cast<std::uint32_t>(d.corners_.size());
    d.corners_.insert(d.corners_.end(), {a, b, c});
    const Vec3i lo(cell_of(glm::min(a, glm::min(b, c)), d.cell_));
    const Vec3i hi(cell_of(glm::max(a, glm::max(b, c)), d.cell_));
    for (int z = lo.z; z <= hi.z; ++z) {
      for (int y = lo.y; y <= hi.y; ++y) {
        for (int x = lo.x; x <= hi.x; ++x) {
          d.cells_[key(Vec3i(x, y, z))].push_back(t);
        }
      }
    }
  }
  return d;
}

float MeshDistance::distance(Vec3f p) const {
  // No triangle is filed past kMaxCell, and nothing within reach is more than
  // kRings cells from the query's own, so a point further out than both, or
  // one that is not finite, has nothing within reach.
  const Vec3f cell = cell_of(p, cell_);
  if (!within(cell, kMaxCell + float(kRings))) {
    return reach_;
  }
  const Vec3i centre(cell);
  // Lower best2 to the nearest triangle filed under cell c, if nearer.
  const auto scan = [this, p](Vec3i c, float& best2) {
    const auto it = cells_.find(key(c));
    if (it == cells_.end()) {
      return;
    }
    for (std::uint32_t t : it->second) {
      const Vec3f d = p - closest_point_on_triangle(
                              p, corners_[t], corners_[t + 1], corners_[t + 2]);
      best2 = std::min(best2, dot(d, d));
    }
  };
  // The squared distance along one axis from coordinate v to cell i's slab.
  const auto gap2 = [this](float v, int i) {
    const float lo = float(i) * cell_;
    const float g = v < lo ? lo - v : v - (lo + cell_);
    return g > 0.0f ? g * g : 0.0f;
  };
  // A triangle is filed under every cell its bounding box touches, so its
  // nearest point to p lies in one of them. The query's own cell goes first,
  // so the best found so far -- under a millimetre on a surface -- prunes
  // every neighbour whose box is farther than it.
  const float reach2 = reach_ * reach_;
  float best2 = reach2;
  scan(centre, best2);
  for (int dz = -kRings; dz <= kRings; ++dz) {
    const float gz = gap2(p.z, centre.z + dz);
    if (gz >= best2) {
      continue;
    }
    for (int dy = -kRings; dy <= kRings; ++dy) {
      const float gy = gz + gap2(p.y, centre.y + dy);
      if (gy >= best2) {
        continue;
      }
      for (int dx = -kRings; dx <= kRings; ++dx) {
        if ((dx == 0 && dy == 0 && dz == 0) ||
            gy + gap2(p.x, centre.x + dx) >= best2) {
          continue;
        }
        scan(centre + Vec3i(dx, dy, dz), best2);
      }
    }
  }
  return best2 < reach2 ? std::sqrt(best2) : reach_;
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
  // Nearest rank: the ceil(0.95 n)-th smallest.
  s.p95 = distances[(within * 95 + 99) / 100 - 1];
  s.max = distances[within - 1];
  return s;
}

Result<MeshComparison> compare_meshes(const mesh::Mesh& reference,
                                      const mesh::Mesh& test,
                                      const CompareOptions& options) {
  VR_TRY(check_options(options));
  VR_TRY(check_surface(reference, options.reach));
  VR_TRY(check_surface(test, options.reach));
  std::vector<float> acc;
  {
    VR_ASSIGN(const MeshDistance to_reference,
              MeshDistance::create(reference, options.reach));
    acc = distances_to(sample_points(test, options.stride), to_reference);
  }
  VR_ASSIGN(const MeshDistance to_test,
            MeshDistance::create(test, options.reach));
  std::vector<float> cov =
      distances_to(sample_points(reference, options.stride), to_test);
  return combine(std::move(acc), std::move(cov), options);
}

ReferenceMesh::ReferenceMesh(const CompareOptions& options,
                             MeshDistance surface, std::vector<Vec3f> points)
    : options_(options),
      surface_(std::move(surface)),
      points_(std::move(points)) {}

Result<ReferenceMesh> ReferenceMesh::create(const mesh::Mesh& reference,
                                            const CompareOptions& options) {
  VR_TRY(check_options(options));
  VR_ASSIGN(MeshDistance surface,
            MeshDistance::create(reference, options.reach));
  return ReferenceMesh(options, std::move(surface),
                       sample_points(reference, options.stride));
}

Result<MeshComparison> ReferenceMesh::compare(const mesh::Mesh& test) const {
  VR_ASSIGN(const MeshDistance to_test,
            MeshDistance::create(test, options_.reach));
  std::vector<float> acc =
      distances_to(sample_points(test, options_.stride), surface_);
  std::vector<float> cov = distances_to(points_, to_test);
  return combine(std::move(acc), std::move(cov), options_);
}

}  // namespace volumetric_kit::recon::eval
