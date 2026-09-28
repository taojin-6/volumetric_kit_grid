// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Host-only test for eval/mesh_distance.hpp: the closest point in every region
// of a triangle, the reach clamp, the summary statistics, accuracy / coverage
// and the F-score between meshes whose true distance is known -- a plane
// against a copy of itself shifted by a known amount, and against one with
// half its triangles gone -- and every refusal. CPU-only.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

#include "volumetric_kit/recon/eval/mesh_distance.hpp"
#include "volumetric_kit/recon/mesh/mesh.hpp"

namespace vr = volumetric_kit::recon;
namespace eval = volumetric_kit::recon::eval;
namespace mesh = volumetric_kit::recon::mesh;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

bool near(vr::Vec3f a, vr::Vec3f b, float eps = 1e-6f) {
  return vr::length(a - b) <= eps;
}

bool near(double a, double b, double eps = 1e-6) {
  return std::fabs(a - b) <= eps;
}

// An n x n grid of `cell`-sized squares in the plane z = `z`, two triangles
// each, starting at the origin. `keep_x` drops every square starting at or past
// that x.
mesh::Mesh plane(int n, float cell, float z, float keep_x = 1e9f) {
  mesh::Mesh m;
  for (int j = 0; j <= n; ++j) {
    for (int i = 0; i <= n; ++i) {
      mesh::Vertex v{};
      v.position = vr::Vec3f(float(i) * cell, float(j) * cell, z);
      m.vertices.push_back(v);
    }
  }
  const auto at = [n](int i, int j) {
    return static_cast<std::uint32_t>(j * (n + 1) + i);
  };
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      if (float(i) * cell >= keep_x) {
        continue;
      }
      m.indices.insert(m.indices.end(), {at(i, j), at(i + 1, j), at(i, j + 1)});
      m.indices.insert(m.indices.end(),
                       {at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)});
    }
  }
  return m;
}

int closest_point_case() {
  const vr::Vec3f a(0, 0, 0), b(1, 0, 0), c(0, 1, 0);
  // Face interior: straight down.
  CHECK(near(eval::closest_point_on_triangle({0.25f, 0.25f, 1.0f}, a, b, c),
             {0.25f, 0.25f, 0.0f}));
  // Each vertex region.
  CHECK(near(eval::closest_point_on_triangle({-1, -1, 0.5f}, a, b, c), a));
  CHECK(near(eval::closest_point_on_triangle({2, -1, 0}, a, b, c), b));
  CHECK(near(eval::closest_point_on_triangle({-1, 2, 0}, a, b, c), c));
  // Each edge region.
  CHECK(near(eval::closest_point_on_triangle({0.5f, -1, 0}, a, b, c),
             {0.5f, 0.0f, 0.0f}));
  CHECK(near(eval::closest_point_on_triangle({-1, 0.5f, 0}, a, b, c),
             {0.0f, 0.5f, 0.0f}));
  CHECK(near(eval::closest_point_on_triangle({1, 1, 0}, a, b, c),
             {0.5f, 0.5f, 0.0f}));
  // A point on the triangle is its own closest point.
  CHECK(near(eval::closest_point_on_triangle({0.2f, 0.3f, 0}, a, b, c),
             {0.2f, 0.3f, 0.0f}));
  // A degenerate triangle collapses to its segment.
  CHECK(near(eval::closest_point_on_triangle({0.5f, 1, 0}, a, b, b),
             {0.5f, 0.0f, 0.0f}));
  return 0;
}

int distance_case() {
  vr::Result<eval::MeshDistance> r =
      eval::MeshDistance::create(plane(20, 0.01f, 0.0f), 0.05f);  // 0.2 m
  CHECK(r.ok());
  // The mesh was a temporary: the index copied what it needs.
  const eval::MeshDistance& d = r.value();
  CHECK(d.reach() == 0.05f);
  CHECK(near(double(d.distance({0.03f, 0.07f, 0.02f})), 0.02, 1e-6));
  CHECK(near(double(d.distance({0.1f, 0.1f, -0.013f})), 0.013, 1e-6));
  // Off the edge of the square: to the edge, not to the plane.
  CHECK(near(double(d.distance({0.23f, 0.1f, 0.0f})), 0.03, 1e-6));
  // Nothing within reach reads as the reach, exactly.
  CHECK(d.distance({0.1f, 0.1f, 0.2f}) == 0.05f);
  CHECK(d.distance({5.0f, 5.0f, 0.0f}) == 0.05f);
  // An empty mesh is nothing within reach anywhere.
  vr::Result<eval::MeshDistance> empty =
      eval::MeshDistance::create(mesh::Mesh{}, 0.05f);
  CHECK(empty.ok() && empty.value().distance({0, 0, 0}) == 0.05f);
  return 0;
}

int summarize_case() {
  const eval::DistanceStats s =
      eval::summarize({0.4f, 0.1f, 0.3f, 0.2f, 1.0f, 1.0f}, 1.0f);
  CHECK(s.count == 6);
  CHECK(s.beyond_reach == 2);  // the two at the reach
  CHECK(near(s.mean, 0.25, 1e-6));
  CHECK(near(s.rms, std::sqrt(0.075), 1e-6));
  CHECK(near(s.max, 0.4, 1e-6));
  CHECK(near(s.p95, 0.4, 1e-6));
  const eval::DistanceStats none = eval::summarize({}, 1.0f);
  CHECK(none.count == 0 && none.mean == 0.0);
  return 0;
}

int compare_case() {
  const mesh::Mesh reference = plane(40, 0.005f, 0.0f);  // 0.2 m, 5 mm cells
  // The same plane 3 mm up: every point of either is exactly 3 mm from the
  // other, in both directions.
  const mesh::Mesh shifted = plane(40, 0.005f, 0.003f);
  eval::CompareOptions opt;
  opt.reach = 0.02f;
  vr::Result<eval::MeshComparison> c =
      eval::compare_meshes(reference, shifted, opt);
  CHECK(c.ok());
  for (const eval::DistanceStats& s :
       {c.value().accuracy, c.value().coverage}) {
    CHECK(s.count == reference.vertices.size());
    CHECK(s.beyond_reach == 0);
    CHECK(near(s.mean, 0.003, 1e-6));
    CHECK(near(s.max, 0.003, 1e-6));
  }
  CHECK(c.value().fscore.threshold == 0.0f && c.value().fscore.f == 0.0);

  // Identical: zero everywhere.
  vr::Result<eval::MeshComparison> same =
      eval::compare_meshes(reference, reference, opt);
  CHECK(same.ok());
  CHECK(same.value().accuracy.max == 0.0 && same.value().coverage.max == 0.0);

  // Half the test mesh gone (every square at x >= 0.1 m): what remains is
  // still exactly on the surface, but the reference past the cut plus the
  // reach is uncovered.
  const mesh::Mesh half = plane(40, 0.005f, 0.0f, 0.1f);
  vr::Result<eval::MeshComparison> h =
      eval::compare_meshes(reference, half, opt);
  CHECK(h.ok());
  CHECK(h.value().accuracy.max == 0.0);
  CHECK(h.value().coverage.beyond_reach > reference.vertices.size() / 3);
  CHECK(h.value().coverage.beyond_reach < reference.vertices.size() / 2);

  // A stride measures every stride-th vertex.
  opt.stride = 7;
  vr::Result<eval::MeshComparison> sparse =
      eval::compare_meshes(reference, shifted, opt);
  CHECK(sparse.ok());
  CHECK(sparse.value().accuracy.count == (reference.vertices.size() + 6) / 7);
  CHECK(near(sparse.value().accuracy.mean, 0.003, 1e-6));
  return 0;
}

// The F-score: every point of the shifted plane is 3 mm off, so a threshold
// above that is perfect and one below it is zero; half a mesh has full
// precision and the recall of the half it kept.
int fscore_case() {
  const mesh::Mesh reference = plane(40, 0.005f, 0.0f);
  const mesh::Mesh shifted = plane(40, 0.005f, 0.003f);
  eval::CompareOptions opt;
  opt.reach = 0.02f;
  opt.fscore_threshold = 0.004f;
  vr::Result<eval::MeshComparison> above =
      eval::compare_meshes(reference, shifted, opt);
  CHECK(above.ok());
  CHECK(above.value().fscore.threshold == 0.004f);
  CHECK(above.value().fscore.precision == 1.0);
  CHECK(above.value().fscore.recall == 1.0);
  CHECK(above.value().fscore.f == 1.0);
  opt.fscore_threshold = 0.002f;
  vr::Result<eval::MeshComparison> below =
      eval::compare_meshes(reference, shifted, opt);
  CHECK(below.ok());
  CHECK(below.value().fscore.precision == 0.0);
  CHECK(below.value().fscore.recall == 0.0);
  CHECK(below.value().fscore.f == 0.0);  // not 0 / 0

  // Half the mesh: every point it has is on the surface (precision 1), and
  // it covers exactly the reference points out to where its triangles end.
  // That edge is read off the mesh rather than assumed: 20 * 0.005f rounds
  // below 0.1f, so the cut keeps a 21st column of squares and ends at 0.105.
  const mesh::Mesh half = plane(40, 0.005f, 0.0f, 0.1f);
  float edge = 0.0f;
  for (std::uint32_t i : half.indices) {
    edge = std::max(edge, half.vertices[i].position.x);
  }
  std::size_t covered = 0;
  for (const mesh::Vertex& v : reference.vertices) {
    covered += v.position.x <= edge ? 1u : 0u;
  }
  const double expected = double(covered) / double(reference.vertices.size());
  CHECK(expected > 0.5 && expected < 0.55);  // the premise: about half
  opt.fscore_threshold = 0.001f;
  vr::Result<eval::MeshComparison> h =
      eval::compare_meshes(reference, half, opt);
  CHECK(h.ok());
  CHECK(h.value().fscore.precision == 1.0);
  CHECK(h.value().fscore.recall == expected);
  CHECK(near(h.value().fscore.f, 2.0 * expected / (1.0 + expected), 1e-12));
  return 0;
}

int refusals_case() {
  const mesh::Mesh good = plane(4, 0.01f, 0.0f);
  // Reach not finite and positive.
  for (float reach : {0.0f, -0.01f, std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity()}) {
    CHECK(!eval::MeshDistance::create(good, reach).ok());
  }
  // Indices not in triangles, or past the vertices.
  mesh::Mesh ragged = good;
  ragged.indices.pop_back();
  CHECK(!eval::MeshDistance::create(ragged, 0.02f).ok());
  mesh::Mesh wild = good;
  wild.indices[4] = static_cast<std::uint32_t>(wild.vertices.size());
  CHECK(!eval::MeshDistance::create(wild, 0.02f).ok());
  // compare_meshes refuses the same meshes, either side.
  CHECK(!eval::compare_meshes(wild, good).ok());
  CHECK(!eval::compare_meshes(good, wild).ok());
  // A stride of 0, and an F-score threshold negative, not finite or past the
  // reach (where every distance reads as the reach).
  eval::CompareOptions opt;
  opt.stride = 0;
  CHECK(!eval::compare_meshes(good, good, opt).ok());
  for (float tau : {-0.001f, 0.021f, std::numeric_limits<float>::quiet_NaN()}) {
    eval::CompareOptions o;
    o.reach = 0.02f;
    o.fscore_threshold = tau;
    CHECK(!eval::compare_meshes(good, good, o).ok());
  }
  eval::CompareOptions at_reach;
  at_reach.reach = 0.02f;
  at_reach.fscore_threshold = 0.02f;  // the reach itself is allowed
  CHECK(eval::compare_meshes(good, good, at_reach).ok());
  return 0;
}

}  // namespace

int main() {
  if (closest_point_case() != 0) return 1;
  if (distance_case() != 0) return 1;
  if (summarize_case() != 0) return 1;
  if (compare_case() != 0) return 1;
  if (fscore_case() != 0) return 1;
  if (refusals_case() != 0) return 1;
  std::printf("eval mesh distance: OK\n");
  return 0;
}
