// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Host-only test for eval/mesh_distance.hpp: the closest point in every region
// of a triangle, on every degenerate one and on thin ones against a
// double-precision answer, the pruned cell search against a scan of every
// triangle, the reach clamp, the summary statistics, accuracy / coverage and
// the F-score between meshes whose true distance is known -- a plane against a
// copy of itself shifted by a known amount, and against one with half its
// triangles gone -- what counts as surface, figures independent of the mesh's
// order, the reusable reference, and every refusal. CPU-only.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include "volumetric_kit/recon/eval/mesh_distance.hpp"
#include "volumetric_kit/recon/mesh/mesh.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
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

// Both ways, as a sweep measures: the reference indexed, the test judged.
vkc::Result<eval::MeshComparison> compare(
    const mesh::Mesh& reference, const mesh::Mesh& test,
    const eval::CompareOptions& options = {}) {
  VKC_ASSIGN(const eval::ReferenceMesh indexed,
             eval::ReferenceMesh::create(reference, options));
  return indexed.compare(test);
}

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

// `m` with its vertices moved one place down (the first to the end) and its
// triangles in reverse order: the same surface, as another run's atomics might
// emit it. A rotation rather than a reversal, which on 41^2 vertices maps
// every seventh index onto every seventh index.
mesh::Mesh reordered(const mesh::Mesh& m) {
  const auto n = static_cast<std::uint32_t>(m.vertices.size());
  mesh::Mesh r;
  r.vertices.assign(m.vertices.begin() + 1, m.vertices.end());
  r.vertices.push_back(m.vertices.front());
  for (std::size_t t = m.indices.size(); t >= 3; t -= 3) {
    for (std::size_t k = t - 3; k < t; ++k) {
      r.indices.push_back((m.indices[k] + n - 1) % n);
    }
  }
  return r;
}

// A deterministic stream of floats in [lo, hi): a fixed LCG, so every
// standard library draws the same points.
struct Lcg {
  std::uint32_t state = 12345u;
  float operator()(float lo, float hi) {
    state = state * 1664525u + 1013904223u;
    return lo + (hi - lo) * float(state >> 8) / float(1u << 24);
  }
};

// The distance from p to triangle abc in double, where a thin triangle at room
// coordinates is still well conditioned: to the face when p projects inside
// it, else to the nearest edge.
double triangle_distance(const vr::Vec3f& pf, const vr::Vec3f& af,
                         const vr::Vec3f& bf, const vr::Vec3f& cf) {
  const glm::dvec3 p(pf), a(af), b(bf), c(cf);
  const auto segment = [&p](const glm::dvec3& u, const glm::dvec3& v) {
    const glm::dvec3 uv = v - u;
    const double len2 = glm::dot(uv, uv);
    const double t =
        len2 > 0.0 ? std::clamp(glm::dot(p - u, uv) / len2, 0.0, 1.0) : 0.0;
    return glm::length(p - (u + uv * t));
  };
  const glm::dvec3 n = glm::cross(b - a, c - a);
  const double n2 = glm::dot(n, n);
  if (n2 > 0.0) {
    const glm::dvec3 q = p - n * (glm::dot(p - a, n) / n2);
    if (glm::dot(glm::cross(b - a, q - a), n) >= 0.0 &&
        glm::dot(glm::cross(c - b, q - b), n) >= 0.0 &&
        glm::dot(glm::cross(a - c, q - c), n) >= 0.0) {
      return glm::length(p - q);
    }
  }
  return std::min({segment(a, b), segment(b, c), segment(c, a)});
}

bool identical(const eval::DistanceStats& a, const eval::DistanceStats& b) {
  return a.count == b.count && a.beyond_reach == b.beyond_reach &&
         a.mean == b.mean && a.rms == b.rms && a.p95 == b.p95 && a.max == b.max;
}

bool identical(const eval::MeshComparison& a, const eval::MeshComparison& b) {
  return identical(a.accuracy, b.accuracy) &&
         identical(a.coverage, b.coverage) &&
         a.fscore.threshold == b.fscore.threshold &&
         a.fscore.precision == b.fscore.precision &&
         a.fscore.recall == b.fscore.recall && a.fscore.f == b.fscore.f;
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
  // A degenerate triangle collapses to its segment, whichever two corners
  // coincide: with the first two, Ericson's edge region read 0 / 0.
  const vr::Vec3f p(0.001f, 0.5f, 0.0f);
  const vr::Vec3f on_ac(0.0f, 0.5f, 0.0f);
  for (const auto& [x, y, z] :
       {std::tuple{a, a, c}, std::tuple{a, c, a}, std::tuple{c, a, a},
        std::tuple{a, c, c}, std::tuple{c, a, c}, std::tuple{c, c, a}}) {
    CHECK(near(eval::closest_point_on_triangle(p, x, y, z), on_ac));
  }
  // Three distinct corners on one line: the segment they span.
  const vr::Vec3f mid(0.0f, 0.25f, 0.0f);
  for (const auto& [x, y, z] :
       {std::tuple{a, mid, c}, std::tuple{mid, a, c}, std::tuple{c, mid, a}}) {
    CHECK(near(eval::closest_point_on_triangle({0.001f, 0.75f, 0}, x, y, z),
               {0.0f, 0.75f, 0.0f}));
  }
  // All three one point: that point.
  CHECK(near(eval::closest_point_on_triangle({1, 1, 1}, c, c, c), c));
  return 0;
}

// Thin triangles at room coordinates, 1 cm across: slivers whose third corner
// sits a sine of 1e-6 to 1e-2 off the opposite edge, each corner in turn
// first, and needles with a 1 um edge. Marching cubes emits both. Ericson's
// region test was off on about one in 300 000 of these, by up to 90 um, so
// the sample is large enough to meet a few. The answer must be the
// double-precision one to within float rounding at 3 m (0.24 um).
int thin_triangle_case() {
  Lcg rng;
  const vr::Vec3f base(3.2f, 1.7f, 2.9f);
  const auto around = [&rng](float r) {
    return vr::Vec3f(rng(-r, r), rng(-r, r), rng(-r, r));
  };
  const auto matches = [](vr::Vec3f p, vr::Vec3f a, vr::Vec3f b, vr::Vec3f c) {
    const vr::Vec3f q = eval::closest_point_on_triangle(p, a, b, c);
    return std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z) &&
           std::fabs(double(vr::length(p - q)) -
                     triangle_distance(p, a, b, c)) <= 0.5e-6;
  };
  for (int i = 0; i < 50000; ++i) {
    const vr::Vec3f a = base + around(0.01f);
    const vr::Vec3f b = a + around(0.01f);
    const vr::Vec3f p = base + around(0.04f);
    for (float sine : {1e-6f, 1e-5f, 3e-5f, 1e-4f, 1e-3f, 1e-2f}) {
      const vr::Vec3f ab = b - a;
      const vr::Vec3f off = glm::normalize(vr::cross(ab, around(1.0f)));
      const vr::Vec3f c = a + ab * rng(0.2f, 0.8f) +
                          off * (sine * vr::length(ab) * rng(0.5f, 1.5f));
      CHECK(matches(p, a, b, c));
      CHECK(matches(p, c, a, b));
      CHECK(matches(p, b, c, a));
    }
    CHECK(matches(p, a, a + around(1e-6f), b));
  }
  return 0;
}

// The search prunes neighbouring cells by distance; it must find what a scan
// of every triangle finds. A bumpy sheet of 1 cm triangles, some collapsed to
// a segment, against points scattered within and beyond the reach.
int search_case() {
  Lcg rng;
  mesh::Mesh m = plane(30, 0.01f, 0.0f);
  for (mesh::Vertex& v : m.vertices) {
    v.position += vr::Vec3f(rng(-0.004f, 0.004f), rng(-0.004f, 0.004f),
                            rng(-0.006f, 0.006f));
  }
  for (std::size_t t = 0; t < m.indices.size(); t += 3 * 17) {
    m.indices[t + 1] = m.indices[t];  // a segment, still surface
  }
  const float reach = 0.04f;
  vkc::Result<eval::MeshDistance> d = eval::MeshDistance::create(m, reach);
  CHECK(d.ok());
  int within = 0;
  for (int i = 0; i < 3000; ++i) {
    const vr::Vec3f p(rng(-0.05f, 0.35f), rng(-0.05f, 0.35f),
                      rng(-0.05f, 0.05f));
    float best = reach;
    for (std::size_t t = 0; t < m.indices.size(); t += 3) {
      best = std::min(
          best, vr::length(p - eval::closest_point_on_triangle(
                                   p, m.vertices[m.indices[t]].position,
                                   m.vertices[m.indices[t + 1]].position,
                                   m.vertices[m.indices[t + 2]].position)));
    }
    CHECK(near(double(d.value().distance(p)), double(best), 1e-7));
    within += best < reach ? 1 : 0;
  }
  CHECK(within > 1000 && within < 3000);  // both sides of the reach
  return 0;
}

int distance_case() {
  vkc::Result<eval::MeshDistance> r =
      eval::MeshDistance::create(plane(20, 0.01f, 0.0f), 0.05f);  // 0.2 m
  CHECK(r.ok());
  // The mesh was a temporary: the index copied what it needs.
  const eval::MeshDistance& d = r.value();
  CHECK(near(double(d.distance({0.03f, 0.07f, 0.02f})), 0.02, 1e-6));
  CHECK(near(double(d.distance({0.1f, 0.1f, -0.013f})), 0.013, 1e-6));
  // Off the edge of the square: to the edge, not to the plane.
  CHECK(near(double(d.distance({0.23f, 0.1f, 0.0f})), 0.03, 1e-6));
  // Nothing within reach reads as the reach, exactly.
  CHECK(d.distance({0.1f, 0.1f, 0.2f}) == 0.05f);
  CHECK(d.distance({5.0f, 5.0f, 0.0f}) == 0.05f);
  // An empty mesh is nothing within reach anywhere.
  vkc::Result<eval::MeshDistance> empty =
      eval::MeshDistance::create(mesh::Mesh{}, 0.05f);
  CHECK(empty.ok() && empty.value().distance({0, 0, 0}) == 0.05f);

  // A mesh triangle whose first two corners coincide is the segment to the
  // third, not a hole.
  mesh::Mesh sliver;
  for (const vr::Vec3f& q : {vr::Vec3f(0, 0, 0), vr::Vec3f(0.5f, 0.5f, 0.5f),
                             vr::Vec3f(0, 0.01f, 0)}) {
    mesh::Vertex v{};
    v.position = q;
    sliver.vertices.push_back(v);
  }
  sliver.indices = {0, 0, 2};
  vkc::Result<eval::MeshDistance> s = eval::MeshDistance::create(sliver, 0.02f);
  CHECK(s.ok());
  CHECK(near(double(s.value().distance({0.001f, 0.005f, 0.0f})), 0.001, 1e-6));

  // A triangle collapsed to one point is not surface.
  sliver.indices = {2, 2, 2};
  vkc::Result<eval::MeshDistance> point =
      eval::MeshDistance::create(sliver, 0.02f);
  CHECK(point.ok());
  CHECK(point.value().distance({0.0f, 0.01f, 0.001f}) == 0.02f);

  // A query that is not finite, or past the index's range, has nothing
  // within reach -- and does not overflow the cell arithmetic getting there.
  const float inf = std::numeric_limits<float>::infinity();
  CHECK(d.distance({inf, 0, 0}) == 0.05f);
  CHECK(d.distance({0, -inf, 0}) == 0.05f);
  CHECK(d.distance({0, 0, std::numeric_limits<float>::quiet_NaN()}) == 0.05f);
  CHECK(d.distance({1e30f, 0, 0}) == 0.05f);
  CHECK(d.distance({0, -3e38f, 0}) == 0.05f);
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
  // The p95 is the nearest rank, ceil(0.95 n): the 19th of 20, not the max.
  std::vector<float> twenty;
  for (int i = 1; i <= 20; ++i) {
    twenty.push_back(float(i));
  }
  const eval::DistanceStats t = eval::summarize(twenty, 100.0f);
  CHECK(t.p95 == 19.0 && t.max == 20.0);
  // And one sample is its own p95.
  CHECK(eval::summarize({0.5f}, 1.0f).p95 == 0.5);
  return 0;
}

int compare_case() {
  const mesh::Mesh reference = plane(40, 0.005f, 0.0f);  // 0.2 m, 5 mm cells
  // The same plane 3 mm up: every point of either is exactly 3 mm from the
  // other, in both directions.
  const mesh::Mesh shifted = plane(40, 0.005f, 0.003f);
  eval::CompareOptions opt;
  opt.reach = 0.02f;
  vkc::Result<eval::MeshComparison> c = compare(reference, shifted, opt);
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
  vkc::Result<eval::MeshComparison> same = compare(reference, reference, opt);
  CHECK(same.ok());
  CHECK(same.value().accuracy.max == 0.0 && same.value().coverage.max == 0.0);

  // Half the test mesh gone (every square at x >= 0.1 m): what remains is
  // still exactly on the surface, but the reference past the cut plus the
  // reach is uncovered.
  const mesh::Mesh half = plane(40, 0.005f, 0.0f, 0.1f);
  vkc::Result<eval::MeshComparison> h = compare(reference, half, opt);
  CHECK(h.ok());
  CHECK(h.value().accuracy.max == 0.0);
  CHECK(h.value().coverage.beyond_reach > reference.vertices.size() / 3);
  CHECK(h.value().coverage.beyond_reach < reference.vertices.size() / 2);

  // A stride measures about one vertex in seven...
  opt.stride = 7;
  vkc::Result<eval::MeshComparison> sparse = compare(reference, shifted, opt);
  CHECK(sparse.ok());
  const std::size_t n = reference.vertices.size();
  CHECK(sparse.value().accuracy.count > n / 10);
  CHECK(sparse.value().accuracy.count < n / 5);
  CHECK(near(sparse.value().accuracy.mean, 0.003, 1e-6));
  // ... chosen by position, so the same surfaces stored in another order give
  // the same figures to the bit, where a stride over the array would not. The
  // test mesh is bumped so every vertex has a distance of its own; on a flat
  // grid, many subsets share one distribution.
  mesh::Mesh bumpy = half;
  for (std::size_t v = 0; v < bumpy.vertices.size(); ++v) {
    bumpy.vertices[v].position.z = 1e-4f * float(v % 17);
  }
  vkc::Result<eval::MeshComparison> again =
      compare(reordered(reference), reordered(bumpy), opt);
  vkc::Result<eval::MeshComparison> first = compare(reference, bumpy, opt);
  CHECK(again.ok() && first.ok());
  CHECK(identical(again.value(), first.value()));
  return 0;
}

// Only the surface is measured: vertices no triangle uses, and triangles
// collapsed to a point, are not surface.
int surface_case() {
  const mesh::Mesh reference = plane(20, 0.005f, 0.0f);
  eval::CompareOptions opt;
  opt.reach = 0.02f;
  opt.fscore_threshold = 0.001f;
  vkc::Result<eval::MeshComparison> clean = compare(reference, reference, opt);
  CHECK(clean.ok());

  mesh::Mesh stale = reference;
  mesh::Vertex far{};
  far.position = vr::Vec3f(0.05f, 0.05f, 0.01f);  // 1 cm off, within reach
  stale.vertices.push_back(far);                  // referenced by nothing
  const auto retired = static_cast<std::uint32_t>(stale.vertices.size());
  for (int k = 0; k < 3; ++k) {  // the default kernel's retired triangle
    mesh::Vertex o{};
    o.position = vr::Vec3f(0.05f, 0.05f, -0.01f);
    stale.vertices.push_back(o);
  }
  stale.indices.insert(stale.indices.end(),
                       {retired, retired + 1, retired + 2});
  stale.indices.insert(stale.indices.end(), {0, 0, 0});  // the shared one's
  // A vertex nothing uses may even be garbage.
  mesh::Vertex nan{};
  nan.position = vr::Vec3f(std::numeric_limits<float>::quiet_NaN());
  stale.vertices.push_back(nan);

  const mesh::Mesh* const clean_mesh = &reference;
  const mesh::Mesh* const stale_mesh = &stale;
  for (const auto& [ref, test] :
       {std::pair{clean_mesh, stale_mesh}, std::pair{stale_mesh, clean_mesh}}) {
    vkc::Result<eval::MeshComparison> c = compare(*ref, *test, opt);
    CHECK(c.ok());
    CHECK(identical(c.value(), clean.value()));
  }
  return 0;
}

// A reference indexed once gives what a fresh one gives, for each mesh judged
// against it, and outlives the mesh it was built from.
int reference_case() {
  eval::CompareOptions opt;
  opt.reach = 0.02f;
  opt.stride = 3;
  opt.fscore_threshold = 0.004f;
  vkc::Result<eval::ReferenceMesh> ref =
      eval::ReferenceMesh::create(plane(40, 0.005f, 0.0f), opt);
  CHECK(ref.ok());
  CHECK(ref.value().options().stride == 3);
  const mesh::Mesh reference = plane(40, 0.005f, 0.0f);
  for (const mesh::Mesh& test :
       {plane(40, 0.005f, 0.003f), plane(40, 0.005f, 0.0f, 0.1f)}) {
    vkc::Result<eval::MeshComparison> once = ref.value().compare(test);
    vkc::Result<eval::MeshComparison> each = compare(reference, test, opt);
    CHECK(once.ok() && each.ok());
    CHECK(identical(once.value(), each.value()));
  }
  // It refuses a mesh MeshDistance refuses, either side, and bad options.
  mesh::Mesh wild = reference;
  wild.indices[4] = static_cast<std::uint32_t>(wild.vertices.size());
  CHECK(!ref.value().compare(wild).ok());
  CHECK(!eval::ReferenceMesh::create(wild, opt).ok());
  eval::CompareOptions no_stride = opt;
  no_stride.stride = 0;
  CHECK(!eval::ReferenceMesh::create(reference, no_stride).ok());
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
  vkc::Result<eval::MeshComparison> above = compare(reference, shifted, opt);
  CHECK(above.ok());
  CHECK(above.value().fscore.threshold == 0.004f);
  CHECK(above.value().fscore.precision == 1.0);
  CHECK(above.value().fscore.recall == 1.0);
  CHECK(above.value().fscore.f == 1.0);
  opt.fscore_threshold = 0.002f;
  vkc::Result<eval::MeshComparison> below = compare(reference, shifted, opt);
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
  vkc::Result<eval::MeshComparison> h = compare(reference, half, opt);
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
  // A comparison refuses the same meshes, either side.
  CHECK(!compare(wild, good).ok());
  CHECK(!compare(good, wild).ok());
  // A corner that is not finite, or too far out for the cell keys: each once
  // left the cell loop unbounded.
  for (float bad : {std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN(), 1e30f}) {
    mesh::Mesh off = good;
    off.vertices[off.indices[0]].position.z = bad;
    CHECK(!eval::MeshDistance::create(off, 0.02f).ok());
    CHECK(!compare(good, off).ok());
  }
  // A reach too small for the triangles, refused before it is paid for: one
  // 1 cm triangle at a micron once took seconds and gigabytes.
  mesh::Mesh one;
  for (const vr::Vec3f& q : {vr::Vec3f(0, 0, 0), vr::Vec3f(0.01f, 0, 0.01f),
                             vr::Vec3f(0, 0.01f, 0.005f)}) {
    mesh::Vertex v{};
    v.position = q;
    one.vertices.push_back(v);
  }
  one.indices = {0, 1, 2};
  const auto t0 = std::chrono::steady_clock::now();
  CHECK(!eval::MeshDistance::create(one, 1e-6f).ok());
  CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(1));
  CHECK(eval::MeshDistance::create(one, 0.01f).ok());  // 3 x 3 x 2 cells
  // A bad reach is named as the reach, not as the threshold left at 0.
  eval::CompareOptions bad_reach;
  bad_reach.reach = -0.01f;
  vkc::Result<eval::MeshComparison> r = compare(good, good, bad_reach);
  CHECK(!r.ok());
  CHECK(r.status().message().find("reach must be") != std::string::npos);
  // A stride of 0, and an F-score threshold negative, not finite or past the
  // reach (where every distance reads as the reach).
  eval::CompareOptions opt;
  opt.stride = 0;
  CHECK(!compare(good, good, opt).ok());
  for (float tau : {-0.001f, 0.021f, std::numeric_limits<float>::quiet_NaN()}) {
    eval::CompareOptions o;
    o.reach = 0.02f;
    o.fscore_threshold = tau;
    CHECK(!compare(good, good, o).ok());
  }
  eval::CompareOptions at_reach;
  at_reach.reach = 0.02f;
  at_reach.fscore_threshold = 0.02f;  // the reach itself is allowed
  CHECK(compare(good, good, at_reach).ok());
  return 0;
}

}  // namespace

int main() {
  if (closest_point_case() != 0) return 1;
  if (thin_triangle_case() != 0) return 1;
  if (search_case() != 0) return 1;
  if (distance_case() != 0) return 1;
  if (summarize_case() != 0) return 1;
  if (compare_case() != 0) return 1;
  if (fscore_case() != 0) return 1;
  if (surface_case() != 0) return 1;
  if (reference_case() != 0) return 1;
  if (refusals_case() != 0) return 1;
  std::printf("eval mesh distance: OK\n");
  return 0;
}
