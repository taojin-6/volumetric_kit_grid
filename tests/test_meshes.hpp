// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Host-side triangle meshes for the GPU tests that take one in (volume's
// triangle allocation and tsdf's mesh integration): the index-buffer mesh they
// share, their fixtures, and the closest point on a triangle -- written out
// independently of the GLSL the kernels use, so the two are not the same
// expression checked against itself.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "volumetric_kit/recon/core/math/vector_types.hpp"

namespace test_meshes {

namespace vr = volumetric_kit::recon;

struct Mesh {
  std::vector<vr::Vec3f> v;
  std::vector<std::uint32_t> i;
  std::uint32_t vertex_count() const {
    return static_cast<std::uint32_t>(v.size());
  }
  std::uint32_t triangle_count() const {
    return static_cast<std::uint32_t>(i.size() / 3);
  }
};

inline vr::Vec3f corner(const Mesh& m, std::size_t t, int k) {
  return m.v[m.i[3 * t + k]];
}

// Closest point on a triangle: barycentric projection with a clamp to the
// edges, a different formulation from the shader's region test. A degenerate
// triangle -- which marching cubes emits where a sample is exactly zero --
// projects to NaN and falls through to its edges, and a zero-length edge
// clamps to its endpoint, so the answer stays finite whenever the corners are.
inline vr::Vec3f closest_point(vr::Vec3f p, vr::Vec3f a, vr::Vec3f b,
                               vr::Vec3f c) {
  const vr::Vec3f ab = b - a;
  const vr::Vec3f ac = c - a;
  const vr::Vec3f n = vr::cross(ab, ac);
  const float nn = vr::dot(n, n);
  // Project onto the plane, then read off barycentrics as signed sub-areas.
  const vr::Vec3f q = p - n * (vr::dot(n, p - a) / nn);
  const float u = vr::dot(n, vr::cross(c - b, q - b)) / nn;
  const float v = vr::dot(n, vr::cross(a - c, q - c)) / nn;
  const float w = vr::dot(n, vr::cross(ab, q - a)) / nn;
  if (u >= 0.0f && v >= 0.0f && w >= 0.0f) {
    return q;  // inside
  }
  // Outside: the nearest point on the nearest edge segment.
  auto on_segment = [](vr::Vec3f pt, vr::Vec3f s0, vr::Vec3f s1) {
    const vr::Vec3f d = s1 - s0;
    const float dd = vr::dot(d, d);
    float t = (dd > 0.0f) ? vr::dot(pt - s0, d) / dd : 0.0f;
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    return s0 + d * t;
  };
  const vr::Vec3f e0 = on_segment(p, a, b);
  const vr::Vec3f e1 = on_segment(p, b, c);
  const vr::Vec3f e2 = on_segment(p, c, a);
  const float d0 = vr::dot(p - e0, p - e0);
  const float d1 = vr::dot(p - e1, p - e1);
  const float d2 = vr::dot(p - e2, p - e2);
  if (d0 <= d1 && d0 <= d2) return e0;
  return (d1 <= d2) ? e1 : e2;
}

// Flip every face whose normal points toward `centre` -- correct for a convex
// solid, which is all it is used on (the dented cube is oriented before it is
// dented; moving a vertex does not change a winding).
inline void orient_outward(Mesh& m, vr::Vec3f centre) {
  for (std::size_t t = 0; t < m.triangle_count(); ++t) {
    const vr::Vec3f a = corner(m, t, 0), b = corner(m, t, 1),
                    c = corner(m, t, 2);
    if (vr::dot(vr::cross(b - a, c - a), (a + b + c) / 3.0f - centre) < 0.0f) {
      std::swap(m.i[3 * t + 1], m.i[3 * t + 2]);
    }
  }
}

// A cube of side `side` at `origin`, 12 triangles, wound outward: flat faces
// and the 90-degree edges and corners a K-coefficient block transform smooths.
inline Mesh cube(vr::Vec3f origin, float side) {
  Mesh m;
  for (int k = 0; k < 8; ++k) {
    const vr::Vec3f unit(float((k == 1 || k == 2 || k == 5 || k == 6)),
                         float((k == 2 || k == 3 || k == 6 || k == 7)),
                         float(k >= 4));
    m.v.push_back(origin + side * unit);
  }
  const std::uint32_t quads[6][4] = {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 5, 4},
                                     {3, 2, 6, 7}, {0, 3, 7, 4}, {1, 2, 6, 5}};
  for (const auto& q : quads) {
    m.i.insert(m.i.end(), {q[0], q[1], q[2], q[0], q[2], q[3]});
  }
  orient_outward(m, origin + vr::Vec3f(0.5f * side));
  return m;
}

// The cube with its (1,1,1) corner pushed in to 0.6 of the way along the
// diagonal: concave edges and a concave vertex, the case a convex fixture
// cannot reach. It is oriented before it is dented, and moving a vertex does
// not change a winding.
inline Mesh dented_cube(vr::Vec3f origin, float side) {
  Mesh m = cube(origin, side);
  m.v[6] = origin + vr::Vec3f(0.6f * side);
  return m;
}

}  // namespace test_meshes
