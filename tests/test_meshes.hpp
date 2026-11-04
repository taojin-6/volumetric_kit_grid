// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Host-side triangle meshes for the GPU tests that take one in (volume's
// triangle allocation and voxelize's mesh integration): the index-buffer mesh
// they share, their fixtures, and the closest point on a triangle -- written
// out independently of the GLSL the kernels use, so the two are not the same
// expression checked against itself.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "volumetric_kit/grid/base/math/vector_types.hpp"

namespace test_meshes {

namespace vgrid = volumetric_kit::grid;

struct Mesh {
  std::vector<vgrid::Vec3f> v;
  std::vector<std::uint32_t> i;
  std::uint32_t vertex_count() const {
    return static_cast<std::uint32_t>(v.size());
  }
  std::uint32_t triangle_count() const {
    return static_cast<std::uint32_t>(i.size() / 3);
  }
};

inline vgrid::Vec3f corner(const Mesh& m, std::size_t t, int k) {
  return m.v[m.i[3 * t + k]];
}

// Closest point on a triangle: barycentric projection with a clamp to the
// edges, a different formulation from the shader's region test. A degenerate
// triangle -- which marching cubes emits where a sample is exactly zero --
// projects to NaN and falls through to its edges, and a zero-length edge
// clamps to its endpoint, so the answer stays finite whenever the corners are.
inline vgrid::Vec3f closest_point(vgrid::Vec3f p, vgrid::Vec3f a,
                                  vgrid::Vec3f b, vgrid::Vec3f c) {
  const vgrid::Vec3f ab = b - a;
  const vgrid::Vec3f ac = c - a;
  const vgrid::Vec3f n = vgrid::cross(ab, ac);
  const float nn = vgrid::dot(n, n);
  // Project onto the plane, then read off barycentrics as signed sub-areas.
  const vgrid::Vec3f q = p - n * (vgrid::dot(n, p - a) / nn);
  const float u = vgrid::dot(n, vgrid::cross(c - b, q - b)) / nn;
  const float v = vgrid::dot(n, vgrid::cross(a - c, q - c)) / nn;
  const float w = vgrid::dot(n, vgrid::cross(ab, q - a)) / nn;
  if (u >= 0.0f && v >= 0.0f && w >= 0.0f) {
    return q;  // inside
  }
  // Outside: the nearest point on the nearest edge segment.
  auto on_segment = [](vgrid::Vec3f pt, vgrid::Vec3f s0, vgrid::Vec3f s1) {
    const vgrid::Vec3f d = s1 - s0;
    const float dd = vgrid::dot(d, d);
    float t = (dd > 0.0f) ? vgrid::dot(pt - s0, d) / dd : 0.0f;
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    return s0 + d * t;
  };
  const vgrid::Vec3f e0 = on_segment(p, a, b);
  const vgrid::Vec3f e1 = on_segment(p, b, c);
  const vgrid::Vec3f e2 = on_segment(p, c, a);
  const float d0 = vgrid::dot(p - e0, p - e0);
  const float d1 = vgrid::dot(p - e1, p - e1);
  const float d2 = vgrid::dot(p - e2, p - e2);
  if (d0 <= d1 && d0 <= d2) return e0;
  return (d1 <= d2) ? e1 : e2;
}

// Flip every face whose normal points toward `centre` -- correct for a convex
// solid, which is all it is used on (the dented cube is oriented before it is
// dented; moving a vertex does not change a winding).
inline void orient_outward(Mesh& m, vgrid::Vec3f centre) {
  for (std::size_t t = 0; t < m.triangle_count(); ++t) {
    const vgrid::Vec3f a = corner(m, t, 0), b = corner(m, t, 1),
                       c = corner(m, t, 2);
    if (vgrid::dot(vgrid::cross(b - a, c - a), (a + b + c) / 3.0f - centre) <
        0.0f) {
      std::swap(m.i[3 * t + 1], m.i[3 * t + 2]);
    }
  }
}

// A cube of side `side` at `origin`, 12 triangles, wound outward: flat faces
// and the 90-degree edges and corners a K-coefficient block transform smooths.
inline Mesh cube(vgrid::Vec3f origin, float side) {
  Mesh m;
  for (int k = 0; k < 8; ++k) {
    const vgrid::Vec3f unit(float((k == 1 || k == 2 || k == 5 || k == 6)),
                            float((k == 2 || k == 3 || k == 6 || k == 7)),
                            float(k >= 4));
    m.v.push_back(origin + side * unit);
  }
  const std::uint32_t quads[6][4] = {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 5, 4},
                                     {3, 2, 6, 7}, {0, 3, 7, 4}, {1, 2, 6, 5}};
  for (const auto& q : quads) {
    m.i.insert(m.i.end(), {q[0], q[1], q[2], q[0], q[2], q[3]});
  }
  orient_outward(m, origin + vgrid::Vec3f(0.5f * side));
  return m;
}

// The cube with its (1,1,1) corner pushed in to 0.6 of the way along the
// diagonal: concave edges and a concave vertex, the case a convex fixture
// cannot reach. It is oriented before it is dented, and moving a vertex does
// not change a winding.
inline Mesh dented_cube(vgrid::Vec3f origin, float side) {
  Mesh m = cube(origin, side);
  m.v[6] = origin + vgrid::Vec3f(0.6f * side);
  return m;
}

// A sphere of radius `r` at `centre`, its radius displaced by up to 4 mm of
// noise: `rings` x `segments` quads, two triangles each -- ~9 mm edges at
// r = 0.25 m and 125 x 200, about the density of a scanned body. The pole
// rows are degenerate (a ring of coincident vertices), as a scan's slivers
// are.
inline Mesh displaced_sphere(vgrid::Vec3f centre, float r, std::uint32_t rings,
                             std::uint32_t segments, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> noise(-0.004f, 0.004f);
  Mesh m;
  for (std::uint32_t i = 0; i <= rings; ++i) {
    const float theta = 3.14159265f * float(i) / float(rings);
    for (std::uint32_t j = 0; j < segments; ++j) {
      const float phi = 2.0f * 3.14159265f * float(j) / float(segments);
      const float rr = r + noise(rng);
      m.v.push_back(centre +
                    rr * vgrid::Vec3f(std::sin(theta) * std::cos(phi),
                                      std::cos(theta),
                                      std::sin(theta) * std::sin(phi)));
    }
  }
  for (std::uint32_t i = 0; i < rings; ++i) {
    for (std::uint32_t j = 0; j < segments; ++j) {
      const std::uint32_t a = i * segments + j;
      const std::uint32_t b = i * segments + (j + 1) % segments;
      const std::uint32_t c = a + segments;
      const std::uint32_t d = b + segments;
      m.i.insert(m.i.end(), {a, c, b, b, c, d});
    }
  }
  return m;
}

// `n` random triangles in a 1.6 m box around `centre`, edges up to `edge`.
inline Mesh soup(vgrid::Vec3f centre, std::uint32_t n, float edge,
                 std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> pos(-0.8f, 0.8f);
  std::uniform_real_distribution<float> off(-edge, edge);
  Mesh m;
  for (std::uint32_t t = 0; t < n; ++t) {
    const vgrid::Vec3f p = centre + vgrid::Vec3f(pos(rng), pos(rng), pos(rng));
    for (int k = 0; k < 3; ++k) {
      m.v.push_back(k == 0 ? p
                           : p + vgrid::Vec3f(off(rng), off(rng), off(rng)));
      m.i.push_back(static_cast<std::uint32_t>(m.v.size() - 1));
    }
  }
  return m;
}

// `n` triangles whose vertices lie on multiples of half a 5 mm voxel within
// `range` half-voxels of the origin, so world / voxel_size lands on (or next
// to) a .5 that rounds to even, after the band's subtract and add.
inline Mesh ties(std::uint32_t n, std::uint32_t seed, int range = 200) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> k(-range, range);
  std::uniform_int_distribution<int> d(-6, 6);
  Mesh m;
  for (std::uint32_t t = 0; t < n; ++t) {
    const int x = k(rng), y = k(rng), z = k(rng);
    for (int v = 0; v < 3; ++v) {
      const int dx = v == 0 ? 0 : d(rng), dy = v == 0 ? 0 : d(rng),
                dz = v == 0 ? 0 : d(rng);
      m.v.push_back(vgrid::Vec3f(float(x + dx), float(y + dy), float(z + dz)) *
                    0.0025f);
      m.i.push_back(static_cast<std::uint32_t>(m.v.size() - 1));
    }
  }
  return m;
}

}  // namespace test_meshes
