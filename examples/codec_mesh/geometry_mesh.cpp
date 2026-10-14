// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "geometry_mesh.hpp"

#include <array>
#include <cmath>
#include <vector>

namespace vr_example {
namespace {

using Point3d = std::array<double, 3>;

Point3d point(const vr::Vec3f& p) { return {p.x, p.y, p.z}; }
Point3d subtract(const Point3d& a, const Point3d& b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
double dot(const Point3d& a, const Point3d& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Point3d cross(const Point3d& a, const Point3d& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}
Point3d unit(const Point3d& v) {
  const double length = std::sqrt(dot(v, v));
  return {v[0] / length, v[1] / length, v[2] / length};
}

}  // namespace

vr::mesh::Mesh geometry_mesh(const vr::io::TriangleMesh& geometry) {
  vr::mesh::Mesh mesh;
  mesh.indices = geometry.indices;
  mesh.vertices.reserve(geometry.positions.size());
  std::vector<Point3d> normals(geometry.positions.size());
  for (std::size_t t = 0; t < geometry.indices.size(); t += 3) {
    const auto a = geometry.indices[t];
    const auto b = geometry.indices[t + 1];
    const auto c = geometry.indices[t + 2];
    const auto normal = cross(
        subtract(point(geometry.positions[b]), point(geometry.positions[a])),
        subtract(point(geometry.positions[c]), point(geometry.positions[a])));
    for (const auto index : {a, b, c}) {
      for (std::size_t axis = 0; axis < 3; ++axis)
        normals[index][axis] += normal[axis];
    }
  }
  std::size_t index = 0;
  for (const auto& p : geometry.positions) {
    const auto normal = dot(normals[index], normals[index]) > 0.0
                            ? unit(normals[index])
                            : Point3d{0, 1, 0};
    mesh.vertices.push_back(
        {p,
         {float(normal[0]), float(normal[1]), float(normal[2])},
         {1, 0, 0, 1},
         {-1, -1},
         {1, 1, 1, 1}});
    ++index;
  }
  return mesh;
}

}  // namespace vr_example
