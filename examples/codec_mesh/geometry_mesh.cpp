// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "geometry_mesh.hpp"

#include <vector>

#include <glm/glm.hpp>

namespace vr_example {

vr::mesh::Mesh geometry_mesh(const vr::io::TriangleMesh& geometry) {
  vr::mesh::Mesh mesh;
  mesh.indices = geometry.indices;
  mesh.vertices.reserve(geometry.positions.size());
  std::vector<vr::Vec3d> normals(geometry.positions.size(), vr::Vec3d(0.0));
  for (std::size_t t = 0; t < geometry.indices.size(); t += 3) {
    const auto a = geometry.indices[t];
    const auto b = geometry.indices[t + 1];
    const auto c = geometry.indices[t + 2];
    const vr::Vec3d pa(geometry.positions[a]);
    const vr::Vec3d normal = glm::cross(vr::Vec3d(geometry.positions[b]) - pa,
                                        vr::Vec3d(geometry.positions[c]) - pa);
    for (const auto index : {a, b, c}) normals[index] += normal;
  }
  for (std::size_t i = 0; i < geometry.positions.size(); ++i) {
    const vr::Vec3d normal = glm::dot(normals[i], normals[i]) > 0.0
                                 ? glm::normalize(normals[i])
                                 : vr::Vec3d(0.0, 1.0, 0.0);
    mesh.vertices.push_back({geometry.positions[i],
                             vr::Vec3f(normal),
                             {1, 0, 0, 1},
                             {-1, -1},
                             {1, 1, 1, 1}});
  }
  return mesh;
}

}  // namespace vr_example
