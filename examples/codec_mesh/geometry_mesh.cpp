// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "geometry_mesh.hpp"

#include <vector>

#include <glm/glm.hpp>

namespace vgrid_example {

vgrid::mesh::Mesh geometry_mesh(const vgrid::mesh_io::TriangleMesh& geometry) {
  vgrid::mesh::Mesh mesh;
  mesh.indices = geometry.indices;
  mesh.vertices.reserve(geometry.positions.size());
  std::vector<vgrid::Vec3d> normals(geometry.positions.size(),
                                    vgrid::Vec3d(0.0));
  for (std::size_t t = 0; t < geometry.indices.size(); t += 3) {
    const auto a = geometry.indices[t];
    const auto b = geometry.indices[t + 1];
    const auto c = geometry.indices[t + 2];
    const vgrid::Vec3d pa(geometry.positions[a]);
    const vgrid::Vec3d normal =
        glm::cross(vgrid::Vec3d(geometry.positions[b]) - pa,
                   vgrid::Vec3d(geometry.positions[c]) - pa);
    for (const auto index : {a, b, c}) normals[index] += normal;
  }
  for (std::size_t i = 0; i < geometry.positions.size(); ++i) {
    const vgrid::Vec3d normal = glm::dot(normals[i], normals[i]) > 0.0
                                    ? glm::normalize(normals[i])
                                    : vgrid::Vec3d(0.0, 1.0, 0.0);
    mesh.vertices.push_back({geometry.positions[i],
                             vgrid::Vec3f(normal),
                             {1, 0, 0, 1},
                             {-1, -1},
                             {1, 1, 1, 1}});
  }
  return mesh;
}

}  // namespace vgrid_example
