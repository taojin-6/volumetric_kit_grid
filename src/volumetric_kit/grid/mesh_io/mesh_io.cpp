// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/io/mesh_io.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <new>
#include <utility>

#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>

namespace volumetric_kit::recon::io {
namespace {

Result<bool> reflected(const aiMatrix4x4& m) {
  const std::array<ai_real, 16> values = {m.a1, m.a2, m.a3, m.a4, m.b1, m.b2,
                                          m.b3, m.b4, m.c1, m.c2, m.c3, m.c4,
                                          m.d1, m.d2, m.d3, m.d4};
  for (const auto value : values) {
    if (!std::isfinite(value)) {
      return Status::invalid_argument("load_mesh: nonfinite node transform");
    }
  }
  if (m.d1 != 0 || m.d2 != 0 || m.d3 != 0 || m.d4 != 1) {
    return Status::invalid_argument("load_mesh: non-affine node transform");
  }
  const double determinant =
      double(m.a1) * (double(m.b2) * m.c3 - double(m.b3) * m.c2) -
      double(m.a2) * (double(m.b1) * m.c3 - double(m.b3) * m.c1) +
      double(m.a3) * (double(m.b1) * m.c2 - double(m.b2) * m.c1);
  if (!std::isfinite(determinant) || determinant == 0) {
    return Status::invalid_argument("load_mesh: singular node transform");
  }
  return determinant < 0;
}

Result<std::array<float, 3>> transform_position(const aiMatrix4x4& m,
                                                const aiVector3D& p) {
  const std::array<double, 3> transformed = {
      double(m.a1) * p.x + double(m.a2) * p.y + double(m.a3) * p.z + m.a4,
      double(m.b1) * p.x + double(m.b2) * p.y + double(m.b3) * p.z + m.b4,
      double(m.c1) * p.x + double(m.c2) * p.y + double(m.c3) * p.z + m.c4};
  std::array<float, 3> result{};
  for (std::size_t axis = 0; axis < result.size(); ++axis) {
    const double value = transformed[axis];
    if (!std::isfinite(value) ||
        std::abs(value) > std::numeric_limits<float>::max()) {
      return Status::invalid_argument(
          "load_mesh: nonfinite/overflowing position");
    }
    result[axis] = static_cast<float>(value);
    if (result[axis] == 0) result[axis] = 0;  // Canonicalize negative zero.
  }
  return result;
}

Result<TriangleMesh> import_mesh(const std::string& path) {
  if (path.empty() || path.find('\0') != std::string::npos) {
    return Status::invalid_argument("load_mesh: empty path or embedded NUL");
  }
  std::error_code error;
  const auto file = std::filesystem::status(path, error);
  if (error == std::errc::no_such_file_or_directory ||
      (!error && !std::filesystem::exists(file))) {
    return Status::not_found("load_mesh: missing file: " + path);
  }
  if (error) return Status::io_error("load_mesh: " + error.message());
  if (!std::filesystem::is_regular_file(file)) {
    return Status::invalid_argument("load_mesh: expected a regular file");
  }

  Assimp::Importer importer;
  // Avoid repair, approximate vertex joining and automatic scene scaling.
  // Exact joining below ignores discarded UV/normal/material seams.
  const aiScene* scene = importer.ReadFile(
      path, aiProcess_Triangulate | aiProcess_ValidateDataStructure);
  if (scene == nullptr) {
    return Status::io_error("load_mesh: " + path + ": " +
                            importer.GetErrorString());
  }
  if (scene->mRootNode == nullptr || scene->mNumMeshes == 0) {
    return Status::invalid_argument("load_mesh: asset has no mesh geometry");
  }
  if (scene->mNumAnimations != 0) {
    return Status::unsupported(
        "load_mesh: animation requires an evaluated static asset");
  }

  struct Node {
    const aiNode* source;
    aiMatrix4x4 parent;
  };
  std::vector<Node> pending{{scene->mRootNode, aiMatrix4x4{}}};
  std::map<std::array<float, 3>, std::uint32_t> positions;
  TriangleMesh result;
  while (!pending.empty()) {
    const Node node = pending.back();
    pending.pop_back();
    const aiMatrix4x4 world = node.parent * node.source->mTransformation;
    VR_ASSIGN(const bool flip, reflected(world));
    for (unsigned index = 0; index < node.source->mNumMeshes; ++index) {
      const unsigned mesh_index = node.source->mMeshes[index];
      if (mesh_index >= scene->mNumMeshes ||
          scene->mMeshes[mesh_index] == nullptr) {
        return Status::invalid_argument("load_mesh: invalid scene mesh index");
      }
      const aiMesh& mesh = *scene->mMeshes[mesh_index];
      if (mesh.HasBones() || mesh.mNumAnimMeshes != 0) {
        return Status::unsupported(
            "load_mesh: skinning/morphs require an evaluated static asset");
      }
      if (mesh.mNumVertices == 0 || mesh.mVertices == nullptr) {
        return Status::invalid_argument("load_mesh: mesh has no positions");
      }
      std::vector<std::uint32_t> remap(mesh.mNumVertices);
      for (unsigned vertex = 0; vertex < mesh.mNumVertices; ++vertex) {
        VR_ASSIGN(const auto position,
                  transform_position(world, mesh.mVertices[vertex]));
        auto found = positions.find(position);
        if (found == positions.end()) {
          if (result.positions.size() >=
              std::numeric_limits<std::uint32_t>::max()) {
            return Status::out_of_memory(
                "load_mesh: geometry exceeds 32-bit indexing");
          }
          const auto id = static_cast<std::uint32_t>(result.positions.size());
          found = positions.emplace(position, id).first;
          result.positions.push_back({position[0], position[1], position[2]});
        }
        remap[vertex] = found->second;
      }
      for (unsigned face_index = 0; face_index < mesh.mNumFaces; ++face_index) {
        const aiFace& face = mesh.mFaces[face_index];
        if (face.mNumIndices != 3 || face.mIndices == nullptr) {
          return Status::unsupported(
              "load_mesh: asset contains non-triangle primitives");
        }
        if (result.indices.size() / 3 >=
            std::numeric_limits<std::uint32_t>::max()) {
          return Status::out_of_memory("load_mesh: too many triangles");
        }
        std::array<std::uint32_t, 3> triangle{};
        for (std::size_t corner = 0; corner < triangle.size(); ++corner) {
          if (face.mIndices[corner] >= remap.size()) {
            return Status::invalid_argument(
                "load_mesh: out-of-range vertex index");
          }
          triangle[corner] = remap[face.mIndices[corner]];
        }
        if (flip) std::swap(triangle[1], triangle[2]);
        result.indices.insert(result.indices.end(), triangle.begin(),
                              triangle.end());
      }
    }
    // Reverse the push order to preserve the source's child traversal order.
    for (unsigned child = node.source->mNumChildren; child != 0; --child) {
      const aiNode* next = node.source->mChildren[child - 1];
      if (next == nullptr)
        return Status::invalid_argument("load_mesh: null scene node");
      pending.push_back({next, world});
    }
  }
  if (result.indices.empty()) {
    return Status::invalid_argument("load_mesh: asset has no triangles");
  }
  return result;
}

}  // namespace

Result<TriangleMesh> load_mesh(const std::string& path) {
  try {
    return import_mesh(path);
  } catch (const std::bad_alloc&) {
    return Status::out_of_memory({});
  } catch (...) {
    return Status::io_error({});
  }
}

}  // namespace volumetric_kit::recon::io
