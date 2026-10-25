// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/io/mesh_io.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <assimp/DefaultIOSystem.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/version.h>
#include <assimp/Importer.hpp>

#include "volumetric_kit/core/base/exception_status.hpp"
#include "volumetric_kit/core/base/file_io.hpp"
#include "volumetric_kit/recon/io/textured_mesh.hpp"

namespace volumetric_kit::recon::io {
namespace {

// A vertex: its position and, in a textured load, its texture coordinates
// (zero otherwise). Each is finite and none is negative zero, so equal bits
// mean equal values.
using VertexKey = std::array<float, 5>;

struct VertexKeyHash {
  std::size_t operator()(const VertexKey& key) const noexcept {
    std::size_t seed = 0;
    for (const float value : key) {
      std::uint32_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      seed ^= std::hash<std::uint32_t>{}(bits) + 0x9e3779b9u + (seed << 6) +
              (seed >> 2);
    }
    return seed;
  }
};

core::Result<bool> reflected(const aiMatrix4x4& m) {
  const std::array<ai_real, 16> values = {m.a1, m.a2, m.a3, m.a4, m.b1, m.b2,
                                          m.b3, m.b4, m.c1, m.c2, m.c3, m.c4,
                                          m.d1, m.d2, m.d3, m.d4};
  for (const auto value : values) {
    if (!std::isfinite(value)) {
      return core::Status::invalid_argument(
          "load_mesh: nonfinite node transform");
    }
  }
  if (m.d1 != 0 || m.d2 != 0 || m.d3 != 0 || m.d4 != 1) {
    return core::Status::invalid_argument(
        "load_mesh: non-affine node transform");
  }
  const double determinant =
      double(m.a1) * (double(m.b2) * m.c3 - double(m.b3) * m.c2) -
      double(m.a2) * (double(m.b1) * m.c3 - double(m.b3) * m.c1) +
      double(m.a3) * (double(m.b1) * m.c2 - double(m.b2) * m.c1);
  if (!std::isfinite(determinant) || determinant == 0) {
    return core::Status::invalid_argument("load_mesh: singular node transform");
  }
  return determinant < 0;
}

core::Result<std::array<float, 3>> transform_position(const aiMatrix4x4& m,
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
      return core::Status::invalid_argument(
          "load_mesh: nonfinite/overflowing position");
    }
    result[axis] = static_cast<float>(value);
    if (result[axis] == 0) result[axis] = 0;  // Canonicalize negative zero.
  }
  return result;
}

// Assimp's own file access, noting each OBJ material library it opens.
class MaterialLibraries final : public Assimp::IOSystem {
 public:
  explicit MaterialLibraries(std::vector<std::string>* opened)
      : opened_(opened) {}

  bool Exists(const char* file) const override { return files_.Exists(file); }
  char getOsSeparator() const override { return files_.getOsSeparator(); }
  Assimp::IOStream* Open(const char* file, const char* mode) override {
    Assimp::IOStream* handle = files_.Open(file, mode);
    std::string extension = std::filesystem::path(file).extension().string();
    for (char& c : extension) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (handle != nullptr && extension == ".mtl") opened_->push_back(file);
    return handle;
  }
  void Close(Assimp::IOStream* handle) override { files_.Close(handle); }
  bool ComparePaths(const char* one, const char* second) const override {
    return files_.ComparePaths(one, second);
  }

 private:
  Assimp::DefaultIOSystem files_;
  std::vector<std::string>* opened_;
};

// The asset as load_mesh reads it, its errors named "load_mesh". A textured
// load joins vertices on their texture coordinates too, and reports its own
// refusals only for an asset load_mesh accepts.
core::Result<TexturedMesh> import_mesh(const std::string& path, bool textured) {
  VKC_TRY(check_mesh_file(path));

  std::vector<std::string> libraries;
  Assimp::Importer importer;
  // The importer owns and deletes it.
  if (textured) importer.SetIOHandler(new MaterialLibraries(&libraries));
  // Keep source coordinates: Collada and FBX importers otherwise apply the
  // file's unit and up axis. Literal names build against releases that lack a
  // property. Assimp before 5.3 still applies Collada's <unit>, and 5.4.x
  // converts FBX regardless.
  importer.SetPropertyBool("IMPORT_COLLADA_IGNORE_UP_DIRECTION", true);
  importer.SetPropertyBool("IMPORT_COLLADA_IGNORE_UNIT_SIZE", true);
  importer.SetPropertyBool("AI_CONFIG_IMPORT_FBX_IGNORE_UP_DIRECTION", true);
  // Avoid repair, approximate vertex joining and automatic scene scaling.
  // Exact joining below ignores normal and material seams, and UV seams but
  // in a textured load.
  const aiScene* scene = importer.ReadFile(
      path, aiProcess_Triangulate | aiProcess_ValidateDataStructure);
  if (scene == nullptr) {
    return core::Status::io_error("load_mesh: " + path + ": " +
                                  importer.GetErrorString());
  }
  if (scene->mRootNode == nullptr || scene->mNumMeshes == 0) {
    return core::Status::invalid_argument(
        "load_mesh: asset has no mesh geometry");
  }
  if (scene->mNumAnimations != 0) {
    return core::Status::unsupported(
        "load_mesh: animation requires an evaluated static asset");
  }

  struct Node {
    const aiNode* source;
    aiMatrix4x4 parent;
  };
  std::vector<Node> pending{{scene->mRootNode, aiMatrix4x4{}}};
  std::size_t source_vertices = 0;
  for (unsigned i = 0; i < scene->mNumMeshes; ++i) {
    if (scene->mMeshes[i] != nullptr)
      source_vertices += scene->mMeshes[i]->mNumVertices;
  }
  std::unordered_map<VertexKey, std::uint32_t, VertexKeyHash> vertices;
  vertices.reserve(source_vertices);
  TexturedMesh textured_result;
  TriangleMesh& result = textured_result.geometry;
  core::Status refusal;  // A textured load's first own refusal.
  const auto refuse = [&refusal](core::Status status) {
    if (refusal.ok()) refusal = std::move(status);
  };
  std::optional<std::string> diffuse;
  while (!pending.empty()) {
    const Node node = pending.back();
    pending.pop_back();
    const aiMatrix4x4 world = node.parent * node.source->mTransformation;
    // Only transforms that place geometry are validated; helper, camera or
    // hidden nodes without meshes do not refuse the asset.
    bool flip = false;
    if (node.source->mNumMeshes != 0) {
      VKC_ASSIGN(flip, reflected(world));
    }
    for (unsigned index = 0; index < node.source->mNumMeshes; ++index) {
      const unsigned mesh_index = node.source->mMeshes[index];
      if (mesh_index >= scene->mNumMeshes ||
          scene->mMeshes[mesh_index] == nullptr) {
        return core::Status::invalid_argument(
            "load_mesh: invalid scene mesh index");
      }
      const aiMesh& mesh = *scene->mMeshes[mesh_index];
      if (mesh.HasBones() || mesh.mNumAnimMeshes != 0) {
        return core::Status::unsupported(
            "load_mesh: skinning/morphs require an evaluated static asset");
      }
      if (mesh.mNumVertices == 0 || mesh.mVertices == nullptr) {
        return core::Status::invalid_argument(
            "load_mesh: mesh has no positions");
      }
      const aiVector3D* uvs = textured ? mesh.mTextureCoords[0] : nullptr;
      if (textured) {
        if (uvs == nullptr) {
          refuse(core::Status::not_found(
              "load_textured_mesh: a mesh has no texture coordinates: " +
              path));
        }
        std::string map;
        aiString file;
        if (mesh.mMaterialIndex >= scene->mNumMaterials ||
            scene->mMaterials[mesh.mMaterialIndex] == nullptr) {
          refuse(core::Status::invalid_argument(
              "load_textured_mesh: invalid material index"));
        } else if (scene->mMaterials[mesh.mMaterialIndex]->GetTexture(
                       aiTextureType_DIFFUSE, 0, &file) == AI_SUCCESS) {
          map = file.C_Str();
        }
        if (!diffuse) {
          diffuse = map;
        } else if (*diffuse != map) {
          refuse(core::Status::unsupported(
              "load_textured_mesh: meshes name different diffuse maps"));
        }
      }
      std::vector<std::uint32_t> remap(mesh.mNumVertices);
      for (unsigned vertex = 0; vertex < mesh.mNumVertices; ++vertex) {
        VKC_ASSIGN(const auto position,
                   transform_position(world, mesh.mVertices[vertex]));
        VertexKey key{position[0], position[1], position[2], 0, 0};
        if (uvs != nullptr) {
          const float u = static_cast<float>(uvs[vertex].x);
          const float v = static_cast<float>(uvs[vertex].y);
          if (!std::isfinite(u) || !std::isfinite(v)) {
            refuse(core::Status::invalid_argument(
                "load_textured_mesh: nonfinite texture coordinate"));
          } else {
            key[3] = u == 0 ? 0 : u;  // Canonicalize negative zero.
            key[4] = v == 0 ? 0 : v;
          }
        }
        auto found = vertices.find(key);
        if (found == vertices.end()) {
          if (result.positions.size() >=
              std::numeric_limits<std::uint32_t>::max()) {
            return core::Status::out_of_memory(
                "load_mesh: geometry exceeds 32-bit indexing");
          }
          const auto id = static_cast<std::uint32_t>(result.positions.size());
          found = vertices.emplace(key, id).first;
          result.positions.push_back({key[0], key[1], key[2]});
          if (textured) textured_result.uvs.push_back({key[3], key[4]});
        }
        remap[vertex] = found->second;
      }
      for (unsigned face_index = 0; face_index < mesh.mNumFaces; ++face_index) {
        const aiFace& face = mesh.mFaces[face_index];
        if (face.mNumIndices != 3 || face.mIndices == nullptr) {
          return core::Status::unsupported(
              "load_mesh: asset contains non-triangle primitives");
        }
        if (result.indices.size() / 3 >=
            std::numeric_limits<std::uint32_t>::max()) {
          return core::Status::out_of_memory("load_mesh: too many triangles");
        }
        std::array<std::uint32_t, 3> triangle{};
        for (std::size_t corner = 0; corner < triangle.size(); ++corner) {
          if (face.mIndices[corner] >= remap.size()) {
            return core::Status::invalid_argument(
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
        return core::Status::invalid_argument("load_mesh: null scene node");
      pending.push_back({next, world});
    }
  }
  if (result.indices.empty()) {
    return core::Status::invalid_argument("load_mesh: asset has no triangles");
  }
  VKC_TRY(refusal);
  if (!diffuse || diffuse->empty()) return textured_result;
  // Assimp names a texture held in the asset "*<index>".
  if ((*diffuse)[0] == '*') {
    return core::Status::unsupported(
        "load_textured_mesh: embedded diffuse map: " + path);
  }
  // Either separator divides the map's path, as when Assimp opens a file.
  std::string map = *diffuse;
  std::replace(map.begin(), map.end(), '\\', '/');
  // An OBJ's texture paths are relative to its material library; other
  // formats define their materials in the asset itself.
  const std::filesystem::path asset = std::filesystem::path(path).parent_path();
  std::filesystem::path base = asset;
  if (!libraries.empty()) {
    const std::filesystem::path library =
        std::filesystem::path(libraries.front()).parent_path();
    for (const std::string& other : libraries) {
      if (std::filesystem::path(other).parent_path() != library) {
        return core::Status::unsupported(
            "load_textured_mesh: material libraries in more than one "
            "directory: " +
            path);
      }
    }
    base = library;
    // Assimp 6 prefixes the library's directory as the OBJ names it, even to
    // an absolute path; earlier releases report the path as the library
    // writes it. An absolute prefix (a bare OBJ name, a library named by an
    // absolute path) already makes the map absolute.
    const std::filesystem::path relative = library.lexically_relative(asset);
    const std::string prefix = relative.generic_string() + '/';
    if (aiGetVersionMajor() >= 6 && !relative.empty() &&
        map.compare(0, prefix.size(), prefix) == 0) {
      map.erase(0, prefix.size());
    }
  }
  textured_result.diffuse_texture = (base / map).string();
  return textured_result;
}

}  // namespace

core::Status check_mesh_file(const std::string& path) try {
  VKC_TRY(core::check_path("check_mesh_file", path));
  std::error_code error;
  const auto file = std::filesystem::status(path, error);
  if (error == std::errc::no_such_file_or_directory ||
      (!error && !std::filesystem::exists(file))) {
    return core::Status::not_found("check_mesh_file: missing file: " + path);
  }
  if (error) {
    return core::Status::io_error("check_mesh_file: " + path + ": " +
                                  error.message());
  }
  if (!std::filesystem::is_regular_file(file)) {
    return core::Status::invalid_argument(
        "check_mesh_file: not a regular file: " + path);
  }
  return {};
} catch (...) {
  return core::exception_status("check_mesh_file", core::Status::Code::IoError);
}

core::Result<TriangleMesh> load_mesh(const std::string& path) {
  try {
    VKC_ASSIGN(TexturedMesh mesh, import_mesh(path, false));
    return std::move(mesh.geometry);
  } catch (...) {
    return core::exception_status("load_mesh", core::Status::Code::IoError);
  }
}

core::Result<TexturedMesh> load_textured_mesh(const std::string& path) {
  try {
    return import_mesh(path, true);
  } catch (...) {
    return core::exception_status("load_textured_mesh",
                                  core::Status::Code::IoError);
  }
}

}  // namespace volumetric_kit::recon::io
