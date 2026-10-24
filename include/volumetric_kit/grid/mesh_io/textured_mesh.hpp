// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file io/textured_mesh.hpp
/// @brief A static triangle mesh with its texture coordinates and diffuse
///        map, imported through the optional Assimp backend.

#include <string>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/io/assimp_export.hpp"
#include "volumetric_kit/recon/io/triangle_mesh.hpp"

namespace volumetric_kit::recon::io {

/// @brief Triangle geometry with one texture coordinate per vertex, split at
///        UV seams.
struct TexturedMesh {
  /// Positions duplicated where a vertex's UVs differ.
  TriangleMesh geometry;
  /// One per position.
  std::vector<Vec2f> uvs;
  /// The material's diffuse map, resolved against the material file; empty
  /// if none.
  std::string diffuse_texture;
};

/// @brief Load a static asset with its texture coordinates and the path of
///        its diffuse map.
///
/// Imports as @ref load_mesh does, with the same triangles in the same order,
/// but two corners share a vertex only where both their positions and their
/// first UV channel's coordinates are equal, so a position on a UV seam is
/// repeated once per distinct UV. Coordinates are as Assimp reports them,
/// with v running up the image (OBJ's convention), unclamped.
///
/// The diffuse map is the one every mesh's material names, resolved against
/// the file that defines the material: an OBJ's material library, otherwise
/// the asset. A backslash in its path separates directories, as a slash
/// does. It is not opened; @ref load_color_packed decodes it.
///
/// @param path Asset path without embedded NUL; the files it names, such as
///             material libraries, resolve relative to it.
/// @return The mesh; @ref load_mesh's error for an asset it refuses;
///         otherwise NotFound when a mesh has no texture coordinates,
///         InvalidArgument for a nonfinite coordinate, and Unsupported when
///         meshes name different diffuse maps, the map is embedded in the
///         asset, or material libraries lie in more than one directory.
VR_IO_ASSIMP_API core::Result<TexturedMesh> load_textured_mesh(
    const std::string& path);

}  // namespace volumetric_kit::recon::io
