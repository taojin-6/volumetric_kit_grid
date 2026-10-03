// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file io/mesh_io.hpp
/// @brief Static triangle geometry imported through the optional Assimp
/// backend.

#include <cstdint>
#include <string>
#include <vector>

#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/io/assimp_export.hpp"

namespace volumetric_kit::recon::io {

/// @brief Owned, indexed triangle geometry in the imported scene's coordinates.
///
/// Positions do not implicitly carry metre units. Asset formats and importers
/// may define their own unit or axis conventions; the caller chooses any
/// further conversion. Normals, UVs, materials and textures are not retained.
struct TriangleMesh {
  std::vector<Vec3f> positions;  ///< Finite positions after node transforms.
  std::vector<std::uint32_t> indices;  ///< Three in-range indices per triangle.
};

/// @brief Load a static asset as one owned triangle mesh.
///
/// Assimp selects the file importer and triangulates polygon faces. All node
/// instances are expanded with their composed transforms; a reflection
/// reverses triangle indices to preserve the source's orientation. Exactly
/// coincident positions are shared across the result (including UV/material
/// seams); nearby positions remain distinct. Duplicate or degenerate triangles
/// are preserved, so successful import does not certify a closed manifold.
///
/// This is host asset I/O: it creates no Vulkan resources or GPU submissions.
/// It does not choose a physical scale, normalize height, repair topology or
/// evaluate animation. Importer unit and up-axis conversions are disabled
/// where the installed Assimp allows it: Assimp before 5.3 still scales
/// Collada by its `<unit>`, and Assimp 5.4.x converts FBX axes and units. The
/// supported formats are those in the installed Assimp library. Link @c
/// volumetric_kit::recon_io_assimp, enabled with
/// @c VR_WITH_ASSIMP=ON; Assimp types and ownership stay private.
///
/// @param path Asset path without embedded NUL; sidecar files resolve relative
///             to this file.
/// @return Owned geometry on success; InvalidArgument for an empty path,
///         malformed geometry or an invalid transform on a mesh instance;
///         NotFound for a missing file; Unsupported for animation, skinning,
///         morphs or non-triangle primitives; IoError for import/read failures;
///         OutOfMemory for allocation failure or geometry exceeding 32-bit
///         indexing.
VR_IO_ASSIMP_API Result<TriangleMesh> load_mesh(const std::string& path);

}  // namespace volumetric_kit::recon::io
