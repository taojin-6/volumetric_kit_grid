// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file io/mesh_io.hpp
/// @brief Static triangle geometry imported through the optional Assimp
/// backend.

#include <string>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/io/assimp_export.hpp"
#include "volumetric_kit/recon/io/triangle_mesh.hpp"

namespace volumetric_kit::recon::io {

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
/// @return Owned geometry on success, its positions finite and after node
///         transforms; InvalidArgument for an empty path, malformed geometry
///         or an invalid transform on a mesh instance; NotFound for a missing
///         file; Unsupported for animation, skinning, morphs or non-triangle
///         primitives; IoError for import/read failures; OutOfMemory for
///         allocation failure or geometry exceeding 32-bit indexing.
VR_IO_ASSIMP_API core::Result<TriangleMesh> load_mesh(const std::string& path);

}  // namespace volumetric_kit::recon::io
