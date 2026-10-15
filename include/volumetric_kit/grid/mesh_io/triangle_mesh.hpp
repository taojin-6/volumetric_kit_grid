// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file io/triangle_mesh.hpp
/// @brief Owned, indexed triangle geometry: what @ref load_mesh returns and
///        the normalization helpers take.

#include <cstdint>
#include <vector>

#include "volumetric_kit/recon/core/math/vector_types.hpp"

namespace volumetric_kit::recon::io {

/// @brief Owned, indexed triangle geometry in its source's coordinates.
///
/// Positions do not implicitly carry metre units. Asset formats and importers
/// may define their own unit or axis conventions; the caller chooses any
/// further conversion, such as the opt-in helpers of
/// `io/mesh_normalization.hpp`. Normals, UVs, materials and textures are not
/// retained.
struct TriangleMesh {
  std::vector<Vec3f> positions;  ///< Vertex positions, source coordinates.
  std::vector<std::uint32_t> indices;  ///< Three in-range indices per triangle.
};

}  // namespace volumetric_kit::recon::io
