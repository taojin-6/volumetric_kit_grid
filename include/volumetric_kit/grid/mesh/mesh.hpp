// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file mesh/mesh.hpp
/// @brief Host triangle-mesh containers: the per-vertex payload and the
///        interleaved vertex + index arrays the `mesh` tier produces.

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "volumetric_kit/recon/core/math/vector_types.hpp"

namespace volumetric_kit::recon::mesh {

/// @brief One mesh vertex: a world-space position and its surface normal.
///
/// This is also the on-device layout the marching-cubes kernel writes: under
/// scalar block layout a GLSL `struct { vec3 position; vec3 normal; }` is
/// byte-identical to this POD (position @0, normal @12, 24 B), so the host
/// reads the vertex buffer back with a plain `memcpy` -- the same
/// one-struct-across- CPU/GLSL discipline the volume tier applies to @ref
/// volume::HashEntry (the 2026-07-05 ABI). The `static_assert`s below pin the
/// host packing; the GLSL mirror keeps its `layout(scalar)` definition in
/// lockstep.
///
/// Scoped to what marching cubes emits today. The gfx ingestion vertex
/// (`position, normal, tangent, uv0, color`) is a wider shape; converting to it
/// -- synthesizing tangents, widening color -- is the interop-seam converter's
/// job in a later slice, not a field on this struct.
struct Vertex {
  Vec3f position;  ///< World-space position (metres).
  Vec3f normal;    ///< Unit surface normal (zero for degenerate geometry).
};
static_assert(sizeof(Vertex) == 24, "Vertex must be 24 bytes");
static_assert(offsetof(Vertex, position) == 0, "Vertex layout drift");
static_assert(offsetof(Vertex, normal) == 12, "Vertex layout drift");
// Uploaded/downloaded by value and mirrored by a GLSL scalar-layout block, so
// it must stay a trivially-copyable, standard-layout POD.
static_assert(std::is_trivially_copyable_v<Vertex>,
              "Vertex must be trivially copyable");
static_assert(std::is_standard_layout_v<Vertex>,
              "Vertex must be standard-layout");

/// @brief A triangle mesh: an interleaved @ref Vertex array plus 32-bit
/// triangle
///        indices (three per triangle), matching gfx's ingestion shape.
///
/// The first marching-cubes slice emits independent triangles -- three fresh
/// vertices per triangle, no shared-edge deduplication -- so @ref indices is
/// the trivial `0,1,2,...` run and `indices.size() == vertices.size()`. The
/// index array is carried regardless so a later edge-hash dedup slice can
/// shrink
/// @ref vertices without changing this container or its consumers.
struct Mesh {
  std::vector<Vertex> vertices;        ///< Interleaved vertices.
  std::vector<std::uint32_t> indices;  ///< Triangle indices (3 per triangle).

  /// @return `true` when the mesh has no triangles.
  bool empty() const noexcept { return indices.empty(); }
  /// @return The number of triangles (`indices.size() / 3`).
  std::size_t triangle_count() const noexcept { return indices.size() / 3; }
};

}  // namespace volumetric_kit::recon::mesh
