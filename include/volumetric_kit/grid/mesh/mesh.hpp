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
#include "volumetric_kit/recon/core/vulkan.hpp"

namespace volumetric_kit::recon::mesh {

/// @brief One mesh vertex: world-space position + normal, and the hybrid
///        appearance the reconstruction handoff carries -- a per-vertex color
///        and an atlas texture coordinate.
///
/// This is also the on-device layout the marching-cubes kernel writes: under
/// scalar block layout a GLSL
/// `struct { vec3 position; vec3 normal; vec4 color; vec2 uv0; }` is
/// byte-identical to this POD (position @0, normal @12, color @24, uv0 @40,
/// 48 B), so the host reads the vertex buffer back with a plain `memcpy` -- the
/// same one-struct-across-CPU/GLSL discipline the volume tier applies to @ref
/// volume::HashEntry (the 2026-07-05 ABI). The `static_assert`s below pin the
/// host packing; the GLSL mirror keeps its `layout(scalar)` definition in
/// lockstep.
///
/// @ref color and @ref uv0 encode the **hybrid** color path shared with the
/// renderer's `HybridMeshPipeline`: where projective texturing won a triangle a
/// camera, @ref uv0 holds an atlas coordinate; where it did not, @ref uv0 is
/// the
/// `(-1, -1)` sentinel and the shader falls back to @ref color (the color TSDF
/// integration fused into the volume, interpolated here). Marching cubes fills
/// @ref color and leaves @ref uv0 at the sentinel; the projective-texturing
/// pass (a later slice) overwrites @ref uv0 for textured triangles. The gfx
/// ingestion vertex additionally carries a `tangent`; synthesizing it is the
/// interop-seam converter's job, not a field on this struct.
struct Vertex {
  Vec3f position;  ///< World-space position (metres).
  Vec3f normal;    ///< Unit surface normal (an arbitrary unit axis where the
                   ///< SDF gradient vanishes; never the zero vector).
  Vec4f color;     ///< Per-vertex RGBA, opaque (alpha 1) -- the vertex-color
                   ///< fallback used where @ref uv0 is the sentinel.
  Vec2f uv0;       ///< Atlas texture coordinate, or the `(-1, -1)` sentinel
                   ///< meaning "no atlas; use @ref color".
};
static_assert(sizeof(Vertex) == 48, "Vertex must be 48 bytes");
static_assert(offsetof(Vertex, position) == 0, "Vertex layout drift");
static_assert(offsetof(Vertex, normal) == 12, "Vertex layout drift");
static_assert(offsetof(Vertex, color) == 24, "Vertex layout drift");
static_assert(offsetof(Vertex, uv0) == 40, "Vertex layout drift");
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

/// @brief The same mesh, still in the device buffers that produced it -- the
///        handoff between two GPU passes with no host copy in between.
///
/// The `mesh` tier writes its geometry into a `VkBuffer` and the `texture` tier
/// rewrites @ref Vertex::uv0 in place, so routing one to the other through a
/// host @ref Mesh costs a full readback *and* a full re-upload of the same
/// bytes (~45 MB each at a ~940 k-vertex room scan) for no reason. A
/// @ref DeviceMesh names those buffers instead, and the host copy happens once,
/// when a caller actually needs a @ref Mesh.
///
/// **Borrowed, not owned.** The buffers belong to the producing extractor,
/// which reuses them across calls, so a @ref DeviceMesh is valid only until the
/// next extract on that same extractor -- exactly like a pointer into a
/// container that the next insert may invalidate. Copying the struct copies the
/// handles, not the storage.
///
/// The vertices are independent triangles (no shared vertices), so
/// @ref indices is the identity run `0, 1, 2, ...`; it exists because the
/// consuming kernels address vertices through it, and because the renderer
/// wants a real index buffer at the interop seam.
struct DeviceMesh {
  VkBuffer vertices = VK_NULL_HANDLE;  ///< Interleaved @ref Vertex array.
  VkBuffer indices = VK_NULL_HANDLE;   ///< `uint32` indices, 3 per triangle.
  std::uint32_t vertex_count = 0;      ///< Live vertices (`3 * triangles`).
  std::uint32_t triangle_count = 0;    ///< Live triangles.

  /// @return `true` when the mesh has no triangles.
  bool empty() const noexcept { return triangle_count == 0; }
  /// @return `true` when both buffers are present.
  bool valid() const noexcept {
    return vertices != VK_NULL_HANDLE && indices != VK_NULL_HANDLE;
  }
};

}  // namespace volumetric_kit::recon::mesh
