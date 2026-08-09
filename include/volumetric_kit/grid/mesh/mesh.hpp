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

/// @brief One mesh vertex: world-space position + normal, and the hybrid
///        appearance the reconstruction handoff carries -- a per-vertex color
///        and an atlas texture coordinate.
///
/// This is also the on-device layout the marching-cubes kernel writes: under
/// scalar block layout a GLSL
/// `struct { vec3 position; vec3 normal; vec4 tangent; vec2 uv0; vec4 color; }`
/// is byte-identical to this POD (position @0, normal @12, tangent @24,
/// uv0 @40, color @48, 64 B), so the host reads the vertex buffer back with a
/// plain `memcpy` -- the same one-struct-across-CPU/GLSL discipline the volume
/// tier applies to @ref volume::HashEntry (the 2026-07-05 ABI). The
/// `static_assert`s below pin the host packing; the GLSL mirror keeps its
/// `layout(scalar)` definition in lockstep.
///
/// @ref color and @ref uv0 encode the **hybrid** color path shared with the
/// renderer's `HybridMeshPipeline`: where projective texturing won a triangle a
/// camera, @ref uv0 holds an atlas coordinate; where it did not, @ref uv0 is
/// the
/// `(-1, -1)` sentinel and the shader falls back to @ref color (the color TSDF
/// integration fused into the volume, interpolated here). Marching cubes fills
/// @ref color and leaves @ref uv0 at the sentinel; the projective-texturing
/// pass (a later slice) overwrites @ref uv0 for textured triangles. @ref
/// tangent is here only to hold the renderer's layout in place: meshing cannot
/// derive one, so the kernel writes a fixed placeholder into it.
struct Vertex {
  Vec3f position;  ///< World-space position (metres).
  Vec3f normal;    ///< Unit surface normal (an arbitrary unit axis where the
                   ///< SDF gradient vanishes; never the zero vector).
  Vec4f tangent;   ///< xyz tangent, w handedness. Carried for the renderer's
                   ///< vertex layout, not produced by meshing: marching cubes
                   ///< has no surface parameterisation to derive one from, so
                   ///< every vertex gets the same placeholder the host
                   ///< converter used to synthesize. See the layout note above.
  Vec2f uv0;       ///< Atlas texture coordinate, or the `(-1, -1)` sentinel
                   ///< meaning "no atlas; use @ref color".
  Vec4f color;     ///< Per-vertex RGBA in **linear** working values (linear
                   ///< BT.709/D65), opaque (alpha 1) -- the vertex-color
                   ///< fallback used where @ref uv0 is the sentinel. Linear
                   ///< because a float color is linear (`core/color_space.hpp`)
                   ///< and because that is what glTF specifies for `COLOR_0`
                   ///< and what the renderer's shading multiply needs, so the
                   ///< seam converts nothing. Marching cubes decodes the
                   ///< canonical-encoded voxel attribute at the corner gather,
                   ///< since interpolating along the cell edge is an average.
                   ///< An 8-bit *export* must re-encode (PLY does; glTF does
                   ///< not).
};
// Byte-for-byte `volumetric_kit::gfx::assets::Vertex`. These offsets are the
// interop-seam contract, not an internal detail: the renderer's vertex-input
// description reads position/normal/uv0/color at exactly these offsets with
// this stride, so a mesh the kernel wrote crosses the seam unconverted.
// Changing one without the other silently misreads every vertex.
// The buffers can also be made *bindable*: a consumer passes the usage it needs
// through mesh/marching_cubes.hpp's MarchingCubesConfig, which is where the
// remaining seam-B gaps are enumerated -- being byte-compatible and bindable is
// still short of being drawable in place.
// TODO(mesh): they remain *host-visible*, which is right on a unified-memory
// GPU (Apple silicon reports a host-visible DEVICE_LOCAL heap, so there is
// nothing to stage) but not on a discrete one, where drawing vertices across
// PCIe every frame is the slow path. Give them a device-local home with a
// staging copy when a discrete-GPU consumer of seam B exists to measure it --
// doing it now would add a copy to the one platform that does not need one.
static_assert(sizeof(Vertex) == 64, "Vertex must be 64 bytes");
static_assert(offsetof(Vertex, position) == 0, "Vertex layout drift");
static_assert(offsetof(Vertex, normal) == 12, "Vertex layout drift");
static_assert(offsetof(Vertex, tangent) == 24, "Vertex layout drift");
static_assert(offsetof(Vertex, uv0) == 40, "Vertex layout drift");
static_assert(offsetof(Vertex, color) == 48, "Vertex layout drift");
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
/// `indices.size()` is always three per triangle. Whether it equals
/// `vertices.size()` is **not** a property of this container: marching cubes
/// emits independent triangles by default -- three fresh vertices each, so
/// @ref indices is the trivial `0,1,2,...` run -- but
/// @ref MarchingCubesConfig::share_vertices lets several triangles index one
/// vertex, which is exactly the case the index array was carried for. Walk the
/// triangles through @ref indices, never as consecutive vertex triples.
struct Mesh {
  std::vector<Vertex> vertices;        ///< Interleaved vertices.
  std::vector<std::uint32_t> indices;  ///< Triangle indices (3 per triangle).

  /// @return `true` when the mesh has no triangles.
  bool empty() const noexcept { return indices.empty(); }
  /// @return The number of triangles (`indices.size() / 3`).
  std::size_t triangle_count() const noexcept { return indices.size() / 3; }
};

}  // namespace volumetric_kit::recon::mesh
