// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file mesh/device_mesh.hpp
/// @brief The device-resident view of a mesh: the `VkBuffer`s a GPU pass wrote
///        it into, handed to the next GPU pass without a host copy.
///
/// Kept out of `mesh/mesh.hpp` so that header stays host-only -- an exporter or
/// any other pure-CPU consumer of @ref Mesh should not have to see Vulkan.

#include <cstdint>

#include "volumetric_kit/recon/core/vulkan.hpp"

namespace volumetric_kit::recon::mesh {

/// @brief A mesh still in the device buffers that produced it -- the handoff
///        between two GPU passes with no host copy in between.
///
/// The `mesh` tier writes its geometry into a `VkBuffer` and the `texture` tier
/// rewrites `Vertex::uv0` in place, so routing one to the other through a host
/// `Mesh` costs a full readback *and* a full re-upload of the same bytes
/// (~45 MB each at a ~940 k-vertex room scan) for no reason. A @ref DeviceMesh
/// names those buffers instead, and the host copy happens once, when a caller
/// actually needs a `Mesh`.
///
/// **Borrowed, not owned.** The buffers belong to the producing extractor,
/// which reuses them across calls, so a @ref DeviceMesh is valid only until the
/// next extract on that same extractor -- exactly like a pointer into a
/// container that the next insert may invalidate. Copying the struct copies the
/// handles, not the storage.
///
/// @ref generation is what makes that rule *enforceable*. Comparing buffer
/// handles cannot detect the common case: the producer reuses one grow-only
/// arena, so a superseded view names the very same `VkBuffer` as the live one
/// and would compare equal while the contents had been replaced. The producer
/// therefore stamps a counter it bumps on every extract, and
/// `MarchingCubes::download` accepts only its current value.
///
/// The index buffer exists because the consuming kernels address vertices
/// through it, and because the renderer wants a real one at the interop seam.
/// Whether it is the identity run `0, 1, 2, ...` depends on @ref
/// shares_vertices -- do not assume either way.
struct DeviceMesh {
  VkBuffer vertices = VK_NULL_HANDLE;  ///< Interleaved `Vertex` array.
  VkBuffer indices = VK_NULL_HANDLE;   ///< `uint32` indices, 3 per triangle.
  /// A single `VkDrawIndexedIndirectCommand` describing this mesh's draw, for
  /// `vkCmdDrawIndexedIndirect`.
  ///
  /// The producer's kernel counts *indices* into `indexCount`, so the command
  /// is written by the extraction itself rather than assembled afterwards from
  /// a triangle count -- which is what lets a consumer draw without the count
  /// ever passing through it. @ref triangle_count and @ref vertex_count say the
  /// same thing for a consumer that wants to know; the command exists so one
  /// does not have to.
  VkBuffer indirect = VK_NULL_HANDLE;
  /// Live vertices in @ref vertices. `3 * triangle_count` when @ref
  /// shares_vertices is false and roughly a quarter of that when it is true --
  /// read it, do not derive it.
  std::uint32_t vertex_count = 0;
  std::uint32_t triangle_count = 0;  ///< Live triangles.
  /// Whether several triangles may index the same vertex.
  ///
  /// False (the default and what this tier always did) means every triangle
  /// owns three private vertices written consecutively, so @ref indices is the
  /// identity run `0, 1, 2, ...` and @ref vertex_count is `3 * triangle_count`.
  /// True is @ref MarchingCubesConfig::share_vertices: neither holds.
  ///
  /// Published because a *consumer* has to act on it and cannot otherwise ask.
  /// `texture::ProjectiveTexturer` decides visibility per triangle and writes
  /// `Vertex::uv0` per vertex, which is well-defined only while a vertex
  /// belongs to one triangle -- so it refuses a mesh carrying this rather than
  /// letting the write order decide the result. Same reason @ref vertex_usage
  /// and @ref sharing_mode are here: the producer's configuration is not
  /// visible from a `VkBuffer`, and an obligation stated in prose is not a
  /// contract.
  bool shares_vertices = false;
  /// Usage flags @ref vertices was created with -- always `STORAGE_BUFFER`,
  /// plus whatever the producer's consumer asked for. Carried so a consumer can
  /// *check* that the binding it is about to make is permitted, rather than
  /// assuming the flags it published reached the producer: Vulkan cannot be
  /// asked what a `VkBuffer` was created with, and binding one that lacks the
  /// bit is a validation-layer-only diagnostic -- undefined behaviour with
  /// layers off, which is the shipping configuration.
  VkBufferUsageFlags vertex_usage = 0;
  /// Usage flags @ref indices was created with; see @ref vertex_usage.
  VkBufferUsageFlags index_usage = 0;
  /// Usage flags @ref indirect was created with; see @ref vertex_usage. Always
  /// carries `INDIRECT_BUFFER` beside the `STORAGE_BUFFER` the kernel counts
  /// through.
  VkBufferUsageFlags indirect_usage = 0;
  /// Sharing mode all three buffers were created with (one producer config
  /// covers them, so they never differ).
  ///
  /// Published for the same reason as @ref vertex_usage, with more at stake:
  /// reading a `VK_SHARING_MODE_EXCLUSIVE` buffer from a queue family that does
  /// not own it is *undefined*, where a missing usage bit is at least a
  /// validation diagnostic. A consumer on a second family must see
  /// `VK_SHARING_MODE_CONCURRENT` here before it binds any of them -- and on
  /// Apple, where Metal has no ownership concept, getting it wrong is undefined
  /// in the way that appears to work.
  ///
  /// @note What is deliberately *not* published is the memory placement: all
  ///       three buffers are host-visible. That is free on a unified-memory GPU
  ///       and costs a PCIe fetch per indirect draw on a discrete one -- and,
  ///       for a consumer binding @ref vertices and @ref indices as geometry, a
  ///       whole-mesh fetch out of system RAM per drawn frame. A recorded trade
  ///       rather than a hidden one; see the `TODO(mesh)` on
  ///       @ref MarchingCubesConfig.
  VkSharingMode sharing_mode = VK_SHARING_MODE_EXCLUSIVE;
  /// Which extract on the producing object this view came from. Producers
  /// number their extracts from 1, so the default 0 never matches a real one.
  std::uint64_t generation = 0;
  /// Points at the producer's live extract counter, so @ref is_current can
  /// compare @ref generation against it. Null on a default-constructed view.
  ///
  /// A *pointer* rather than a copy is the whole point: a copied number is the
  /// snapshot @ref generation already is, and says nothing about whether the
  /// producer has moved on. Non-owning, and it points into the producing
  /// object, so it borrows the producer's lifetime and address exactly as the
  /// buffer handles borrow its storage -- a view must not outlive its producer,
  /// and must not be held across a move of it.
  const std::uint64_t* live_generation = nullptr;

  /// @return `true` when this view is the producer's *newest* extract.
  ///
  /// **This is the single-slot validity rule, not the ring's.** At
  /// `MarchingCubesConfig::slot_count == 1` the two coincide and this is the
  /// check every consumer needs before it *uses* these handles, not only
  /// before it copies them out: one arena is reused in place, so a later
  /// extract has overwritten it -- or, if it grew, destroyed the old
  /// `VkBuffer` synchronously, making a bind a use-after-free of a Vulkan
  /// object, undefined with validation layers off (the shipping configuration,
  /// and the only one available on iOS).
  ///
  /// Above one slot, "newest" and "still mine" come apart, and this predicate
  /// answers the first. A ring consumer *expects* to hold older generations --
  /// that is what the extra slots buy -- and its validity condition is instead
  /// "not yet released": a view stays good until the consumer itself passes
  /// its generation to @ref MarchingCubes::release_through. That condition is
  /// the consumer's own declaration, so the producer cannot check it and this
  /// predicate must not be read as it; a ring consumer that gated on
  /// `is_current()` would drop every frame its producer had already run ahead
  /// of. `fuse_viewer` is the worked example.
  ///
  /// @warning Not synchronized. It dereferences the producer's live counter,
  ///          which the producing thread increments on every extract, so
  ///          calling it while another thread may be extracting is a data
  ///          race. Producer-thread use only -- which is where the tier's own
  ///          callers (@ref MarchingCubes::download,
  ///          @ref ProjectiveTexturer::texture) sit.
  ///
  /// Comparing buffer handles cannot substitute for this. The producer reuses
  /// one grow-only arena per slot, so a superseded view names the very same
  /// `VkBuffer` as the live one and compares equal while the contents have been
  /// replaced.
  bool is_current() const noexcept {
    return live_generation != nullptr && *live_generation == generation;
  }

  /// @return `true` when the mesh has no triangles.
  bool empty() const noexcept { return triangle_count == 0; }
  /// @return `true` when *every* buffer this view names is present.
  ///
  /// All three, not just the geometry: a consumer drawing indirectly reads
  /// @ref indirect and never consults @ref triangle_count, so a predicate that
  /// exempted it would wave through a null handle to
  /// `vkCmdDrawIndexedIndirect`. An extract that meshed nothing still names its
  /// command (zeroed), so `valid() && empty()` is the "draw nothing" case
  /// rather than a malformed one.
  bool valid() const noexcept {
    return vertices != VK_NULL_HANDLE && indices != VK_NULL_HANDLE &&
           indirect != VK_NULL_HANDLE;
  }
};

}  // namespace volumetric_kit::recon::mesh
