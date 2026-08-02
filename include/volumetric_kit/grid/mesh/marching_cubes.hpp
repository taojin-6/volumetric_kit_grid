// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file mesh/marching_cubes.hpp
/// @brief GPU marching-cubes iso-surface extraction: owns the compute pipeline
///        and drives a GLSL kernel that turns a dense SDF grid into a triangle
///        @ref Mesh.

#include <cstddef>
#include <cstdint>

#include "volumetric_kit/recon/core/buffer.hpp"
#include "volumetric_kit/recon/core/compute_kernel.hpp"
#include "volumetric_kit/recon/core/descriptor.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/mesh/export.hpp"
#include "volumetric_kit/recon/mesh/mesh.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

namespace volumetric_kit::recon {
class Device;
class Allocator;
}  // namespace volumetric_kit::recon

namespace volumetric_kit::recon::mesh {

/// @brief A dense grid of SDF samples -- the input to the analytic extraction
///        path this first slice proves.
///
/// Samples are @ref volume::Voxel (signed distance + integration weight),
/// stored x-fastest: sample `(x, y, z)` lives at linear index `x + dims.x * (y
/// + dims.y * z)`. The grid spans `dims` samples per axis, so marching cubes
/// walks `(dims - 1)` cells per axis, each cell's eight corners being adjacent
/// samples. Using @ref volume::Voxel (not a bare `float`) keeps this the same
/// payload the `tsdf` tier will fill on the sparse hash map -- the weight gates
/// out cells that touch an unintegrated voxel.
///
/// @note This dense entry point is the path-proving building block; the
///       @ref MarchingCubes::extract overload taking a @ref
///       volume::VoxelBlockGrid meshes the real sparse volume (with cross-block
///       neighbour sampling at block boundaries). The per-cell kernel is
///       identical -- only the corner-sampling differs.
struct DenseGrid {
  Vec3i dims{};             ///< Sample count per axis (cells = `dims - 1`).
  float voxel_size = 0.0f;  ///< Metres between adjacent samples.
  Vec3f origin{};           ///< World position of sample index `(0, 0, 0)`.
};

/// @brief Where one sparse @ref MarchingCubes::extract call spent its time, and
///        the sizes that explain it.
///
/// Opt-in and explicit: the caller passes one of these to @ref
/// MarchingCubes::extract to have it filled, and `nullptr` (the default)
/// measures nothing. The tier keeps no profiler, no global sink, and no
/// timing state between calls -- a caller that wants a running view (the
/// viewer's overlay) aggregates these itself.
///
/// The spans are **wall-clock**, and the GPU ones are end-to-end: the dispatch
/// goes through @ref Device::submit_single_time, which blocks on a fence, so
/// @ref dispatch_ms covers host record *plus* device execution rather than
/// either alone.
///
/// Meshing is currently whole-volume and worst-case-sized, so the counters
/// matter as much as the spans: @ref triangle_capacity is what this call sized
/// for (5 triangles per cell), typically orders of magnitude above what @ref
/// emitted_triangles fills, and @ref arena_bytes is what the extractor is
/// holding to serve it.
struct ExtractTimings {
  /// Compacting the hash map's active block list (a dispatch + readback).
  double compact_ms = 0.0;
  /// Building the host-side 2x2x2 neighbour lookup table.
  double neighbour_lut_ms = 0.0;
  /// Allocating + filling the active-block and neighbour input buffers.
  double input_upload_ms = 0.0;
  /// Sizing the worst-case vertex arena + zeroing the atomic counter. Near
  /// zero once the retained arena already fits the call -- the steady state,
  /// since the arena is reused across extracts (see @ref MarchingCubes).
  double arena_alloc_ms = 0.0;
  /// Writing the kernel's descriptor bindings.
  double descriptor_ms = 0.0;
  /// The marching-cubes dispatch, including the blocking fence wait.
  double dispatch_ms = 0.0;
  /// Reading the counter back and copying the vertices into the host mesh.
  double readback_ms = 0.0;

  /// Active blocks meshed -- the dispatch's real size (occupancy, not the
  /// map's capacity).
  std::uint32_t active_blocks = 0;
  /// Worst-case triangle capacity the arena was sized for.
  std::uint32_t triangle_capacity = 0;
  /// Triangles the kernel actually emitted.
  std::uint32_t emitted_triangles = 0;
  /// Bytes the extractor's vertex arena currently holds. This is *resident*
  /// size, not this call's allocation: the arena is retained across extracts,
  /// so a steady-state call allocates nothing and still reports it, and a
  /// grown arena can exceed what @ref triangle_capacity alone would need.
  std::uint64_t arena_bytes = 0;

  /// @return The sum of every phase, in milliseconds.
  double total_ms() const noexcept {
    return compact_ms + neighbour_lut_ms + input_upload_ms + arena_alloc_ms +
           descriptor_ms + dispatch_ms + readback_ms;
  }
};

/// @brief Owns the marching-cubes compute pipelines and extracts an iso-surface
///        into a host @ref Mesh -- from a dense @ref DenseGrid or straight off
///        a sparse @ref volume::VoxelBlockGrid.
///
/// Built on the `core` compute foundation (@ref Allocator, @ref Buffer,
/// @ref ComputeKernel, @ref Device::submit_single_time), mirroring the volume
/// tier's @ref volume::VoxelHashMap. The kernel runs one invocation per cell,
/// builds the cube index from the eight corner signs, interpolates a vertex on
/// each crossed edge, and appends independent triangles through an atomic bump
/// counter -- the simple, correct form; shared-edge dedup and an incremental
/// block-mesh pool are later slices. Normals come from the SDF gradient -- one
/// central difference over the cell's eight corners, shared by that cell's
/// vertices -- so they point outward (increasing distance). Each vertex also
/// carries the hybrid appearance the renderer consumes: a @ref Vertex::color
/// interpolated from the optional per-sample color input (opaque white when
/// absent), and a @ref Vertex::uv0 left at the `(-1, -1)` sentinel -- the
/// projective-texturing pass (a later slice) fills real atlas coordinates.
///
/// @note An extractor **retains its vertex arena between calls**, growing it
///       when a call needs more and never shrinking it. The arena is sized for
///       the worst case of 5 triangles per cell, so it can reach hundreds of
///       megabytes on a large volume and stays resident for this object's
///       lifetime -- reuse is what makes a steady-state extract pay nothing for
///       its output storage (it was ~90% of a sparse extract when allocated per
///       call), at the cost of holding the peak. Destroy the extractor to
///       release it; @ref ExtractTimings::arena_bytes reports what it holds.
///
/// @warning The @ref Device and @ref Allocator passed to @ref create must
///          outlive this object; it stores references to them.
class VR_MESH_API MarchingCubes {
 public:
  /// @brief Create the extractor on @p device, building its pipeline and
  ///        binding it to @p allocator for its input, vertex-arena, and
  ///        counter buffers.
  /// @param device     The compute device (must outlive this object).
  /// @param allocator  The allocator its buffers come from (must outlive this).
  /// @return The extractor, or a non-OK @ref Status if a pipeline, layout, or
  ///         descriptor allocation fails.
  static Result<MarchingCubes> create(Device& device, Allocator& allocator);

  // Rule of zero: every owned member (Buffer / ComputeKernel / pool) self-frees
  // and self-resets on move, so the defaulted moves are correct. Nothing here
  // caches a *copy* of an owned member's state -- the arena's capacity is
  // derived from vertex_arena_ (see arena_capacity) rather than tracked
  // alongside it, so a defaulted move cannot leave the two disagreeing.
  // device_ / allocator_ are borrowed pointers, so a defaulted move leaving the
  // moved-from extractor pointing at them is harmless -- it reports valid() ==
  // false and is only destroyed.
  ~MarchingCubes() = default;
  MarchingCubes(MarchingCubes&&) noexcept = default;
  MarchingCubes& operator=(MarchingCubes&&) noexcept = default;
  MarchingCubes(const MarchingCubes&) = delete;
  MarchingCubes& operator=(const MarchingCubes&) = delete;

  /// @brief Extract the @p iso iso-surface from @p samples over @p grid.
  /// @param samples  The dense SDF grid, x-fastest (see @ref DenseGrid).
  /// @param count    Number of samples; must equal `dims.x * dims.y * dims.z`.
  /// @param grid     The grid dimensions, spacing, and world origin.
  /// @param iso      The iso-value to extract (0 for a raw signed-distance
  ///                 field).
  /// @param colors   Optional per-sample RGB, parallel to @p samples (same
  /// count
  ///                 and x-fastest layout) -- the color the `tsdf` tier fuses
  ///                 into the volume. When non-null, each vertex's
  ///                 @ref Vertex::color is interpolated from it at the edge
  ///                 crossing; when null, vertices are opaque white.
  ///                 @ref Vertex::uv0 is always the `(-1, -1)` sentinel
  ///                 (projective texturing fills it in a later slice).
  /// @return The extracted mesh (empty when the surface misses the grid), or a
  ///         non-OK @ref Status: @ref Status::Code::InvalidArgument for a
  ///         moved-from extractor, a null/mis-sized sample array, or a grid
  ///         that is not at least `2x2x2` samples; a backend error if a buffer
  ///         or the dispatch fails.
  Result<Mesh> extract(const volume::Voxel* samples, std::size_t count,
                       const DenseGrid& grid, float iso = 0.0f,
                       const Vec3u8* colors = nullptr);

  /// @brief Extract the @p iso iso-surface straight off a sparse
  ///        @ref volume::VoxelBlockGrid, meshing every active block.
  ///
  /// Runs one invocation per voxel of each active block (the block iteration
  /// the `tsdf` integrator uses), each voxel the base corner of one
  /// marching-cubes cell. A cell on a block's `+face` reaches its far corners
  /// into neighbouring blocks; those samples are resolved through a host-built
  /// 2x2x2 neighbour table (this block plus its seven `+x/+y/+z` neighbours,
  /// built from the compacted active set), so the kernel needs no device-side
  /// hash probe. The per-cell body is identical to the dense @ref extract --
  /// independent triangles, one gradient normal per cell, reversed winding, and
  /// the same hybrid @ref Vertex::color / @ref Vertex::uv0 appearance.
  ///
  /// @param grid  A grid carrying `float` `tsdf` + `weight` attributes (see
  /// @ref
  ///              volume::VoxelBlockGrid::create); a voxel whose weight is at
  ///              or below the unintegrated threshold drops any cell that
  ///              touches it. When the grid also carries a `uint32` packed-RGB
  ///              `color` attribute, each vertex's @ref Vertex::color is
  ///              interpolated from it; otherwise vertices are opaque white.
  ///              @ref Vertex::uv0 is always the `(-1, -1)` sentinel
  ///              (projective texturing fills it in a later slice).
  /// @param iso   The iso-value to extract (0 for a raw signed-distance field).
  /// @param timings  Optional; when non-null, receives this call's per-phase
  ///                 wall-clock breakdown and size counters (see @ref
  ///                 ExtractTimings). `nullptr` measures nothing.
  /// @return The extracted mesh (empty when no active block holds a surface),
  /// or
  ///         a non-OK @ref Status: @ref Status::Code::InvalidArgument for a
  ///         moved-from extractor, a moved-from @p grid, or a grid missing a
  ///         `float` `tsdf`/`weight` attribute, or if the active set is too
  ///         large for a single 1-D dispatch; a backend error if a buffer or
  ///         the dispatch fails.
  Result<Mesh> extract(volume::VoxelBlockGrid& grid, float iso = 0.0f,
                       ExtractTimings* timings = nullptr);

  /// @return `true` if this owns a live kernel (`false` when moved-from).
  bool valid() const noexcept { return kernel_.valid(); }

 private:
  MarchingCubes() = default;

  // Borrowed (must outlive this). Pointers, not references, so a moved-from
  // extractor is left in a defined (empty) state.
  Device* device_ = nullptr;
  Allocator* allocator_ = nullptr;

  // Cached maxComputeWorkGroupCount[0]: the ceiling on a 1-D dispatch's
  // groupCountX (Vulkan guarantees only >= 65535), so extract() can reject an
  // over-large grid cleanly instead of risking a device-lost.
  std::uint32_t max_workgroup_count_x_ = 0;

  // Cached maxStorageBufferRange: the ceiling on a storage-buffer binding, so
  // extract() can reject a worst-case vertex arena that would exceed it with a
  // clean Status instead of an opaque allocation failure.
  std::uint32_t max_storage_buffer_range_ = 0;

  // The marching-cubes lookup tables, uploaded once and bound at set binding 0
  // of both kernels for every extract (the counterpart to the volume tier's
  // persistent bindings). The input buffers are per-extract; the vertex arena
  // and counter are retained (below). All of them are (re)written into the
  // remaining bindings before a dispatch, so a regrown arena's new handle is
  // always the one bound.
  Buffer tables_;
  // A 1-element dummy bound to the sparse kernel's color slot when a grid
  // carries no `color` attribute, so that descriptor stays valid (the has_color
  // push flag tells the kernel to ignore it). Mirrors the tsdf integrator's
  // color dummy.
  Buffer color_dummy_;

  // The vertex arena + atomic triangle counter the kernels write, kept ACROSS
  // extract calls and grown only when a call needs more than the last one.
  //
  // These were allocated per call, which measured as ~90% of a sparse extract
  // (~50 ms of a 55 ms call on Replica room0): the arena is sized for the
  // worst case of 5 triangles per cell, so it runs to hundreds of megabytes,
  // and creating it every frame makes the driver fault in and zero that many
  // fresh pages while the dispatch that fills it costs ~2 ms. Reusing one
  // allocation makes a steady-state extract pay nothing for its output
  // storage. Only the counter is reset per call (4 bytes); the arena's stale
  // contents past the emitted range are never read, since the counter bounds
  // the readback.
  Buffer vertex_arena_;
  Buffer counter_;

  // The two marching-cubes kernels -- each its descriptor-set layout, pipeline,
  // and a set allocated from the shared pool_ (see @ref ComputeKernel): the
  // dense analytic-grid path and the sparse VoxelBlockGrid path.
  ComputeKernel kernel_;
  ComputeKernel kernel_sparse_;
  DescriptorPool pool_;

  // Triangle capacity @ref vertex_arena_ is currently sized for (0 when it
  // holds no buffer, including after a move -- Buffer zeroes its size). Derived
  // rather than stored so the capacity can never disagree with the buffer it
  // describes; the division is exact because the arena is only ever allocated
  // as a whole number of triangles.
  std::uint32_t arena_capacity() const noexcept {
    return static_cast<std::uint32_t>(vertex_arena_.size() /
                                      (3 * sizeof(Vertex)));
  }

  // Size @ref vertex_arena_ / @ref counter_ for a dispatch emitting at most
  // @p capacity triangles, reallocating only when the current arena is too
  // small, and zero the counter. Returns a non-OK Status when @p capacity's
  // arena would exceed the device's maxStorageBufferRange.
  Status ensure_output_buffers(std::uint32_t capacity);
};

}  // namespace volumetric_kit::recon::mesh
