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
#include "volumetric_kit/recon/mesh/device_mesh.hpp"
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
/// viewer's overlay) aggregates these itself. Every field is **overwritten**
/// on each call, so one instance may be reused across frames without carrying
/// a previous call's numbers forward.
///
/// The spans are **wall-clock**, and the GPU ones are end-to-end: the dispatch
/// goes through @ref Device::submit_single_time, which blocks on a fence, so
/// @ref dispatch_ms covers host record *plus* device execution rather than
/// either alone.
///
/// Meshing is whole-volume, and the arena is fitted to the surface rather than
/// to the 5-triangles-per-cell ceiling, so the counters explain the spans:
/// @ref triangle_capacity is what the dispatch ran with (the arena's full
/// capacity, so `emitted_triangles / triangle_capacity` is its fill ratio),
/// @ref dispatches says whether the call had to refit and re-run, and
/// @ref arena_bytes is what the extractor is holding.
struct ExtractTimings {
  /// Compacting the hash map's active block list (a dispatch + readback).
  double compact_ms = 0.0;
  /// Building the host-side 2x2x2 neighbour lookup table.
  double neighbour_lut_ms = 0.0;
  /// Allocating + filling the active-block and neighbour input buffers.
  double input_upload_ms = 0.0;
  /// Sizing the vertex arena + zeroing the atomic counter, including a refit
  /// after an undersized guess (see @ref dispatches). Near zero once the
  /// retained arena already fits the call -- the steady state, since the arena
  /// is reused across extracts (see @ref MarchingCubes).
  double arena_alloc_ms = 0.0;
  /// Writing the kernel's descriptor bindings.
  double descriptor_ms = 0.0;
  /// The marching-cubes dispatch(es), including the blocking fence wait --
  /// summed over both when a refit forced a second one (@ref dispatches).
  double dispatch_ms = 0.0;
  /// Getting the result back to the caller: the 4-byte counter read after each
  /// dispatch, plus the vertex copy into the host mesh when one is made. @ref
  /// MarchingCubes::extract makes one, so this covers both; @ref
  /// MarchingCubes::extract_device does not, so there it is the counter alone.
  double readback_ms = 0.0;

  /// Active blocks meshed -- the dispatch's real size (occupancy, not the
  /// map's capacity).
  std::uint32_t active_blocks = 0;
  /// Marching-cubes dispatches this call ran: 1 in the steady state, 2 when
  /// the planned capacity was under what the field emitted, so the arena was
  /// refitted to the measured count and the surface re-run. A run that keeps
  /// reporting 2 means the planner is not tracking the surface -- which is
  /// invisible in @ref dispatch_ms alone, since that sums both.
  std::uint32_t dispatches = 0;
  /// Triangle capacity the dispatch ran with -- the retained arena's full
  /// capacity, so `emitted_triangles / triangle_capacity` is its fill ratio.
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

/// @brief Extra buffer usage the mesh's *consumer* requires.
///
/// The kernel itself needs only `STORAGE_BUFFER`. A consumer that wants to read
/// the arena in place rather than be handed a host copy — a renderer binding it
/// as vertex + index buffers is the motivating case — needs its own usage bits
/// on the same allocation, and only it knows which.
///
/// So this tier does not name them. The **consumer publishes** what it requires
/// and the application passes it in, exactly as the create/adopt device seam
/// works: each library states its needs, neither is compiled against the other,
/// and the app satisfies the union. Hardcoding a sibling's flags here would be
/// this tier guessing at an API it cannot see.
///
/// @code
/// // The app, which knows both:
/// const auto want = vg::pipelines::HybridMeshPipeline::mesh_requirements();
/// mesh::MarchingCubesConfig config;
/// config.extra_vertex_usage = want.vertex_usage;
/// config.extra_index_usage = want.index_usage;
/// @endcode
///
/// @note Usage bits are declarative — they permit a binding and cost nothing to
///       carry — so asking for more than you use is harmless. Memory
///       *residency* is a different question and not one of these: the buffers
///       are host-visible, which is free on a unified-memory GPU and a PCIe
///       round trip on a discrete one (see the `TODO(mesh)` in mesh.hpp).
struct MarchingCubesConfig {
  /// Added to the vertex arena's usage, beyond `STORAGE_BUFFER`.
  VkBufferUsageFlags extra_vertex_usage = 0;
  /// Added to the index run's usage, beyond `STORAGE_BUFFER`.
  VkBufferUsageFlags extra_index_usage = 0;
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
/// block-mesh pool are later slices (the TODOs below).
/// Normals come from the SDF gradient -- one
/// central difference over the cell's eight corners, shared by that cell's
/// vertices -- so they point outward (increasing distance). Each vertex also
/// carries the hybrid appearance the renderer consumes: a @ref Vertex::color
/// interpolated from the optional per-sample color input (opaque white when
/// absent), and a @ref Vertex::uv0 left at the `(-1, -1)` sentinel -- the
/// projective-texturing pass (a later slice) fills real atlas coordinates.
///
/// @note An extractor **retains its vertex arena between calls**, growing it
///       when a call needs more and never shrinking it -- reuse is what makes a
///       steady-state extract pay nothing for its output storage (it was ~90%
///       of a sparse extract when allocated per call), at the cost of holding
///       the peak for this object's lifetime. The sparse @ref extract fits the
///       arena to what the surface actually emits rather than to the
///       5-triangles-per-cell ceiling it can never reach, so what stays
///       resident is the mesh's real size plus headroom; the dense @ref extract
///       still sizes for its (caller-bounded) worst case and grows the shared
///       arena to it. Destroy the extractor to release it; @ref
///       ExtractTimings::arena_bytes reports what it holds.
///
/// @warning The @ref Device and @ref Allocator passed to @ref create must
///          outlive this object; it stores references to them.
//
// TODO(mesh): shared-edge vertex dedup, so the index buffer stops being the
// identity run and the arena shrinks toward the unique-vertex count.
// TODO(mesh): an incremental block-mesh pool, re-meshing only the blocks the
// integrator touched instead of the whole volume each call. Deferred
// deliberately, not forgotten: profiling put the dispatch itself at ~2 ms, so
// the pool would optimise what was already fast (CLAUDE.md's ExtractTimings
// finding).
class VR_MESH_API MarchingCubes {
 public:
  /// @brief Create the extractor on @p device, building its pipeline and
  ///        binding it to @p allocator for its input, vertex-arena, and
  ///        counter buffers.
  /// @param device     The compute device (must outlive this object).
  /// @param allocator  The allocator its buffers come from (must outlive this).
  /// @return The extractor, or a non-OK @ref Status if a pipeline, layout, or
  ///         descriptor allocation fails.
  /// @brief Create the extractor.
  /// @param device     The compute device (must outlive this object).
  /// @param allocator  The allocator its buffers come from (must outlive this).
  /// @param config     Extra buffer usage a *consumer* of the mesh needs; see
  ///                   @ref MarchingCubesConfig. Defaults to none, which is
  ///                   what a recon-only consumer wants.
  static Result<MarchingCubes> create(Device& device, Allocator& allocator,
                                      const MarchingCubesConfig& config = {});

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
  ///         `float` `tsdf`/`weight` attribute, if the active set is too large
  ///         for a single 1-D dispatch, or if the surface needs a vertex arena
  ///         past the device's `maxStorageBufferRange`; @ref
  ///         Status::Code::OutOfMemory if the refitted arena overflowed again
  ///         (see @ref ExtractTimings::dispatches); a backend error if a buffer
  ///         or the dispatch fails.
  ///
  /// @warning Whether it succeeds or not, this call **overwrites the vertex
  ///          arena**, so any @ref DeviceMesh from an earlier extract on this
  ///          object is invalidated the moment it starts -- a failure is not a
  ///          rollback.
  Result<Mesh> extract(volume::VoxelBlockGrid& grid, float iso = 0.0f,
                       ExtractTimings* timings = nullptr);

  /// @brief Extract as @ref extract does, but leave the result in this
  ///        extractor's device buffers instead of copying it to the host.
  ///
  /// The pass that consumes the mesh next -- `texture::ProjectiveTexturer`, or
  /// the renderer at the interop seam -- can bind these buffers directly, so
  /// the readback and the matching re-upload both disappear. Call @ref download
  /// when a host @ref Mesh is finally needed; @ref extract is exactly this
  /// followed by that.
  ///
  /// @param grid  As @ref extract.
  /// @param iso   As @ref extract.
  /// @param timings  As @ref extract, except @ref ExtractTimings::readback_ms
  ///                 covers only the 4-byte counter read, not a vertex copy.
  /// @return A @ref DeviceMesh **borrowing** this extractor's buffers -- valid
  ///         only until the next extract on this object, which overwrites them
  ///         -- or the same failures @ref extract reports (including its
  ///         @ref Status::Code::OutOfMemory case, and its warning that a failed
  ///         call still invalidates an earlier @ref DeviceMesh).
  Result<DeviceMesh> extract_device(volume::VoxelBlockGrid& grid,
                                    float iso = 0.0f,
                                    ExtractTimings* timings = nullptr);

  /// @brief Copy a @ref DeviceMesh's live vertices + indices into a host
  ///        @ref Mesh.
  /// @param device_mesh  A mesh from @ref extract_device on *this* extractor,
  ///                     not yet invalidated by a later extract.
  /// @return The host mesh, or @ref Status::Code::InvalidArgument if
  ///         @p device_mesh is not this extractor's newest extract -- it was
  ///         superseded by a later one, or came from a different extractor.
  ///         The check is by @ref DeviceMesh::generation, not by buffer handle:
  ///         the arena is reused in place, so a superseded view names the same
  ///         `VkBuffer` and a handle comparison would accept it.
  Result<Mesh> download(const DeviceMesh& device_mesh) const;

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
  // a vertex arena that would exceed it is rejected with a clean Status
  // instead of an opaque allocation failure.
  std::uint32_t max_storage_buffer_range_ = 0;
  // What a consumer asked for at create; applied on every arena grow, so a
  // regrown buffer stays bindable by whoever is already holding views of it.
  MarchingCubesConfig config_{};

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
  // (~50 ms of a 55 ms call on Replica room0): the arena was sized for the
  // worst case of 5 triangles per cell, so it ran to hundreds of megabytes,
  // and creating it every frame makes the driver fault in and zero that many
  // fresh pages while the dispatch that fills it costs ~2 ms. Reusing one
  // allocation makes a steady-state extract pay nothing for its output
  // storage, and fitting it to the surface (see plan_capacity) keeps what
  // stays resident close to the mesh's real size. Only the counter is reset
  // per call (4 bytes); the arena's stale contents past the emitted range are
  // never read, since the counter bounds the readback.
  Buffer vertex_arena_;
  Buffer counter_;
  // The identity index run 0,1,2,... covering @ref vertex_arena_. The kernels
  // emit independent triangles, so this never varies in content -- it is filled
  // once per grow and then reused, which is why it can live beside the arena
  // instead of being rebuilt per call. It exists because the consuming passes
  // (projective texturing, and the renderer at the interop seam) address
  // vertices through an index buffer.
  Buffer index_run_;
  // Numbers the extracts, so a DeviceMesh can say which one it came from.
  // Pre-incremented, so the first extract is generation 1 and a default-
  // constructed (or foreign) DeviceMesh at 0 never passes for a live one.
  // Bumped when a call is first about to touch the arena, NOT when it
  // succeeds: a call that overwrites the arena and then fails must still
  // invalidate every outstanding DeviceMesh, or download() would hand the
  // caller this call's geometry under the previous call's counts.
  std::uint64_t generation_ = 0;

  // Triangles per active block the last completed sparse extract measured, and
  // the input to the next call's capacity plan. 0 = nothing measured yet (a
  // fresh extractor, or a last extract that meshed nothing), which falls back
  // to kSeedTrisPerBlock.
  std::uint32_t tris_per_block_ = 0;

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

  // Capacity to *try* for a dispatch over @p num_active blocks whose
  // theoretical ceiling is @p worst_case triangles: the last extract's
  // measured triangles-per-block scaled by *this* call's active set, never
  // below what the arena already holds (planning under that would drop
  // triangles there was room for), and never above the ceiling. Scaling by the
  // current active set is what moves the plan ahead of a growing surface, so a
  // scan does not discover each size increase by overflowing. Adds no headroom
  // of its own -- ensure_output_buffers owns that, so the two do not compound.
  std::uint32_t plan_capacity(std::uint32_t num_active,
                              std::uint64_t worst_case) const;

  // Prepare the output buffers for a dispatch emitting at most @p capacity
  // triangles: size @ref vertex_arena_ (growing geometrically, never
  // shrinking) and @ref index_run_, and reset @ref counter_ to zero.
  //
  // The counter reset is part of the contract, not a side effect: it runs on
  // every call, including one that reallocates nothing, because a retry after
  // a refit must start its dispatch from zero or it would accumulate onto the
  // first dispatch's count. Returns a non-OK Status when @p capacity's arena
  // would exceed the device's maxStorageBufferRange -- checked against the
  // request itself, before any growth headroom, so a surface that legitimately
  // fits is never rejected because the growth policy overshot.
  Status ensure_output_buffers(std::uint32_t capacity);
};

}  // namespace volumetric_kit::recon::mesh
