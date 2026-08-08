// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file mesh/marching_cubes.hpp
/// @brief GPU marching-cubes iso-surface extraction: owns the compute pipeline
///        and drives a GLSL kernel that turns a dense SDF grid into a triangle
///        @ref Mesh.

#include <cstddef>
#include <cstdint>

#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/buffer.hpp"
#include "volumetric_kit/recon/core/compute_kernel.hpp"
#include "volumetric_kit/recon/core/descriptor.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/core/vulkan.hpp"
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

/// @brief Indices one extracted triangle contributes: three, since marching
///        cubes emits independent triangles and the index run is the identity.
///
/// Named because it is the conversion between the two units this tier trades
/// in. The kernel's append atomic bumps the draw command's `indexCount` by
/// exactly this per triangle, so the counter it maintains *is* the command --
/// the host divides by it on every readback, multiplies by it to size the arena
/// and to bound the 32-bit counter, and `marching_cubes_common.glsl` mirrors
/// it. An open-coded `3` at any one of those sites is a unit error that
/// compiles.
inline constexpr std::uint32_t kIndicesPerTriangle = 3;

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
  /// Sizing the vertex arena + resetting the draw command, including a refit
  /// after an undersized guess (see @ref dispatches). Near zero once the
  /// retained arena already fits the call -- the steady state, since the arena
  /// is reused across extracts (see @ref MarchingCubes).
  double arena_alloc_ms = 0.0;
  /// Writing the kernel's descriptor bindings.
  double descriptor_ms = 0.0;
  /// The marching-cubes dispatch(es), including the blocking fence wait --
  /// summed over both when a refit forced a second one (@ref dispatches).
  double dispatch_ms = 0.0;
  /// Getting the result back to the caller: the 20-byte draw-command read after
  /// each dispatch, plus the vertex copy into the host mesh when one is made.
  /// @ref MarchingCubes::extract makes one, so this covers both; @ref
  /// MarchingCubes::extract_device does not, so there it is the command alone.
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
  /// Bytes the extractor's vertex arenas currently hold, summed over every
  /// slot. This is *resident* size, not this call's allocation: an arena is
  /// retained across extracts, so a steady-state call allocates nothing and
  /// still reports it, and a grown arena can exceed what @ref
  /// triangle_capacity alone would need.
  ///
  /// Summed rather than reporting the slot this call wrote, because each of
  /// @ref MarchingCubesConfig::slot_count slots carries its own arena -- one of
  /// them is what the *next* extract may grow, not what the extractor costs.
  /// The index runs are not included (they are a sixteenth of an arena, four
  /// bytes per index against a 64-byte vertex).
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
/// the arena in place rather than be handed a host copy -- a renderer binding
/// it as vertex + index buffers is the motivating case -- needs its own usage
/// bits on the same allocation, and only it knows which. So this tier does not
/// name them: the application, which knows both sides, passes them in, the way
/// the create/adopt device seam has each library state its needs without either
/// being compiled against the other.
///
/// The flags actually applied come back on @ref DeviceMesh::vertex_usage /
/// @ref DeviceMesh::index_usage, so a consumer verifies rather than assumes --
/// binding a buffer that lacks the bit is a validation-layer-only diagnostic.
///
/// @warning Whatever is passed here reaches `vkCreateBuffer` directly, so ask
///          only for bits the device supports. `SHADER_DEVICE_ADDRESS` is
///          rejected by @ref MarchingCubes::create (this repo's @ref Device
///          never enables `bufferDeviceAddress`); other feature- or
///          extension-gated bits are the embedder's to get right, and asking
///          for one the device lacks fails buffer creation.
//
// TODO(mesh): usage is *necessary* for interop seam B, not sufficient. What a
// renderer binding this arena still needs, none of which a create-time flag can
// supply:
//   (1) Lifetime. SETTLED 2026-08-03: MarchingCubesConfig::slot_count gives
//       each outstanding extract its own arena and index run, and an extract
//       only writes, grows or frees a slot the consumer has released through
//       release_through. slot_count = 1 keeps the old single-arena behaviour.
//       Note this takes the ring of slots DESIGN.md's seam B specifies but
//       *not* its timeline semaphore: a host-side release report replaces the
//       GPU wait, because a command buffer waiting on a value the sibling has
//       not signalled deadlocks against a swapchain rebuild, which drains the
//       queue while holding the submit mutex.
//   (2) Sharing. SETTLED 2026-08-03: BufferDesc::queue_families picks the mode
//       from the families a caller names (Buffer::sharing_mode reads back what
//       it got), and MarchingCubesConfig::queue_families now carries them, so
//       every buffer this tier hands out -- arena, index run and indirect
//       command alike -- is CONCURRENT exactly where a sibling reads it.
//   (3) Visibility. SETTLED 2026-08-03: core's shared dispatch() barrier now
//       also reaches VERTEX_INPUT / VERTEX_ATTRIBUTE_READ / INDEX_READ and
//       DRAW_INDIRECT / INDIRECT_COMMAND_READ -- the first two only where the
//       queue family advertises graphics, since Vulkan forbids naming
//       VERTEX_INPUT on a compute-only one. A cross-queue handoff still needs
//       its semaphore, which carries visibility on its own.
//   (4) Indirect draw. SETTLED 2026-08-03: the kernel's append atomic *is* the
//       draw command's indexCount (it bumps by three per triangle), the command
//       carries INDIRECT_BUFFER beside STORAGE_BUFFER, and DeviceMesh publishes
//       it, so vkCmdDrawIndexedIndirect runs off the bytes the kernel wrote.
//       The count still round-trips to the host, but only because *this tier*
//       needs it to refit an undersized arena -- a consumer no longer does.
//
// TODO(mesh): the command lives in host-visible memory, because the refit
// protocol reads it back on every extract and resets it before every dispatch.
// On a unified-memory GPU that is free; on a discrete one the command processor
// fetches those 20 bytes across PCIe on every indirect draw. Device-local would
// invert the cost (two transfers per extract to buy a local fetch per draw), so
// this waits for a discrete-GPU consumer to measure it rather than guessing --
// the same wait-for-the-second-consumer rule the rest of this config follows.
// Recorded on DeviceMesh::sharing_mode rather than left to be inferred, because
// the hazard here is not the cost but that the cost is invisible on the only
// hardware this repo's CI runs.
struct MarchingCubesConfig {
  /// Added to the vertex arena's usage, beyond `STORAGE_BUFFER`.
  VkBufferUsageFlags extra_vertex_usage = 0;
  /// Added to the index run's usage, beyond `STORAGE_BUFFER`.
  VkBufferUsageFlags extra_index_usage = 0;
  /// Added to the indirect command's usage, beyond the `STORAGE_BUFFER` the
  /// kernel counts through and the `INDIRECT_BUFFER` a draw reads it as (both
  /// unconditional -- 20 bytes, and a consumer that never draws indirectly pays
  /// a usage bit nobody reads).
  VkBufferUsageFlags extra_indirect_usage = 0;

  /// @brief Queue families that will access this extractor's output buffers.
  ///
  /// Applies to all three -- vertex arena, index run and indirect command --
  /// since a renderer drawing the mesh reads every one of them. Left empty (the
  /// default) they are `VK_SHARING_MODE_EXCLUSIVE`, which is right for a
  /// recon-only consumer and byte-identical to what this tier always did.
  ///
  /// A consumer sharing the mesh with a sibling on another queue family must
  /// name both here. Reading an EXCLUSIVE buffer from a family that does not
  /// own it is undefined, and on Apple -- where the shared-device bootstrap
  /// hands recon and gfx *different* families, and where Metal has no ownership
  /// concept to violate -- it is undefined in the way that appears to work.
  /// Pass both indices unconditionally: duplicates collapse, so a device whose
  /// two consumers land on one family gets EXCLUSIVE for free.
  ///
  /// @see BufferDesc::queue_families, which this is copied into. Held by value
  ///      rather than as a pointer because @ref MarchingCubes stores the config
  ///      and re-reads it on every arena grow, long after @ref
  ///      MarchingCubes::create returned.
  std::uint32_t queue_families[BufferDesc::kMaxQueueFamilies] = {};
  /// Entries in @ref queue_families; more than `kMaxQueueFamilies` distinct is
  /// rejected by @ref MarchingCubes::create.
  std::uint32_t queue_family_count = 0;

  /// @brief How many extracts may be outstanding at once.
  ///
  /// One (the default) is the behaviour this always had: a single grow-only
  /// arena reused in place, so a @ref DeviceMesh is valid only until the next
  /// extract and @ref DeviceMesh::generation is what enforces it.
  ///
  /// That is unusable for a renderer drawing the arena directly. The next
  /// extract overwrites the memory an in-flight draw is reading, and a grow
  /// *frees* it -- `vmaDestroyBuffer` runs immediately, with no fence wait,
  /// because this tier's own work is fence-blocked inside `submit_single_time`
  /// and never needed one.
  ///
  /// More than one gives each extract its own arena and index run, and turns
  /// on the release contract: the consumer calls
  /// @ref MarchingCubes::release_through as its frames complete, and an
  /// extract only ever writes -- or grows, or frees -- a slot that has been
  /// released. That is what makes the immediate destroy safe again, and it is
  /// why this needs no fence queue inside the library and no semaphore across
  /// the seam. A cross-library GPU wait is the one thing the shared-queue
  /// design forbids outright.
  ///
  /// Size it to the consumer's frames in flight plus one -- and no higher than
  /// that, because **each slot costs a full vertex arena**. They are sized
  /// independently (a slot grows when the surface it is handed needs more) and
  /// none of them ever shrinks, so resident output memory is roughly
  /// `slot_count` times a single arena. @ref ExtractTimings::arena_bytes
  /// reports the sum, not one slot's share.
  ///
  /// Extracting with every slot outstanding is a contract violation, reported
  /// rather than silently overwriting a live draw.
  std::uint32_t slot_count = 1;
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
///       still sizes for its (caller-bounded) worst case and grows whichever
///       slot it lands on to that. At @ref MarchingCubesConfig::slot_count
///       above one there is one such arena *per slot*, each sized
///       independently. Destroy the extractor to release them; @ref
///       ExtractTimings::arena_bytes reports their total.
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
  ///        draw-command buffers.
  /// @param device     The compute device (must outlive this object).
  /// @param allocator  The allocator its buffers come from (must outlive this).
  /// @param config     Extra buffer usage and queue families a *consumer* of
  ///                   the mesh needs; see @ref MarchingCubesConfig. Defaults
  ///                   to none, which is what a recon-only consumer wants.
  /// @return The extractor, or a non-OK @ref Status if @p config asks for an
  ///         unsupported usage bit, or a pipeline, layout, or descriptor
  ///         allocation fails.
  static Result<MarchingCubes> create(Device& device, Allocator& allocator,
                                      const MarchingCubesConfig& config = {});

  /// @brief Report that every mesh up to and including @p generation has been
  ///        read, so its slot may be written again.
  ///
  /// The consumer half of @ref MarchingCubesConfig::slot_count. Call it as the
  /// work reading a @ref DeviceMesh completes -- for a renderer, when the frame
  /// that drew it retires.
  ///
  /// Host-side by design. The alternative, a semaphore the extract waits on, is
  /// the one thing the shared-queue arrangement forbids: a command buffer
  /// waiting on a value the sibling library has not signalled deadlocks against
  /// a swapchain rebuild, which drains the queue while holding the submit
  /// mutex. Reporting completion after the fact costs nothing and cannot
  /// deadlock.
  ///
  /// Monotonic: a generation already released stays released, and an older
  /// value than the newest reported is ignored rather than un-releasing
  /// anything. With a single slot this records the value and changes no
  /// behaviour -- there, a @ref DeviceMesh still dies at the next extract.
  ///
  /// @warning **The caller must synchronize this against the extracting
  ///          thread.** It is not atomic, and the natural consumer is on
  ///          another thread -- a renderer retires the frame that drew a mesh
  ///          on its own thread while fusion extracts on a background one
  ///          (which is exactly how `examples/viewer/fuse_viewer` is built).
  ///          Calling it concurrently with an @ref extract or @ref
  ///          extract_device on the same object is a data race. Serialize it
  ///          with whatever already guards the handoff of a @ref DeviceMesh
  ///          from the extracting thread to the consuming one; that mutex is
  ///          held for a `std::uint64_t` store, so the contention is nil.
  ///          Made a documented contract rather than a `std::atomic` member
  ///          deliberately: an atomic is not movable, and this class's
  ///          rule-of-zero defaulted moves are load-bearing (see below), so
  ///          one would cost hand-written move operations across every member
  ///          to remove a lock the consumer is already holding.
  void release_through(std::uint64_t generation) noexcept;

  // Rule of zero: every owned member (Buffer / ComputeKernel / pool) self-frees
  // and self-resets on move, so the defaulted moves are correct. Nothing here
  // caches a *copy* of an owned member's state -- the arena's capacity is
  // derived from arena() (see arena_capacity) rather than tracked
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
  ///         for a single 1-D dispatch, or if the surface's *measured* triangle
  ///         count needs a vertex arena past the device's
  ///         `maxStorageBufferRange` (the predicted capacity is clamped to that
  ///         limit rather than rejected, so an over-estimate costs one refit
  ///         dispatch instead of failing an extract that would have fit); @ref
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
  ///                 covers only the 20-byte command read, not a vertex copy.
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
  /// @return The host mesh, or @ref Status::Code::InvalidArgument for a
  ///         moved-from extractor, or if @p device_mesh is not this extractor's
  ///         newest extract -- it was superseded by a later one, or came from a
  ///         different extractor. The currency check is by
  ///         @ref DeviceMesh::generation, not by buffer handle: with one slot
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
  // What a consumer asked for at create; applied on every arena grow, not just
  // the first, so a regrown buffer carries the same usage.
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

  // The vertex arena + the draw command the kernels append through, kept ACROSS
  // extract calls and grown only when a call needs more than the last one.
  //
  // These were allocated per call, which measured as ~90% of a sparse extract
  // (~50 ms of a 55 ms call on Replica room0): the arena was sized for the
  // worst case of 5 triangles per cell, so it ran to hundreds of megabytes,
  // and creating it every frame makes the driver fault in and zero that many
  // fresh pages while the dispatch that fills it costs ~2 ms. Reusing one
  // allocation makes a steady-state extract pay nothing for its output
  // storage, and fitting it to the surface (see plan_capacity) keeps what
  // stays resident close to the mesh's real size. Only the draw command is
  // reset per call (20 bytes); the arena's stale contents past the emitted
  // range are never read, since the command's indexCount bounds the readback.
  //
  // One slot is the single reused arena described above. Several make a ring,
  // so a consumer can still be drawing generation N while N+1 is extracted --
  // see MarchingCubesConfig::slot_count.
  struct Slot {
    Buffer arena;
    // The identity index run 0,1,2,... covering @ref arena. The kernels emit
    // independent triangles, so this never varies in content -- it is filled
    // once per grow and then reused, which is why it can live beside the arena
    // instead of being rebuilt per call. It exists because the consuming
    // passes (projective texturing, and the renderer at the interop seam)
    // address vertices through an index buffer.
    Buffer index_run;
    // The `VkDrawIndexedIndirectCommand` this slot's draw is issued from, and
    // the atomic the kernel counts into: `indexCount` is field 0, so the two
    // are the same 20 bytes rather than a counter plus a command built from it.
    // Per slot, not shared, because it *is* part of the mesh -- a renderer
    // reading slot N's command while N+1 is extracted is the whole point of the
    // ring.
    Buffer indirect;
    // The extract that last *published a DeviceMesh out of* this slot; 0 until
    // one has. Compared against released_through_ to tell "still being read"
    // from "free to reuse", so it is written where a mesh is handed out, not
    // where the buffers are written: a slot marked with a generation nothing
    // ever received is one nothing can release.
    std::uint64_t generation = 0;
  };
  // A fixed array, not a vector, and that is load-bearing rather than a
  // micro-optimisation. This class promises that a *self*-move leaves it
  // intact -- `mc = std::move(mc)` is exercised directly -- and every other
  // member keeps that promise, because Buffer and the pipeline wrappers all
  // survive self-assignment. std::vector does not: self-move-assignment leaves
  // it valid but unspecified, and libc++ empties it, so the extractor would
  // pass valid() and then index nothing. An array of members that each survive
  // makes the aggregate survive too.
  //
  // All kMaxSlots are constructed regardless of slot_count_, which costs a
  // little over a kilobyte of null Buffer handles on an extractor using one --
  // against arenas measured in hundreds of megabytes. The buffers themselves
  // are filled lazily by the first extract that reaches each slot, so an unused
  // slot allocates no device memory at all.
  static constexpr std::size_t kMaxSlots = 8;
  Slot slots_[kMaxSlots];
  // Live entries in slots_. Never zero on a created extractor, and unchanged by
  // a move -- which is what keeps arena_capacity() answering 0 after one rather
  // than reading past the end.
  std::size_t slot_count_ = 1;
  // Which slot the most recent extract wrote, and therefore which one a
  // DeviceMesh handed out now names. Advances before each extract touches
  // anything, in the same breath as generation_; with one slot it never moves.
  std::size_t slot_ = 0;
  // The newest generation the consumer has finished reading, as reported by
  // release_through. Zero means "nothing released yet", which is correct at
  // rest: slot generations start at 0 too, and an extract has to be able to
  // claim a slot that has never been written.
  std::uint64_t released_through_ = 0;

  Buffer& arena() noexcept { return slots_[slot_].arena; }
  const Buffer& arena() const noexcept { return slots_[slot_].arena; }
  Buffer& index_run() noexcept { return slots_[slot_].index_run; }
  const Buffer& index_run() const noexcept { return slots_[slot_].index_run; }
  Buffer& indirect() noexcept { return slots_[slot_].indirect; }
  const Buffer& indirect() const noexcept { return slots_[slot_].indirect; }

  // Numbers the extracts, so a DeviceMesh can say which one it came from.
  // Pre-incremented, so the first extract is generation 1 and a default-
  // constructed (or foreign) DeviceMesh at 0 never passes for a live one.
  // Bumped immediately after an extract claims its slot, NOT when it succeeds:
  // a call that overwrites the arena and then fails must still invalidate every
  // outstanding DeviceMesh, or download() would hand the caller this call's
  // geometry under the previous call's counts. Immediately after, because
  // download() reads slot_ on the strength of comparing this -- see
  // claim_output_slot.
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

  // Triangle capacity the current slot's arena is sized for (0 when it holds no
  // buffer). Derived rather than stored so the capacity can never disagree with
  // the buffer it describes; the division is exact because the arena is only
  // ever allocated as a whole number of triangles.
  //
  std::uint32_t arena_capacity() const noexcept {
    return static_cast<std::uint32_t>(arena().size() /
                                      (kIndicesPerTriangle * sizeof(Vertex)));
  }

  // Vertex-arena bytes the whole ring is holding -- what ExtractTimings
  // reports. Every slot carries its own arena, so the current one's size is
  // this object's cost divided by slot_count_, not its cost.
  std::uint64_t resident_arena_bytes() const noexcept {
    std::uint64_t total = 0;
    for (std::size_t i = 0; i < slot_count_; ++i)
      total += slots_[i].arena.size();
    return total;
  }

  // Capacity to *try* for a dispatch over @p num_active blocks whose
  // theoretical ceiling is @p worst_case triangles: the last extract's
  // measured triangles-per-block scaled by *this* call's active set, clamped
  // to the ceiling. Scaling by the current active set is what moves the plan
  // ahead of a growing surface, so a scan does not discover each size increase
  // by overflowing. Adds no headroom of its own -- ensure_output_buffers owns
  // that, so the two do not compound.
  //
  // It reads no slot, which is a property to preserve rather than an accident:
  // floored at the current slot's capacity it compounded 1.5x per extract
  // across a ring (see plan_capacity's definition). Being slot-independent is
  // also why it may be called before or after claim_output_slot.
  std::uint32_t plan_capacity(std::uint32_t num_active,
                              std::uint64_t worst_case) const;

  // Pick the slot the extract about to run will write, or refuse because every
  // slot is still outstanding. Called once at the top of each extract, with
  // `++generation_` as the very next statement -- both halves load-bearing, and
  // argued where it is defined. In short: before the bump, so a refusal leaves
  // every outstanding DeviceMesh as valid as it was (this is the path that
  // exists to protect a consumer's live mesh, so it must not retire it);
  // immediately before it, so nothing fallible sits between slot_ moving and
  // generation_ moving, which is the pair download() reads as one statement.
  Status claim_output_slot();

  // Give back the slot stamped with @p generation without moving
  // released_through_. The host extract overloads' answer to "this call
  // published no DeviceMesh, so nothing outside it can release the slot":
  // release_through would do it with the *consumer's* high-water mark and so
  // also retire every older slot, including one a DeviceMesh from the same
  // extractor is still being drawn out of. A no-op on generation 0, which is
  // every untouched slot's stamp.
  void free_slot_of(std::uint64_t generation) noexcept;

  // Prepare the output buffers for a dispatch emitting at most @p capacity
  // triangles: size arena() (growing geometrically, never shrinking) and
  // index_run(), and reset the draw command via ensure_indirect_command().
  //
  // The command reset is part of the contract, not a side effect: it runs on
  // every call, including one that reallocates nothing, because a retry after
  // a refit must start its dispatch from zero or it would accumulate onto the
  // first dispatch's count. Returns a non-OK Status when @p capacity's arena
  // would exceed the device's maxStorageBufferRange -- checked against the
  // request itself, before any growth headroom, so a surface that legitimately
  // fits is never rejected because the growth policy overshot.
  //
  // It does NOT stamp the claimed slot: that belongs beside the DeviceMesh a
  // publishing return builds, so a call that fails here or after leaves the
  // slot exactly as claimable as it found it.
  Status ensure_output_buffers(std::uint32_t capacity);

  // Create this slot's draw command if it has none, and reset it to "draw
  // nothing yet": indexCount 0 for the kernel to accumulate into, and the four
  // fields that make the result a *drawable* command rather than a number a
  // host has to build one from. Split out of ensure_output_buffers because the
  // empty-active-set path needs the command without needing an arena.
  Status ensure_indirect_command();

  // Bound the command's indexCount by what the arena can actually hold.
  //
  // The kernel counts triangles it *dropped* as well as ones it wrote -- that
  // over-count is deliberate, and is what the host refits from -- so between
  // an undersized dispatch and its retry the command names more indices than
  // exist. Every path that returns while that is still true must call this, or
  // it hands back a command that reads past the end of both the arena and the
  // index run. The success path is already in range by construction (the loop
  // exits only when the count fits), so this is the *failure* paths' guarantee,
  // not theirs.
  void clamp_indirect_to_arena() noexcept;
};

}  // namespace volumetric_kit::recon::mesh
