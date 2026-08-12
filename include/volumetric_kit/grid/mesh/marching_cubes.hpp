// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file mesh/marching_cubes.hpp
/// @brief GPU marching-cubes iso-surface extraction: owns the compute pipeline
///        and drives a GLSL kernel that turns a dense SDF grid into a triangle
///        @ref Mesh.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

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

/// @brief Indices one extracted triangle contributes: three.
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
///       volume::VoxelBlockGrid meshes the real sparse volume (whose cells at a
///       block boundary sample corners out of the neighbouring blocks, resolved
///       by an on-device hash probe). The per-cell kernel is identical -- only
///       the corner-sampling differs.
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
/// Meshing is whole-volume, and the buffers are fitted to the surface rather
/// than to the 5-triangles-per-cell ceiling, so the counters explain the spans:
/// @ref triangle_capacity / @ref vertex_capacity are what the dispatch ran with
/// (one slot's index run and vertex arena, so each pairs with its `emitted_`
/// counter as that buffer's fill ratio), @ref dispatches says whether the call
/// had to refit and re-run, and @ref arena_bytes is what the extractor is
/// holding across the whole ring.
struct ExtractTimings {
  /// Compacting the hash map's active block list (a dispatch + readback).
  double compact_ms = 0.0;
  /// Allocating + filling the active-block input buffer.
  double input_upload_ms = 0.0;
  /// Sizing the vertex arena + resetting the draw command, including a refit
  /// after an undersized guess (see @ref dispatches). Near zero once the
  /// retained arena already fits the call -- the steady state, since the arena
  /// is reused across extracts (see @ref MarchingCubes).
  ///
  /// Also carries the per-block span table's upkeep when
  /// @ref MarchingCubesConfig::track_block_spans is on: growing it *and*
  /// stamping the active set at the end of a successful extract. The stamping
  /// is O(active blocks) of host work with no dispatch of its own, so it
  /// belongs to a phase or it belongs to no row at all -- and @ref total_ms
  /// sums these six rather than measuring the call, so a row it is outside of
  /// is time nothing reports.
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
  /// Whether this call re-meshed only the changed blocks, or fell back to a
  /// full extract.
  ///
  /// `false` for every @ref MarchingCubes::extract_device call, and for an
  /// @ref MarchingCubes::extract_device_incremental one that could not make
  /// the trade soundly -- the first extract against a grid, a topology change,
  /// @ref MarchingCubesConfig::share_vertices, a
  /// @ref MarchingCubesConfig::slot_count above one, an arena that had to
  /// grow, or flags the integrator will not vouch for. Each of those is
  /// invisible to the caller and each turns the feature off *permanently*
  /// (a config flag) or *silently* (a fallback), so the answer is reported
  /// rather than left to be inferred: @ref dispatches counts refit rounds and
  /// reads 1 on both paths, so it cannot be read as this.
  bool incremental = false;
  /// Blocks this call actually re-meshed, counted on-device.
  ///
  /// Meaningful only while @ref incremental is `true`; a full pass re-meshes
  /// its whole @ref active_blocks and reports 0 here rather than paying an
  /// atomic per workgroup to restate that. `remeshed_blocks / active_blocks`
  /// is the fraction of the surface the fuse actually moved, dilated into the
  /// `+{0,1}^3` neighbourhood the kernel has to redo -- which is the number
  /// the whole feature trades against, and one no caller can compute (the
  /// dilation happens on-device, off shared memory the host never sees).
  std::uint32_t remeshed_blocks = 0;
  /// Marching-cubes dispatches this call ran: 1 in the steady state, 2 when
  /// the planned capacity was under what the field emitted, so the arena was
  /// refitted to the measured count and the surface re-run. A run that keeps
  /// reporting 2 means the planner is not tracking the surface -- which is
  /// invisible in @ref dispatch_ms alone, since that sums both.
  std::uint32_t dispatches = 0;
  /// Triangle capacity the dispatch ran with -- one slot's index run, so
  /// `emitted_triangles / triangle_capacity` is that buffer's fill ratio.
  std::uint32_t triangle_capacity = 0;
  /// Triangles the kernel actually emitted.
  std::uint32_t emitted_triangles = 0;
  /// Cells per block the sparse kernel could **not** cache a triangle count
  /// for, so they were gathered twice instead of once -- correct, measurably
  /// slower, and otherwise invisible.
  ///
  /// The sparse kernel visits a cell twice: once to count (signs only), once to
  /// emit. Between them it caches each cell's triangle count in one byte of a
  /// private register, so the ~92% of cells that emit nothing are rejected
  /// without touching memory rather than by a second gather. That cache holds
  /// four counts per invocation, which covers `block_size` 8 whole; a block
  /// with more cells than it holds still meshes **correctly**, but every cell
  /// past it pays a second full gather -- at `block_size` 16 that is 75% of the
  /// block, roughly 1.8 gathers per cell against 1.1.
  ///
  /// Reported rather than refused, because nothing is wrong with the mesh --
  /// but a limit the caller cannot see is this library's to surface (see the
  /// 2026-08-04 decision). **0 for `block_size` 8**, the only shape any in-tree
  /// caller uses, and 0 for the dense @ref extract and under
  /// @ref MarchingCubesConfig::share_vertices, neither of which uses that cache
  /// (sharing is *refused* above its own limit instead).
  std::uint32_t uncached_cells_per_block = 0;
  /// Vertex capacity the dispatch ran with -- one slot's vertex arena, so
  /// `emitted_vertices / vertex_capacity` is that buffer's fill ratio.
  ///
  /// Reported separately from @ref triangle_capacity because the two stop being
  /// proportional under @ref MarchingCubesConfig::share_vertices, which is the
  /// whole point of that flag. The arena is the buffer that dominates
  /// @ref arena_bytes, so it is the one whose fill actually explains the
  /// memory.
  std::uint32_t vertex_capacity = 0;
  /// Vertices the kernel actually emitted -- exactly `3 * emitted_triangles`
  /// with sharing off, and roughly a quarter of that with it on.
  std::uint32_t emitted_vertices = 0;
  /// Bytes the extractor's output buffers currently hold -- vertex arenas *and*
  /// index runs -- summed over every slot. This is *resident* size, not this
  /// call's allocation: the buffers are retained across extracts, so a
  /// steady-state call allocates nothing and still reports them, and a grown
  /// buffer can exceed what this call's capacities alone would need.
  ///
  /// Summed rather than reporting the slot this call wrote, because each of
  /// @ref MarchingCubesConfig::slot_count slots carries its own pair -- one of
  /// them is what the *next* extract may grow, not what the extractor costs.
  ///
  /// The index runs are included, which they were not while a triangle owned
  /// three private vertices and the run was a sixteenth of the arena it
  /// covered. Vertex sharing removes exactly that proportionality: at the ~750
  /// vertices per 1000 triangles in-block sharing settles at, the arena is
  /// ~48 B/triangle against the run's 12 B, so leaving the run out would
  /// under-report resident output memory by ~20% -- on the instrument the
  /// ring's runaway growth was diagnosed with.
  ///
  /// Under @ref MarchingCubesConfig::track_block_spans it also includes
  /// **both** halves of the span table -- the device-side spans and the
  /// host-side stamps beside them, 24 bytes per block between them, sized by
  /// the grid rather than the surface. Counting one and not the other
  /// under-reported that feature by a third, which is the same defect as
  /// leaving the index runs out.
  std::uint64_t arena_bytes = 0;

  /// @return The sum of every phase, in milliseconds.
  double total_ms() const noexcept {
    return compact_ms + input_upload_ms + arena_alloc_ms + descriptor_ms +
           dispatch_ms + readback_ms;
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
// TODO(mesh): every buffer this tier hands out lives in host-visible memory --
// core::storage_buffer allocates them all that way -- and that placement is a
// bigger deal for a seam-B consumer than the usage bits above.
//   * The command, 20 bytes, because the refit protocol reads it back on every
//     extract and resets it before every dispatch. On a unified-memory GPU that
//     is free; on a discrete one the command processor fetches it across PCIe
//     on every indirect draw. Device-local would invert the cost (two transfers
//     per extract to buy a local fetch per draw).
//   * The arena and index run, which is the one that scales: a renderer binding
//     them as geometry makes the vertex-input stage pull the WHOLE mesh out of
//     system RAM on every presented frame -- ~64 MiB at the 991 k vertices
//     `fuse_viewer` reaches on room0, and ~16 MiB of that once
//     share_vertices is on, since the fetch is per vertex and in-block sharing
//     removes ~4 in 5 of them. That is free on unified memory, which is the
//     only hardware this measured on, and on a discrete GPU it would cost more
//     than the host round trip seam B exists to delete.
//   * The index run's ACCESS PATTERN moves with share_vertices, and that is a
//     second, independent placement decision. Off, the run is the identity and
//     the host only ever writes it, so it is allocated SequentialWrite -- which
//     on a discrete GPU asks VMA for device-local host-visible (BAR) memory,
//     exactly right for the buffer a renderer binds as INDEX_BUFFER. On, only
//     the kernel knows the mapping, so download() has to read it back and the
//     run becomes HostAccess::Random -- plain HOST_CACHED system RAM, no
//     device-local bit, on the buffer the draw fetches every frame. Recorded
//     rather than resolved: the honest fix is a device-local run plus a staging
//     copy for the host path, which is the same measurement the two bullets
//     above wait on.
// Both wait on the same thing -- a discrete-GPU consumer to measure a
// device-local arena plus staging against them -- rather than being guessed at,
// the wait-for-the-second-consumer rule the rest of this config follows. It is
// recorded here and on DeviceMesh::sharing_mode rather than left to be
// inferred, because the hazard is not the cost but that the cost is invisible
// on the only hardware this repo's CI runs.
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

  /// @brief Share a vertex between the cells that meet on an edge, instead of
  ///        giving every triangle three private ones.
  ///
  /// Off (the default) is what this tier always did: marching cubes appends
  /// three vertices per triangle, so a closed surface carries roughly six times
  /// the vertices it needs and the arena is sized for all of them.
  ///
  /// On, the sparse kernel shares within a block -- each cell emits a vertex
  /// only for the three edges it *owns*, and its neighbours index that one.
  /// Edges on a block's `+face` are still duplicated, because sharing them
  /// would need the neighbouring workgroup's shared memory, which is why the
  /// saving is about **4x** rather than the 6x full sharing would give.
  /// Triangles are unaffected: a shared vertex and a duplicated one are
  /// interpolated by the same code from the same two corner samples, in the
  /// same canonical endpoint order, so the surface is bit-identical and only
  /// the vertex count moves.
  ///
  /// Read at @ref MarchingCubes::create and fixed for that object's lifetime:
  /// it selects which of two compiled kernels is built, rather than switching a
  /// branch inside one. Sharing needs ~8 KiB of `shared` arrays, and a `shared`
  /// array is reserved at pipeline creation whatever a push constant later says
  /// -- so a single kernel would make the *off* path pay sharing's threadgroup
  /// budget, and its residency, for a feature it does not use.
  ///
  /// Two consequences a consumer can observe. The index run stops being the
  /// identity `0,1,2,...` (a vertex belongs to several triangles now), which is
  /// published as @ref DeviceMesh::shares_vertices; and
  /// @ref ExtractTimings::emitted_vertices stops being `3 * emitted_triangles`.
  ///
  /// @note Compatible with `texture::ProjectiveTexturer`, which it was not
  ///       until that pass moved to a per-*vertex* dispatch. The
  ///       incompatibility was never really about sharing: the texturer decided
  ///       visibility per *triangle* and wrote @ref Vertex::uv0 per *vertex*,
  ///       so a vertex belonging to several triangles that disagreed was
  ///       written by whichever thread ran last. Every input to that verdict is
  ///       a property of the vertex alone, so the pass now dispatches one
  ///       thread per vertex -- one writer each, nothing to race -- and refuses
  ///       nothing. It also got cheaper doing it, since a shared vertex used to
  ///       be projected once per referencing triangle.
  ///
  ///       What it costs is the all-three-vertices gate: a triangle straddling
  ///       the visibility boundary is no longer refused whole, so the textured
  ///       region grows by up to one triangle at an occlusion silhouette. See
  ///       `texture::ProjectiveTexturer` for the encoding that bounds it.
  ///
  /// @note A packed multi-camera atlas is a different matter and still wants a
  ///       per-*primitive* camera id: a triangle whose vertices index different
  ///       sub-rects of a pack cannot be expressed per vertex under any
  ///       encoding. That is the reason this flag stays published on
  ///       @ref DeviceMesh -- along with a consumer needing to know whether
  ///       `v = 3t` when it sizes an arena -- not a residual incompatibility.
  ///
  /// @note @ref MarchingCubes::extract_device_incremental runs under this. It
  ///       used to fall back to a full extract, on the grounds that a relocated
  ///       block can only retire what it leaves behind while it owns its
  ///       vertices three-per-triangle. That reads the cost backwards: this
  ///       kernel owns its index run, so it retires a dead triangle in 12 bytes
  ///       against the default kernel's 192, and the dead vertices need no
  ///       retiring at all because in-block sharing leaves them unreachable
  ///       once no triangle names them. What it does need is both counts
  ///       fitting before a block reuses its ranges in place, since the two are
  ///       reserved independently and a triangle indexes into the vertex range
  ///       beside it.
  ///
  /// @note Refused per *extract*, not at @ref MarchingCubes::create, for a grid
  ///       whose `voxels_per_block` exceeds 512 -- the sharing kernel's shared
  ///       per-cell table is sized for `block_size` 8, the only shape any
  ///       in-tree caller uses, and the block size arrives with the grid rather
  ///       than with this config. Both sparse @ref extract overloads report it.
  ///
  /// @note Applies to the sparse @ref extract overloads only. The dense one
  ///       meshes an arbitrary caller-supplied grid with no block structure to
  ///       share within, and is unchanged -- including its identity index run.
  bool share_vertices = false;

  /// @brief Publish @ref MarchingCubes::block_spans -- where each block's
  ///        geometry landed -- for the sparse @ref MarchingCubes::extract
  ///        overloads.
  ///
  /// Off by default because it is not free and most callers never read it. The
  /// table is sized by the **grid**, not by the surface: `num_blocks` entries
  /// of 16 bytes, which is 24 MB at @ref volume::VoxelGridParams::defaults and
  /// doubles with every @ref volume::VoxelHashMap::resize, held for this
  /// object's lifetime. A host-side array of the same length carries the
  /// per-slot stamp @ref MarchingCubes::block_span_valid answers from, for
  /// another 8 bytes per block; both are counted in
  /// @ref ExtractTimings::arena_bytes, so the figure there is what the feature
  /// actually costs rather than the visible half of it. With this off the
  /// kernel is told not to write the table, nothing is allocated, and the
  /// per-block stamping loop does not run, so a caller who did not ask measures
  /// nothing -- the same bargain
  /// `tsdf::TsdfIntegratorConfig::track_dirty_blocks` strikes for the flags it
  /// gates, and for the same reason.
  ///
  /// @note Applies to the sparse @ref extract overloads only. The dense one
  ///       meshes a caller-supplied grid with no block structure to describe.
  bool track_block_spans = false;
};

/// @brief Where one block's geometry landed in the extract that meshed it.
///
/// Counted in vertices and in TRIANGLES -- not indices -- because a triangle is
/// what a block owns and what a re-mesh replaces; multiply by
/// @ref kIndicesPerTriangle for the index run. The four numbers are independent
/// under @ref MarchingCubesConfig::share_vertices and locked at `v = 3t`
/// without it, which is exactly the ratio sharing breaks.
///
/// Mirrored field-for-field by the `BlockSpan` of
/// `shaders/marching_cubes_block_span.glsl`, which both sparse kernels write
/// **by field name**. The offsets below pin that ABI: four same-typed members
/// make every permutation the same size, so `sizeof` alone cannot see a
/// transposition (see the 2026-07-05 scalar-block-layout decision).
struct BlockSpan {
  std::uint32_t vertex_base = 0;     ///< First vertex the block owns.
  std::uint32_t vertex_count = 0;    ///< Vertices it owns from there.
  std::uint32_t triangle_base = 0;   ///< First **triangle**, not index.
  std::uint32_t triangle_count = 0;  ///< Triangles it owns from there.
};
static_assert(sizeof(BlockSpan) == 16, "BlockSpan must be 16 bytes");
static_assert(offsetof(BlockSpan, vertex_base) == 0, "BlockSpan ABI");
static_assert(offsetof(BlockSpan, vertex_count) == 4, "BlockSpan ABI");
static_assert(offsetof(BlockSpan, triangle_base) == 8, "BlockSpan ABI");
static_assert(offsetof(BlockSpan, triangle_count) == 12, "BlockSpan ABI");
// block_spans() reinterprets mapped device memory as an array of these, which
// is defined only for a trivially copyable standard-layout type.
static_assert(std::is_standard_layout_v<BlockSpan>,
              "BlockSpan must be standard layout");
static_assert(std::is_trivially_copyable_v<BlockSpan>,
              "BlockSpan must be trivially copyable");

/// @brief The blocks a fuse changed, as the device buffer holding them.
///
/// Deliberately opaque: `tsdf` produces this and `mesh` only reads it, so
/// passing the handle rather than the integrator keeps `mesh` off `tsdf`
/// entirely -- the tier order forbids the include, and this is what replaces
/// it. Fill all three fields from the integrator that wrote the flags; each
/// has an accessor, and each answers the same question about the same fuse.
///
/// One `uint32_t` per **block slot** (`BlockIndex::ptr / voxels_per_block`),
/// non-zero where that block's voxels changed. This is the *changed* set; the
/// extractor dilates it into the re-mesh set on-device, since a cell reads
/// corners at `base + {0,1}^3`.
struct DirtyBlocks {
  /// The flag buffer, from `TsdfIntegrator::dirty_flags_buffer()`. Null when
  /// the integrator has nothing usable to offer, which makes the extract fall
  /// back to a full one rather than fail.
  VkBuffer flags = VK_NULL_HANDLE;
  /// Block slots @p flags addresses, from
  /// `TsdfIntegrator::dirty_flags_capacity()`. The kernel's only bound on the
  /// buffer, and what its descriptor range is written from -- so it must be
  /// the capacity of *this* @p flags buffer, not one cached from before the
  /// integrator grew it. A slot past it is treated as changed, the
  /// conservative direction.
  std::uint32_t capacity = 0;
  /// The `volume::VoxelBlockGrid::topology_epoch` the flags were accumulated
  /// against, from `TsdfIntegrator::dirty_epoch()`. Checked against the grid
  /// being meshed, because a flag is keyed by block *slot* and a slot means
  /// nothing across a `remove()`/`clear()` (which hands the index to a
  /// different block) or across grids. The token is globally unique, so this
  /// one comparison answers both -- exactly as it does for the span table.
  std::uint64_t epoch = 0;
};

/// @brief Owns the marching-cubes compute pipelines and extracts an iso-surface
///        into a host @ref Mesh -- from a dense @ref DenseGrid or straight off
///        a sparse @ref volume::VoxelBlockGrid.
///
/// Built on the `core` compute foundation (@ref Allocator, @ref Buffer,
/// @ref ComputeKernel, @ref Device::submit_single_time), mirroring the volume
/// tier's @ref volume::VoxelHashMap. The kernel runs one invocation per cell,
/// builds the cube index from the eight corner signs, and interpolates a vertex
/// on each crossed edge. How the triangles reach the arena differs by path:
///
/// - The **dense** @ref extract appends each triangle independently through an
///   atomic bump counter -- it has no block structure to reserve against.
/// - The **sparse** @ref extract runs one workgroup per active block, which
///   counts the block's output, reserves one range for all of it with a single
///   atomic, and only then writes. A block's triangles therefore land
///   **contiguously** in the arena rather than interleaved with every other
///   block's in flight. That is the precondition for meshing only the blocks a
///   fuse changed -- with an interleaved arena there is no range to leave in
///   place -- and it costs ~10% on the dispatch, taken deliberately (see the
///   2026-08-09 incremental-extraction decision). **Both** sparse kernels do
///   it: @ref MarchingCubesConfig::share_vertices selects one that reserves two
///   ranges rather than one, since a shared vertex breaks `v = 3t`, and it
///   measured no cost there. Either range is published as
///   @ref block_spans when @ref MarchingCubesConfig::track_block_spans asks for
///   it, which is how a caller outside the extractor indexes a block's
///   geometry.
///
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
// TODO(mesh): extend incremental extraction past
// MarchingCubesConfig::slot_count
// == 1. @ref extract_device_incremental refuses a ring, because a re-meshed
// block writes into the arena the LAST extract filled and a ring hands this one
// a different slot -- so the clean blocks' triangles are in the wrong buffer.
// The two ways out are to reuse a block's range only for slots the consumer has
// released through @ref release_through, or to copy the retained run forward
// into the newly claimed slot (a vkCmdCopyBuffer of `watermark * 3` vertices).
// Until then the feature is off in exactly the configuration seam B uses, which
// is why @ref ExtractTimings::incremental exists to say so.

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

  /// @brief Where the last successful sparse extract put each block's geometry.
  ///
  /// Indexed by **block slot** -- `volume::BlockIndex::ptr / voxels_per_block`
  /// -- and meaningful only for the blocks in the active set of the extract
  /// that wrote it. A slot means nothing against a different grid, and nothing
  /// after a `remove()` or `clear()`: the block heap is LIFO, so a reused slot
  /// names a different block.
  ///
  /// @warning **Every entry reads as a well-formed span, including the ones
  ///          this extract did not write.** Nothing is cleared on the way past:
  ///          a grow copies the whole existing table forward (a
  ///          `volume::VoxelHashMap::resize` preserves block indices, so those
  ///          spans are still true) and zeroes only the tail it added, so a
  ///          slot the last extract did not mesh holds whatever an *earlier*
  ///          one left there -- bases into an arena that has since been
  ///          rewritten, and possibly reallocated. There is no value here that
  ///          means "not mine", and an empty span does not: that is what a
  ///          block which meshed and emitted nothing writes.
  ///
  /// So this array is not something to iterate and interpret. Read it only at
  /// slots @ref block_span_valid has answered `true` for, having first checked
  /// @ref block_spans_generation against the @ref DeviceMesh::generation whose
  /// arena you are about to index. Those two questions are the contract; this
  /// pointer is only how the answer is fetched.
  ///
  /// @warning **Borrowed, and invalidated by the next @ref extract or
  ///          @ref extract_device on this object** -- exactly like a
  ///          @ref DeviceMesh, and for the same reason: a grid whose
  ///          `num_blocks` grew reallocates this table, which frees the pages
  ///          this points at. Do not cache the pointer across a call. Compare
  ///          @ref block_spans_generation against the
  ///          @ref DeviceMesh::generation you hold to know whether the table
  ///          still describes *your* mesh -- above one
  ///          @ref MarchingCubesConfig::slot_count it will not, because the
  ///          arena is per slot and this table is not.
  ///
  /// @return `block_span_capacity()` entries, or `nullptr` when
  ///         @ref MarchingCubesConfig::track_block_spans is off, on a
  ///         moved-from extractor, before the first sparse extract, or when the
  ///         last extract (sparse or dense) did not leave a table describing it
  ///         -- a failed one, or one that meshed nothing.
  const BlockSpan* block_spans() const noexcept;

  /// @brief Entries @ref block_spans addresses.
  ///
  /// Derived from the buffer rather than tracked beside it, so the count and
  /// the pointer cannot disagree -- including on a moved-from extractor, where
  /// the defaulted move leaves both empty.
  std::uint32_t block_span_capacity() const noexcept {
    return static_cast<std::uint32_t>(block_spans_.size() / sizeof(BlockSpan));
  }

  /// @brief The generation @ref block_spans describes, or 0 if it describes
  ///        nothing.
  ///
  /// The same counter @ref DeviceMesh::generation carries, so the two are
  /// directly comparable: a consumer holding generation `g` learns that the
  /// table is about some *other* extract the moment this stops equalling `g`.
  /// There is one table for the whole ring -- it is one dispatch's worth of
  /// state, not a mesh a consumer still holds -- so at
  /// @ref MarchingCubesConfig::slot_count above one this is the check that
  /// keeps a span from being read against the wrong slot's arena.
  std::uint64_t block_spans_generation() const noexcept {
    return block_spans_generation_;
  }

  /// @brief Does slot @p slot carry a span **the extract that wrote the current
  ///        table** put there?
  ///
  /// Not "has this slot ever been meshed". A block that drops out of the active
  /// set keeps the stamp its last extract wrote, and the span under it goes on
  /// naming an arena range later extracts have handed to other blocks, so the
  /// question has to be about the *published* table or the answer is worse than
  /// useless. False for such a slot, false for one no extract has meshed, and
  /// false for **every** slot once the anchor breaks -- a different grid, or a
  /// `remove()` / `clear()` that moved @ref
  /// volume::VoxelBlockGrid::topology_epoch. The heap is LIFO, so a reused slot
  /// names a different block and its span would otherwise be read as that
  /// block's geometry, under `Status::ok`.
  ///
  /// A `resize()` does **not** break it: resizing preserves block indices, so
  /// the spans stay true and the table simply grows.
  ///
  /// Takes the grid rather than trusting the caller to re-extract after a
  /// topology change: between a `remove()` and the next extract the stamps are
  /// still set, so a query that did not re-check the anchor would report a
  /// stale span as live. That is a staleness the caller cannot see, which makes
  /// it this tier's to check (the 2026-08-04 rule). It holds however the
  /// topology moved -- @ref volume::VoxelBlockGrid::remove and the raw @ref
  /// volume::VoxelHashMap::remove reached through @ref
  /// volume::VoxelBlockGrid::map both move the token, since it belongs to the
  /// table that frees the index.
  ///
  /// @warning This is the per-*slot* half of the question and **not the whole
  ///          of it.** It answers false whenever @ref block_spans_generation is
  ///          0, so it can never report a slot live while @ref block_spans
  ///          returns `nullptr` -- but it does not know which
  ///          @ref DeviceMesh *you* hold. There is one table for the whole
  ///          ring, so above one @ref MarchingCubesConfig::slot_count a
  ///          consumer still drawing generation `g` will find this `true` for a
  ///          table describing `g+1`'s arena. Check `block_spans_generation()
  ///          == your_mesh.generation` first; only then does a per-slot answer
  ///          mean anything.
  ///
  /// Always false when @ref MarchingCubesConfig::track_block_spans is off, on a
  /// moved-from extractor, and for a moved-from @p grid -- which owns no blocks
  /// however the token reads.
  ///
  /// @param grid The grid the spans are expected to describe.
  /// @param slot `volume::BlockIndex::ptr / voxels_per_block`.
  bool block_span_valid(const volume::VoxelBlockGrid& grid,
                        std::uint32_t slot) const noexcept;

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
  /// Being a single high-water mark is what shapes the consumer's side of the
  /// contract, so it is worth stating plainly: above one slot, **this** -- not
  /// @ref DeviceMesh::is_current -- is what bounds a view's life. A view stays
  /// good until its own generation is reported here, which is why a ring
  /// consumer can hold and draw a view the producer has already run past. The
  /// flip side is that a generation the consumer takes and then abandons keeps
  /// its slot until some *newer* generation is reported, because the mark
  /// cannot skip one; a consumer that can drop a taken mesh must therefore
  /// bound how many it drops, or it exhausts the ring and every later extract
  /// is refused.
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
  /// Runs one **workgroup** per active block, striding over the block's voxels,
  /// each voxel the base corner of one marching-cubes cell. A cell on a block's
  /// `+face` reaches its far corners into neighbouring blocks; the kernel
  /// resolves that 2x2x2 neighbourhood (this block plus its seven `+x/+y/+z`
  /// neighbours) **itself**, probing @p grid's hash table on-device, eight
  /// probes amortised over the block's cells. The per-cell body is identical to
  /// the dense @ref extract -- independent triangles, one gradient normal per
  /// cell, reversed winding, and the same hybrid @ref Vertex::color /
  /// @ref Vertex::uv0 appearance.
  ///
  /// @warning The probe is lock-free and unfenced, so this call requires
  ///          @p grid's hash table to be **quiescent**: no `allocate` /
  ///          `remove` / `clear` / `resize` dispatch on it may be in flight, on
  ///          this thread or any other. Within one thread that holds by
  ///          construction (every `volume` dispatch blocks on its fence), so it
  ///          binds only a caller that fuses and meshes concurrently -- which
  ///          must serialise the two itself. The same precondition as
  ///          @ref volume::VoxelHashMap::entries_buffer, stated here because
  ///          this is where a caller meets it; the host-built neighbour table
  ///          this replaced needed it too, having read the same table through a
  ///          compacted snapshot.
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

  /// @brief Extract, re-meshing only the blocks @p dirty says a fuse changed.
  ///
  /// The blocks whose `+{0,1}^3` neighbourhood carries no change keep the
  /// triangles they already have, at the offsets @ref block_spans already
  /// names, and cost one workgroup that returns before gathering a single
  /// corner. A changed block re-meshes into the range it already owns when the
  /// new count fits, and appends past the watermark when it does not.
  ///
  /// **Falls back to a full extract**, silently and by design, whenever an
  /// incremental one would be wrong rather than merely slower. Falling back
  /// rather than refusing is the point: a caller fusing a live scan cannot
  /// predict a topology change, and the correct response to one is to re-mesh
  /// everything, not to fail. The full list, every entry of which is something
  /// the caller cannot see:
  ///
  /// - the first extract against a grid, which is what *establishes* the
  ///   arena and spans an incremental pass reads;
  /// - a `remove()`/`clear()` since then, which hands a block slot to a
  ///   different block, so neither @ref DirtyBlocks::epoch nor the span
  ///   table's own anchor still matches the grid;
  /// - flags the integrator will not vouch for (a null
  ///   @ref DirtyBlocks::flags, or an epoch that has moved);
  /// - an arena that has to grow for this call, since a grow reallocates and
  ///   nothing copies the clean blocks' triangles forward;
  /// - an overflow refit, whose retry has already lost the pre-call spans;
  /// - an arena whose occupancy has drifted too far past its live surface,
  ///   where a full pass is what compacts the retired triangles away;
  /// - @ref MarchingCubesConfig::slot_count above one (see the `TODO(mesh)`
  ///   above this class), since the retained triangles are in the slot the
  ///   *last* extract filled and a ring hands this one a different slot.
  ///
  /// Which one it got is reported as @ref ExtractTimings::incremental, beside
  /// the @ref ExtractTimings::remeshed_blocks that says how much it saved.
  ///
  /// @warning An in-place re-mesh writes bytes an outstanding generation may
  ///          still be drawing. Every index stays in range and every vertex
  ///          stays a real vertex, so this is not a memory error -- but a
  ///          consumer holding a @ref DeviceMesh across the call can catch one
  ///          block mid-update. That is the trade this overload exists to make
  ///          measurable; @ref extract_device is unchanged and does not make
  ///          it.
  ///
  /// Runs under @ref MarchingCubesConfig::share_vertices, which reuses **two**
  /// ranges rather than one: a triangle indexes into the vertex range beside
  /// it, so a block reuses in place only when *both* counts fit and relocates
  /// both when either does not. Retirement is cheaper there, not dearer -- that
  /// kernel owns its index run, so a dead triangle is retired by pointing its
  /// three indices at one vertex (12 bytes, zero area, culled before
  /// rasterisation), where the default kernel's run is the identity it cannot
  /// touch and it must overwrite 192 bytes of vertices to say the same thing.
  /// The dead *vertices* need no writing at all: sharing is in-block, so once
  /// no triangle references them they are unreachable rather than merely
  /// unused.
  ///
  /// @param grid     The sparse volume to mesh, as @ref extract_device takes
  ///                 it.
  /// @param iso      The iso-value to extract at (0 for a TSDF surface).
  /// @param dirty    The blocks the fuse changed; see @ref DirtyBlocks for
  ///                 where each field comes from. Read for the duration of
  ///                 this call and not retained.
  /// @param timings  Optional; filled as @ref extract_device fills it, plus
  ///                 @ref ExtractTimings::incremental and
  ///                 @ref ExtractTimings::remeshed_blocks.
  /// @return The mesh in this extractor's device buffers, borrowed exactly as
  ///         @ref extract_device's is, or that overload's @ref Status on any
  ///         of the failures it can report.
  Result<DeviceMesh> extract_device_incremental(
      volume::VoxelBlockGrid& grid, float iso, const DirtyBlocks& dirty,
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

  /// @brief Extract as @ref extract_device does, but mesh only @p blocks --
  ///        the caller's own compacted subset of the grid's active set.
  ///
  /// The motivating subset is a camera's: @ref
  /// volume::VoxelHashMap::compact_active_blocks_in_frustum culls the active
  /// set to what a view can see, and a scanning device that renders a small
  /// part of a large volume then meshes only that part. Nothing here is
  /// specific to a frustum, though -- a region of interest, a chunk queue or a
  /// level-of-detail selection are the same call.
  ///
  /// This is the *only* difference from @ref extract_device: the set arrives
  /// instead of being compacted, so @ref ExtractTimings::compact_ms reads 0 and
  /// every other phase runs unchanged and shrinks with the set. The arena is
  /// rebuilt from this dispatch alone, so a block outside @p blocks contributes
  /// no triangles to the mesh and no bytes to the arena -- which is what makes
  /// this worth doing on a memory-bound device, rather than only a cheaper
  /// dispatch.
  ///
  /// @note The mesh does **not** hole at the cull boundary. The kernel resolves
  ///       each block's 2x2x2 neighbourhood by probing the hash table
  ///       on-device, so a block on the edge of @p blocks still samples correct
  ///       corner values out of neighbours that were never dispatched; the
  ///       surface simply ends there. A host-built neighbour table -- what this
  ///       tier used before the 2026-08-08 decision -- could not have done
  ///       this.
  ///
  /// @note Culling does not make the *compaction* cheaper. The frustum kernel
  ///       still scans every hash-table slot; what shrinks is the readback, the
  ///       upload, this dispatch, the arena, and whatever draws or textures the
  ///       result.
  ///
  /// @warning Deliberately **not** offered on @ref
  ///          extract_device_incremental, and the two do not compose today: an
  ///          incremental pass keeps the triangles of every block it does not
  ///          re-mesh, and a block outside @p blocks is simply not dispatched,
  ///          so it would keep its geometry below the watermark and go on being
  ///          drawn. That is not wrong, but it gives back the arena and draw
  ///          savings and leaves only the dispatch one -- so the two are kept
  ///          as separate answers to the same cost rather than stacked.
  ///
  /// @param grid     The sparse volume to mesh, as @ref extract_device takes
  ///                 it.
  /// @param iso      The iso-value to extract at (0 for a TSDF surface).
  /// @param blocks   The blocks to mesh: a subset of @p grid's active set, with
  ///                 @ref volume::BlockList::epoch filled from
  ///                 `grid.topology_epoch()` when it was compacted. Read for
  ///                 the duration of this call and not retained. An empty list
  ///                 is legal and meshes nothing, exactly as an empty map does.
  /// @param timings  As @ref extract_device, except @ref
  ///                 ExtractTimings::compact_ms is 0 and @ref
  ///                 ExtractTimings::active_blocks reports @p blocks's count --
  ///                 which is the instrument that says what the cull bought.
  /// @return The mesh in this extractor's device buffers, borrowed exactly as
  ///         @ref extract_device's is, or that overload's @ref Status, plus
  ///         @ref Status::Code::InvalidArgument when @p blocks is internally
  ///         inconsistent (a null pointer with a non-zero count) or was
  ///         compacted against a topology @p grid has since left behind. The
  ///         second is refused rather than meshed: the block heap is LIFO, so a
  ///         `remove()` since the compaction has handed those block pointers to
  ///         different blocks, and the extract would silently mesh whatever
  ///         voxels now live there.
  Result<DeviceMesh> extract_device(volume::VoxelBlockGrid& grid, float iso,
                                    const volume::BlockList& blocks,
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
  // Per-block spans, indexed by block slot (`BlockIndex::ptr /
  // voxels_per_block`) and sized to the grid's `num_blocks`. Grown on demand,
  // never shrunk, and NOT per slot: it describes where the *current* extract
  // put each block, which is one dispatch's worth of state rather than a mesh a
  // consumer still holds. Allocated only when config_.track_block_spans is on;
  // otherwise the kernel is told not to write it and block_spans_dummy_ keeps
  // the binding valid.
  Buffer block_spans_;
  // A 1-element stand-in bound at the span binding when tracking is off, so
  // that descriptor stays valid without paying num_blocks * 16 bytes for a
  // table nobody asked for. Mirrors color_dummy_ above, and the `write_spans`
  // push flag is what keeps the kernel off it.
  Buffer block_spans_dummy_;
  // The generation block_spans_ describes; 0 when it describes nothing. Set
  // only once an extract has succeeded, and cleared by anything that leaves the
  // table not describing the mesh this object last handed out -- a failed
  // sparse extract, an empty one, or a dense one. Comparable against
  // DeviceMesh::generation, which is the point: it is what makes the one table
  // safe to read beside a ring of arenas.
  std::uint64_t block_spans_generation_ = 0;
  // What `block_spans_` is anchored to: the
  // volume::VoxelHashMap::topology_epoch token of the map whose slots the spans
  // are keyed by. A slot only names a block against one table at one topology,
  // and the block heap is LIFO, so after a remove() a reused slot names a
  // DIFFERENT block and every span keyed by it is a lie that still typechecks.
  //
  // ONE token and no grid pointer beside it, because the token is drawn from a
  // process-wide counter: no two grids, and no two topologies of one grid, ever
  // share one, so equality here already means "the same grid, unchanged". A
  // pointer would be strictly worse than redundant -- this class hands out no
  // way to un-anchor and must never dereference a borrowed grid, so a grid
  // destroyed after an extract leaves a dangling address that the next grid
  // built in that storage matches exactly.
  //
  // Where block_spans_generation_ retires the WHOLE table when it stops
  // describing the mesh this object handed out, this retires individual slots
  // that no longer name the block their span was written for. Both are needed:
  // the first is about which extract the table belongs to, the second about
  // which block a slot means.
  std::uint64_t span_epoch_ = 0;
  // Which extract wrote each slot's span, as a value of `span_serial_`, which
  // is bumped once per sparse extract that reaches ensure_block_spans. A stamp
  // is live only while it EQUALS the current serial: nothing clears a stamp, so
  // "non-zero" would mean "ever meshed" and report a block dropped from the
  // active set as still described by a table rewritten since. Zero is "never
  // written", which no serial equals.
  //
  // block_span_capacity() entries -- read off `block_spans_` rather than stored
  // beside it, so the length cannot outlive the buffer it parallels (a count
  // kept in a member survives a move that empties the buffer). Null unless
  // config_.track_block_spans is on: this is num_blocks * 8 bytes of HOST
  // memory (12 MB at VoxelGridParams::defaults), so it falls under the same
  // "nothing measured for a caller who did not ask" rule as the table it
  // describes.
  //
  // A unique_ptr and deliberately not a std::vector: the defaulted moves below
  // promise a self-move leaves this object intact, and vector's self-move-
  // assignment is unspecified (libc++ empties it), which would leave valid()
  // true beside a stamp array that reports every slot dead. unique_ptr's
  // assignment is specified as release-then-reset, which is self-safe.
  std::unique_ptr<std::uint64_t[]> span_stamp_;
  std::uint64_t span_serial_ = 0;
  // What the retained arena holds, from the extract that last published a mesh
  // out of it -- and the ONLY thing an incremental extract is allowed to trust
  // about the arena's existing contents.
  //
  // One struct rather than three members because the three are only ever
  // meaningful together, and every path that invalidates one invalidates all
  // three: a default-constructed value is "no incremental state", which is what
  // makes the next extract a full one. It is cleared at the top of BOTH extract
  // paths, beside block_spans_generation_ and for the same reason -- a dense
  // extract claims the same slot, and there are several ways down from there
  // that publish nothing -- and re-established only on the publishing return,
  // so no failure can leave it describing geometry the failed call destroyed.
  //
  // Scalars throughout, so they survive a self-move like every other member.
  struct ArenaState {
    // Triangles the arena holds, so the next incremental pass appends past
    // them instead of restarting at zero. This is OCCUPANCY, not the live
    // surface: it includes the ranges relocating blocks retired to
    // zero-area triangles. Zero means there is no state to trust.
    std::uint32_t watermark = 0;
    // The grid topology `watermark` and the spans were written against. A
    // remove()/clear() moves the token and hands block slots to different
    // blocks, which invalidates both at once.
    std::uint64_t epoch = 0;
    // span_serial_ of the extract that wrote them, so a table re-anchored
    // since (or one a dense extract stepped over) cannot be read as this
    // arena's. The epoch alone does not catch that: a resize deliberately
    // does not move it.
    std::uint64_t serial = 0;
    // Vertices the arena holds, the counterpart of `watermark` for the buffer
    // sharing actually binds on. Its own number rather than one derived from
    // the triangle count, because share_vertices breaks v = 3t -- which is the
    // whole reason that kernel allocates vertices through a counter of its
    // own. Without sharing it tracks 3 * watermark and nothing reads it: that
    // kernel writes each triangle's three vertices at `tri * 3` and never
    // touches the counter.
    std::uint32_t vertex_watermark = 0;
  };
  ArenaState arena_state_{};

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
    // The index run covering @ref arena. It exists because the consuming passes
    // (projective texturing, and the renderer at the interop seam) address
    // vertices through an index buffer.
    //
    // WHO fills it depends on MarchingCubesConfig::share_vertices. Off, every
    // triangle owns three private vertices written at `tri * 3`, so the run is
    // the identity 0,1,2,..., never varies in content, and the host fills it
    // once per grow. On, a vertex is referenced by several triangles from
    // several cells and only the kernel knows which, so the kernel writes it
    // every dispatch and download() reads it back.
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
  // member keeps that promise, because Buffer, unique_ptr and the pipeline
  // wrappers all survive self-assignment. std::vector does not:
  // self-move-assignment leaves it valid but unspecified, and libc++ empties
  // it, so the extractor would pass valid() and then index nothing. An array of
  // members that each survive makes the aggregate survive too -- which is the
  // rule every member added here has to be checked against (span_stamp_ is a
  // unique_ptr for exactly this reason).
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

  // Vertices the last completed sparse extract emitted per 1000 triangles, and
  // the input to the next call's arena budget. Scaled by 1000 rather than kept
  // as a float so the plan is exactly reproducible. 0 = nothing measured yet,
  // which falls back to kSeedVertsPer1000.
  //
  // Without sharing this settles at exactly 3000 and the arena is sized as it
  // always was. With sharing it converges toward ~750: a closed surface has
  // about half as many vertices as triangles, less what a block seam
  // duplicates.
  std::uint32_t verts_per_1000_tris_ = 0;

  // The two marching-cubes kernels -- each its descriptor-set layout, pipeline,
  // and a set allocated from the shared pool_ (see @ref ComputeKernel): the
  // dense analytic-grid path and the sparse VoxelBlockGrid path.
  ComputeKernel kernel_;
  ComputeKernel kernel_sparse_;
  DescriptorPool pool_;

  // Triangles the current slot's INDEX RUN can hold (0 when it holds no
  // buffer). Derived from the buffer rather than stored, so the capacity can
  // never disagree with what it describes; the division is exact because the
  // run is only ever allocated as a whole number of triangles.
  //
  // Read off the index run and not the arena, which is where it used to come
  // from: that was valid only while the arena held exactly three vertices per
  // triangle. Vertex sharing breaks that proportionality, so each capacity now
  // comes from the buffer that actually bounds it.
  std::uint32_t arena_capacity() const noexcept {
    return static_cast<std::uint32_t>(
        index_run().size() / (kIndicesPerTriangle * sizeof(std::uint32_t)));
  }

  // Vertices the current slot's arena can actually hold. Derived from the
  // buffer, like arena_capacity, so the two can never disagree with it -- and
  // stated separately because vertex sharing breaks the "three per triangle"
  // identity that let one stand in for the other.
  std::uint32_t arena_vertex_capacity() const noexcept {
    return static_cast<std::uint32_t>(arena().size() / sizeof(Vertex));
  }

  // Triangles the current slot can actually MESH, which is what the dispatch is
  // pushed and what the emitted count is clamped to.
  //
  // Not simply arena_capacity(): with sharing off the kernel writes each
  // triangle's three vertices at `tri * 3`, so the arena bounds the triangle
  // count too -- and the two buffers grow independently now, so the index run
  // can legitimately hold more triangles than the arena has vertices for.
  // Sharing on, the kernel claims vertices through a counter it bounds itself,
  // so the run is the only limit.
  std::uint32_t usable_triangle_capacity() const noexcept {
    if (config_.share_vertices) return arena_capacity();
    return std::min(arena_capacity(),
                    arena_vertex_capacity() / kIndicesPerTriangle);
  }

  // Output bytes the whole ring is holding -- what ExtractTimings::arena_bytes
  // reports. Every slot carries its own arena AND index run, so the current
  // slot's size is a fraction of this object's cost, not its cost.
  //
  // The span table counts too, and it is not a rounding error: it is sized by
  // the GRID rather than by the surface (num_blocks * 16, which is 24 MB at
  // VoxelGridParams::defaults against room0's ~38 MB of triangles), it is held
  // for this object's lifetime, and this is the instrument the ring's runaway
  // growth was diagnosed with. Omitting a component of what stays resident is
  // the same defect that folding the index runs in here fixed.
  //
  // ...and so does the host-side stamp array that parallels it, another
  // num_blocks * 8 (12 MB at the same defaults), for the same reason and by the
  // same argument. It is derived from the table's own capacity rather than
  // measured, because the two are allocated in lockstep by ensure_block_spans.
  std::uint64_t resident_output_bytes() const noexcept {
    std::uint64_t total = block_spans_.size() +
                          static_cast<std::uint64_t>(block_span_capacity()) *
                              sizeof(std::uint64_t);
    for (std::size_t i = 0; i < slot_count_; ++i)
      total += slots_[i].arena.size() + slots_[i].index_run.size();
    return total;
  }

  // The one sparse extract, with the three public entry points differing only
  // in what they have to offer it: dirty flags, an active set, or neither.
  //
  // @p dirty is null for extract_device and points at the caller's struct for
  // extract_device_incremental, which is borrowed for this call alone. A
  // PARAMETER rather than a member the wrapper arms and disarms around a
  // delegated call: as a member, any exception unwinding out of here -- and
  // this function allocates a vector, a 12 MB stamp array and several
  // std::strings -- left a caller-owned VkBuffer latched on the extractor, so
  // the *plain* extract_device would then bind it, possibly after the
  // integrator that owned it was destroyed. As a parameter that state cannot be
  // represented.
  //
  // @p blocks is null when this call compacts the whole active set itself, and
  // points at the caller's subset otherwise -- borrowed for this call alone,
  // and a parameter for the same reason @p dirty is, only more so: it is a bare
  // host pointer into a std::vector the caller owns, so latching it on the
  // extractor would leave a dangling read for the NEXT extract rather than a
  // stale one for this.
  //
  // Whether the pass is actually incremental is decided HERE, not by which
  // entry point was called: every clause is something the caller cannot see.
  Result<DeviceMesh> extract_device_impl(volume::VoxelBlockGrid& grid,
                                         float iso, const DirtyBlocks* dirty,
                                         const volume::BlockList* blocks,
                                         ExtractTimings* timings);

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
  //
  // @p seed_triangles and @p seed_vertices are forwarded to
  // ensure_indirect_command; every caller but the incremental one passes 0.
  Status ensure_output_buffers(std::uint32_t triangle_capacity,
                               std::uint32_t vertex_capacity,
                               std::uint32_t seed_triangles = 0,
                               std::uint32_t seed_vertices = 0);

  // Re-anchor the span table on @p grid and grow it to that grid's num_blocks,
  // carrying the existing spans forward and zeroing only the new tail. A no-op
  // unless config_.track_block_spans is on, and never shrinks: num_blocks only
  // rises, because a VoxelHashMap::resize preserves block indices.
  //
  // Takes the grid rather than a block count because the anchor and the size
  // both come from it, and they have to move together: a table grown for one
  // grid while still stamped for another would report another grid's slots as
  // live.
  //
  // Separate from ensure_output_buffers because the two are sized by different
  // things -- that one by the surface this call measured, this one by the grid
  // it is meshing -- but called beside it, so both allocations land in the same
  // ExtractTimings row.
  Status ensure_block_spans(const volume::VoxelBlockGrid& grid);

  // Vertices to budget for a dispatch planned at @p triangle_capacity
  // triangles: the last extract's measured density, seeded when there is none.
  // Reads no slot, for the same reason plan_capacity does not -- a budget that
  // consulted the buffer it is about to grow ratchets across the output ring.
  std::uint32_t plan_vertex_capacity(std::uint32_t triangle_capacity) const;

  // The most triangles whose vertices fit one storage-buffer binding, under the
  // same density plan_vertex_capacity uses -- so plan_capacity can bound its
  // request by BOTH output buffers and never hand ensure_output_buffers a
  // vertex request it would reject rather than clamp.
  std::uint64_t triangles_fitting_arena() const;

  // Create this slot's draw command if it has none, and reset it to "draw
  // nothing yet": indexCount 0 for the kernel to accumulate into, and the four
  // fields that make the result a *drawable* command rather than a number a
  // host has to build one from. Split out of ensure_output_buffers because the
  // empty-active-set path needs the command without needing an arena.
  //
  // @p seed_triangles starts indexCount above zero, which is what makes an
  // incremental dispatch APPEND: the arena still holds that many triangles from
  // the last extract, and the kernel's atomic hands out slots past them. A
  // PARAMETER rather than a member the reset consumes: as a member it survived
  // every path that returned before the reset -- both of ensure_output_buffers'
  // range guards sit above it -- and a later dense or empty extract then
  // inherited a live indexCount over an arena it had rewritten, which is the
  // exact staleness the reset exists to prevent. Passed explicitly, there is
  // nothing to strand.
  //
  // @p seed_vertices does the same for the vertex counter in the scratch word
  // past the command, and is a separate number for the same reason
  // ArenaState::vertex_watermark is: sharing breaks v = 3t, so the vertices an
  // incremental pass appends past cannot be derived from the triangles. Only
  // the sharing kernel allocates through that counter, so every other caller
  // passes 0 and the word stays the plain zero it has always been.
  Status ensure_indirect_command(std::uint32_t seed_triangles,
                                 std::uint32_t seed_vertices);

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
  void disarm_indirect_command() noexcept;
};

}  // namespace volumetric_kit::recon::mesh
