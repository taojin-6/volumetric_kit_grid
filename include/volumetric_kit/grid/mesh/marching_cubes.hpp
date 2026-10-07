// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file mesh/marching_cubes.hpp
/// @brief GPU marching-cubes iso-surface extraction: owns the compute pipeline
///        and drives a GLSL kernel that turns a sparse voxel block grid into a
///        triangle @ref Mesh.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/mesh/device_mesh.hpp"
#include "volumetric_kit/recon/mesh/export.hpp"
#include "volumetric_kit/recon/mesh/mesh.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

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

/// @brief Where one extract call spent its time, and the sizes that explain it.
///
/// Opt-in and explicit: the caller passes one of these to an extract entry
/// point to have it filled, and `nullptr` (the default)
/// measures nothing. The tier keeps no profiler, no global sink, and no
/// timing state between calls -- a caller that wants a running view (the
/// viewer's overlay) aggregates these itself. Every field is **overwritten**
/// on each call, so one instance may be reused across frames without carrying
/// a previous call's numbers forward.
///
/// The spans are **wall-clock**, and the GPU ones are end-to-end: the dispatch
/// goes through `Device::submit_single_time`, which blocks on a fence, so
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
  /// Compacting the hash map's active block list on the device, near zero
  /// when the map's last device list still holds.
  double compact_ms = 0.0;
  /// With a caller's subset, allocating the active-block input buffer and
  /// staging the list, whose copy runs in @ref dispatch_ms's submit. Near zero
  /// otherwise: the device list is bound in place.
  double input_upload_ms = 0.0;
  /// Sizing the vertex arena + recording the draw command's reset, which runs
  /// in @ref dispatch_ms's submit, including a refit after an undersized guess
  /// (see @ref dispatches). A grow of the unshared index run submits its
  /// identity here, on its own. Near zero once the
  /// retained arena already fits the call -- the steady state, since the arena
  /// is reused across extracts (see @ref MarchingCubes).
  double arena_alloc_ms = 0.0;
  /// Writing the kernel's descriptor bindings.
  double descriptor_ms = 0.0;
  /// Each attempt's submit, including the blocking fence wait: a host active
  /// list's copy and the command reset, the marching-cubes dispatch, and the
  /// command's readback -- summed over both when a refit forced a second one
  /// (@ref dispatches).
  double dispatch_ms = 0.0;
  /// Getting the result back to the caller: the vertex copy into the host mesh
  /// when one is made. @ref MarchingCubes::extract_host makes one; @ref
  /// MarchingCubes::extract_device does not, and its command readback rides
  /// the dispatch's submit, so there this reads near zero.
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
  /// Triangle capacity the dispatch ran with -- one slot's index run, so
  /// `emitted_triangles / triangle_capacity` is that buffer's fill ratio.
  std::uint32_t triangle_capacity = 0;
  /// Triangles the kernel actually emitted.
  std::uint32_t emitted_triangles = 0;
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
/// The flags actually applied come back on `Device`Mesh::vertex_usage /
/// `Device`Mesh::index_usage, so a consumer verifies rather than assumes --
/// binding a buffer that lacks the bit is a validation-layer-only diagnostic.
///
/// @warning Whatever is passed here reaches `vkCreateBuffer` directly, so ask
///          only for bits the device supports. `SHADER_DEVICE_ADDRESS` is
///          rejected by @ref MarchingCubes::create (this repo's `Device`
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
// Placement: every buffer this tier hands out -- arena, index run and draw
// command -- is device-local, as a renderer binding them wants, and the host
// reaches them only through a CommandBatch (the 2026-09-28 residency
// decision).
struct MarchingCubesConfig {
  /// Added to the vertex arena's usage, beyond `STORAGE_BUFFER` and the
  /// transfer bits every buffer here carries.
  VkBufferUsageFlags extra_vertex_usage = 0;
  /// Added to the index run's usage, beyond the same.
  VkBufferUsageFlags extra_index_usage = 0;
  /// Added to the indirect command's usage, beyond the same and the
  /// `INDIRECT_BUFFER` a draw reads it as (unconditional -- 20 bytes, and a
  /// consumer that never draws indirectly pays a usage bit nobody reads).
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
  std::uint32_t queue_families[core::BufferDesc::kMaxQueueFamilies] = {};
  /// Entries in @ref queue_families; more than `kMaxQueueFamilies` distinct is
  /// rejected by @ref MarchingCubes::create.
  std::uint32_t queue_family_count = 0;

  /// @brief How many extracts may be outstanding at once.
  ///
  /// One (the default) is the behaviour this always had: a single grow-only
  /// arena reused in place, so a `Device`Mesh is valid only until the next
  /// extract and `Device`Mesh::generation is what enforces it.
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
  /// published as `Device`Mesh::shares_vertices; and
  /// @ref ExtractTimings::emitted_vertices stops being `3 * emitted_triangles`.
  ///
  /// @note Compatible with `texture::ProjectiveTexturer`'s single-camera
  ///       pass, which it was not until that pass moved to a per-*vertex*
  ///       dispatch. The
  ///       incompatibility was never really about sharing: the texturer decided
  ///       visibility per *triangle* and wrote @ref Vertex::uv0 per *vertex*,
  ///       so a vertex belonging to several triangles that disagreed was
  ///       written by whichever thread ran last. Every input to that verdict is
  ///       a property of the vertex alone, so the pass now dispatches one
  ///       thread per vertex -- one writer each, nothing to race -- and
  ///       textures a shared mesh. It also got cheaper doing it, since a
  ///       shared vertex used to be projected once per referencing triangle.
  ///
  ///       What it costs is the all-three-vertices gate: a triangle straddling
  ///       the visibility boundary is no longer refused whole, so the textured
  ///       region grows by up to one triangle at an occlusion silhouette. See
  ///       `texture::ProjectiveTexturer` for the encoding that bounds it.
  ///
  /// @note The texturer's several-view atlas is a different matter and
  ///       refuses a mesh built with this: it chooses a view per triangle, and
  ///       a triangle whose vertices index different tiles of the atlas cannot
  ///       be expressed per vertex under any encoding. A per-*primitive* tile
  ///       id would lift that. This is one reason the flag stays published on
  ///       `Device`Mesh, the other being a consumer that needs to know
  ///       whether `v = 3t` when it sizes an arena.
  ///
  /// @note Refused per *extract*, not at @ref MarchingCubes::create, for a grid
  ///       whose `voxels_per_block` exceeds 512 -- the sharing kernel's shared
  ///       per-cell table is sized for `block_size` 8, the only shape any
  ///       in-tree caller uses, and the block size arrives with the grid rather
  ///       than with this config. Every extract entry point reports it.
  bool share_vertices = false;
};

/// @brief Owns the marching-cubes compute pipelines and extracts an iso-surface
///        straight off a sparse @ref volume::VoxelBlockGrid -- into a host
///        @ref Mesh for export, or a device-resident `Device`Mesh for a
///        renderer to draw.
///
/// **Two workflows, and the entry point names say which one you are in.**
///
/// - **Export** -- @ref extract_host returns an owned host @ref Mesh, the
///   complete transaction: it extracts, downloads, and gives its ring slot
///   back. A caller writing a PLY or a glTF wants this and needs to know
///   nothing about the ring. The host copy is the cost, and it is inherent:
///   the vertices have to cross to system memory to be written to a file.
/// - **Live** -- @ref extract_device returns a borrowed `Device`Mesh the
///   renderer draws straight out of, with no host round trip, and the caller
///   releases slots by generation through @ref release_through as its frames
///   retire. The @ref volume::BlockList overload is a mode of this one, not
///   a separate workflow. @ref download bridges the two for a caller that
///   wants both.
///
/// The split is the destination, which is why the return types differ:
/// @ref Mesh owns its vertices, `Device`Mesh names buffers this extractor
/// owns and will overwrite on the next call.
///
/// Built on the `core` compute foundation (`Allocator`, `Buffer`,
/// `ComputeKernel`, `Device::submit_single_time`), mirroring the volume
/// tier's @ref volume::VoxelHashMap. The kernel runs one invocation per cell,
/// builds the cube index from the eight corner signs, and interpolates a vertex
/// on each crossed edge.
///
/// - The kernel runs one workgroup per active block, which
///   counts the block's output, reserves one range for all of it with a single
///   atomic, and only then writes, so a block's triangles land
///   **contiguously** in the arena. @ref MarchingCubesConfig::share_vertices
///   selects a kernel that reserves two ranges rather than one, since a shared
///   vertex breaks `v = 3t`.
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
///       the peak for this object's lifetime. An extract fits the
///       arena to what the surface actually emits rather than to the
///       5-triangles-per-cell ceiling it can never reach, so what stays
///       resident is the mesh's real size plus headroom. At
///       @ref MarchingCubesConfig::slot_count
///       above one there is one such arena *per slot*, each sized
///       independently. Destroy the extractor to release them; @ref
///       ExtractTimings::arena_bytes reports their total.
///
/// @warning The `Device` and `Allocator` passed to @ref create must
///          outlive this object; it stores references to them.
//
// TODO(mesh): shared-edge vertex dedup, so the index buffer stops being the
// identity run and the arena shrinks toward the unique-vertex count.

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
  /// @return The extractor, or a non-OK `Status` if @p config asks for an
  ///         unsupported usage bit, or a pipeline, layout, or descriptor
  ///         allocation fails.
  static core::Result<MarchingCubes> create(
      core::Device& device, core::Allocator& allocator,
      const MarchingCubesConfig& config = {});

  /// @brief Report that every mesh up to and including @p generation has been
  ///        read, so its slot may be written again.
  ///
  /// The consumer half of @ref MarchingCubesConfig::slot_count. Call it as the
  /// work reading a `Device`Mesh completes -- for a renderer, when the frame
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
  /// behaviour -- there, a `Device`Mesh still dies at the next extract.
  ///
  /// Being a single high-water mark is what shapes the consumer's side of the
  /// contract, so it is worth stating plainly: above one slot, **this** -- not
  /// `Device`Mesh::is_current -- is what bounds a view's life. A view stays
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
  ///          Calling it concurrently with an @ref extract_host or @ref
  ///          extract_device on the same object is a data race. Serialize it
  ///          with whatever already guards the handoff of a `Device`Mesh
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

  /// @brief Extract the @p iso iso-surface straight off a sparse
  ///        @ref volume::VoxelBlockGrid, meshing every active block.
  ///
  /// Runs one **workgroup** per active block, striding over the block's voxels,
  /// each voxel the base corner of one marching-cubes cell. A cell on a block's
  /// `+face` reaches its far corners into neighbouring blocks; the kernel
  /// resolves that 2x2x2 neighbourhood (this block plus its seven `+x/+y/+z`
  /// neighbours) **itself**, probing @p grid's hash table on-device, eight
  /// probes amortised over the block's cells. The per-cell body is identical to
  /// every other emitter here -- independent triangles, one gradient normal
  /// per cell, reversed winding, and the same hybrid @ref Vertex::color /
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
  ///         a non-OK `Status`: `Status::Code::InvalidArgument` for a
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
  ///          arena**, so any `Device`Mesh from an earlier extract on this
  ///          object is invalidated the moment it starts -- a failure is not a
  ///          rollback.
  core::Result<Mesh> extract_host(volume::VoxelBlockGrid& grid,
                                  float iso = 0.0f,
                                  ExtractTimings* timings = nullptr);

  /// @brief Extract as @ref extract_host does, but leave the result in this
  ///        extractor's device buffers instead of copying it to the host.
  ///
  /// The pass that consumes the mesh next -- `texture::ProjectiveTexturer`, or
  /// the renderer at the interop seam -- can bind these buffers directly, so
  /// the readback and the matching re-upload both disappear. Call @ref download
  /// when a host @ref Mesh is finally needed.
  ///
  /// That pair is **not** @ref extract_host, and substituting it leaks a ring
  /// slot: @ref extract_host also gives its slot back, which a caller cannot
  /// do for it (a `Result<Mesh>` carries no generation, and @ref
  /// release_through is the *consumer's* high-water mark, so calling it here
  /// would retire slots another `Device`Mesh is still drawn from). Every
  /// `Device`Mesh this hands out is the caller's to release.
  ///
  /// @param grid  As @ref extract_host.
  /// @param iso   As @ref extract_host.
  /// @param timings  As @ref extract_host, except
  ///                 @ref ExtractTimings::readback_ms reads near zero, since
  ///                 no vertex copy is made.
  /// @return A `Device`Mesh **borrowing** this extractor's buffers -- valid
  ///         only until the next extract on this object, which overwrites them
  ///         -- or the same failures @ref extract_host reports (including its
  ///         `Status::Code::OutOfMemory` case, and its warning that a failed
  ///         call still invalidates an earlier `Device`Mesh).
  core::Result<DeviceMesh> extract_device(volume::VoxelBlockGrid& grid,
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
  /// every phase that scales with the active set shrinks with it. The arena is
  /// rebuilt from this dispatch alone, so a block outside @p blocks contributes
  /// no triangles to the mesh and no *live bytes* to the arena -- which is what
  /// makes this worth doing on a memory-bound device, rather than only a
  /// cheaper dispatch.
  ///
  /// @note "No live bytes", not "no bytes". The arena is retained and
  ///       grow-only (see the @ref MarchingCubes note on the ring), so culling
  ///       lowers what the arena *holds*, never what it has already reserved:
  ///       one full extract -- a warm-up frame, or a pose that is not ready
  ///       yet -- sizes it for the whole active set and it stays that size.
  ///       Cull from the first extract to get the resident figure, and read
  ///       @ref ExtractTimings::arena_bytes rather than assuming it.
  ///
  /// @note Two @ref ExtractTimings rows do **not** shrink with the set, so the
  ///       cull will look partly ineffective if they are read as if they did:
  ///       @ref ExtractTimings::readback_ms reads near zero on this path
  ///       and @ref ExtractTimings::descriptor_ms is a fixed set of descriptor
  ///       writes, both per-call constants. What scales is the upload, the
  ///       dispatch, the arena, and whatever draws or textures the result.
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
  /// @warning @p blocks must not name the same block twice. Nothing checks it
  ///          -- a duplicate is something the caller can see and a set-wise
  ///          test is O(count) of host work per frame -- and the consequence is
  ///          silent: one workgroup is dispatched per entry, each reserves its
  ///          own arena range, so the block's surface is emitted twice
  ///          (coincident geometry, z-fighting, double the arena). The
  ///          internal producers cannot emit one; a caller unioning two
  ///          cameras' compactions must merge them first.
  ///
  /// @param grid     The sparse volume to mesh, as @ref extract_device takes
  ///                 it.
  /// @param iso      The iso-value to extract at (0 for a TSDF surface).
  /// @param blocks   The blocks to mesh: a subset of @p grid's active set,
  ///                 duplicate-free, and anchored to @p grid -- build it with
  ///                 @ref volume::VoxelBlockGrid::block_list rather than
  ///                 assembling the triple by hand. Read for the duration of
  ///                 this call and not retained. An empty list is legal and
  ///                 meshes nothing, exactly as an empty map does -- including
  ///                 a default-constructed one, which is exempt from the epoch
  ///                 check because it names no block.
  /// @param timings  As @ref extract_device, except @ref
  ///                 ExtractTimings::compact_ms is 0 and @ref
  ///                 ExtractTimings::active_blocks reports @p blocks's count --
  ///                 which is the instrument that says what the cull bought.
  /// @return The mesh in this extractor's device buffers, borrowed exactly as
  ///         @ref extract_device's is, or that overload's `Status`, plus
  ///         `Status::Code::InvalidArgument` when @p blocks is internally
  ///         inconsistent (a null pointer with a non-zero count), holds more
  ///         blocks than @p grid's heap has slots, or was compacted against a
  ///         topology @p grid has since left behind. The last is refused rather
  ///         than meshed: the block heap is LIFO, so a `remove()` since the
  ///         compaction has handed those block pointers to different blocks,
  ///         and the extract would silently mesh whatever voxels now live
  ///         there.
  ///
  ///         All three are checked before this call claims anything, so --
  ///         unlike the failures @ref extract_host's `@warning` describes --
  ///         they
  ///         **are** a rollback: no output slot is claimed, no generation is
  ///         bumped, and every outstanding `Device`Mesh stays exactly as
  ///         valid as it was. That matters because a consumer culling a frame
  ///         behind hits the epoch refusal on every frame after a `remove()`,
  ///         and having to re-extract and redraw on each one would cost more
  ///         than the cull saves.
  // TODO(mesh): no in-tree consumer culls yet, so the win is correct and
  // unquantified. fuse_viewer is the natural one and needs its render camera
  // published across the fusion-thread boundary; the scanner that motivates it
  // lives in volumetric_kit_ios.
  core::Result<DeviceMesh> extract_device(volume::VoxelBlockGrid& grid,
                                          float iso,
                                          const volume::BlockList& blocks,
                                          ExtractTimings* timings = nullptr);

  /// @brief Copy a `Device`Mesh's live vertices + indices into a host
  ///        @ref Mesh.
  /// @param device_mesh  A mesh from @ref extract_device on *this* extractor,
  ///                     not yet invalidated by a later extract.
  /// @return The host mesh, or `Status::Code::InvalidArgument` for a
  ///         moved-from extractor, or if @p device_mesh is not this extractor's
  ///         newest extract -- it was superseded by a later one, or came from a
  ///         different extractor. The currency check is by
  ///         `Device`Mesh::generation, not by buffer handle: with one slot
  ///         the arena is reused in place, so a superseded view names the same
  ///         `VkBuffer` and a handle comparison would accept it.
  core::Result<Mesh> download(const DeviceMesh& device_mesh) const;

  /// @return `true` if this owns a live kernel (`false` when moved-from).
  bool valid() const noexcept { return kernel_sparse_.valid(); }

 private:
  MarchingCubes() = default;

  // Borrowed (must outlive this). Pointers, not references, so a moved-from
  // extractor is left in a defined (empty) state.
  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;

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
  // of the one kernel this extractor built, for every extract (the counterpart
  // to the volume tier's persistent bindings). The input buffers are
  // per-extract; the vertex arena and counter are retained (below). All of them
  // are (re)written into the remaining bindings before a dispatch, so a regrown
  // arena's new handle is always the one bound.
  core::Buffer tables_;
  // A 1-element dummy bound to the sparse kernel's color slot when a grid
  // carries no `color` attribute, so that descriptor stays valid (the has_color
  // push flag tells the kernel to ignore it). Mirrors the tsdf integrator's
  // color dummy.
  core::Buffer color_dummy_;

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
    core::Buffer arena;
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
    core::Buffer index_run;
    // The `VkDrawIndexedIndirectCommand` this slot's draw is issued from, and
    // the atomic the kernel counts into: `indexCount` is field 0, so the two
    // are the same 20 bytes rather than a counter plus a command built from it.
    // Per slot, not shared, because it *is* part of the mesh -- a renderer
    // reading slot N's command while N+1 is extracted is the whole point of the
    // ring.
    core::Buffer indirect;
    // Whether @ref indirect holds the empty command an empty extract resets it
    // to, so a run of empty extracts submits that reset once.
    bool command_empty = false;
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
  // pass valid() and then index nothing. An array of members that each
  // survive makes the aggregate survive too -- which is the rule every member
  // added here has to be checked against.
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

  // Name one of the current slot's buffers for a GPU capture: `form` is a
  // printf form taking the slot index, as "mesh.arena[%u]". Called wherever
  // that buffer is (re)created -- a name lives on the handle, so a grown arena
  // is anonymous until renamed -- and one buffer at a time rather than all
  // three, so a grow does not re-state the names of the buffers it left alone.
  // A no-op where the device resolved no debug-utils entry points.
  void name_slot_buffer(const core::Buffer& buffer,
                        const char* form) const noexcept;

  core::Buffer& arena() noexcept { return slots_[slot_].arena; }
  const core::Buffer& arena() const noexcept { return slots_[slot_].arena; }
  core::Buffer& index_run() noexcept { return slots_[slot_].index_run; }
  const core::Buffer& index_run() const noexcept {
    return slots_[slot_].index_run;
  }
  core::Buffer& indirect() noexcept { return slots_[slot_].indirect; }
  const core::Buffer& indirect() const noexcept {
    return slots_[slot_].indirect;
  }

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

  // The marching-cubes kernel over a sparse VoxelBlockGrid: its
  // descriptor-set layout, pipeline, and a set allocated from the shared pool_
  // (see `ComputeKernel`). Which of the two sparse variants it holds is
  // fixed at create by MarchingCubesConfig::share_vertices.
  core::ComputeKernel kernel_sparse_;
  core::DescriptorPool pool_;

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
  std::uint64_t resident_output_bytes() const noexcept {
    std::uint64_t total = 0;
    for (std::size_t i = 0; i < slot_count_; ++i)
      total += slots_[i].arena.size() + slots_[i].index_run.size();
    return total;
  }

  // The one sparse extract, with the public entry points differing only in
  // whether they hand it a caller's active set.
  //
  // @p blocks is null when this call compacts the whole active set itself (on
  // the device), and points at the caller's subset otherwise -- borrowed for
  // this call alone, and a parameter rather than a member: it is a bare host
  // pointer into a std::vector the caller owns, so latching it on the
  // extractor would leave a dangling read for the NEXT extract.
  //
  // @p entry is the public name to report failures under -- the only thing
  // about the caller this function keeps, and it keeps it because a diagnostic
  // that names a method the header does not declare leaves a user with nothing
  // to grep. See kEntryHost in the .cpp.
  core::Result<DeviceMesh> extract_device_impl(volume::VoxelBlockGrid& grid,
                                               float iso,
                                               const volume::BlockList* blocks,
                                               ExtractTimings* timings,
                                               const char* entry);

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
  core::Status claim_output_slot(const char* entry);

  // Give back the slot stamped with @p generation without moving
  // released_through_. extract_host's answer to "this call published no
  // DeviceMesh, so nothing outside it can release the slot":
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
  // The reset is recorded into @p batch, ahead of the dispatch that batch will
  // run. The identity index run a grow fills is submitted on its own, before
  // the run is committed.
  core::Status ensure_output_buffers(core::CommandBatch& batch,
                                     std::uint32_t triangle_capacity,
                                     std::uint32_t vertex_capacity,
                                     const char* entry);

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
  // The reset is recorded into @p batch rather than written.
  core::Status ensure_indirect_command(core::CommandBatch& batch);

  // Bound the command's indexCount by what the arena can actually hold.
  //
  // The kernel counts triangles it *dropped* as well as ones it wrote -- that
  // over-count is deliberate, and is what the host refits from -- so between
  // an undersized dispatch and its retry the command names more indices than
  // exist. Every path that returns while that is still true must call this, or
  // it hands back a command that reads past the end of both the arena and the
  // index run. The success path is already in range by construction (the loop
  // exits only when the count fits), so this is the *failure* paths' guarantee,
  // not theirs. Its own submit, best effort, since its callers are already
  // failing.
  void disarm_indirect_command();
};

}  // namespace volumetric_kit::recon::mesh
