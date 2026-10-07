// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file volume/voxel_hash_map.hpp
/// @brief The sparse voxel hash map on the GPU: owns the device buffers and the
///        compute pipelines, allocates blocks, and compacts the active set.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/gpu_timer.hpp"
#include "volumetric_kit/recon/core/camera_params.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/volume/export.hpp"
#include "volumetric_kit/recon/volume/frustum.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/voxel_grid.hpp"

namespace volumetric_kit::recon::volume {

/// @brief On-demand occupancy + health statistics for a @ref VoxelHashMap,
///        computed by @ref VoxelHashMap::diagnostics.
struct HashDiagnostics {
  std::int32_t active_count = 0;      ///< Occupied hash slots.
  std::int32_t overflow_count = 0;    ///< Active slots in collision chains.
  std::int32_t max_chain_length = 0;  ///< Longest collision chain, in hops.
  std::int32_t heap_free_count = 0;   ///< Free blocks left on the heap.
  std::int32_t total_blocks = 0;      ///< Block capacity (grid `num_blocks`).
  float load_factor = 0.0f;           ///< `active_count / total slots`.
  float heap_utilization = 0.0f;      ///< `1 - heap_free / total_blocks`.
};

/// @brief Why a dispatch's failures failed -- the per-reason split behind the
///        single count @ref VoxelHashMap::allocate and friends return.
///
/// Opt-in and caller-owned (pass `nullptr`, the default, and nothing is
/// written) -- the shape @ref mesh::ExtractTimings established, for the same
/// reason: the distinction is invisible from outside and choosing correctly
/// needs it.
///
/// The choice it exists for is **grow or retry**. @ref lock is transient
/// same-bucket contention (a GPU spin-lock livelock within a SIMD group, as
/// when many new blocks land in a few buckets at once) and says nothing
/// about capacity: the table may be nearly empty. @ref chain and @ref heap are
/// genuine capacity limits and are what @ref VoxelHashMap::resize answers.
/// Reading the aggregate as capacity pressure grows the volume -- doubling
/// every attribute array -- over a table that was never full.
struct AllocFailures {
  std::uint32_t total = 0;  ///< Retryable failures the final round reported.
  std::uint32_t lock = 0;   ///< Lost bucket-lock races: contention, not size.
  std::uint32_t chain = 0;  ///< Collision chain full: a capacity limit.
  std::uint32_t heap = 0;   ///< Block heap empty: a capacity limit.
  /// No free non-anchor slot anywhere in the table: a capacity limit, and a
  /// distinct one. Kept apart from @ref heap because the two are not
  /// interchangeable -- the rehash path presets each block's pointer and never
  /// touches the heap, so @ref heap is *provably impossible* there and
  /// reporting it would name a cause that cannot have occurred.
  ///
  /// "Anywhere" is literal: the overflow scan is uncapped, and a sweep that
  /// skipped a free-looking slot because another thread held its bucket reports
  /// @ref lock instead. So a count here is proof the table is out of usable
  /// slots, never a merely *clustered* table mistaken for a full one -- which
  /// matters because the caller's answer to it is to grow, doubling every
  /// attribute array.
  std::uint32_t table = 0;
  /// Non-retryable failures, summed over every round. Reported only by
  /// @ref VoxelHashMap::remove, where a block index that could not be returned
  /// to the free heap is unreachable capacity: neither in the table nor on the
  /// heap. Not resolved by retrying and not fixed by @ref VoxelHashMap::resize.
  std::uint32_t terminal = 0;

  /// @return Whether growing the map could help: a genuine capacity limit was
  ///         hit, rather than only transient lock contention.
  bool capacity_limited() const noexcept {
    return chain > 0 || heap > 0 || table > 0;
  }
};

/// @brief One camera's depth frame, for @ref VoxelHashMap::allocate_from_depth
///        over several cameras at once.
struct DepthInput {
  /// Row-major depth in **metres**, `camera.width * camera.height` samples:
  /// a host array, staged in the call's first round, or a storage buffer,
  /// bound in place.
  core::StorageInput depth;
  DepthCameraParams camera;  ///< Intrinsics, depth range, size and pose.
};

/// @brief A compacted active set on the device, or a frustum-culled subset of
///        it: @ref count @ref BlockIndex entries at the start of @ref buffer,
///        bound by their exact range.
///
/// Borrowed from the map that compacted it, and stamped so that map can say
/// whether it still holds: @ref VoxelHashMap::check_device_block_list for the
/// active set, @ref VoxelHashMap::check_device_block_subset for either kind.
struct DeviceBlockList {
  const core::Buffer* buffer = nullptr;  ///< The map's own list; borrowed.
  std::uint32_t count = 0;               ///< Entries in it.
  /// The map's @ref VoxelHashMap::topology_epoch when it was compacted.
  std::uint64_t epoch = 0;
  /// Which of the map's compactions wrote it.
  std::uint64_t serial = 0;
  /// The map's free blocks then, which an allocation since moves.
  std::uint32_t heap_free = 0;
};

/// @brief Owns the device-side sparse voxel hash table -- the hash-entry index,
///        the free-block heap, and the per-bucket locks -- plus the compute
///        pipelines that operate on them, and drives block allocation and
///        active-block compaction.
///
/// Built on the `core` compute foundation (`Allocator`, `Buffer`,
/// `ComputePipeline`, `Device::submit_single_time`). The GLSL kernels
/// read the hash structs through scalar block layout (the 2026-07-05 ABI), so
/// the host @ref HashEntry / @ref BlockIndex and their shader mirrors agree
/// byte-for-byte. Covers init, allocate-from-coords / -depth / -triangles,
/// remove, compact / compact-in-frusta, diagnostics, and an
/// **index-preserving** @ref resize (the GPU rehash that keeps each block's
/// `ptr`).
///
/// @warning The `Device` and `Allocator` passed to @ref create must
///          outlive this object; it stores references to them.
class VR_VOLUME_API VoxelHashMap {
 public:
  /// @brief Create the hash map on @p device with the given grid, allocating
  /// its
  ///        buffers from @p allocator and running the init kernel.
  /// @param device     The compute device (must outlive this object).
  /// @param allocator  The allocator its buffers come from (must outlive this).
  /// @param grid       The grid resolution + hash-table shape.
  /// @return The hash map, or a non-OK `Status` if a buffer, pipeline, or
  /// the
  ///         init dispatch fails; `Status::Code::InvalidArgument` for a grid
  ///         that fails @ref VoxelGridParams::validate.
  static core::Result<VoxelHashMap> create(core::Device& device,
                                           core::Allocator& allocator,
                                           const VoxelGridParams& grid);

  // Rule of zero for the owned members: every Buffer / pipeline / layout / pool
  // self-frees and self-resets on move, so the defaulted moves are correct and
  // move-only follows from those members. device_ / allocator_ are borrowed
  // (non-owning), so a defaulted move leaving the moved-from map pointing at
  // them is harmless -- that map reports valid() == false and is only
  // destroyed.
  ~VoxelHashMap() = default;
  VoxelHashMap(VoxelHashMap&&) noexcept = default;
  VoxelHashMap& operator=(VoxelHashMap&&) noexcept = default;
  VoxelHashMap(const VoxelHashMap&) = delete;
  VoxelHashMap& operator=(const VoxelHashMap&) = delete;

  /// @brief Allocate voxel blocks at the given block coordinates (the `ptr`
  ///        field of each @ref BlockIndex is ignored -- only `coord` is read).
  /// @param coords  The block coordinates to insert.
  /// @param count   How many.
  /// @return The number of allocations that failed (0 = all succeeded), or a
  ///         non-OK `Status` if a buffer or the dispatch fails. Allocation
  ///         re-dispatches to converge under contention; a non-zero count means
  ///         a genuine capacity limit (chain full / heap empty) -- grow with
  ///         @ref resize -- but read @p out_failures rather than assuming that,
  ///         since transient lock contention can also leave a residue.
  /// @param out_failures  Optional: receives the per-reason split (see
  ///                      @ref AllocFailures). Untouched when null.
  core::Result<std::uint32_t> allocate(const BlockIndex* coords,
                                       std::uint32_t count,
                                       AllocFailures* out_failures = nullptr);

  /// @brief Allocate voxel blocks from a posed depth frame.
  ///
  /// Each pixel's depth sample is unprojected (via @p camera's pinhole
  /// intrinsics + pose) to a world point and the block containing it, and
  /// each distinct block of a 16 x 16 pixel tile is dilated once into the
  /// surrounding `(2*tb+1)^3` truncation band the TSDF integrates
  /// (`tb` = @ref truncation_blocks). Out-of-range
  /// (`< min_depth` / `> max_depth`) and non-finite samples are skipped;
  /// already-present blocks are left untouched, so overlapping bands merge and
  /// re-running the same frame allocates nothing new.
  /// @param depth   Row-major depth image in **metres**, `width * height`
  ///                samples (the caller applies any raw sensor depth-scale).
  /// @param camera  Intrinsics, valid-depth range, dimensions, and pose.
  /// @param out_failures  Optional: receives the per-reason split. **Consult it
  ///                      before growing the map.** A non-zero count does *not*
  ///                      by itself mean bucket/heap pressure -- a frame of
  ///                      mostly new blocks (the first, or a fast pan) makes
  ///                      many of them at once, so a residue of pure
  ///                      @ref AllocFailures::lock failures is possible on a
  ///                      table with ample room. @ref
  ///                      AllocFailures::capacity_limited is the test @ref
  ///                      resize answers.
  /// @param metrics  Optional `StageMetrics` collecting an `"allocate"` host
  ///                  row and, from timestamp spans around the dispatches, its
  ///                  device half. `nullptr` measures nothing. The retry loop
  ///                  contributes one span per round and they **accumulate**
  ///                  under one name, which is the honest total: a contended
  ///                  frame genuinely dispatches several times, and reporting
  ///                  only the last would hide exactly the cost that makes
  ///                  contention worth seeing.
  ///
  ///                  Because rows accumulate by name, a caller that *also*
  ///                  wraps this call in a `StageScope("allocate")` of its own
  ///                  counts the host span twice. Wrap only what this does not
  ///                  cover -- a @ref resize between retries, say -- and give
  ///                  that its own row.
  /// @return The number of block allocations that failed (0 = all succeeded),
  ///         or a non-OK `Status` if a buffer or the dispatch fails, or the
  ///         map is moved-from / @p depth is null.
  core::Result<std::uint32_t> allocate_from_depth(
      const float* depth, const DepthCameraParams& camera,
      AllocFailures* out_failures = nullptr,
      core::StageMetrics* metrics = nullptr);

  /// @brief @ref allocate_from_depth from a depth image already on the device.
  ///
  /// The same allocation, but @p depth is bound where it lives -- a GPU
  /// pre-processing pass's output -- rather than uploaded from the host.
  /// @param depth  A storage buffer holding the image in metres, at least
  ///               `camera.width * camera.height` floats, row-major. Borrowed
  ///               for the call; the writer's dispatch must have finished,
  ///               which a `dispatch` on this device guarantees.
  /// @return As the host overload; `Status::Code::InvalidArgument` also for
  ///         a @p depth that is empty, not a storage buffer, has unknown or
  ///         non-device-local memory, or is smaller than
  ///         the image.
  core::Result<std::uint32_t> allocate_from_depth(
      const core::Buffer& depth, const DepthCameraParams& camera,
      AllocFailures* out_failures = nullptr,
      core::StageMetrics* metrics = nullptr);

  /// @brief @ref allocate_from_depth for several cameras' frames at once: one
  ///        submit a round rather than one a frame.
  ///
  /// Each round dispatches every frame, in order, in one batch, and reads the
  /// tally back once. A retry round dispatches them all again, the blocks
  /// already present taking the lookup's lock-free path. The rounds are the
  /// call's, not each frame's: a set has the retry budget one frame has.
  /// Once every frame's blocks are in, they are the blocks allocating the
  /// frames one after another gives, each perhaps in another slot; short of
  /// that, a capacity limit can leave other frames' blocks out.
  /// @param frames        The frames. Every one is checked, as the
  ///                      one-frame overloads check theirs, before any work;
  ///                      none allocates nothing.
  /// @param out_failures  As @ref allocate_from_depth, over every frame.
  /// @param metrics       As @ref allocate_from_depth: one `"allocate"` row,
  ///                      a device span per frame per round.
  /// @return As @ref allocate_from_depth, the failures summed over the frames.
  core::Result<std::uint32_t> allocate_from_depth(
      const std::vector<DepthInput>& frames,
      AllocFailures* out_failures = nullptr,
      core::StageMetrics* metrics = nullptr);

  /// @brief Allocate the voxel blocks a triangle mesh's truncation band covers.
  ///
  /// A block is allocated when its centre lies within `trunc_dist` plus the
  /// block's half-diagonal of some triangle -- conservative, so a block holding
  /// **any** voxel within `trunc_dist` of the surface is never missed, which is
  /// exactly the set a mesh-to-SDF pass then writes.
  ///
  /// Not expressible as dilating each vertex into the `(2*tb+1)^3` band, as
  /// @ref allocate_from_depth dilates a point: a triangle wider than that band
  /// would leave an unallocated hole through its middle -- and the band is
  /// one block (40 mm) with 5 mm voxels and a 40 mm band, which any mesh that
  /// is not a dense scan exceeds routinely. Nor is it a per-triangle dispatch:
  /// triangle size is unbounded, and one lane owning a large triangle's whole
  /// bounding box is the dispatch shape that hangs a mobile GPU. The work is
  /// therefore split per *candidate block*, which costs a host pass over the
  /// triangles to count them (`StageMetrics` reports it in the row's CPU half).
  ///
  /// A triangle is skipped, costing nothing, when it holds a non-finite vertex
  /// or has zero area; a zero-area triangle is dropped here rather than guarded
  /// in the kernel because it is the one input that makes the closest-point
  /// solve divide by zero. Already-present blocks are untouched, so overlapping
  /// triangles de-duplicate at the table.
  /// @param vertices      World-space vertex positions, metres.
  /// @param vertex_count  How many @p vertices.
  /// @param indices       `3 * triangle_count` indices into @p vertices.
  /// @param triangle_count  How many triangles.
  /// @param out_failures  Optional: receives the per-reason split (see
  ///                      @ref AllocFailures). Untouched when null.
  /// @param metrics       Optional: receives an `"allocate"` row, as
  ///                      @ref allocate_from_depth.
  /// @return The number of block allocations that failed (0 = all succeeded),
  ///         or a non-OK `Status`: `Status::Code::InvalidArgument` for a
  ///         moved-from map, a null @p vertices / @p indices, an index at or
  ///         past @p vertex_count, or a candidate-block total past 2^32 (a
  ///         mesh grossly mis-scaled against the grid -- metres read as
  ///         millimetres, say); or whatever a buffer or the dispatch returns.
  core::Result<std::uint32_t> allocate_from_triangles(
      const Vec3f* vertices, std::uint32_t vertex_count,
      const std::uint32_t* indices, std::uint32_t triangle_count,
      AllocFailures* out_failures = nullptr,
      core::StageMetrics* metrics = nullptr);

  /// @brief Remove voxel blocks at the given block coordinates (only `coord` is
  ///        read); absent coordinates are ignored. Returns each freed block to
  ///        the heap.
  ///
  /// Must not run concurrently with @ref allocate — the heap requires alloc and
  /// free in separate dispatches, which the fence between calls guarantees for
  /// single-threaded use.
  /// @param coords  The block coordinates to remove.
  /// @param count   How many.
  /// @warning This frees the block index but does **not** clear the per-voxel
  ///          attribute data it addressed -- the block index does not know what
  ///          attributes, if any, a consumer declared. The free heap is LIFO,
  ///          so the next allocation re-draws this very index onto the same
  ///          attribute range and the old surface's `tsdf`/`weight`/`color`
  ///          resurrect under the new geometry. Use
  ///          @ref VoxelBlockGrid::remove, which clears them first, on any grid
  ///          that carries attributes.
  /// @return The number of removals that failed (0 = all done), or a non-OK
  ///         `Status` if a buffer or the dispatch fails. A non-zero count
  ///         here is **not** capacity pressure. It is the sum of blocks
  ///         removed from the table whose index the free heap refused, i.e.
  ///         leaked capacity (@ref AllocFailures::terminal), and coords
  ///         bucket-lock contention kept through every round
  ///         (@ref AllocFailures::lock), which are still in the table: call
  ///         again to remove them.
  /// @param out_failures  Optional: receives the per-reason split (see
  ///                      @ref AllocFailures). Untouched when null.
  core::Result<std::uint32_t> remove(const BlockIndex* coords,
                                     std::uint32_t count,
                                     AllocFailures* out_failures = nullptr);

  /// @brief Remove blocks from a device buffer of @ref BlockIndex without
  ///        downloading or uploading the coordinate list.
  ///
  /// As with the host overload, only coordinates are read, absent coordinates
  /// are ignored, and attributes are not cleared. The caller must zero any
  /// attributes before removing their blocks. The buffer must remain alive
  /// and unchanged until this synchronous call returns.
  /// @param coords  Storage buffer with known device-local memory holding at
  ///                least @p count block indices.
  /// @param count   Number of entries; zero is a no-op.
  /// @param out_failures  Optional per-reason failure counts.
  /// @return The number of failed removals, or a non-OK `Status` for an
  ///         invalid input, allocation failure, or failed dispatch. An
  ///         empty, refused or zero-count call leaves @ref topology_epoch
  ///         unchanged.
  core::Result<std::uint32_t> remove(const core::Buffer& coords,
                                     std::uint32_t count,
                                     AllocFailures* out_failures = nullptr);

  /// @brief Compact every active block into a host vector of @ref BlockIndex.
  /// @param metrics  Optional `StageMetrics` collecting an `"active set"`
  ///                 row -- host and device. Named with @ref
  ///                 StageMetrics::kBreakdownPrefix when a `StageScope` is
  ///                 already open on @p metrics and plainly when it is not,
  ///                 because which of the two it is depends on the caller, not
  ///                 on this operation: `tsdf`'s `"integrate"` host row wraps
  ///                 this dispatch, so as a sub-row of that its host half must
  ///                 not be summed again (and without the sub-row its device
  ///                 time would be invisible, leaving the gap between
  ///                 `integrate`'s halves reading as submit overhead when it is
  ///                 a second kernel) -- while a caller that compacts at top
  ///                 level is asking for a stage, and a prefixed row there
  ///                 would be left out of `StageMetrics::total_cpu_ms`
  ///                 entirely.
  /// @return The active blocks (order unspecified), or a non-OK `Status`.
  core::Result<std::vector<BlockIndex>> compact_active_blocks(
      core::StageMetrics* metrics = nullptr);

  /// @brief @ref compact_active_blocks, with the list left on the device for a
  ///        kernel to read in place; only its count reaches the host.
  ///
  /// While the last call's list still holds, this returns it, dispatching and
  /// reporting nothing, so the encoder after a mesh extract reuses the
  /// extract's compaction.
  /// @param metrics  As @ref compact_active_blocks, under the same row name.
  /// @return The list, which @ref check_device_block_list accepts until this
  ///         map compacts another, allocates a block, resizes, removes,
  ///         clears or moves; or a non-OK `Status`.
  core::Result<DeviceBlockList> compact_active_blocks_on_device(
      core::StageMetrics* metrics = nullptr);

  /// @brief Whether @p list is still this map's active set, for a consumer
  ///        that needs every block to ask before it binds the list.
  /// @param list  From @ref compact_active_blocks_on_device. An empty one is
  ///              always accepted: it names no block.
  /// @param who   The caller, for the message.
  /// @return OK; or `Status::Code::InvalidArgument` as
  ///         @ref check_device_block_subset refuses, and for a list from
  ///         @ref compact_active_blocks_in_frusta_on_device, which is a subset
  ///         and never the active set.
  core::Status check_device_block_list(const DeviceBlockList& list,
                                       const char* who) const;

  /// @brief Whether @p list still names blocks of this map -- its active set
  ///        or a frustum-culled subset of it -- for a consumer that meshes or
  ///        reads only the blocks listed.
  /// @param list  From @ref compact_active_blocks_on_device or
  ///              @ref compact_active_blocks_in_frusta_on_device. An empty one
  ///              is always accepted: it names no block.
  /// @param who   The caller, for the message.
  /// @return OK; or `Status::Code::InvalidArgument` when this map is
  ///         moved-from, did not compact @p list, has moved since, or has
  ///         since rewritten the list (another compaction of the same kind, or
  ///         a resize), allocated, removed or cleared.
  core::Status check_device_block_subset(const DeviceBlockList& list,
                                         const char* who) const;

  /// @brief Compact the active blocks inside **any** of @p frusta, leaving the
  ///        list on the device; only its count reaches the host.
  ///
  /// One scan of the table: each block's world AABB -- its voxels and the
  /// marching-cubes cells it meshes, which reach a half-voxel past them -- is
  /// tested against each frustum's six planes and kept unless fully outside
  /// one of them (a conservative p-vertex test; see @ref make_frustum_planes),
  /// so a block
  /// several frusta keep is listed once. `tsdf::TsdfIntegrator` fuses the
  /// blocks its cameras reach this way, and `mesh::MarchingCubes` meshes the
  /// blocks a viewing camera sees.
  ///
  /// The list has a buffer of its own: it leaves
  /// @ref compact_active_blocks_on_device's list holding, and
  /// @ref check_device_block_list refuses it, since it is not the active set.
  /// @ref check_device_block_subset accepts it until the next frustum
  /// compaction rewrites it, or the map allocates, resizes, removes, clears or
  /// moves. Nothing is cached.
  /// @param frusta   The frusta; none keeps no block.
  /// @param metrics  As @ref compact_active_blocks, under the same row name.
  /// @return The list, or a non-OK `Status`: `Status::Code::InvalidArgument`
  ///         for a moved-from map or more frusta than one storage-buffer
  ///         binding holds; otherwise a buffer or dispatch failure.
  core::Result<DeviceBlockList> compact_active_blocks_in_frusta_on_device(
      const std::vector<FrustumPlanes>& frusta,
      core::StageMetrics* metrics = nullptr);

  /// @brief Reset the table to empty (re-runs the init kernel).
  /// @return An OK `Status`, or a non-OK one if the init dispatch fails or
  ///         the map is moved-from.
  core::Status clear();

  /// @brief Grow the hash table to @p new_num_buckets buckets (must exceed the
  ///        current count), **preserving each block's index** so per-voxel data
  ///        keyed by @ref BlockIndex::ptr survives.
  ///
  /// Snapshot the active blocks (coordinate + pointer), grow the buffers,
  /// re-init the larger table, then rehash: the @ref rehash_ kernel re-inserts
  /// each block with its *original* pointer (not a fresh heap draw), and the
  /// heap is rebuilt to hold exactly the block indices the snapshot does not
  /// occupy. A block thus keeps its `ptr`, so a @ref VoxelBlockGrid's attribute
  /// arrays (addressed by `ptr`) stay valid across the grow -- but those arrays
  /// are sized for the old `num_blocks` and are **not** grown here; resize a
  /// grid that carries attributes through @ref VoxelBlockGrid::resize, which
  /// grows them first.
  /// @param new_num_buckets  The new bucket count (> the current @ref grid).
  /// @return OK on success, or a non-OK `Status`:
  ///         `Status::Code::InvalidArgument` for a non-growing count;
  ///         `Status::Code::OutOfMemory` if the re-insert overflows.
  core::Status resize(std::int32_t new_num_buckets);

  /// @brief Read the raw hash-entry slots back to the host.
  ///
  /// Low-level / diagnostic: exposes the on-device @ref HashEntry array (all
  /// slots, including free ones) so callers can inspect the table or verify the
  /// host<->shader layout. @ref compact_active_blocks is the normal way to get
  /// the active set.
  /// @return All hash-entry slots (length `num_buckets * bucket_size`), or a
  ///         non-OK `Status` if the map is moved-from.
  core::Result<std::vector<HashEntry>> read_entries();

  /// @brief The device hash-entry array, for a kernel that resolves blocks by
  ///        coordinate itself instead of being handed a host-built table.
  ///
  /// Published because the alternative is worse. A consumer that needs the
  /// neighbourhood of N blocks otherwise pays an O(N) serial host pass of
  /// coordinate lookups plus an upload -- 102 ms of a 133 ms mesh extract at
  /// 107k blocks, measured on an M5 iPad Pro -- to hand the GPU a table it
  /// could have built itself in parallel. `volume/shaders/hash_lookup.glsl` is
  /// the read-only traversal to bind this with; it mirrors `find_block` and
  /// takes the table shape (`num_buckets` / `bucket_size` / `max_chain`) as
  /// arguments.
  ///
  /// @warning Read-only, and only safe while **nothing is mutating the table**
  /// --
  ///          no allocate / remove / clear / resize dispatch in flight. Probing
  ///          concurrently with an insert would race the bucket locks this
  ///          accessor deliberately does not expose. A meshing pass qualifies
  ///          because it is quiescent by construction; a pass that also
  ///          allocates does not.
  /// @warning **Do not cache the handle.** A completed @ref resize replaces the
  ///          entry buffer outright -- the old allocation is destroyed, so a
  ///          handle held in a persistent descriptor set then names freed
  ///          memory, which is a validation-layer-only diagnostic and undefined
  ///          with layers off. Both shipped examples resize mid-scan on
  ///          block-heap overflow. Re-fetch on every use, exactly as @ref
  ///          AttributeView requires across a @ref VoxelBlockGrid::resize; a
  ///          move of the map has the same effect.
  /// @return The entry buffer, or `VK_NULL_HANDLE` on a moved-from map.
  VkBuffer entries_buffer() const noexcept;

  /// @brief Bytes in @ref entries_buffer, for a `VK_WHOLE_SIZE`-free binding.
  ///
  /// Pair it with @ref entries_buffer: bind the range rather than
  /// `VK_WHOLE_SIZE` so the consumer can check it against the device's
  /// `maxStorageBufferRange` first. That limit matters here -- the table is
  /// `num_buckets * bucket_size * sizeof(HashEntry)` and @ref resize doubles
  /// `num_buckets`, while Vulkan guarantees only 2^27 (128 MiB), which is what
  /// Android-class drivers report and no driver this repo tests on does.
  /// @return The size in bytes, or 0 on a moved-from map (so the pair stays
  ///         consistent: a null handle never carries a non-zero range).
  VkDeviceSize entries_buffer_size() const noexcept;

  /// @brief The map's occupancy as a fraction of its block capacity -- a
  ///        constant-time read, safe to call every frame.
  ///
  /// The complement of the free-block heap, `1 - heap_free / num_blocks`, taken
  /// from the host's copy of the heap counter, which every call that moves the
  /// counter reads back in its own submit -- so this costs no dispatch. It
  /// is also the fraction of hash slots in use -- the same quantity
  /// @ref HashDiagnostics::load_factor reports, derived from the heap counter
  /// instead of a slot scan -- because @ref VoxelGridParams::validate forces
  /// `num_blocks == bucket_size * num_buckets` and every occupied slot holds
  /// exactly one heap block.
  ///
  /// This is what lets a caller **grow before it fails**. @ref diagnostics
  /// reports the same number alongside chain health, but scans every slot on
  /// the host (1.5M at the example defaults) and so cannot run per frame; and
  /// growing only once @ref AllocFailures::capacity_limited fires means the map
  /// necessarily spends time at the occupancy where collision chains are
  /// longest and every insert is slowest. Grow at @ref kGrowThreshold rather
  /// than at the cliff.
  /// @return The occupancy in `[0, 1]`, or a non-OK `Status` if the map is
  ///         moved-from.
  core::Result<float> load_factor() const;

  /// @brief The @ref load_factor a caller should grow at rather than run past.
  ///
  /// Linear probing degrades sharply beyond this: each insert walks a longer
  /// chain, and `allocate_from_depth`'s overflow scan -- exhaustive by design,
  /// see the 2026-08-08 decision -- pays a contended atomic for every slot it
  /// walks. Well under 1.0 on purpose; the point is to leave the band where
  /// the map is both slowest and likeliest to fail, not to sit at its edge.
  ///
  /// A named constant here rather than a number each caller picks, because it
  /// is a property of this table: a UI drawing its own ceiling, or an embedder
  /// refusing to allocate past one, otherwise ends up disagreeing with the
  /// guidance @ref load_factor gives right above it -- and neither side is
  /// obviously the wrong one to a reader who sees only one.
  static constexpr float kGrowThreshold = 0.7f;

  /// @brief Compute occupancy + health statistics (active / overflow / chain
  ///        length + heap utilization).
  ///
  /// A host-side scan of the entries, read back whole, plus the device's heap
  /// counter -- O(total slots), so call it for inspection/logging, not per
  /// frame. A GPU-side scan is a perf follow-up for very large tables.
  /// @return The statistics, or a non-OK `Status` (e.g. moved-from map).
  core::Result<HashDiagnostics> diagnostics();

  /// @brief A token identifying this table's *current* block-index assignment:
  ///        it changes whenever a block stops being live (@ref remove, @ref
  ///        clear).
  ///
  /// Exists so a consumer that caches something keyed by block slot can ask
  /// whether that cache still describes this table, which it otherwise cannot:
  /// a removed block's index goes back to a LIFO heap and is re-drawn by the
  /// next allocation, so the same slot silently comes to mean a different block
  /// at a different coordinate. A compacted @ref BlockList is such a cache. The
  /// block stamps are not, since the map zeroes a slot's record as it frees the
  /// block.
  ///
  /// It lives *here*, on the table that hands block indices out and takes them
  /// back, rather than on @ref VoxelBlockGrid -- which is what makes it
  /// impossible to free an index without moving it. A counter kept one tier up
  /// was bumped by @ref VoxelBlockGrid::remove and silently *not* by this
  /// `remove` reached through @ref VoxelBlockGrid::map, so the raw path
  /// defeated every anchor built on it in perfect silence.
  ///
  /// **Globally unique, not a per-map count.** Each value is drawn once from a
  /// process-wide counter -- at @ref create as well as at every removal -- so
  /// no two tables, and no two topologies of one table, ever share one. That is
  /// what lets an anchor be *just* this token: a cache holding the token of a
  /// destroyed map cannot be revived by a new map built at the same address,
  /// the ABA a raw pointer comparison has no way to see. Consequently it does
  /// not count anything; only equality with a previously read value is
  /// meaningful.
  ///
  /// @ref resize deliberately does **not** move it: it preserves every block's
  /// index, so a slot-keyed cache stays correct across a grow (which is the
  /// whole point of the index-preserving rehash).
  /// @return The token; 0 only on a moved-from map, which no live token equals.
  std::uint64_t topology_epoch() const noexcept { return topology_epoch_; }

  /// @brief The map's clock, which every @ref BlockStamp is a reading of.
  ///        Starts at 1; 0 is no tick.
  ///
  /// It counts the passes that write voxels -- each `tsdf::TsdfIntegrator`
  /// integrate call (one per set, for a rig), `tsdf::MeshIntegrator` call and
  /// codec decode -- since each advances it before it stamps `changed`. So a
  /// reader that records the tick it read at sees every later change as
  /// newer, however its calls interleave with the writers'. Ticks compare
  /// modulo 2^32, so an age is right up to 2^31 of them.
  std::uint32_t tick() const noexcept { return tick_; }
  /// @brief Advance @ref tick by one, as a pass that writes voxels does before
  ///        it stamps. Skips 0 when it wraps.
  void advance_tick() noexcept { tick_ = tick_ + 1 == 0 ? 1 : tick_ + 1; }
  /// @brief The device buffer of one @ref BlockStamp per block slot, for a
  ///        kernel that reads or writes a stamp. Every allocation stamps
  ///        `requested`, and every pass that writes voxels `changed`;
  ///        @ref remove, @ref clear and @ref create zero a slot's record, and
  ///        @ref resize keeps each where it is.
  const core::Buffer& stamps_buffer() const noexcept { return stamps_; }
  /// @brief Every block slot's record, read back; the iOS scanner reads its
  ///        `changed` ticks through this.
  core::Result<std::vector<BlockStamp>> read_block_stamps() const;

  /// @return The grid + hash-table parameters this map was built with.
  const VoxelGridParams& grid() const noexcept { return grid_; }
  /// @return `true` if this owns a live table (`false` when moved-from).
  bool valid() const noexcept { return entries_.valid(); }

 private:
  VoxelHashMap() = default;

  /// Draw the next process-wide unique @ref topology_epoch value. Never
  /// returns 0, so a default-initialized member is distinguishable from every
  /// live token.
  static std::uint64_t next_topology_epoch() noexcept;

  /// Run the init kernel, resetting every slot to empty (used by create +
  /// clear).
  core::Status init_table();

  /// Rebuild the free-block heap so it holds exactly the block indices @p
  /// active does *not* occupy, in ascending order, with @ref heap_counter_ set
  /// to that free count. Called by @ref resize after @ref init_table (which
  /// fills the heap with every index) and the rehash (which re-inserts @p
  /// active with their preserved pointers): computed on the host and uploaded
  /// in one batch, since resize runs single-threaded between dispatches.
  core::Status rebuild_heap_excluding(const std::vector<BlockIndex>& active);

  /// @return The hash-table slot count, `num_buckets * bucket_size`.
  std::uint32_t total_entries() const noexcept;

  /// The body of @ref compact_active_blocks: zero the counter, run the
  /// compaction over every hash slot, then read back the appended
  /// @ref BlockIndex list. @ref last_active_count_ is the previous count, which
  /// sizes the list read back in the same batch, and is updated. @p stage,
  /// when non-null, collects the dispatch's device span.
  core::Result<std::vector<BlockIndex>> collect_compacted(
      core::GpuStageScope* stage);
  /// The compaction into @p out, the buffer @p kernel writes, in one submit,
  /// returning the count and reading the list's first @p head_count entries
  /// back into @p head.
  core::Result<std::uint32_t> compact_into_device_list(
      const core::ComputeKernel& kernel, const core::Buffer& out,
      core::GpuStageScope* stage,
      const std::function<core::Status(core::CommandBatch&)>& prepare = {},
      BlockIndex* head = nullptr, std::uint32_t head_count = 0);
  /// Record into @p batch the upload of @p count frusta into
  /// @ref frustum_planes_ -- the count, then the planes from byte 4, the
  /// frustum kernel's scalar layout -- growing and rebinding the buffer first
  /// when it is too small. Called before the batch records the dispatch.
  core::Status upload_frusta(core::CommandBatch& batch,
                             const FrustumPlanes* frusta, std::uint32_t count);

  /// The row label both compaction entry points report under, carrying
  /// `StageMetrics::kBreakdownPrefix` or not according to whether @p metrics
  /// already has a stage open.
  static const char* active_set_row(const core::StageMetrics* metrics) noexcept;

  /// Whether @p list still names this map's blocks: compacted here, and
  /// nothing since has rewritten its buffer, freed a block or allocated one.
  bool holds(const DeviceBlockList& list) const noexcept;

  /// Record what @p dispatches records each round, re-dispatching while the
  /// shared `fail_counts_[kFailTotal]` tally is
  /// non-zero, to converge past transient same-bucket lock contention, until
  /// only capacity failures are left, a capacity limit stops progress, or the
  /// rounds run out. Re-zeroes the tally
  /// each round. The shared tail of every allocate/remove kernel.
  /// Non-retryable failures (`kFailTerminal`) are accumulated across rounds and
  /// added to the returned count; @p out_failures, when non-null, receives the
  /// full per-reason split.
  /// A dispatch given a stage collects one span per round; they accumulate
  /// under its one label, which is the honest total for a frame that genuinely
  /// dispatched several times. A round that fails leaves the rounds before it
  /// recorded, and the scope publishes them on the way out -- they ran.
  /// @p prepare, when set, records the call's parameter uploads into the first
  /// round's batch, ahead of the dispatches. Each round is one submit, and
  /// reads back the heap counter into @ref heap_free_ beside the tally.
  core::Result<std::uint32_t> dispatch_with_retry(
      const std::function<core::Status(core::CommandBatch&)>& dispatches,
      AllocFailures* out_failures,
      const std::function<core::Status(core::CommandBatch&)>& prepare = {});

  /// Create a transient device-local buffer that @p batch fills with @p bytes
  /// of @p data, and bind it at @p binding of @p set. The caller keeps the
  /// returned `Buffer` alive until the batch, and any round after it that
  /// reads the binding, has run.
  core::Result<core::Buffer> upload_to_binding(core::CommandBatch& batch,
                                               const core::DescriptorSet& set,
                                               std::uint32_t binding,
                                               const void* data,
                                               VkDeviceSize bytes);

  /// Shared body of the per-element kernels (@ref allocate, @ref remove):
  /// stage or bind @p count elements
  /// of @p elem_size bytes at input binding (4) of @p kernel's set, then run
  /// @p kernel over them (one thread per element) via @ref dispatch_with_retry.
  /// @p op names the caller for diagnostics. @p removes marks @ref remove: it
  /// moves @ref topology_epoch once the input is accepted, and binds a zeroed
  /// flag per element at binding 6, which the kernel sets on each coord it
  /// finishes so later rounds skip it.
  core::Result<std::uint32_t> run_input_kernel(
      const char* op, const core::StorageInput& input, std::size_t elem_size,
      std::uint32_t count, const core::ComputeKernel& kernel,
      AllocFailures* out_failures, bool removes = false);

  /// Point every set at the persistent buffers (entries / heap / heap_counter /
  /// bucket_mutex / fail_counts / compacted / active_count); run at create and
  /// after a resize swaps them.
  void write_persistent_bindings();

  // Borrowed (must outlive this). Pointers, not references, so a moved-from map
  // is left in a defined (empty) state.
  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;
  VoxelGridParams grid_{};
  // Re-drawn from the process-wide counter by create / remove / clear; see
  // topology_epoch(). A scalar, so the defaulted move copies it into the
  // destination -- which is right: the destination *is* the table the token
  // named. It is left set on the moved-from map too, and harmlessly so, since
  // valid() is what says that map owns nothing.
  std::uint64_t topology_epoch_ = 0;
  // Cached maxComputeWorkGroupCount[0] -- the device cap on a 1-D dispatch's
  // groupCountX; every dispatch rejects an input that would exceed it.
  std::uint32_t max_workgroup_count_x_ = 0;
  // The ceiling on one storage-buffer binding's range, read once at create().
  // The per-call input uploads (a depth frame, a coord or point list) are bound
  // whole, so an over-large one is invalid usage rather than a slow path.
  VkDeviceSize max_storage_buffer_range_ = 0;

  // Persistent device buffers, all device-local; the host reaches them
  // through a CommandBatch (the 2026-09-28 residency decision).
  core::Buffer entries_;
  core::Buffer heap_;
  core::Buffer heap_counter_;
  core::Buffer bucket_mutex_;
  // Persistent scratch, re-zeroed per call rather than re-allocated: allocate
  // fail-counts and the compaction counter are fixed; the compaction output
  // tracks num_blocks, so resize() grows it with the rest.
  core::Buffer fail_counts_;
  core::Buffer compacted_;
  // The frustum compactions' output, so they leave compacted_'s list holding.
  core::Buffer frustum_compacted_;
  core::Buffer active_count_;
  // One BlockStamp per block slot (see stamps_buffer()), and the clock.
  core::Buffer stamps_;
  std::uint32_t tick_ = 1;
  // Persistent camera params for allocate_from_depth (bound at binding 6 of
  // every depth set, rewritten inline ahead of each frame's dispatch);
  // grid-independent, so not in the bundle.
  core::Buffer camera_params_;
  // Persistent frusta for the frustum compactions (bound at binding 3 of
  // compact_frustum_.set, rewritten inline per call, grown by upload_frusta);
  // grid-independent.
  core::Buffer frustum_planes_;
  // The host's copy of heap_counter_, which load_factor() reads: set by
  // init_table and the heap rebuild, and read back by every retry round, the
  // only dispatches that move the counter. Copied by the defaulted move, and
  // harmlessly left on a moved-from map, whose load_factor() is refused.
  std::uint32_t heap_free_ = 0;
  // The last compact_active_blocks count, from which collect_compacted
  // guesses how much of the list to read back beside the next count.
  std::uint32_t last_active_count_ = 0;
  // Bumped by every write to compacted_ (a compaction) and every swap of it (a
  // resize), so a DeviceBlockList can be checked against it; frustum_serial_
  // likewise for frustum_compacted_.
  std::uint64_t compaction_serial_ = 0;
  std::uint64_t frustum_serial_ = 0;
  // The last compact_active_blocks_on_device list, returned again while
  // holds() says it is still the active set.
  DeviceBlockList last_device_list_{};

  // The shared descriptor pool the kernels' sets are allocated from. Declared
  // BEFORE the ComputeKernel members so it is destroyed AFTER them (members
  // tear down in reverse declaration order): a pool must outlive the sets it
  // owns. DescriptorSet is a non-owning view today (freed with the pool, so the
  // order is not yet load-bearing), but this keeps the safe ordering if that
  // ever changes.
  core::DescriptorPool pool_;
  // Device spans for this tier's dispatches; idle -- no query written -- until
  // a caller passes a StageMetrics.
  core::GpuTimer gpu_timer_;
  // One ComputeKernel per shader -- its descriptor-set layout, pipeline, and
  // the set allocated from the shared pool_ (see `ComputeKernel`). The
  // KernelSetBuilder in create() builds all seven and sizes pool_ to them.
  // Every persistent-buffer binding is written once by
  // write_persistent_bindings(); only the genuinely per-call input (coords /
  // depth+camera / triangles) is (re)written before a dispatch. Depth adds the
  // camera-params buffer at binding 6, and triangles adds indices at 6 + the
  // prefix-sum offsets at 7.
  core::ComputeKernel init_;
  core::ComputeKernel allocate_;
  core::ComputeKernel compact_;
  core::ComputeKernel delete_;
  core::ComputeKernel depth_;
  core::ComputeKernel triangles_;
  core::ComputeKernel compact_frustum_;
  // Re-inserts a snapshot of active blocks into the grown table preserving each
  // block's index (insert_block with the block's own pointer, not a fresh heap
  // draw), so per-voxel data survives a resize. Same 6-binding shape as
  // allocate_ (its input at binding 4 is BlockIndex{coord, ptr}, ptr read).
  core::ComputeKernel rehash_;
  // The depth kernel's sets, one a frame of a call, so a round dispatches
  // several cameras in one batch; depth_.set goes unused. Grown to the most
  // frames a call has had; written whole each call, so a resize leaves none
  // stale.
  core::KernelSets depth_sets_;
};

}  // namespace volumetric_kit::recon::volume
