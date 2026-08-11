// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file volume/voxel_hash_map.hpp
/// @brief The sparse voxel hash map on the GPU: owns the device buffers and the
///        compute pipelines, allocates blocks, and compacts the active set.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/buffer.hpp"
#include "volumetric_kit/recon/core/camera_params.hpp"
#include "volumetric_kit/recon/core/compute_kernel.hpp"
#include "volumetric_kit/recon/core/descriptor.hpp"
#include "volumetric_kit/recon/core/gpu_timer.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/core/stage_metrics.hpp"
#include "volumetric_kit/recon/volume/export.hpp"
#include "volumetric_kit/recon/volume/frustum.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/voxel_grid.hpp"

namespace volumetric_kit::recon {
class Device;
}

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
/// same-bucket contention (a GPU spin-lock livelock within a SIMD group, worst
/// for depth, whose adjacent pixels hammer the same block) and says nothing
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

/// @brief Owns the device-side sparse voxel hash table -- the hash-entry index,
///        the free-block heap, and the per-bucket locks -- plus the compute
///        pipelines that operate on them, and drives block allocation and
///        active-block compaction.
///
/// Built on the `core` compute foundation (@ref Allocator, @ref Buffer,
/// @ref ComputePipeline, @ref Device::submit_single_time). The GLSL kernels
/// read the hash structs through scalar block layout (the 2026-07-05 ABI), so
/// the host @ref HashEntry / @ref BlockIndex and their shader mirrors agree
/// byte-for-byte. Covers init, allocate-from-coords / -depth / -points, remove,
/// compact / compact-in-frustum, diagnostics, and an **index-preserving**
/// @ref resize (the GPU rehash that keeps each block's `ptr`).
///
/// @warning The @ref Device and @ref Allocator passed to @ref create must
///          outlive this object; it stores references to them.
class VR_VOLUME_API VoxelHashMap {
 public:
  /// @brief Create the hash map on @p device with the given grid, allocating
  /// its
  ///        buffers from @p allocator and running the init kernel.
  /// @param device     The compute device (must outlive this object).
  /// @param allocator  The allocator its buffers come from (must outlive this).
  /// @param grid       The grid resolution + hash-table shape.
  /// @return The hash map, or a non-OK @ref Status if a buffer, pipeline, or
  /// the
  ///         init dispatch fails; @ref Status::Code::InvalidArgument for a grid
  ///         that fails @ref VoxelGridParams::validate.
  static Result<VoxelHashMap> create(Device& device, Allocator& allocator,
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
  ///         non-OK @ref Status if a buffer or the dispatch fails. Allocation
  ///         re-dispatches to converge under contention; a non-zero count means
  ///         a genuine capacity limit (chain full / heap empty) -- grow with
  ///         @ref resize -- but read @p out_failures rather than assuming that,
  ///         since transient lock contention can also leave a residue.
  /// @param out_failures  Optional: receives the per-reason split (see
  ///                      @ref AllocFailures). Untouched when null.
  Result<std::uint32_t> allocate(const BlockIndex* coords, std::uint32_t count,
                                 AllocFailures* out_failures = nullptr);

  /// @brief Allocate voxel blocks from a posed depth frame.
  ///
  /// One thread per pixel unprojects its depth sample (via @p camera's pinhole
  /// intrinsics + pose) to a world point, finds the block containing it, and
  /// dilates that into the surrounding `(2*tb+1)^3` truncation band the TSDF
  /// integrates (`tb` = @ref truncation_blocks). Out-of-range
  /// (`< min_depth` / `> max_depth`) and non-finite samples are skipped;
  /// already-present blocks are left untouched, so overlapping bands merge and
  /// re-running the same frame allocates nothing new.
  /// @param depth   Row-major depth image in **metres**, `width * height`
  ///                samples (the caller applies any raw sensor depth-scale).
  /// @param camera  Intrinsics, valid-depth range, dimensions, and pose.
  /// @param out_failures  Optional: receives the per-reason split. **Consult it
  ///                      before growing the map.** A non-zero count does *not*
  ///                      by itself mean bucket/heap pressure -- this is the
  ///                      most contended entry point (adjacent pixels dilate
  ///                      into the same block), so a residue of pure
  ///                      @ref AllocFailures::lock failures is expected on a
  ///                      table with ample room. @ref
  ///                      AllocFailures::capacity_limited is the test @ref
  ///                      resize answers.
  /// @param metrics  Optional @ref StageMetrics collecting an `"allocate"` host
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
  ///         or a non-OK @ref Status if a buffer or the dispatch fails, or the
  ///         map is moved-from / @p depth is null.
  Result<std::uint32_t> allocate_from_depth(
      const float* depth, const DepthCameraParams& camera,
      AllocFailures* out_failures = nullptr, StageMetrics* metrics = nullptr);

  /// @brief Allocate voxel blocks from a world-space point cloud.
  ///
  /// One thread per point finds the block containing it and dilates that into
  /// the `(2*tb+1)^3` truncation band, exactly as @ref allocate_from_depth
  /// does per unprojected pixel. Points are in world space (no unprojection);
  /// non-finite points are skipped and already-present blocks are untouched.
  /// @param points  World-space points, metres.
  /// @param count   How many.
  /// @return The number of block allocations that failed (0 = all succeeded),
  ///         or a non-OK @ref Status if a buffer or the dispatch fails, or the
  ///         map is moved-from / @p points is null.
  /// @param out_failures  Optional: receives the per-reason split (see
  ///                      @ref AllocFailures). Untouched when null.
  Result<std::uint32_t> allocate_from_points(
      const Vec3f* points, std::uint32_t count,
      AllocFailures* out_failures = nullptr);

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
  ///         @ref Status if a buffer or the dispatch fails. A non-zero count
  ///         here is **not** capacity pressure: it counts blocks that were
  ///         removed from the table but whose index could not be returned to
  ///         the free heap, i.e. leaked capacity (@ref
  ///         AllocFailures::terminal), plus any coord no attempt could delete.
  /// @param out_failures  Optional: receives the per-reason split (see
  ///                      @ref AllocFailures). Untouched when null.
  Result<std::uint32_t> remove(const BlockIndex* coords, std::uint32_t count,
                               AllocFailures* out_failures = nullptr);

  /// @brief Compact every active block into a host vector of @ref BlockIndex.
  /// @param metrics  Optional @ref StageMetrics collecting an `"active set"`
  ///                 row -- host and device. Named with @ref
  ///                 StageMetrics::kBreakdownPrefix when a @ref StageScope is
  ///                 already open on @p metrics and plainly when it is not,
  ///                 because which of the two it is depends on the caller, not
  ///                 on this operation: `tsdf`'s `"integrate"` host row wraps
  ///                 this dispatch, so as a sub-row of that its host half must
  ///                 not be summed again (and without the sub-row its device
  ///                 time would be invisible, leaving the gap between
  ///                 `integrate`'s halves reading as submit overhead when it is
  ///                 a second kernel) -- while a caller that compacts at top
  ///                 level is asking for a stage, and a prefixed row there
  ///                 would be left out of @ref StageMetrics::total_cpu_ms
  ///                 entirely.
  /// @return The active blocks (order unspecified), or a non-OK @ref Status.
  Result<std::vector<BlockIndex>> compact_active_blocks(
      StageMetrics* metrics = nullptr);

  /// @brief Compact only the active blocks intersecting @p planes -- the
  ///        per-frame streamed working set for a camera view.
  ///
  /// Like @ref compact_active_blocks, but each block's world AABB is tested
  /// against the six frustum planes and dropped if fully outside any of them
  /// (a conservative p-vertex test; the planes are ~10% widened, see
  /// @ref make_frustum_planes). This is what TSDF integration and meshing
  /// consume so only camera-visible blocks are processed.
  /// @param planes  Six inward-normal frustum planes (@ref
  /// make_frustum_planes).
  /// @param metrics  As @ref compact_active_blocks, under the same row name:
  ///                 this is the same round trip against a smaller set, so a
  ///                 caller that switches to it to make the trip cheaper must
  ///                 be able to read what that bought rather than watch the row
  ///                 disappear.
  /// @return The visible active blocks (order unspecified), or a non-OK
  ///         @ref Status if a buffer or the dispatch fails / the map is
  ///         moved-from.
  Result<std::vector<BlockIndex>> compact_active_blocks_in_frustum(
      const FrustumPlanes& planes, StageMetrics* metrics = nullptr);

  /// @brief @ref compact_active_blocks_in_frustum for a depth camera: derives
  ///        the frustum from @p camera's intrinsics, `[min_depth, max_depth]`
  ///        range, and pose, then culls.
  /// @param camera  The same camera passed to @ref allocate_from_depth.
  /// @param metrics  As @ref compact_active_blocks.
  /// @return The visible active blocks, or a non-OK @ref Status.
  Result<std::vector<BlockIndex>> compact_active_blocks_in_frustum(
      const DepthCameraParams& camera, StageMetrics* metrics = nullptr);

  /// @brief Reset the table to empty (re-runs the init kernel).
  /// @return An OK @ref Status, or a non-OK one if the init dispatch fails or
  ///         the map is moved-from.
  Status clear();

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
  /// @return OK on success, or a non-OK @ref Status:
  ///         @ref Status::Code::InvalidArgument for a non-growing count;
  ///         @ref Status::Code::OutOfMemory if the re-insert overflows.
  Status resize(std::int32_t new_num_buckets);

  /// @brief Read the raw hash-entry slots back to the host.
  ///
  /// Low-level / diagnostic: exposes the on-device @ref HashEntry array (all
  /// slots, including free ones) so callers can inspect the table or verify the
  /// host<->shader layout. @ref compact_active_blocks is the normal way to get
  /// the active set.
  /// @return All hash-entry slots (length `num_buckets * bucket_size`), or a
  ///         non-OK @ref Status if the map is moved-from.
  Result<std::vector<HashEntry>> read_entries();

  /// @brief The device hash-entry array, for a kernel that resolves blocks by
  ///        coordinate itself instead of being handed a host-built table.
  ///
  /// Published because the alternative is worse. A consumer that needs the
  /// neighbourhood of N blocks otherwise pays an O(N) serial host pass of
  /// coordinate lookups plus an upload -- 102 ms of a 133 ms mesh extract at
  /// 107k blocks, measured on an M5 iPad Pro -- to hand the GPU a table it
  /// could have built itself in parallel. `volume/shaders/hash_lookup.glsl` is
  /// the read-only traversal to bind this with; it mirrors `block_exists` and
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
  /// from the host-mapped heap counter with a 4-byte copy and no dispatch. It
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
  /// @return The occupancy in `[0, 1]`, or a non-OK @ref Status if the map is
  ///         moved-from.
  Result<float> load_factor() const;

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
  /// A host-side scan of the entries plus the heap counter -- O(total slots),
  /// so call it for inspection/logging, not per frame. A GPU-side scan is a
  /// perf follow-up for very large tables.
  /// @return The statistics, or a non-OK @ref Status (e.g. moved-from map).
  Result<HashDiagnostics> diagnostics();

  /// @brief A token identifying this table's *current* block-index assignment:
  ///        it changes whenever a block stops being live (@ref remove, @ref
  ///        clear).
  ///
  /// Exists so a consumer that caches something keyed by block slot can ask
  /// whether that cache still describes this table, which it otherwise cannot:
  /// a removed block's index goes back to a LIFO heap and is re-drawn by the
  /// next allocation, so the same slot silently comes to mean a different block
  /// at a different coordinate. `tsdf::TsdfIntegrator`'s dirty-block flags and
  /// `mesh::MarchingCubes`' span table are the two such caches.
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
  Status init_table();

  /// Rebuild the free-block heap so it holds exactly the block indices @p
  /// active does *not* occupy, in ascending order, with @ref heap_counter_ set
  /// to that free count. Called by @ref resize after @ref init_table (which
  /// fills the heap with every index) and the rehash (which re-inserts @p
  /// active with their preserved pointers): a plain host write of @ref heap_ +
  /// @ref heap_counter_, since resize runs single-threaded between dispatches.
  void rebuild_heap_excluding(const std::vector<BlockIndex>& active);

  /// @return The hash-table slot count, `num_buckets * bucket_size`.
  std::uint32_t total_entries() const noexcept;

  /// Shared body of the compaction kernels: zero the counter, run @p kernel
  /// over every hash slot, then read back the appended @ref BlockIndex list.
  /// Used by @ref compact_active_blocks (plain) and
  /// @ref compact_active_blocks_in_frustum (whose set also carries the planes).
  /// @p stage, when non-null, collects the dispatch's device span.
  Result<std::vector<BlockIndex>> collect_compacted(const ComputeKernel& kernel,
                                                    GpuStageScope* stage);

  /// The row label both compaction entry points report under, carrying
  /// @ref StageMetrics::kBreakdownPrefix or not according to whether @p metrics
  /// already has a stage open.
  static const char* active_set_row(const StageMetrics* metrics) noexcept;

  /// Dispatch @p kernel (push arg = @p arg) over @p groups groups,
  /// re-dispatching while the shared `fail_counts_[kFailTotal]` tally keeps
  /// dropping to converge past transient same-bucket lock contention. Re-zeroes
  /// the tally each round. The shared tail of every allocate/remove kernel.
  /// Non-retryable failures (`kFailTerminal`) are accumulated across rounds and
  /// added to the returned count; @p out_failures, when non-null, receives the
  /// full per-reason split.
  /// @p stage, when non-null, collects one span per round; they accumulate
  /// under its one label, which is the honest total for a frame that genuinely
  /// dispatched several times. A round that fails leaves the rounds before it
  /// recorded, and the scope publishes them on the way out -- they ran.
  Result<std::uint32_t> dispatch_with_retry(const ComputeKernel& kernel,
                                            std::uint32_t arg,
                                            std::uint32_t groups,
                                            AllocFailures* out_failures,
                                            GpuStageScope* stage = nullptr);

  /// Create a transient host-visible buffer holding @p bytes of @p data and
  /// bind it at @p binding of @p set. The caller keeps the returned @ref Buffer
  /// alive across the (synchronous) dispatch that reads it.
  Result<Buffer> upload_to_binding(const DescriptorSet& set,
                                   std::uint32_t binding, const void* data,
                                   VkDeviceSize bytes);

  /// Shared body of the per-element allocate kernels (@ref allocate,
  /// @ref remove, @ref allocate_from_points): upload @p count elements of
  /// @p elem_size bytes to the input binding (4) of @p kernel's set, then run
  /// @p kernel over them (one thread per element) via @ref dispatch_with_retry.
  /// @p op names the caller for diagnostics.
  Result<std::uint32_t> run_input_kernel(const char* op, const void* data,
                                         std::size_t elem_size,
                                         std::uint32_t count,
                                         const ComputeKernel& kernel,
                                         AllocFailures* out_failures);

  /// Point every set at the persistent buffers (entries / heap / heap_counter /
  /// bucket_mutex / fail_counts / compacted / active_count); run at create and
  /// after a resize swaps them.
  void write_persistent_bindings();

  // Borrowed (must outlive this). Pointers, not references, so a moved-from map
  // is left in a defined (empty) state.
  Device* device_ = nullptr;
  Allocator* allocator_ = nullptr;
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

  // Persistent device buffers. TODO(volume): these are host-visible for this
  // slice; a device-local + staging path is a follow-up perf pass.
  Buffer entries_;
  Buffer heap_;
  Buffer heap_counter_;
  Buffer bucket_mutex_;
  // Persistent scratch, re-zeroed per call rather than re-allocated: allocate
  // fail-counts and the compaction counter are fixed; the compaction output
  // tracks num_blocks, so resize() grows it with the rest.
  Buffer fail_counts_;
  Buffer compacted_;
  Buffer active_count_;
  // Persistent camera params for allocate_from_depth (bound at binding 6 of
  // depth_.set, rewritten per call); grid-independent, so not in the bundle.
  Buffer camera_params_;
  // Persistent frustum planes for compact_active_blocks_in_frustum (bound at
  // binding 3 of compact_frustum_.set, rewritten per call); grid-independent.
  Buffer frustum_planes_;

  // The shared descriptor pool the kernels' sets are allocated from. Declared
  // BEFORE the ComputeKernel members so it is destroyed AFTER them (members
  // tear down in reverse declaration order): a pool must outlive the sets it
  // owns. DescriptorSet is a non-owning view today (freed with the pool, so the
  // order is not yet load-bearing), but this keeps the safe ordering if that
  // ever changes.
  DescriptorPool pool_;
  // Device spans for this tier's dispatches; idle -- no query written -- until
  // a caller passes a StageMetrics.
  GpuTimer gpu_timer_;
  // One ComputeKernel per shader -- its descriptor-set layout, pipeline, and
  // the set allocated from the shared pool_ (see @ref ComputeKernel). The
  // KernelSetBuilder in create() builds all seven and sizes pool_ to them.
  // Every persistent-buffer binding is written once by
  // write_persistent_bindings(); only the genuinely per-call input (coords /
  // points / depth+camera) is (re)written before a dispatch.
  // allocate-from-coords and -from-points have the same 6-binding shape but
  // each owns its kernel; depth adds the camera-params buffer at binding 6 (7
  // bindings).
  ComputeKernel init_;
  ComputeKernel allocate_;
  ComputeKernel compact_;
  ComputeKernel delete_;
  ComputeKernel depth_;
  ComputeKernel points_;
  ComputeKernel compact_frustum_;
  // Re-inserts a snapshot of active blocks into the grown table preserving each
  // block's index (insert_block with the block's own pointer, not a fresh heap
  // draw), so per-voxel data survives a resize. Same 6-binding shape as
  // allocate_ (its input at binding 4 is BlockIndex{coord, ptr}, ptr read).
  ComputeKernel rehash_;
};

}  // namespace volumetric_kit::recon::volume
