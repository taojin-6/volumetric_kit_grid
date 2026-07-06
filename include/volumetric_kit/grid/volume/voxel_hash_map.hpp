// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file volume/voxel_hash_map.hpp
/// @brief The sparse voxel hash map on the GPU: owns the device buffers and the
///        compute pipelines, allocates blocks, and compacts the active set.

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/buffer.hpp"
#include "volumetric_kit/recon/core/compute_pipeline.hpp"
#include "volumetric_kit/recon/core/descriptor.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/volume/export.hpp"
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

/// @brief Pinhole depth-camera intrinsics + pose for
///        @ref VoxelHashMap::allocate_from_depth.
///
/// Uploaded verbatim to the depth-allocation kernel, which reads it through
/// scalar block layout -- so this packs byte-for-byte to the shader's
/// `CameraParams`: the scalars at their natural 4-byte offsets, the `mat4` at
/// offset 32. The `static_assert`s below pin that layout (a drift is a compile
/// error, not silent misprojection).
struct DepthCameraParams {
  float fx;              ///< Focal length x (pixels).
  float fy;              ///< Focal length y (pixels).
  float cx;              ///< Principal point x (pixels).
  float cy;              ///< Principal point y (pixels).
  float min_depth;       ///< Reject samples nearer than this (metres).
  float max_depth;       ///< Reject samples farther than this (metres).
  std::uint32_t width;   ///< Depth image width (pixels).
  std::uint32_t height;  ///< Depth image height (pixels).
  Mat4f cam_to_world;    ///< Camera -> world rigid transform (column-major).
};
static_assert(sizeof(DepthCameraParams) == 96,
              "DepthCameraParams must be 96 bytes");
static_assert(offsetof(DepthCameraParams, min_depth) == 16, "layout drift");
static_assert(offsetof(DepthCameraParams, width) == 24, "layout drift");
static_assert(offsetof(DepthCameraParams, cam_to_world) == 32, "layout drift");
static_assert(std::is_trivially_copyable_v<DepthCameraParams>,
              "DepthCameraParams must be trivially copyable");
static_assert(std::is_standard_layout_v<DepthCameraParams>,
              "DepthCameraParams must be standard-layout");

/// @brief Owns the device-side sparse voxel hash table -- the hash-entry index,
///        the free-block heap, and the per-bucket locks -- plus the compute
///        pipelines that operate on them, and drives block allocation and
///        active-block compaction.
///
/// Built on the `core` compute foundation (@ref Allocator, @ref Buffer,
/// @ref ComputePipeline, @ref Device::submit_single_time). The GLSL kernels
/// read the hash structs through scalar block layout (the 2026-07-05 ABI), so
/// the host @ref HashEntry / @ref BlockIndex and their shader mirrors agree
/// byte-for-byte. Covers init, allocate-from-coords, remove, compact, and
/// resize; diagnostics, the index-preserving GPU rehash, and depth/point
/// allocation follow.
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
  ///         @ref resize.
  Result<std::uint32_t> allocate(const BlockIndex* coords, std::uint32_t count);

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
  /// @return The number of block allocations that failed (0 = all succeeded),
  ///         or a non-OK @ref Status if a buffer or the dispatch fails, or the
  ///         map is moved-from / @p depth is null. A non-zero count means
  ///         bucket/heap pressure -- grow with @ref resize.
  Result<std::uint32_t> allocate_from_depth(const float* depth,
                                            const DepthCameraParams& camera);

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
  Result<std::uint32_t> allocate_from_points(const Vec3f* points,
                                             std::uint32_t count);

  /// @brief Remove voxel blocks at the given block coordinates (only `coord` is
  ///        read); absent coordinates are ignored. Returns each freed block to
  ///        the heap.
  ///
  /// Must not run concurrently with @ref allocate — the heap requires alloc and
  /// free in separate dispatches, which the fence between calls guarantees for
  /// single-threaded use.
  /// @param coords  The block coordinates to remove.
  /// @param count   How many.
  /// @return The number of removals that failed (0 = all done), or a non-OK
  ///         @ref Status if a buffer or the dispatch fails.
  Result<std::uint32_t> remove(const BlockIndex* coords, std::uint32_t count);

  /// @brief Compact every active block into a host vector of @ref BlockIndex.
  /// @return The active blocks (order unspecified), or a non-OK @ref Status.
  Result<std::vector<BlockIndex>> compact_active_blocks();

  /// @brief Reset the table to empty (re-runs the init kernel).
  /// @return An OK @ref Status, or a non-OK one if the init dispatch fails or
  ///         the map is moved-from.
  Status clear();

  /// @brief Grow the hash table to @p new_num_buckets buckets (must exceed the
  ///        current count), preserving the active block set.
  ///
  /// Reuses the proven init / allocate / compact kernels: snapshot the active
  /// coordinates, grow the buffers, re-init the larger table, and re-insert.
  /// This reassigns block indices -- transparent for the coordinate set, but it
  /// does not preserve per-block voxel data.
  /// TODO(tsdf): a block-index-preserving GPU rehash once blocks carry SDF
  /// data.
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

  /// @brief Compute occupancy + health statistics (active / overflow / chain
  ///        length + heap utilization).
  ///
  /// A host-side scan of the entries plus the heap counter -- O(total slots),
  /// so call it for inspection/logging, not per frame. A GPU-side scan is a
  /// perf follow-up for very large tables.
  /// @return The statistics, or a non-OK @ref Status (e.g. moved-from map).
  Result<HashDiagnostics> diagnostics();

  /// @return The grid + hash-table parameters this map was built with.
  const VoxelGridParams& grid() const noexcept { return grid_; }
  /// @return `true` if this owns a live table (`false` when moved-from).
  bool valid() const noexcept { return entries_.valid(); }

 private:
  VoxelHashMap() = default;

  /// Run the init kernel, resetting every slot to empty (used by create +
  /// clear).
  Status init_table();

  /// @return The hash-table slot count, `num_buckets * bucket_size`.
  std::uint32_t total_entries() const noexcept;

  /// Dispatch @p pipeline (bound through @p set, push arg = @p arg) over
  /// @p groups groups, re-dispatching while the shared `fail_counts_[0]` tally
  /// keeps dropping to converge past transient same-bucket lock contention.
  /// Re-zeroes the tally each round. The shared tail of every allocate/remove
  /// kernel.
  Result<std::uint32_t> dispatch_with_retry(const ComputePipeline& pipeline,
                                            DescriptorSet& set,
                                            std::uint32_t arg,
                                            std::uint32_t groups);

  /// Create a transient host-visible buffer holding @p bytes of @p data and
  /// bind it at @p binding of @p set. The caller keeps the returned @ref Buffer
  /// alive across the (synchronous) dispatch that reads it.
  Result<Buffer> upload_to_binding(DescriptorSet& set, std::uint32_t binding,
                                   const void* data, VkDeviceSize bytes);

  /// Shared body of the per-element allocate kernels (@ref allocate,
  /// @ref remove, @ref allocate_from_points): upload @p count elements of
  /// @p elem_size bytes to the input binding (4) of @p set, then run @p
  /// pipeline over them (one thread per element) via @ref dispatch_with_retry.
  /// @p op names the caller for diagnostics.
  Result<std::uint32_t> run_input_kernel(const char* op, const void* data,
                                         std::size_t elem_size,
                                         std::uint32_t count,
                                         DescriptorSet& set,
                                         const ComputePipeline& pipeline);

  /// Point every set at the persistent buffers (entries / heap / heap_counter /
  /// bucket_mutex / fail_counts / compacted / active_count); run at create and
  /// after a resize swaps them.
  void write_persistent_bindings();

  // Borrowed (must outlive this). Pointers, not references, so a moved-from map
  // is left in a defined (empty) state.
  Device* device_ = nullptr;
  Allocator* allocator_ = nullptr;
  VoxelGridParams grid_{};
  // Cached maxComputeWorkGroupCount[0] -- the device cap on a 1-D dispatch's
  // groupCountX; every dispatch rejects an input that would exceed it.
  std::uint32_t max_workgroup_count_x_ = 0;

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
  // depth_set_, rewritten per call); grid-independent, so not in the bundle.
  Buffer camera_params_;

  // One descriptor-set layout + pipeline per kernel; every persistent-buffer
  // binding is written once at create(), and only the genuinely per-call input
  // buffer(s) (coords / points / depth+camera) are (re)written before each
  // dispatch. allocate-from-coords and allocate-from-points share one 6-binding
  // layout (identical shape); depth needs its own 7-binding layout (it adds the
  // camera-params buffer at binding 6).
  DescriptorSetLayout init_layout_;
  DescriptorSetLayout allocate_layout_;
  DescriptorSetLayout compact_layout_;
  DescriptorSetLayout delete_layout_;
  DescriptorSetLayout depth_layout_;
  ComputePipeline init_pipeline_;
  ComputePipeline allocate_pipeline_;
  ComputePipeline compact_pipeline_;
  ComputePipeline delete_pipeline_;
  ComputePipeline depth_pipeline_;
  ComputePipeline points_pipeline_;  // bound through allocate_layout_
  DescriptorPool pool_;
  DescriptorSet init_set_;
  DescriptorSet allocate_set_;
  DescriptorSet compact_set_;
  DescriptorSet delete_set_;
  DescriptorSet depth_set_;
  DescriptorSet points_set_;  // allocated from allocate_layout_
};

}  // namespace volumetric_kit::recon::volume
