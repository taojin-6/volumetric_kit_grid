// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file volume/voxel_hash_map.hpp
/// @brief The sparse voxel hash map on the GPU: owns the device buffers and the
///        compute pipelines, allocates blocks, and compacts the active set.

#include <cstdint>
#include <vector>

#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/buffer.hpp"
#include "volumetric_kit/recon/core/compute_pipeline.hpp"
#include "volumetric_kit/recon/core/descriptor.hpp"
#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/volume/export.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/voxel_grid.hpp"

namespace volumetric_kit::recon {
class Device;
}

namespace volumetric_kit::recon::volume {

/// @brief Owns the device-side sparse voxel hash table -- the hash-entry index,
///        the free-block heap, and the per-bucket locks -- plus the compute
///        pipelines that operate on them, and drives block allocation and
///        active-block compaction.
///
/// Built on the `core` compute foundation (@ref Allocator, @ref Buffer,
/// @ref ComputePipeline, @ref Device::submit_single_time). The GLSL kernels
/// read the hash structs through scalar block layout (the 2026-07-05 ABI), so
/// the host @ref HashEntry / @ref BlockIndex and their shader mirrors agree
/// byte-for-byte. This slice covers init + allocate-from-coords + remove +
/// compact; resize/rehash, diagnostics, and depth/point allocation follow.
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
  ///         with a non-positive dimension.
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
  ///         non-OK @ref Status if a buffer or the dispatch fails. A non-zero
  ///         count means bucket/heap pressure -- resize + retry lands in a
  ///         later slice.
  Result<std::uint32_t> allocate(const BlockIndex* coords, std::uint32_t count);

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

  /// @brief Read the raw hash-entry slots back to the host.
  ///
  /// Low-level / diagnostic: exposes the on-device @ref HashEntry array (all
  /// slots, including free ones) so callers can inspect the table or verify the
  /// host<->shader layout. @ref compact_active_blocks is the normal way to get
  /// the active set.
  /// @return All hash-entry slots (length `num_buckets * bucket_size`), or a
  ///         non-OK @ref Status if the map is moved-from.
  Result<std::vector<HashEntry>> read_entries();

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

  /// Shared body of @ref allocate and @ref remove: upload @p coords, run
  /// @p pipeline (bound through @p set) over them, and read back the
  /// `fail_counts_[0]` tally the coord kernels share (allocate and delete never
  /// run in the same dispatch). @p op names the caller for diagnostics.
  Result<std::uint32_t> run_coord_kernel(const char* op,
                                         const BlockIndex* coords,
                                         std::uint32_t count,
                                         DescriptorSet& set,
                                         const ComputePipeline& pipeline);

  // Borrowed (must outlive this). Pointers, not references, so a moved-from map
  // is left in a defined (empty) state.
  Device* device_ = nullptr;
  Allocator* allocator_ = nullptr;
  VoxelGridParams grid_{};

  // Persistent device buffers. TODO(volume): these are host-visible for this
  // slice; a device-local + staging path is a follow-up perf pass.
  Buffer entries_;
  Buffer heap_;
  Buffer heap_counter_;
  Buffer bucket_mutex_;
  // Persistent scratch (fixed size for the map's lifetime, re-zeroed per call
  // rather than re-allocated): allocate fail-counts, compaction output + count.
  Buffer fail_counts_;
  Buffer compacted_;
  Buffer active_count_;

  // One descriptor-set layout + pipeline per kernel; every persistent-buffer
  // binding is written once at create(), and only the genuinely per-call coords
  // buffer is (re)written before each allocate dispatch.
  DescriptorSetLayout init_layout_;
  DescriptorSetLayout allocate_layout_;
  DescriptorSetLayout compact_layout_;
  DescriptorSetLayout delete_layout_;
  ComputePipeline init_pipeline_;
  ComputePipeline allocate_pipeline_;
  ComputePipeline compact_pipeline_;
  ComputePipeline delete_pipeline_;
  DescriptorPool pool_;
  DescriptorSet init_set_;
  DescriptorSet allocate_set_;
  DescriptorSet compact_set_;
  DescriptorSet delete_set_;
};

}  // namespace volumetric_kit::recon::volume
