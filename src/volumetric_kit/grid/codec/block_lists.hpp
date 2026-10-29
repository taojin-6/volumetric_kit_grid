// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file block_lists.hpp
/// @brief The codec's work on block coordinates, on the device: the
///        encoder's blocks sorted into frame order, and the decoder's frame
///        merged from its reference and diffed against the grid.
///
/// Internal (under src/, never installed). The kernels order coordinates by
/// `volume/shaders/coord_order.glsl`, the mirror of @ref coord_less.

#include <cstdint>
#include <memory>

#include "bitstream.hpp"
#include "dct_transform.hpp"
#include "reference_store.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/recon/codec/export.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

namespace volumetric_kit::recon::codec::detail {

/// @brief Block lists in frame order, made on the device: the encoder's
///        observed blocks sorted by coordinate, and a decoded frame's blocks
///        merged from its reference and new blocks, then diffed against the
///        grid it decodes into.
///
/// The sort is a stable rank-merge sort on @ref coord_less itself, so it
/// takes every int32 coordinate with no key transform. A tile kernel sorts
/// each 256 entries in shared memory; each level after it merges pairs of
/// sorted runs, an entry an invocation, each placed at its index in its run
/// plus a binary search of its sibling run. So a bound of n entries takes
/// `1 + ceil(log2(ceil(n / 256)))` dispatches, and every entry's place depends
/// only on the keys: unique keys, as a hash table's are, give one order
/// whatever order they came in.
///
/// The decoder never sorts. A frame's blocks are a merge of two strictly
/// increasing lists, the reference's survivors and the new blocks, so each
/// block's place is a count -- the survivors below it, from the reader's
/// ranks, plus the new blocks below it, by binary search -- which
/// @ref record_merge writes with each block's sources; a new block on a
/// reference block is row 13's refusal (@ref check_new_block). Then
/// @ref record_diff finds what the grid must lose and gain without sorting
/// it: a grid block whose coordinate a search of the frame misses is gone,
/// and a frame block the hash table does not hold is missing. Both record
/// into the decode's batch and leave their findings on the device
/// (@ref collision, @ref diff_counts), for the decoder's gate and results.
///
/// Every buffer is device-local and retained, grown as a call needs
/// (`ensure_device_scratch`).
///
/// @warning The `Device` and `Allocator` passed to @ref create must
///          outlive this object. Not thread-safe: one call at a time.
class VR_CODEC_API BlockLists {
 public:
  /// @brief Build the kernels.
  /// @return The lists, or a pipeline or pool failure.
  static core::Result<std::unique_ptr<BlockLists>> create(
      core::Device& device, core::Allocator& allocator);

  ~BlockLists() = default;
  BlockLists(const BlockLists&) = delete;
  BlockLists& operator=(const BlockLists&) = delete;
  BlockLists(BlockLists&&) = delete;
  BlockLists& operator=(BlockLists&&) = delete;

  /// @brief Record a stable sort of the first `n` `volume::BlockIndex`
  ///        entries of @p list, by @ref coord_less, into @p out, `n` the
  ///        word @p count names, so it never reaches the host.
  ///
  /// The tile pass and every level the bound needs are recorded over the
  /// bound's groups, a group past `n` returning, and a level past what `n`
  /// needs copies its entries through: the order is `n` entries' sort,
  /// whatever the bound. The last level, or the tile pass when the bound
  /// needs none, writes @p out, so the sorted list lands where its reader
  /// binds it with no copy. @p list is read, never written. The levels bind
  /// sets this object keeps, so a batch holds at most one of these calls.
  /// @param batch  The batch to record into; submitted by the caller.
  /// @param list   A storage buffer of at least the bound's entries.
  /// @param count  `n`, a word a kernel wrote, at most its bound.
  /// @param out    Where the sorted entries go: at least the bound's.
  /// @param stage  Optional scope the caller's stage row is open under; the
  ///               dispatches record their device spans into it.
  /// @return OK, recording nothing for a bound of 0; or
  ///         `Status::Code::InvalidArgument` for a @p list
  ///         `StorageInput::check` refuses, entries past
  ///         `maxStorageBufferRange`, a count word missing, misaligned or past
  ///         its buffer, or an @p out short of the bound; otherwise a buffer
  ///         failure.
  core::Status record_sort(core::CommandBatch& batch, const core::Buffer& list,
                           const volume::DeviceCount& count,
                           const core::Buffer& out,
                           core::GpuStageScope* stage = nullptr);

  /// @brief Make the diff's lists hold a frame of @p n blocks into a grid of
  ///        @p slots hash slots, before anything is recorded.
  /// @return OK; `Status::Code::InvalidArgument` for a list past
  ///         `maxStorageBufferRange`, which grows nothing; or an
  ///         allocation's failure.
  core::Status reserve(std::uint32_t n, std::uint32_t slots);

  /// @brief Record the merge of a decoded frame's @p n blocks into
  ///        @p frame, in frame order: each block's coordinate, `ptr` 0, and
  ///        its sources -- keep `{e, e}`, skip `{e, none}`, zero `{none, e}`,
  ///        new block q `{none, M + q}` -- and row 13's first collision.
  ///
  /// Sound once rows 11 and 12 pass (the reader's check, after the submit):
  /// on a frame they refuse the outputs are garbage, but every write stays
  /// below @p n and the searches end.
  /// @param batch      The decode's batch, after the reader's decode.
  /// @param decoded    The reader's entries: its M modes, its ranks and its
  ///                   A new blocks.
  /// @param reference  The held reference's M blocks, strictly increasing.
  /// @param frame      Where the blocks and sources go, @p n of each.
  /// @param n          N, the frame's blocks.
  /// @param stage      As @ref record_sort.
  /// @return OK; `Status::Code::InvalidArgument` for buffers missing or
  ///         smaller than the counts need; otherwise a buffer failure.
  core::Status record_merge(core::CommandBatch& batch,
                            const ResidentBlocks& decoded,
                            const core::Buffer& reference, FrameBuffers frame,
                            std::uint32_t n,
                            core::GpuStageScope* stage = nullptr);

  /// @brief Record the diff of @p grid's block set against a frame's first
  ///        @p n blocks of @p frame_list, strictly increasing: what the grid
  ///        holds and the frame lacks into @ref gone, with each block's
  ///        `ptr`, and what the frame holds and the grid lacks into
  ///        @ref missing, `ptr` 0, each in no order.
  ///
  /// Binds the hash table as it is when the batch runs, so nothing may
  /// allocate into or remove from @p grid until then.
  /// @param batch       The decode's batch, after the merge.
  /// @param grid        The grid the frame decodes into.
  /// @param frame_list  The frame's blocks, as the merge wrote them.
  /// @param n           N.
  /// @param stage       As @ref record_sort.
  /// @return OK; `Status::Code::InvalidArgument` for a moved-from grid, a
  ///         hash table past `maxStorageBufferRange`, or more blocks or
  ///         slots than @ref reserve made room for; otherwise a buffer
  ///         failure.
  core::Status record_diff(core::CommandBatch& batch,
                           const volume::VoxelBlockGrid& grid,
                           const core::Buffer& frame_list, std::uint32_t n,
                           core::GpuStageScope* stage = nullptr);

  /// @return Row 13's finding once the last merge's batch ran: the first new
  ///         block on a reference entry, and that entry, a u32 each;
  ///         @ref kNoSource for none.
  const core::Buffer& collision() const noexcept { return collision_; }
  /// @return The last diff's counts once its batch ran: the grid blocks
  ///         @ref gone lists, then the frame blocks @ref missing lists, a u32
  ///         each.
  const core::Buffer& diff_counts() const noexcept { return diff_counts_; }
  /// @return The grid blocks the last diff found the frame lacks, the first
  ///         `diff_counts()[0]` of them.
  const core::Buffer& gone() const noexcept { return gone_; }
  /// @return The frame blocks the last diff found the grid lacks, the first
  ///         `diff_counts()[1]` of them.
  const core::Buffer& missing() const noexcept { return missing_; }

 private:
  BlockLists() = default;

  // Borrowed (must outlive this).
  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;
  std::uint32_t max_workgroup_count_x_ = 0;
  VkDeviceSize max_storage_buffer_range_ = 0;

  // Declared before pool_, so the pool is destroyed first.
  core::ComputeKernel sort_tile_kernel_;
  core::ComputeKernel sort_merge_kernel_;
  core::ComputeKernel frame_merge_kernel_;
  core::ComputeKernel grid_diff_kernel_;
  core::DescriptorPool pool_;
  // The merge levels' sets: sorted_[0] into sorted_[1], back, and the last
  // level's half into the caller's output.
  core::KernelSets sort_merge_sets_;

  // Device-local, retained: the sort's two halves, which its levels
  // ping-pong between; the diff's lists; and the merge's collision and the
  // diff's counts.
  core::Buffer sorted_[2];
  core::Buffer gone_;
  core::Buffer missing_;
  core::Buffer collision_;
  core::Buffer diff_counts_;
};

}  // namespace volumetric_kit::recon::codec::detail
