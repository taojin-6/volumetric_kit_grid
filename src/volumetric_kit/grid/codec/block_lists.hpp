// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file block_lists.hpp
/// @brief The codec's work on block coordinates, on the device: the
///        encoder's blocks sorted into frame order.
///
/// Internal (under src/, never installed). The kernels order coordinates by
/// `volume/shaders/coord_order.glsl`, the mirror of @ref coord_less.

#include <cstdint>
#include <memory>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/recon/codec/export.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"

namespace volumetric_kit::recon::codec::detail {

/// @brief Block lists put in frame order on the device: the encoder's
///        observed blocks, sorted by coordinate.
///
/// The sort is a stable rank-merge sort on @ref coord_less itself, so it
/// takes every int32 coordinate with no key transform. A tile kernel sorts
/// each 256 entries in shared memory; each level after it merges pairs of
/// sorted runs, an entry an invocation, each placed at its index in its run
/// plus a binary search of its sibling run. So n entries take
/// `1 + ceil(log2(ceil(n / 256)))` dispatches, and every entry's place depends
/// only on the keys: unique keys, as a hash table's are, give one order
/// whatever order they came in.
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

  /// @brief Record a stable sort of the first @p n `volume::BlockIndex`
  ///        entries of @p list, by @ref coord_less, into @p batch.
  ///
  /// @p list is read, never written. The levels bind sets this object keeps,
  /// so a batch holds at most one of these calls.
  /// @param batch  The batch to record into; submitted by the caller.
  /// @param list   A storage buffer of at least @p n entries.
  /// @param n      The entries to sort.
  /// @param stage  Optional scope the caller's stage row is open under; the
  ///               dispatches record their device spans into it.
  /// @return This object's buffer that holds the sorted entries once
  ///         @p batch has run, valid until the next call (one with no
  ///         entries for @p n 0, which records nothing); or
  ///         `Status::Code::InvalidArgument` for a @p list
  ///         `StorageInput::check` refuses or entries past
  ///         `maxStorageBufferRange`; otherwise a buffer failure.
  core::Result<const core::Buffer*> record_sort(
      core::CommandBatch& batch, const core::Buffer& list, std::uint32_t n,
      core::GpuStageScope* stage = nullptr);

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
  core::DescriptorPool pool_;
  // The merge levels' sets, one a direction: sorted_[0] into sorted_[1],
  // then back.
  core::KernelSets sort_merge_sets_;

  // Device-local, retained: the sort's two halves, which its levels
  // ping-pong between.
  core::Buffer sorted_[2];
};

}  // namespace volumetric_kit::recon::codec::detail
