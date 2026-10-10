// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file table_sizes.hpp
/// @brief The sizes of a @ref VoxelHashMap's buffers that scale with its
///        grid.
///
/// Internal (under src/, never installed): the map allocates its buffers at
/// these sizes, and @ref VoxelBlockGrid::bytes_at reports what a grid of
/// another size would take, so the two read one table.

#include <cstdint>

#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"

namespace volumetric_kit::recon::volume {

/// The map's grid-sized buffers, in bytes, for one grid.
struct TableSizes {
  VkDeviceSize entries = 0;            ///< A @ref HashEntry a slot.
  VkDeviceSize heap = 0;               ///< A block index a block.
  VkDeviceSize bucket_mutex = 0;       ///< A lock a bucket.
  VkDeviceSize compacted = 0;          ///< A @ref BlockIndex a block.
  VkDeviceSize frustum_compacted = 0;  ///< A @ref BlockIndex a block.
  VkDeviceSize stamps = 0;             ///< A @ref BlockStamp a block.

  /// @return Every buffer above, together.
  VkDeviceSize total() const noexcept {
    return entries + heap + bucket_mutex + compacted + frustum_compacted +
           stamps;
  }
};

/// @param num_buckets  The grid's bucket count.
/// @param bucket_size  Its slots a bucket; a grid holds a block a slot
///                     (@ref VoxelGridParams::validate).
/// @return Its buffers' sizes.
inline TableSizes table_sizes(std::uint64_t num_buckets,
                              std::uint64_t bucket_size) noexcept {
  const VkDeviceSize blocks = num_buckets * bucket_size;
  TableSizes sizes;
  sizes.entries = blocks * sizeof(HashEntry);
  sizes.heap = blocks * sizeof(std::uint32_t);
  sizes.bucket_mutex = num_buckets * sizeof(std::int32_t);
  sizes.compacted = blocks * sizeof(BlockIndex);
  sizes.frustum_compacted = blocks * sizeof(BlockIndex);
  sizes.stamps = blocks * sizeof(BlockStamp);
  return sizes;
}

}  // namespace volumetric_kit::recon::volume
