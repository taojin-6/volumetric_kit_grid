// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file hash_map_internal.hpp
/// @brief @ref VoxelHashMap::Internal: the hash-entry array, for the voxel
///        grid's own kernels that resolve blocks by coordinate on the device.
///
/// Internal (under src/, never installed): the codec, mesh and voxelize tiers
/// include it on their private `src/` path and read the table through
/// `volume/shaders/hash_lookup.glsl` or `hash_common.glsl`, whose layout is
/// internal too. A consumer's kernel gets the grid's contract alone
/// (`grid_types.glsl`), never a slot.

#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/recon/volume/voxel_hash_map.hpp"

#include "table_sizes.hpp"

namespace volumetric_kit::recon::volume {

class VoxelHashMap::Internal {
 public:
  Internal() = delete;

  /// @brief The device hash-entry array, for a kernel that resolves blocks by
  ///        coordinate itself instead of being handed a host-built table.
  ///
  /// The alternative is worse. A kernel that needs the neighbourhood of N
  /// blocks otherwise pays an O(N) serial host pass of coordinate lookups
  /// plus an upload -- 102 ms of a 133 ms mesh extract at 107k blocks,
  /// measured on an M5 iPad Pro -- to hand the GPU a table it could have
  /// built itself in parallel. `volume/shaders/hash_lookup.glsl` is the
  /// read-only traversal, mirroring `find_block` and taking the table shape
  /// as arguments.
  ///
  /// @warning Read-only, and only safe while **nothing is mutating the
  ///          table** -- no allocate / remove / clear / resize dispatch in
  ///          flight. Probing concurrently with an insert would race the
  ///          bucket locks this deliberately does not expose. A meshing pass
  ///          qualifies because it is quiescent by construction; a pass that
  ///          also allocates does not.
  /// @warning **Do not cache the handle.** A completed @ref resize replaces
  ///          the entry buffer outright -- the old allocation is destroyed, so
  ///          a handle held in a persistent descriptor set then names freed
  ///          memory, which is a validation-layer-only diagnostic and
  ///          undefined with layers off. Both shipped examples resize mid-scan
  ///          on block-heap overflow. Re-fetch on every use, exactly as
  ///          @ref AttributeView requires across a @ref VoxelBlockGrid::resize;
  ///          a move of the map has the same effect.
  /// @return The entry buffer, or `VK_NULL_HANDLE` on a moved-from map.
  static VkBuffer entries_buffer(const VoxelHashMap& map) noexcept {
    return map.entries_.valid() ? map.entries_.handle() : VK_NULL_HANDLE;
  }

  /// @brief Bytes in @ref entries_buffer, for a `VK_WHOLE_SIZE`-free binding.
  ///
  /// Pair it with @ref entries_buffer: bind the range rather than
  /// `VK_WHOLE_SIZE` so the caller can check it against the device's
  /// `maxStorageBufferRange` first. That limit matters here -- the table holds
  /// an entry a slot and @ref resize doubles `num_buckets`, while Vulkan
  /// guarantees only 2^27 (128 MiB), which is what Android-class drivers
  /// report and no driver this repo tests on does.
  /// @return The size in bytes, or 0 on a moved-from map (so the pair stays
  ///         consistent: a null handle never carries a non-zero range).
  static VkDeviceSize entries_buffer_size(const VoxelHashMap& map) noexcept {
    // Gated on the same `valid()` as entries_buffer(), not on grid_ alone:
    // grid_ is a POD the defaulted move COPIES, so a moved-from map still
    // knows its table shape while owning no buffer. Reporting the shape there
    // would pair a VK_NULL_HANDLE with a non-zero range in exactly the
    // descriptor write these two exist to spell -- invalid usage this repo
    // cannot detect (neither nullDescriptor nor robustBufferAccess is
    // enabled), and one that looks well-formed in a debugger.
    if (!map.entries_.valid()) return 0;
    return table_sizes(std::uint64_t(map.grid_.num_buckets),
                       std::uint64_t(map.grid_.bucket_size))
        .entries;
  }
};

}  // namespace volumetric_kit::recon::volume
