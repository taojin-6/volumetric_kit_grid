// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// A test's block coordinates handed to the volume tier as it takes them, on
// the device: uploaded at the test boundary, then allocated or removed through
// the device-list overloads, as a kernel's output would be.

#include <cstdint>
#include <vector>

#include "buffer_readback.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"
#include "volumetric_kit/recon/volume/voxel_grid.hpp"
#include "volumetric_kit/recon/volume/voxel_hash_map.hpp"

namespace vr_test {

namespace detail {

// The first `count` of `coords` as a device list; none for a zero count,
// which every device-list call takes as a no-op.
inline vkc::Result<vkc::Buffer> device_coords(
    const Gpu& gpu, const volumetric_kit::recon::volume::BlockIndex* coords,
    std::uint32_t count) {
  if (count == 0) return vkc::Buffer{};
  return upload_device_buffer(
      gpu.device, gpu.allocator, coords,
      VkDeviceSize(count) * sizeof(volumetric_kit::recon::volume::BlockIndex));
}

}  // namespace detail

// VoxelHashMap::allocate over the first `count` of `coords`.
inline vkc::Result<std::uint32_t> allocate_blocks(
    const Gpu& gpu, volumetric_kit::recon::volume::VoxelHashMap& map,
    const volumetric_kit::recon::volume::BlockIndex* coords,
    std::uint32_t count,
    volumetric_kit::recon::volume::AllocFailures* out_failures = nullptr) {
  VKC_ASSIGN(const vkc::Buffer list, detail::device_coords(gpu, coords, count));
  return map.allocate(list, count, out_failures);
}

// VoxelHashMap::allocate over all of `coords`.
inline vkc::Result<std::uint32_t> allocate_blocks(
    const Gpu& gpu, volumetric_kit::recon::volume::VoxelHashMap& map,
    const std::vector<volumetric_kit::recon::volume::BlockIndex>& coords,
    volumetric_kit::recon::volume::AllocFailures* out_failures = nullptr) {
  return allocate_blocks(gpu, map, coords.data(),
                         static_cast<std::uint32_t>(coords.size()),
                         out_failures);
}

// VoxelHashMap::remove over the first `count` of `coords`: the index alone,
// the attributes left as they are.
inline vkc::Result<std::uint32_t> remove_blocks(
    const Gpu& gpu, volumetric_kit::recon::volume::VoxelHashMap& map,
    const volumetric_kit::recon::volume::BlockIndex* coords,
    std::uint32_t count,
    volumetric_kit::recon::volume::AllocFailures* out_failures = nullptr) {
  VKC_ASSIGN(const vkc::Buffer list, detail::device_coords(gpu, coords, count));
  return map.remove(list, count, out_failures);
}

// VoxelHashMap::remove over all of `coords`.
inline vkc::Result<std::uint32_t> remove_blocks(
    const Gpu& gpu, volumetric_kit::recon::volume::VoxelHashMap& map,
    const std::vector<volumetric_kit::recon::volume::BlockIndex>& coords,
    volumetric_kit::recon::volume::AllocFailures* out_failures = nullptr) {
  return remove_blocks(gpu, map, coords.data(),
                       static_cast<std::uint32_t>(coords.size()), out_failures);
}

// VoxelBlockGrid::remove over the first `count` of `coords`: zeroed, then
// freed.
inline vkc::Result<std::uint32_t> remove_blocks(
    const Gpu& gpu, volumetric_kit::recon::volume::VoxelBlockGrid& grid,
    const volumetric_kit::recon::volume::BlockIndex* coords,
    std::uint32_t count,
    volumetric_kit::recon::volume::AllocFailures* out_failures = nullptr) {
  VKC_ASSIGN(const vkc::Buffer list, detail::device_coords(gpu, coords, count));
  return grid.remove(list, count, out_failures);
}

// VoxelBlockGrid::remove over all of `coords`.
inline vkc::Result<std::uint32_t> remove_blocks(
    const Gpu& gpu, volumetric_kit::recon::volume::VoxelBlockGrid& grid,
    const std::vector<volumetric_kit::recon::volume::BlockIndex>& coords,
    volumetric_kit::recon::volume::AllocFailures* out_failures = nullptr) {
  return remove_blocks(gpu, grid, coords.data(),
                       static_cast<std::uint32_t>(coords.size()), out_failures);
}

// How many of `coords` a fresh map of `grid`'s table shape refuses for chain
// room when its chains may not grow (max_chain 1), each coord allocated in a
// call of its own so no lock race enters the count. Coords that share a
// bucket on the device fill its slots and the rest are refused, so this is the
// device's own word on which coords collide -- what a chain fixture chosen by
// the tests' hash oracle (volume_hash_oracle.hpp) rests on, and what a GLSL
// hash drifted from it would change.
inline vkc::Result<std::uint32_t> chain_refusals(
    const Gpu& gpu, volumetric_kit::recon::volume::VoxelGridParams grid,
    const std::vector<volumetric_kit::recon::volume::BlockIndex>& coords) {
  grid.max_chain = 1;
  VKC_ASSIGN(volumetric_kit::recon::volume::VoxelHashMap map,
             volumetric_kit::recon::volume::VoxelHashMap::create(
                 gpu.device, gpu.allocator, grid));
  std::uint32_t refused = 0;
  for (const volumetric_kit::recon::volume::BlockIndex& coord : coords) {
    volumetric_kit::recon::volume::AllocFailures why{};
    VKC_TRY(allocate_blocks(gpu, map, &coord, 1, &why).status());
    refused += why.chain;
  }
  return refused;
}

}  // namespace vr_test
