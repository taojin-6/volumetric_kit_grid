// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for the sparse voxel hash map: build a VoxelHashMap, allocate a
// known set of block coordinates, compact the active set, and verify it
// round-trips -- including reading the raw HashEntry slots back to prove the
// host<->shader scalar-layout ABI matches on-device. This exercises the real
// driver (MoltenVK on Apple, the NVIDIA ICD on the Linux CI box); it exits 0
// (skip) where no device is present.

#include <cstdint>
#include <cstdio>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/device.hpp"
#include "volumetric_kit/recon/core/instance.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/core/vulkan.hpp"
#include "volumetric_kit/recon/volume/hash.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/voxel_grid.hpp"
#include "volumetric_kit/recon/volume/voxel_hash_map.hpp"

namespace vr = volumetric_kit::recon;
namespace vol = volumetric_kit::recon::volume;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

using Coord = std::tuple<int, int, int>;

}  // namespace

int main() {
  vr::Result<vr::Instance> instance = vr::Instance::create({});
  if (!instance) {
    std::fprintf(stderr, "no Vulkan instance (%s); skipping\n",
                 instance.status().message().c_str());
    return 0;
  }
  vr::Result<VkPhysicalDevice> gpu = instance.value().select_physical_device();
  if (!gpu) {
    std::fprintf(stderr, "no compute-capable device (%s); skipping\n",
                 gpu.status().message().c_str());
    return 0;
  }
  vr::Result<vr::Device> device =
      vr::Device::create(instance.value().handle(), gpu.value(), {});
  if (!device) {
    std::fprintf(stderr, "device create failed: %s\n",
                 device.status().message().c_str());
    return 1;
  }
  vr::Result<vr::Allocator> allocator =
      vr::Allocator::create(instance.value().handle(), device.value());
  if (!allocator) {
    std::fprintf(stderr, "allocator create failed: %s\n",
                 allocator.status().message().c_str());
    return 1;
  }

  // A small grid: 1024 buckets x 8, 8192-block heap -- plenty for the test set,
  // and cheap to init.
  vol::VoxelGridParams grid{};
  grid.voxel_size = 0.005f;
  grid.block_size = 8;
  grid.voxels_per_block = 512;
  grid.trunc_dist = 0.04f;
  grid.bucket_size = 8;
  grid.num_buckets = 1024;
  grid.num_blocks = 8192;
  grid.max_chain = 128;

  vr::Result<vol::VoxelHashMap> map_result =
      vol::VoxelHashMap::create(device.value(), allocator.value(), grid);
  if (!map_result) {
    std::fprintf(stderr, "VoxelHashMap::create failed: %s\n",
                 map_result.status().message().c_str());
    return 1;
  }
  vol::VoxelHashMap map = std::move(map_result).value();

  // A 3x3x3 cube of block coordinates around the origin (27 distinct blocks).
  std::vector<vol::BlockIndex> coords;
  std::set<Coord> want;
  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      for (int z = -1; z <= 1; ++z) {
        vol::BlockIndex block{};
        block.coord = vr::Vec3i(x, y, z);
        coords.push_back(block);
        want.insert({x, y, z});
      }
    }
  }

  // Allocate; every insert should succeed.
  vr::Result<std::uint32_t> failures =
      map.allocate(coords.data(), static_cast<std::uint32_t>(coords.size()));
  CHECK(failures.ok());
  CHECK(failures.value() == 0);

  // Idempotent: re-inserting the same coords allocates nothing new and fails
  // nothing (the block-exists fast path dedups).
  vr::Result<std::uint32_t> failures2 =
      map.allocate(coords.data(), static_cast<std::uint32_t>(coords.size()));
  CHECK(failures2.ok() && failures2.value() == 0);

  // Compact: exactly the input set comes back.
  vr::Result<std::vector<vol::BlockIndex>> active = map.compact_active_blocks();
  CHECK(active.ok());
  CHECK(active.value().size() == want.size());
  std::set<Coord> got;
  for (const vol::BlockIndex& block : active.value()) {
    got.insert({block.coord.x, block.coord.y, block.coord.z});
    CHECK(block.ptr >= 0);
  }
  CHECK(got == want);

  // Scalar-ABI round-trip: read the raw HashEntry slots. Exactly 27 are
  // occupied, each pos is one of the inputs (proving `pos` lands at the host
  // offset the shader wrote), and each ptr is a whole block (a multiple of
  // voxels_per_block).
  vr::Result<std::vector<vol::HashEntry>> entries = map.read_entries();
  CHECK(entries.ok());
  int occupied = 0;
  for (const vol::HashEntry& entry : entries.value()) {
    if (entry.ptr != vol::kFreeEntry && entry.ptr != vol::kLockEntry) {
      ++occupied;
      CHECK(want.count({entry.pos.x, entry.pos.y, entry.pos.z}) == 1);
      CHECK(entry.ptr % grid.voxels_per_block == 0);
    }
  }
  CHECK(occupied == static_cast<int>(want.size()));

  // clear() empties the table.
  CHECK(map.clear().ok());
  vr::Result<std::vector<vol::BlockIndex>> after_clear =
      map.compact_active_blocks();
  CHECK(after_clear.ok());
  CHECK(after_clear.value().empty());

  // Move-only: the source is emptied, the destination lives.
  vol::VoxelHashMap moved = std::move(map);
  CHECK(!map.valid());
  CHECK(moved.valid());

  std::printf(
      "recon volume hash-map test passed: allocated + compacted %zu blocks; "
      "HashEntry scalar-layout round-trip matched on-device\n",
      want.size());
  return 0;
}
