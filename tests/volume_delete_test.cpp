// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for VoxelHashMap::remove: allocate a cube of blocks, delete a
// subset, and verify the survivors remain while the deleted ones are gone --
// then re-allocate the deleted coords to confirm their blocks went back to the
// heap (chain splice / successor pull-up + heap free). Runs on the real driver
// (MoltenVK / NVIDIA); exits 0 (skip) where no device is present.

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

// Collect the active-block coords into a set.
vr::Result<std::set<Coord>> active_set(vol::VoxelHashMap& map) {
  vr::Result<std::vector<vol::BlockIndex>> active = map.compact_active_blocks();
  if (!active) {
    return active.status();
  }
  std::set<Coord> out;
  for (const vol::BlockIndex& block : active.value()) {
    out.insert({block.coord.x, block.coord.y, block.coord.z});
  }
  return out;
}

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

  // A 3x3x3 cube of blocks; the 8 corners (all axes at +/-1) are the delete
  // set.
  std::vector<vol::BlockIndex> all;
  std::vector<vol::BlockIndex> corners;
  std::set<Coord> want_all;
  std::set<Coord> want_corners;
  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      for (int z = -1; z <= 1; ++z) {
        vol::BlockIndex block{};
        block.coord = vr::Vec3i(x, y, z);
        all.push_back(block);
        want_all.insert({x, y, z});
        if (x != 0 && y != 0 && z != 0) {
          corners.push_back(block);
          want_corners.insert({x, y, z});
        }
      }
    }
  }
  CHECK(corners.size() == 8);

  // Allocate the full cube.
  vr::Result<std::uint32_t> alloc_fail =
      map.allocate(all.data(), static_cast<std::uint32_t>(all.size()));
  CHECK(alloc_fail.ok() && alloc_fail.value() == 0);

  // Remove the 8 corners.
  vr::Result<std::uint32_t> remove_fail =
      map.remove(corners.data(), static_cast<std::uint32_t>(corners.size()));
  CHECK(remove_fail.ok() && remove_fail.value() == 0);

  // The survivors are exactly the non-corners.
  vr::Result<std::set<Coord>> after_remove = active_set(map);
  CHECK(after_remove.ok());
  CHECK(after_remove.value().size() == want_all.size() - want_corners.size());
  for (const Coord& corner : want_corners) {
    CHECK(after_remove.value().count(corner) == 0);
  }
  for (const Coord& coord : want_all) {
    const bool is_corner = want_corners.count(coord) != 0;
    CHECK(after_remove.value().count(coord) == (is_corner ? 0u : 1u));
  }

  // Removing an absent coord is a no-op (0 failures, set unchanged).
  vol::BlockIndex absent{};
  absent.coord = vr::Vec3i(100, 100, 100);
  vr::Result<std::uint32_t> noop = map.remove(&absent, 1);
  CHECK(noop.ok() && noop.value() == 0);
  vr::Result<std::set<Coord>> unchanged = active_set(map);
  CHECK(unchanged.ok() && unchanged.value() == after_remove.value());

  // Heap reuse: re-allocating the removed corners succeeds (their blocks were
  // returned to the heap) and restores the full cube.
  vr::Result<std::uint32_t> realloc_fail =
      map.allocate(corners.data(), static_cast<std::uint32_t>(corners.size()));
  CHECK(realloc_fail.ok() && realloc_fail.value() == 0);
  vr::Result<std::set<Coord>> restored = active_set(map);
  CHECK(restored.ok());
  CHECK(restored.value() == want_all);

  std::printf(
      "recon volume delete test passed: removed %zu of %zu blocks, survivors "
      "verified, heap reuse restored the set\n",
      want_corners.size(), want_all.size());
  return 0;
}
