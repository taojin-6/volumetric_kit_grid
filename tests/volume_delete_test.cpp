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

#include "buffer_readback.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/voxel_grid.hpp"
#include "volumetric_kit/recon/volume/voxel_hash_map.hpp"

#include "gpu_test.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
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
vkc::Result<std::set<Coord>> active_set(vol::VoxelHashMap& map) {
  vkc::Result<std::vector<vol::BlockIndex>> active =
      map.compact_active_blocks();
  if (!active) {
    return active.status();
  }
  std::set<Coord> out;
  for (const vol::BlockIndex& block : active.value()) {
    out.insert({block.coord.x, block.coord.y, block.coord.z});
  }
  return out;
}

// Collect the active block pointers into a set (to prove heap reuse: reused
// blocks draw the same pointers, a leaked free-list hands out fresh ones).
vkc::Result<std::set<std::int32_t>> active_ptrs(vol::VoxelHashMap& map) {
  vkc::Result<std::vector<vol::BlockIndex>> active =
      map.compact_active_blocks();
  if (!active) {
    return active.status();
  }
  std::set<std::int32_t> out;
  for (const vol::BlockIndex& block : active.value()) {
    out.insert(block.ptr);
  }
  return out;
}

int gpu_main(vr_test::GpuContext& gpu) {
  vol::VoxelGridParams grid{};
  grid.voxel_size = 0.005f;
  grid.block_size = 8;
  grid.voxels_per_block = 512;
  grid.trunc_dist = 0.04f;
  // Force the delete paths on-device: bucket_size = 2 makes the corner buckets
  // (the 8 corners hash into just two buckets, 4 apiece) overflow into
  // collision chains, so removing the corners drives the chain splice +
  // successor pull-up rather than only the trivial primary-slot clear.
  // num_blocks must equal bucket_size * num_buckets
  // (VoxelGridParams::validate), so heap reuse can't be forced by starving
  // capacity -- it is proven instead by the block-pointer set (a reused block
  // draws its old pointer back; a leaked one hands out a fresh).
  grid.bucket_size = 2;
  grid.num_buckets = 1024;
  grid.num_blocks = 2048;  // == bucket_size * num_buckets (grid invariant)
  grid.max_chain = 128;

  vkc::Result<vol::VoxelHashMap> map_result =
      vol::VoxelHashMap::create(gpu.device, gpu.allocator, grid);
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
  vkc::Result<std::uint32_t> alloc_fail =
      map.allocate(all.data(), static_cast<std::uint32_t>(all.size()));
  CHECK(alloc_fail.ok() && alloc_fail.value() == 0);

  // Snapshot the block pointers in use, to prove heap reuse after the
  // remove/re-allocate round-trip below.
  vkc::Result<std::set<std::int32_t>> ptrs_before = active_ptrs(map);
  CHECK(ptrs_before.ok() && ptrs_before.value().size() == want_all.size());

  // Remove the 8 corners.
  vkc::Result<std::uint32_t> remove_fail =
      map.remove(corners.data(), static_cast<std::uint32_t>(corners.size()));
  CHECK(remove_fail.ok() && remove_fail.value() == 0);

  // The survivors are exactly the non-corners.
  vkc::Result<std::set<Coord>> after_remove = active_set(map);
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
  vkc::Result<std::uint32_t> noop = map.remove(&absent, 1);
  CHECK(noop.ok() && noop.value() == 0);
  vkc::Result<std::set<Coord>> unchanged = active_set(map);
  CHECK(unchanged.ok() && unchanged.value() == after_remove.value());

  // Removing already-removed coords is also a no-op: their freed slots read as
  // absent under the lock, so no block is double-freed back onto the heap (a
  // double-free would later surface as a duplicate in the pointer set).
  vkc::Result<std::uint32_t> double_remove =
      map.remove(corners.data(), static_cast<std::uint32_t>(corners.size()));
  CHECK(double_remove.ok() && double_remove.value() == 0);
  vkc::Result<std::set<Coord>> still_gone = active_set(map);
  CHECK(still_gone.ok() && still_gone.value() == after_remove.value());

  // Heap reuse: re-allocating the removed corners restores the full cube AND
  // draws back exactly the block pointers that were freed. If the freed blocks
  // had leaked, the re-allocate would hand out fresh pointers and the set would
  // differ (or fail outright once the heap ran dry).
  vkc::Result<std::uint32_t> realloc_fail =
      map.allocate(corners.data(), static_cast<std::uint32_t>(corners.size()));
  CHECK(realloc_fail.ok() && realloc_fail.value() == 0);
  vkc::Result<std::set<Coord>> restored = active_set(map);
  CHECK(restored.ok());
  CHECK(restored.value() == want_all);
  vkc::Result<std::set<std::int32_t>> ptrs_after = active_ptrs(map);
  CHECK(ptrs_after.ok());
  CHECK(ptrs_after.value() == ptrs_before.value());

  // Device coordinates follow exactly the same retry/heap protocol. Duplicate
  // and absent entries must not free twice; an occupied trailing entry beyond
  // count must not be read. This list has no host mapping.
  std::vector<vol::BlockIndex> device_coords = corners;
  device_coords.insert(device_coords.end(), corners.begin(), corners.end());
  device_coords.push_back(absent);
  const auto device_count = static_cast<std::uint32_t>(device_coords.size());
  device_coords.push_back(vol::BlockIndex{});  // centre (0,0,0): must survive
  auto device_list = vr_test::upload_device_buffer(
      gpu.device, gpu.allocator, device_coords.data(),
      device_coords.size() * sizeof(vol::BlockIndex));
  CHECK(device_list.ok() && device_list->mapped() == nullptr);
  auto cached = map.compact_active_blocks_on_device();
  CHECK(cached.ok() &&
        map.check_device_block_list(cached.value(), "test").ok());
  auto device_removed = map.remove(device_list.value(), device_count);
  CHECK(device_removed.ok() && device_removed.value() == 0);
  CHECK(!map.check_device_block_list(cached.value(), "test").ok());
  auto device_survivors = active_set(map);
  CHECK(device_survivors.ok() &&
        device_survivors.value() == after_remove.value());
  device_removed = map.remove(device_list.value(), device_count);
  CHECK(device_removed.ok() && device_removed.value() == 0);
  realloc_fail =
      map.allocate(corners.data(), static_cast<std::uint32_t>(corners.size()));
  CHECK(realloc_fail.ok() && realloc_fail.value() == 0);
  auto device_ptrs = active_ptrs(map);
  CHECK(device_ptrs.ok() && device_ptrs.value() == ptrs_before.value());

  // An empty, zero-count or refused call frees nothing, so it must leave a
  // cached block list valid.
  auto kept = map.compact_active_blocks_on_device();
  CHECK(kept.ok());
  const vkc::Buffer empty;
  CHECK(map.remove(empty, 0).ok());
  CHECK(map.remove(device_list.value(), 0).ok());
  CHECK(map.remove(corners.data(), 0).ok());
  CHECK(map.remove(empty, 1).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  CHECK(map.remove(device_list.value(), device_count + 2).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  vkc::BufferDesc transfer_desc;
  transfer_desc.size = sizeof(vol::BlockIndex);
  transfer_desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  transfer_desc.memory = vkc::MemoryUsage::DeviceOnly;
  auto transfer_only = gpu.allocator.create_buffer(transfer_desc);
  CHECK(transfer_only.ok());
  CHECK(map.remove(transfer_only.value(), 1).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  CHECK(map.check_device_block_list(kept.value(), "test").ok());

  // Thousands of blocks freed in one call all go back to the heap, and are
  // drawn off it again with no pass calling a half-free heap empty. A capped
  // compare-and-swap loop lost about 960 of 2 048 removed blocks for good, and
  // made allocation report kFailHeap.
  {
    vol::VoxelGridParams wide = grid;
    wide.bucket_size = 8;
    wide.num_buckets = 4096;
    wide.num_blocks = 8 * 4096;
    vkc::Result<vol::VoxelHashMap> made =
        vol::VoxelHashMap::create(gpu.device, gpu.allocator, wide);
    CHECK(made.ok());
    vol::VoxelHashMap big = std::move(made).value();
    std::vector<vol::BlockIndex> slab;
    std::vector<vol::BlockIndex> half;
    for (int i = 0; i < 16384; ++i) {
      vol::BlockIndex block{};
      block.coord = vr::Vec3i(i % 32, (i / 32) % 32, i / 1024);
      slab.push_back(block);
      if (i % 2 == 0) half.push_back(block);
    }
    // Allocation re-drives what lock contention turned away, as a caller does,
    // but with half the map free no pass may report a capacity limit.
    const auto place = [&](const std::vector<vol::BlockIndex>& blocks) {
      std::uint32_t left = 1;
      for (int pass = 0; pass < 8 && left != 0; ++pass) {
        vol::AllocFailures failures{};
        vkc::Result<std::uint32_t> r = big.allocate(
            blocks.data(), std::uint32_t(blocks.size()), &failures);
        if (!r || failures.capacity_limited()) return false;
        left = r.value();
      }
      return left == 0;
    };
    const auto occupancy_is = [&](float want) {
      vkc::Result<float> occupancy = big.load_factor();
      return occupancy.ok() && occupancy.value() == want;
    };
    CHECK(place(slab));
    auto half_on_device =
        vr_test::upload_device_buffer(gpu.device, gpu.allocator, half.data(),
                                      half.size() * sizeof(vol::BlockIndex));
    CHECK(half_on_device.ok());
    vkc::Result<std::set<std::int32_t>> slab_ptrs = active_ptrs(big);
    CHECK(slab_ptrs.ok() && slab_ptrs.value().size() == slab.size());
    for (int cycle = 0; cycle < 3; ++cycle) {
      // One call removes them all: a retry round skips what earlier rounds
      // removed, and contention alone never ends the rounds. Lost blocks
      // (terminal) and blocks left behind fail on separate lines.
      vol::AllocFailures failures{};
      vkc::Result<std::uint32_t> removed =
          cycle % 2 == 0
              ? big.remove(half_on_device.value(), std::uint32_t(half.size()),
                           &failures)
              : big.remove(half.data(), std::uint32_t(half.size()), &failures);
      CHECK(removed.ok());
      CHECK(failures.terminal == 0);
      CHECK(removed.value() == 0);
      CHECK(occupancy_is(0.25f));
      CHECK(place(half));
      CHECK(occupancy_is(0.5f));
      // The heap is LIFO, so the freed blocks come back, each to one coord.
      vkc::Result<std::set<std::int32_t>> ptrs = active_ptrs(big);
      CHECK(ptrs.ok() && ptrs.value() == slab_ptrs.value());
    }
  }

  // Contention that outlasts every round is reported as lock contention, and
  // loses nothing. The table is one bucket, so every thread queues on one lock,
  // which settles fewer than 100 of 1 024 coords over all ten rounds on an RTX
  // 5090 or an M5 Max. lavapipe, on the CPU, settles them all, and there the
  // checks hold trivially. The 8 allocated coords lead the list; the rest are
  // absent, which the kernel still takes the lock to learn.
  {
    vol::VoxelGridParams one = grid;
    one.bucket_size = 8;
    one.num_buckets = 1;
    one.num_blocks = 8;
    vkc::Result<vol::VoxelHashMap> made =
        vol::VoxelHashMap::create(gpu.device, gpu.allocator, one);
    CHECK(made.ok());
    vol::VoxelHashMap tight = std::move(made).value();
    std::vector<vol::BlockIndex> coords(1024);
    for (std::size_t i = 0; i < coords.size(); ++i) {
      coords[i].coord = vr::Vec3i(static_cast<int>(i), 0, 0);
    }
    // Re-drives until nothing is left, as a caller does.
    const auto settle = [&](bool remove) {
      std::uint32_t left = 1;
      for (int pass = 0; pass < 8 && left != 0; ++pass) {
        vkc::Result<std::uint32_t> r = remove
                                           ? tight.remove(coords.data(), 8)
                                           : tight.allocate(coords.data(), 8);
        if (!r) return false;
        left = r.value();
      }
      return left == 0;
    };
    CHECK(settle(false));
    vol::AllocFailures failures{};
    vkc::Result<std::uint32_t> left = tight.remove(
        coords.data(), static_cast<std::uint32_t>(coords.size()), &failures);
    CHECK(left.ok());
    CHECK(failures.lock == left.value());
    CHECK(failures.terminal == 0);
    // Nothing was lost: the blocks still allocated remove, and free the heap.
    CHECK(settle(true));
    vkc::Result<float> occupancy = tight.load_factor();
    CHECK(occupancy.ok() && occupancy.value() == 0.0f);
  }

  std::printf(
      "recon volume delete test passed: removed %zu of %zu blocks, survivors "
      "verified, heap reuse restored the set\n",
      want_corners.size(), want_all.size());
  return 0;
}

}  // namespace

int main() { return vr_test::run_on_gpu(gpu_main); }
