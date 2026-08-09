// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for the sparse voxel hash map: build a VoxelHashMap, allocate a
// known set of block coordinates, compact the active set, and verify it
// round-trips -- including reading the raw HashEntry slots back to prove the
// host<->shader scalar-layout ABI matches on-device (both `pos` @4 and, via a
// second collision-forcing table, the overflow-chain `offset` @16). Also checks
// distinct heap slots, idempotent re-alloc, clear, and move semantics. This
// exercises the real driver (MoltenVK on Apple, the NVIDIA ICD on the Linux CI
// box); it exits 0 (skip) where no device is present.

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

  // Compact: exactly the input set comes back, each block drawing a DISTINCT
  // heap slot (a double-pop would hand two coords the same block -- both would
  // still look valid/aligned, so the distinctness check is what catches it).
  vr::Result<std::vector<vol::BlockIndex>> active = map.compact_active_blocks();
  CHECK(active.ok());
  CHECK(active.value().size() == want.size());
  std::set<Coord> got;
  std::set<int> ptrs;
  for (const vol::BlockIndex& block : active.value()) {
    got.insert({block.coord.x, block.coord.y, block.coord.z});
    CHECK(block.ptr >= 0);
    ptrs.insert(block.ptr);
  }
  CHECK(got == want);
  CHECK(ptrs.size() == want.size());

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

  // --- Overflow / collision-chain coverage ----------------------------------
  // The 3x3x3 cube above scatters across 1024 buckets and never fills one, so
  // it exercises neither the overflow chain nor the HashEntry.offset field (all
  // offsets stay kNoOffset). Build a second map on a tiny table and force many
  // coords into ONE bucket so the allocator spills into the collision chain,
  // then prove the chain round-trips -- i.e. `offset` lands at its
  // scalar-layout byte (@16) and is traversable on-device.
  {
    vol::VoxelGridParams cg{};
    cg.voxel_size = 0.005f;
    cg.block_size = 8;
    cg.voxels_per_block = 512;
    cg.trunc_dist = 0.04f;
    cg.bucket_size = 4;
    cg.num_buckets = 8;
    cg.num_blocks = 32;  // bucket_size * num_buckets
    cg.max_chain = 16;

    vr::Result<vol::VoxelHashMap> cmap_result =
        vol::VoxelHashMap::create(device.value(), allocator.value(), cg);
    CHECK(cmap_result.ok());
    vol::VoxelHashMap cmap = std::move(cmap_result).value();

    // Seven coords that all hash to bucket 0 -- more than bucket_size (4), so
    // the surplus must spill into the overflow chain.
    const std::size_t kColliding = 7;
    std::vector<vol::BlockIndex> cc;
    std::set<Coord> cwant;
    for (int x = 1; cc.size() < kColliding; ++x) {
      vr::Vec3i coord(x, 0, 0);
      if (vol::hash_bucket(coord, cg.num_buckets) == 0u) {
        vol::BlockIndex block{};
        block.coord = coord;
        cc.push_back(block);
        cwant.insert({x, 0, 0});
      }
    }

    // Insert one per dispatch: a single dispatch of same-bucket coords has all
    // threads hammer bucket 0's spin lock (the known SIMD-contention failure
    // mode, which legitimately returns failures for the host to retry), so
    // build the chain deterministically one coord at a time -- exactly the
    // cross-dispatch retry the host is expected to do.
    for (const vol::BlockIndex& block : cc) {
      vr::Result<std::uint32_t> f = cmap.allocate(&block, 1);
      CHECK(f.ok() && f.value() == 0);
    }

    // All seven come back, each with a distinct heap block.
    vr::Result<std::vector<vol::BlockIndex>> cactive =
        cmap.compact_active_blocks();
    CHECK(cactive.ok());
    CHECK(cactive.value().size() == cwant.size());
    std::set<Coord> cgot;
    std::set<int> cptrs;
    for (const vol::BlockIndex& block : cactive.value()) {
      cgot.insert({block.coord.x, block.coord.y, block.coord.z});
      cptrs.insert(block.ptr);
    }
    CHECK(cgot == cwant);
    CHECK(cptrs.size() == cwant.size());

    // The chain is real: bucket 0's anchor (the last slot of its bucket) must
    // carry a non-zero offset -- i.e. the surplus coords were linked through
    // HashEntry.offset, proving that field round-trips at its scalar-layout
    // byte (the 3x3x3 test, all offsets kNoOffset, never checks it).
    vr::Result<std::vector<vol::HashEntry>> centries = cmap.read_entries();
    CHECK(centries.ok());
    const std::size_t anchor =
        static_cast<std::size_t>((0 + 1) * cg.bucket_size - 1);
    CHECK(centries.value()[anchor].offset != vol::kNoOffset);

    // Idempotent under collisions: re-inserting the same coords must find the
    // chained ones by TRAVERSING `offset` and allocate nothing new -- if offset
    // were mis-read the chained coords would be re-inserted and the count grow.
    vr::Result<std::uint32_t> cfail2 =
        cmap.allocate(cc.data(), static_cast<std::uint32_t>(cc.size()));
    CHECK(cfail2.ok() && cfail2.value() == 0);
    vr::Result<std::vector<vol::BlockIndex>> cactive2 =
        cmap.compact_active_blocks();
    CHECK(cactive2.ok());
    CHECK(cactive2.value().size() == cwant.size());
  }

  // --- The entries accessors, which a consumer is told to use as a pair
  // ------- A live map reports a real handle and the table's byte size; the two
  // are written into one VkDescriptorBufferInfo by an out-of-tier kernel that
  // probes the table itself (mesh's sparse marching cubes).
  CHECK(map.entries_buffer() != VK_NULL_HANDLE);
  CHECK(map.entries_buffer_size() ==
        static_cast<VkDeviceSize>(grid.num_buckets) *
            static_cast<VkDeviceSize>(grid.bucket_size) *
            sizeof(vol::HashEntry));

  // --- Move-only ------------------------------------------------------------
  // Move-construct: the source empties, the destination lives.
  vol::VoxelHashMap moved = std::move(map);
  CHECK(!map.valid());
  CHECK(moved.valid());

  // Both entries accessors must agree with valid(). grid_ is a POD the
  // defaulted move COPIES, so the size accessor can still see the table shape
  // after the buffer is gone -- and the pair it is meant to be used with would
  // then feed {VK_NULL_HANDLE, 0, size} to vkUpdateDescriptorSets, invalid
  // usage this repo enables neither nullDescriptor nor robustBufferAccess to
  // survive. Reverting the size accessor's valid() gate fails here.
  CHECK(map.entries_buffer() == VK_NULL_HANDLE);
  CHECK(map.entries_buffer_size() == 0);
  CHECK(moved.entries_buffer() != VK_NULL_HANDLE);
  CHECK(moved.entries_buffer_size() > 0);

  // Move-assign over a live object: build a second map and move it over
  // `moved`; the source empties and the destination stays live (its prior
  // buffers / pipelines are released by the move-assign -- ASan turns a
  // leak/double-free here into a failure).
  vr::Result<vol::VoxelHashMap> other_result =
      vol::VoxelHashMap::create(device.value(), allocator.value(), grid);
  CHECK(other_result.ok());
  vol::VoxelHashMap other = std::move(other_result).value();
  moved = std::move(other);
  CHECK(!other.valid());
  CHECK(moved.valid());

  // Self-move (laundered through a pointer to dodge -Wself-move under -Werror):
  // the guarded move leaves the object intact.
  vol::VoxelHashMap* alias = &moved;
  moved = std::move(*alias);
  CHECK(moved.valid());

  std::printf(
      "recon volume hash-map test passed: allocated + compacted %zu blocks; "
      "HashEntry scalar-layout round-trip (incl. overflow-chain offset) "
      "matched on-device\n",
      want.size());
  return 0;
}
