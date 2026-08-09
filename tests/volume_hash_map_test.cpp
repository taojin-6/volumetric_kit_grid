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

// Find `n` distinct block coords that all hash to `target_bucket`, so they land
// in one primary bucket and its collision chain. Mirrors the helper in
// volume_diagnostics_test.cpp, and bounded for the same reason: an open
// `for (int x = 1; out.size() < n; ++x)` terminates only by success, so
// changing a hash prime, a bucket count, or the fixed y/z pair turns a failing
// test into an infinite loop with signed-overflow UB instead of a red build. A
// short return fails the caller's CHECK, which is a diagnosis.
//
// Sweeps from 1 rather than 0 to keep (0,0,0) out of every fixture: it is also
// the value a freshly-initialised `HashEntry::pos` holds, and block_exists
// documents that coord as the one a stale read could false-match.
std::vector<vol::BlockIndex> coords_in_bucket(int target_bucket,
                                              int num_buckets, std::size_t n) {
  std::vector<vol::BlockIndex> out;
  for (int x = 1; x < 64 && out.size() < n; ++x) {
    for (int y = 1; y < 64 && out.size() < n; ++y) {
      for (int z = 1; z < 64 && out.size() < n; ++z) {
        const vr::Vec3i c(x, y, z);
        if (static_cast<int>(vol::hash_bucket(c, num_buckets)) ==
            target_bucket) {
          vol::BlockIndex block{};
          block.coord = c;
          out.push_back(block);
        }
      }
    }
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
    std::vector<vol::BlockIndex> cc =
        coords_in_bucket(0, cg.num_buckets, kColliding);
    CHECK(cc.size() == kColliding);
    std::set<Coord> cwant;
    for (const vol::BlockIndex& block : cc) {
      cwant.insert({block.coord.x, block.coord.y, block.coord.z});
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

  // --- The overflow sweep reaches, and its capacity reasons are exact -------
  // allocate_in_overflow sweeps the WHOLE table for a free non-anchor slot.
  // What had made that unaffordable was its cost per slot -- a contended
  // atomicCompSwap taken before it had even looked -- so it now reads each
  // candidate unlocked first and locks only one that looks free. Two properties
  // need pinning: that the sweep really does reach a distant slot, and that
  // when it does fail it names the cause it can prove.
  {
    vol::VoxelGridParams pg{};
    pg.voxel_size = 0.005f;
    pg.block_size = 8;
    pg.voxels_per_block = 512;
    pg.trunc_dist = 0.04f;
    pg.bucket_size = 8;
    pg.num_buckets = 256;
    pg.num_blocks = 2048;  // bucket_size * num_buckets
    pg.max_chain = 512;    // > kDeepChain, so the chain is never the limit

    vr::Result<vol::VoxelHashMap> pmap_result =
        vol::VoxelHashMap::create(device.value(), allocator.value(), pg);
    CHECK(pmap_result.ok());
    vol::VoxelHashMap pmap = std::move(pmap_result).value();

    // Reach. One bucket's chain is driven to 300 on a table that stays ~85%
    // free, so every insert is a slot an exhaustive sweep finds and a windowed
    // one does not. The depth is the discriminator: overflow nodes pack in from
    // the bucket's end skipping one anchor per 8 slots, so the 292nd sits 333
    // iterations out -- past any fixed window that would have been small enough
    // to be worth having. A cap of 256 fails this at roughly the 232nd coord.
    const std::size_t kDeepChain = 300;
    std::vector<vol::BlockIndex> deep =
        coords_in_bucket(0, pg.num_buckets, kDeepChain);
    CHECK(deep.size() == kDeepChain);

    // One per dispatch, as above: a single dispatch of same-bucket coords
    // contends the spin lock and legitimately reports retryable failures, which
    // would muddle what this is measuring.
    for (const vol::BlockIndex& block : deep) {
      vr::Result<std::uint32_t> f = pmap.allocate(&block, 1);
      CHECK(f.ok() && f.value() == 0);
    }

    // All 300 landed, each on its own block.
    vr::Result<std::vector<vol::BlockIndex>> pactive =
        pmap.compact_active_blocks();
    CHECK(pactive.ok());
    CHECK(pactive.value().size() == kDeepChain);
    std::set<int> pptrs;
    for (const vol::BlockIndex& block : pactive.value()) {
      pptrs.insert(block.ptr);
    }
    CHECK(pptrs.size() == kDeepChain);

    // ...and the chain that links them is traversable, which the size check
    // above cannot show: compact_active_blocks is a flat per-slot scan that
    // never walks `offset`, so a node written with a broken link still appears
    // in the active set and still counts 300. Re-allocating the whole set in
    // ONE dispatch has to find every one of them by hopping the chain, so a bad
    // link re-inserts instead and the count grows.
    vr::Result<std::uint32_t> prefail =
        pmap.allocate(deep.data(), static_cast<std::uint32_t>(deep.size()));
    CHECK(prefail.ok() && prefail.value() == 0);
    vr::Result<std::vector<vol::BlockIndex>> pactive2 =
        pmap.compact_active_blocks();
    CHECK(pactive2.ok());
    CHECK(pactive2.value().size() == kDeepChain);

    vr::Result<std::vector<vol::HashEntry>> pentries = pmap.read_entries();
    CHECK(pentries.ok());
    const auto panchor = static_cast<std::size_t>(pg.bucket_size - 1);
    CHECK(pentries.value()[panchor].offset != vol::kNoOffset);

    // Occupancy is readable without the O(total slots) diagnostics scan -- the
    // cheap signal a caller needs to grow *before* it starts failing.
    vr::Result<float> plf = pmap.load_factor();
    CHECK(plf.ok());
    CHECK(plf.value() > 0.14f && plf.value() < 0.15f);  // 300 / 2048
  }

  // Exactness of the two capacity reasons, on a table small enough to sweep by
  // hand: 2 buckets x 2 entries. Slot 1 and slot 3 are the per-bucket chain
  // anchors, which an overflow insert skips, so slots 0 and 2 are the only ones
  // it may ever take.
  {
    vol::VoxelGridParams tg{};
    tg.voxel_size = 0.005f;
    tg.block_size = 8;
    tg.voxels_per_block = 512;
    tg.trunc_dist = 0.04f;
    tg.bucket_size = 2;
    tg.num_buckets = 2;
    tg.num_blocks = 4;  // bucket_size * num_buckets
    tg.max_chain = 8;

    vr::Result<vol::VoxelHashMap> tmap_result =
        vol::VoxelHashMap::create(device.value(), allocator.value(), tg);
    CHECK(tmap_result.ok());
    vol::VoxelHashMap tmap = std::move(tmap_result).value();

    std::vector<vol::BlockIndex> b0 = coords_in_bucket(0, tg.num_buckets, 4);
    CHECK(b0.size() == 4);

    // Three coords on bucket 0 take its primary pair (slots 0 and 1 -- the
    // anchor is closed only to *overflow*) and then spill into slot 2. One
    // block is left on the heap and one slot is free, but that slot is bucket
    // 1's anchor.
    for (std::size_t i = 0; i < 3; ++i) {
      vr::Result<std::uint32_t> f = tmap.allocate(&b0[i], 1);
      CHECK(f.ok() && f.value() == 0);
    }

    // So the fourth sweeps all four slots and legitimately finds nothing it may
    // use. kFailTable is the whole claim under test: not kFailHeap (a block is
    // still on the heap, and load_factor proves it), and not a sweep that gave
    // up early -- a windowed probe reports the same code for "I stopped
    // looking", which is what makes the reason unusable for deciding to grow.
    vr::Result<float> tlf = tmap.load_factor();
    CHECK(tlf.ok() && tlf.value() < 1.0f);
    vol::AllocFailures tf{};
    vr::Result<std::uint32_t> f4 = tmap.allocate(&b0[3], 1, &tf);
    CHECK(f4.ok() && f4.value() > 0);
    CHECK(tf.table > 0);
    CHECK(tf.heap == 0);
    CHECK(tf.capacity_limited());

    // Fill that last anchor through bucket 1's primary path, emptying the heap.
    std::vector<vol::BlockIndex> b1 = coords_in_bucket(1, tg.num_buckets, 1);
    CHECK(b1.size() == 1);
    vr::Result<std::uint32_t> f5 = tmap.allocate(&b1[0], 1);
    CHECK(f5.ok() && f5.value() == 0);
    vr::Result<float> tfull = tmap.load_factor();
    CHECK(tfull.ok() && tfull.value() == 1.0f);

    // Now the identical insert must report kFailHeap instead. Nothing about the
    // sweep changed for it -- the same four slots reach the same conclusion --
    // so this pins the early-out that answers an empty heap with one atomic
    // load rather than a full sweep, and pins that it attributes the failure
    // the way allocate_in_primary already names this same state. Without the
    // early-out the sweep runs to exhaustion and reports `table` here, failing
    // the pair.
    vol::AllocFailures hf{};
    vr::Result<std::uint32_t> f6 = tmap.allocate(&b0[3], 1, &hf);
    CHECK(f6.ok() && f6.value() > 0);
    CHECK(hf.heap > 0);
    CHECK(hf.table == 0);
  }

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
