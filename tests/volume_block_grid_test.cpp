// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for VoxelBlockGrid: declare independent per-voxel attributes (SoA),
// verify they are distinct, correctly-sized device buffers keyed by the block
// pool, that the composed VoxelHashMap still allocates, that resize grows the
// attribute arrays while preserving per-voxel data at the same block pointer,
// and the error / move paths. Runs on the real driver (MoltenVK / NVIDIA);
// exits 0 (skip) where no device is present.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/device.hpp"
#include "volumetric_kit/recon/core/instance.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/core/vulkan.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"
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

vol::VoxelGridParams small_grid() {
  vol::VoxelGridParams grid{};
  grid.voxel_size = 0.005f;
  grid.block_size = 8;
  grid.voxels_per_block = 512;
  grid.trunc_dist = 0.04f;
  grid.bucket_size = 8;
  grid.num_buckets = 256;
  grid.num_blocks = 256 * 8;
  grid.max_chain = 128;
  return grid;
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

  const vol::VoxelGridParams grid = small_grid();
  const std::uint64_t voxels =
      static_cast<std::uint64_t>(grid.num_blocks) * grid.voxels_per_block;

  // Declare two independent float attributes (SoA): tsdf + weight.
  const vol::AttributeSpec attrs[] = {{"tsdf", sizeof(float)},
                                      {"weight", sizeof(float)}};
  vr::Result<vol::VoxelBlockGrid> grid_result = vol::VoxelBlockGrid::create(
      device.value(), allocator.value(), grid, attrs, 2);
  if (!grid_result) {
    std::fprintf(stderr, "VoxelBlockGrid::create failed: %s\n",
                 grid_result.status().message().c_str());
    return 1;
  }
  vol::VoxelBlockGrid vbg = std::move(grid_result).value();

  // Attribute presence + lookup.
  CHECK(vbg.valid());
  CHECK(vbg.has_attribute("tsdf"));
  CHECK(vbg.has_attribute("weight"));
  CHECK(!vbg.has_attribute("color"));

  vr::Result<vol::AttributeView> tsdf = vbg.attribute("tsdf");
  vr::Result<vol::AttributeView> weight = vbg.attribute("weight");
  CHECK(tsdf.ok() && weight.ok());
  CHECK(vbg.attribute("color").status().domain() ==
        vr::Status::Code::InvalidArgument);

  // Each attribute is its own buffer, sized to the whole voxel pool.
  CHECK(tsdf.value().element_size == sizeof(float));
  CHECK(tsdf.value().element_count == voxels);
  CHECK(weight.value().element_count == voxels);
  CHECK(tsdf.value().buffer != weight.value().buffer);  // distinct SoA arrays
  CHECK(tsdf.value().buffer->valid());
  CHECK(tsdf.value().buffer->size() == voxels * sizeof(float));

  // SoA independence: writing one attribute leaves the other untouched. The
  // buffers are host-visible + zero-initialised, so a plain host round-trip
  // exercises the mapping (no device work needed for the storage itself).
  auto* tsdf_data = static_cast<float*>(tsdf.value().buffer->mapped());
  auto* weight_data = static_cast<float*>(weight.value().buffer->mapped());
  CHECK(tsdf_data != nullptr && weight_data != nullptr);
  const std::uint64_t last = voxels - 1;
  CHECK(tsdf_data[0] == 0.0f && weight_data[0] == 0.0f);  // zero-initialised
  tsdf_data[0] = 1.5f;
  tsdf_data[last] = 2.5f;
  weight_data[0] = 10.0f;
  weight_data[last] = 20.0f;
  CHECK(tsdf_data[0] == 1.5f && tsdf_data[last] == 2.5f);
  CHECK(weight_data[0] == 10.0f && weight_data[last] == 20.0f);

  // The composed block index still allocates: a 3x3x3 cube, indexed by ptr.
  std::vector<vol::BlockIndex> cube;
  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      for (int z = -1; z <= 1; ++z) {
        vol::BlockIndex block{};
        block.coord = vr::Vec3i(x, y, z);
        cube.push_back(block);
      }
    }
  }
  vr::Result<std::uint32_t> allocated =
      vbg.map().allocate(cube.data(), static_cast<std::uint32_t>(cube.size()));
  CHECK(allocated.ok() && allocated.value() == 0);
  vr::Result<std::vector<vol::BlockIndex>> active =
      vbg.map().compact_active_blocks();
  CHECK(active.ok() && active.value().size() == cube.size());

  // A block's ptr keys into the attribute arrays at [ptr, ptr +
  // voxels_per_block): ptr is voxel-granular (block_idx * voxels_per_block), so
  // two distinct blocks address disjoint ranges. Probing two blocks (not one
  // same-index round-trip) proves the scaling -- a bare-block-index ptr would
  // fail the modulo check and alias its neighbours.
  CHECK(active.value().size() >= 2);
  const std::int32_t vpb = grid.voxels_per_block;
  const std::int32_t ptr_a = active.value()[0].ptr;
  const std::int32_t ptr_b = active.value()[1].ptr;
  CHECK(ptr_a != ptr_b);
  CHECK(ptr_a % vpb == 0 && ptr_b % vpb == 0);  // voxel-granular block base
  CHECK(ptr_a >= 0 && static_cast<std::uint64_t>(ptr_a) + vpb <= voxels);
  CHECK(ptr_b >= 0 && static_cast<std::uint64_t>(ptr_b) + vpb <= voxels);
  // Fill block A's whole tsdf range, then set only block B's base: disjoint
  // ranges mean B never reaches A's last voxel and A's fill never reaches B.
  for (std::int32_t i = 0; i < vpb; ++i) {
    tsdf_data[ptr_a + i] = -0.02f;
  }
  tsdf_data[ptr_b] = 0.75f;
  CHECK(tsdf_data[ptr_a + vpb - 1] == -0.02f);  // B did not overwrite A
  CHECK(tsdf_data[ptr_b] == 0.75f);             // A did not overwrite B
  // SoA independence at a live block: writing block A's weight leaves its tsdf
  // untouched (independent buffers, not interleaved AoS).
  weight_data[ptr_a] = 30.0f;
  CHECK(weight_data[ptr_a] == 30.0f && tsdf_data[ptr_a] == -0.02f);

  // Resize with attribute preservation: VoxelBlockGrid::resize grows every
  // attribute array AND rehashes the map preserving block indices, so the
  // per-voxel data written above survives at the SAME ptr. Snapshot the live
  // block count first.
  vr::Result<std::vector<vol::BlockIndex>> before =
      vbg.map().compact_active_blocks();
  CHECK(before.ok());
  const std::size_t before_count = before.value().size();

  const std::int32_t new_buckets = grid.num_buckets * 4;
  CHECK(vbg.resize(new_buckets).ok());
  CHECK(vbg.map().grid().num_buckets == new_buckets);  // the map grew
  const std::uint64_t new_voxels =
      static_cast<std::uint64_t>(grid.bucket_size) *
      static_cast<std::uint64_t>(new_buckets) * grid.voxels_per_block;

  // The attribute buffers grew to the new pool (a resize swaps them), so
  // re-fetch the views: element_count now tracks the grown grid, buffer-derived
  // as ever.
  vr::Result<vol::AttributeView> tsdf_grown = vbg.attribute("tsdf");
  vr::Result<vol::AttributeView> weight_grown = vbg.attribute("weight");
  CHECK(tsdf_grown.ok() && weight_grown.ok());
  CHECK(tsdf_grown.value().element_count == new_voxels);  // grew, not frozen
  CHECK(weight_grown.value().element_count == new_voxels);
  CHECK(tsdf_grown.value().buffer->size() == new_voxels * sizeof(float));

  // The data written before the grow survives at the same pointers (indices
  // preserved): block A's filled tsdf range, block B's base, block A's weight.
  auto* tsdf_grown_data =
      static_cast<float*>(tsdf_grown.value().buffer->mapped());
  auto* weight_grown_data =
      static_cast<float*>(weight_grown.value().buffer->mapped());
  CHECK(tsdf_grown_data[ptr_a] == -0.02f);
  CHECK(tsdf_grown_data[ptr_a + vpb - 1] == -0.02f);  // whole range survived
  CHECK(tsdf_grown_data[ptr_b] == 0.75f);
  CHECK(weight_grown_data[ptr_a] == 30.0f);

  // The active set is intact, and the grown table still allocates a fresh block
  // whose attribute range lands in the new pool and reads back zero.
  const vr::Vec3i fresh_coord(40, 40, 40);
  vol::BlockIndex fresh{};
  fresh.coord = fresh_coord;
  vr::Result<std::uint32_t> fresh_alloc = vbg.map().allocate(&fresh, 1);
  CHECK(fresh_alloc.ok() && fresh_alloc.value() == 0);
  vr::Result<std::vector<vol::BlockIndex>> with_fresh =
      vbg.map().compact_active_blocks();
  CHECK(with_fresh.ok() && with_fresh.value().size() == before_count + 1);
  std::int32_t fresh_ptr = -1;
  for (const vol::BlockIndex& blk : with_fresh.value()) {
    if (blk.coord == fresh_coord) {
      fresh_ptr = blk.ptr;
    }
  }
  CHECK(fresh_ptr >= 0);
  CHECK(static_cast<std::uint64_t>(fresh_ptr) + vpb <= new_voxels);  // in pool
  CHECK(tsdf_grown_data[fresh_ptr] == 0.0f);  // fresh block: zeroed attribute

  // --- remove() clears the per-voxel data before the index goes back.
  // The free heap is LIFO, so the next allocation re-draws the index just
  // freed onto the identical attribute range. Left uncleared, the removed
  // surface's tsdf/weight resurrect under the new geometry at full fused
  // weight -- and the integrator's `color_attr == 0` first-observation gate,
  // which exists to stop a wrong first colour, is bypassed by the stale
  // non-zero value. Written through the grid, which owns both halves;
  // map().remove() is the raw path that does not clear (documented there).
  {
    // Give the block being removed a distinctive, definitely-non-zero range.
    tsdf_grown_data[fresh_ptr] = -0.5f;
    tsdf_grown_data[fresh_ptr + vpb - 1] = -0.5f;
    weight_grown_data[fresh_ptr] = 5.0f;

    vr::Result<std::uint32_t> removed = vbg.remove(&fresh, 1);
    CHECK(removed.ok() && removed.value() == 0);
    // Cleared at the removed block's own range, both attributes, whole block.
    CHECK(tsdf_grown_data[fresh_ptr] == 0.0f);
    CHECK(tsdf_grown_data[fresh_ptr + vpb - 1] == 0.0f);
    CHECK(weight_grown_data[fresh_ptr] == 0.0f);
    // ...and only there: a block that was not removed keeps its data.
    CHECK(tsdf_grown_data[ptr_a] == -0.02f);
    CHECK(weight_grown_data[ptr_a] == 30.0f);

    // The reuse this protects against: re-allocating draws the same index back
    // (LIFO) and it reads as a fresh block rather than the removed one.
    vr::Result<std::uint32_t> realloc = vbg.map().allocate(&fresh, 1);
    CHECK(realloc.ok() && realloc.value() == 0);
    vr::Result<std::vector<vol::BlockIndex>> after_reuse =
        vbg.map().compact_active_blocks();
    CHECK(after_reuse.ok());
    std::int32_t reused_ptr = -1;
    for (const vol::BlockIndex& blk : after_reuse.value()) {
      if (blk.coord == fresh_coord) {
        reused_ptr = blk.ptr;
      }
    }
    CHECK(reused_ptr == fresh_ptr);              // LIFO: the same index back
    CHECK(tsdf_grown_data[reused_ptr] == 0.0f);  // and it is clean
  }

  // --- clear() zeroes every attribute, for the same reason: it returns every
  // block to the heap, so every range is about to be re-drawn.
  {
    tsdf_grown_data[ptr_a] = -0.02f;
    CHECK(vbg.clear().ok());
    CHECK(tsdf_grown_data[ptr_a] == 0.0f);
    CHECK(weight_grown_data[ptr_a] == 0.0f);
    vr::Result<std::vector<vol::BlockIndex>> empty =
        vbg.map().compact_active_blocks();
    CHECK(empty.ok() && empty.value().empty());
  }

  // --- An impossible grow is rejected on the parameters, not after paying for
  // it. resize() used to build every enlarged attribute buffer beside the live
  // ones and only *then* call map_.resize, which validates -- so an
  // arithmetically illegal grow committed the memory first and threw it away to
  // report an invalid_argument. At this grid (bucket_size 8, 512 voxels/block)
  // the request below is ~17 GB per attribute; the point is that it never
  // reaches VMA.
  //
  // 2^20 buckets puts num_blocks * voxels_per_block past INT32_MAX, the bound
  // BlockIndex::ptr must fit -- illegal regardless of how much memory the
  // machine has, so this is deterministic rather than a resource test.
  //
  // Which of the two up-front guards rejects it is deliberately not asserted:
  // the grid-contract check and the maxStorageBufferRange check both run before
  // the allocation loop, and they cannot be separated on real hardware (an
  // attribute array large enough to overflow the pointer bound is >= 8 GB, far
  // past any device's binding limit). What is pinned is the property that
  // matters -- rejected, before allocating, with the grid untouched.
  {
    const std::int32_t before_buckets = vbg.map().grid().num_buckets;
    vr::Result<vol::AttributeView> before_tsdf = vbg.attribute("tsdf");
    CHECK(before_tsdf.ok());
    const std::uint64_t before_size = before_tsdf.value().buffer->size();

    const vr::Status huge = vbg.resize(1 << 20);
    CHECK(!huge.ok());
    CHECK(huge.domain() == vr::Status::Code::InvalidArgument);
    // All-or-nothing: the live grid is exactly as it was.
    CHECK(vbg.map().grid().num_buckets == before_buckets);
    vr::Result<vol::AttributeView> after_tsdf = vbg.attribute("tsdf");
    CHECK(after_tsdf.ok());
    CHECK(after_tsdf.value().buffer->size() == before_size);
  }

  // --- attribute() refuses an array that no longer covers the live grid.
  // Resizing through the raw map() handle grows the table (preserving block
  // indices) and leaves the attribute arrays at their old size, so a block
  // allocated into the grown capacity addresses past them -- and both consumers
  // bind these VK_WHOLE_SIZE with robustBufferAccess enabled nowhere, so the
  // write is undefined rather than clamped. attribute() is the funnel every
  // consumer passes through, so this is where the desync becomes a Status.
  {
    const std::int32_t bigger = vbg.map().grid().num_buckets * 2;
    CHECK(vbg.map().resize(bigger).ok());  // the raw path, not vbg.resize
    CHECK(vbg.map().grid().num_buckets == bigger);
    vr::Result<vol::AttributeView> stale = vbg.attribute("tsdf");
    CHECK(!stale.ok());
    CHECK(stale.status().domain() == vr::Status::Code::InvalidArgument);
  }

  // Error paths: null list with a count, empty name, zero element size, and a
  // duplicate name are each rejected before any buffer is allocated.
  CHECK(vol::VoxelBlockGrid::create(device.value(), allocator.value(), grid,
                                    nullptr, 1)
            .status()
            .domain() == vr::Status::Code::InvalidArgument);
  const vol::AttributeSpec empty_name[] = {{"", sizeof(float)}};
  CHECK(vol::VoxelBlockGrid::create(device.value(), allocator.value(), grid,
                                    empty_name, 1)
            .status()
            .domain() == vr::Status::Code::InvalidArgument);
  const vol::AttributeSpec zero_size[] = {{"bad", 0}};
  CHECK(vol::VoxelBlockGrid::create(device.value(), allocator.value(), grid,
                                    zero_size, 1)
            .status()
            .domain() == vr::Status::Code::InvalidArgument);
  const vol::AttributeSpec dup[] = {{"tsdf", 4}, {"tsdf", 4}};
  CHECK(vol::VoxelBlockGrid::create(device.value(), allocator.value(), grid,
                                    dup, 2)
            .status()
            .domain() == vr::Status::Code::InvalidArgument);

  // A grid with no attributes is valid and costs no per-voxel memory.
  vr::Result<vol::VoxelBlockGrid> bare = vol::VoxelBlockGrid::create(
      device.value(), allocator.value(), grid, nullptr, 0);
  CHECK(bare.ok());
  CHECK(bare.value().valid() && !bare.value().has_attribute("tsdf"));

  // Move-only: the moved-from grid is left empty; a moved-to grid is live.
  vol::VoxelBlockGrid moved = std::move(vbg);
  CHECK(moved.valid() && moved.has_attribute("tsdf"));
  CHECK(!vbg.valid());  // NOLINT(bugprone-use-after-move) -- asserting empty

  // Self-move must preserve a grid that STILL owns its attribute buffers: a
  // defaulted memberwise move-assign would free them (std::vector self-move),
  // so operator= guards it. Launder through a pointer to dodge -Wself-move.
  vol::VoxelBlockGrid* alias = &moved;
  moved = std::move(*alias);
  CHECK(moved.valid() && moved.has_attribute("tsdf") &&
        moved.has_attribute("weight"));

  // Move-assign over a live grid frees the destination's attributes first.
  moved = std::move(bare).value();
  CHECK(moved.valid() && !moved.has_attribute("tsdf"));

  std::printf(
      "recon volume block grid test passed: 2 SoA attributes (%llu voxels "
      "each), independent storage, composed map allocates, resize grew them to "
      "%llu voxels preserving per-voxel data\n",
      static_cast<unsigned long long>(voxels),
      static_cast<unsigned long long>(new_voxels));
  return 0;
}
