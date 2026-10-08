// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GridGrowth on real grids: it grows ahead only past kGrowThreshold, within
// the ceiling, and not when growth is off; a known headroom too small for the
// grown grid declines it, a reading at the limit included, while an unknown
// one never does; a refusal, for memory or a resize that ran out of it, holds
// at its size for retry_after ticks and is then asked again, by either entry
// point; and bytes_at and max_num_buckets read the grid's own layout. Exits 0
// (skip) where no device is present.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/recon/core/device_requirements.hpp"
#include "volumetric_kit/recon/volume/grid_growth.hpp"
#include "volumetric_kit/recon/volume/hash.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"
#include "volumetric_kit/recon/volume/voxel_hash_map.hpp"

#include "no_device.hpp"

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

using vol::GrowthOutcome;

vol::VoxelGridParams params(std::int32_t num_buckets) {
  vol::VoxelGridParams grid{};
  grid.voxel_size = 0.05f;
  grid.block_size = 8;
  grid.voxels_per_block = 512;
  grid.trunc_dist = 0.15f;
  grid.bucket_size = 8;
  grid.num_buckets = num_buckets;
  grid.num_blocks = 8 * num_buckets;
  grid.max_chain = 128;
  return grid;
}

vkc::Result<vol::VoxelBlockGrid> make_grid(vkc::Device& device,
                                           vkc::Allocator& allocator,
                                           const vol::VoxelGridParams& grid) {
  const vol::AttributeSpec attrs[] = {{"tsdf", sizeof(float)},
                                      {"weight", sizeof(float)},
                                      {"color", sizeof(std::uint32_t)}};
  return vol::VoxelBlockGrid::create(device, allocator, grid, attrs, 3);
}

// `count` distinct blocks along x, allocated into `grid`.
int fill(vol::VoxelBlockGrid& grid, int count) {
  std::vector<vol::BlockIndex> coords(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) coords[std::size_t(i)].coord = {i, 0, 0};
  auto failed = grid.map().allocate(coords.data(), std::uint32_t(count));
  CHECK(failed.ok() && failed.value() == 0);
  return 0;
}

// A grid of four buckets (32 blocks) holding `blocks` of them.
vkc::Result<vol::VoxelBlockGrid> filled(vkc::Device& device,
                                        vkc::Allocator& allocator, int blocks) {
  VKC_ASSIGN(vol::VoxelBlockGrid grid, make_grid(device, allocator, params(4)));
  if (fill(grid, blocks) != 0) {
    return vkc::Status::io_error("allocation failed");
  }
  return grid;
}

}  // namespace

int main() {
  auto instance = vkc::Instance::create({});
  if (!instance) {
    return vr_test::no_device("no Vulkan instance",
                              instance.status().message());
  }
  auto gpu = instance->select_physical_device(vr::device_requirements());
  if (!gpu) {
    return vr_test::no_device("no compute-capable device",
                              gpu.status().message());
  }
  auto device = vkc::Device::create(*instance, *gpu, vr::device_requirements());
  CHECK(device.ok());
  auto allocator = vkc::Allocator::create(instance->handle(), *device);
  CHECK(allocator.ok());
  vkc::Device& dev = *device;
  vkc::Allocator& alloc = *allocator;

  // The layout the grid reports: three 4-byte attributes a voxel, and the
  // table's entry, heap slot, two compacted lists and stamps a block, and a
  // lock a bucket.
  {
    auto grid = make_grid(dev, alloc, params(4));
    CHECK(grid.ok());
    const std::uint64_t per_block =
        512 * 12 + sizeof(vol::HashEntry) + sizeof(std::uint32_t) +
        2 * sizeof(vol::BlockIndex) + sizeof(vol::BlockStamp);
    CHECK(grid->bytes_at(4) == 32 * per_block + 4 * sizeof(std::int32_t));
    CHECK(grid->bytes_at(1000) == 8000 * per_block + 1000 * 4);
    CHECK(grid->bytes_at(0) == 0);
    // Block pointers within int32, and each 4-byte attribute in one binding.
    const std::uint64_t range = vkc::max_storage_buffer_range(dev);
    const std::uint64_t blocks = std::min<std::uint64_t>(
        std::numeric_limits<std::int32_t>::max() / 512, range / (512 * 4));
    CHECK(grid->max_num_buckets() == std::int32_t(blocks / 8));
    vol::VoxelBlockGrid moved(std::move(grid).value());
    vol::GridGrowth growth;
    CHECK(growth.grow_ahead(*grid).status().domain() ==
          vkc::Status::Code::InvalidArgument);
    CHECK(grid->max_num_buckets() == 0 && grid->bytes_at(4) == 0);
    CHECK(moved.max_num_buckets() == std::int32_t(blocks / 8));
  }

  // Not due at or below the threshold; past it, a doubling that keeps every
  // block, timed as a "resize" row.
  {
    auto grid = filled(dev, alloc, 22);  // 22 of 32: 0.6875
    CHECK(grid.ok());
    vol::GridGrowth growth;
    auto event = growth.grow_ahead(*grid);
    CHECK(event.ok() && event->outcome == GrowthOutcome::NotDue);
    CHECK(event->load_factor == 22.0f / 32.0f);
    CHECK(grid->grid().num_buckets == 4);
    CHECK(fill(*grid, 23) == 0);  // 23 of 32: 0.71875
    vkc::StageMetrics rows;
    event = growth.grow_ahead(*grid, &rows);
    CHECK(event.ok() && event->outcome == GrowthOutcome::Grew);
    CHECK(event->from_buckets == 4 && event->to_buckets == 8);
    CHECK(event->needed_bytes == grid->bytes_at(8));
    CHECK(grid->grid().num_buckets == 8);
    CHECK(grid->map().load_factor().value() == 23.0f / 64.0f);
    CHECK(rows.rows().size() == 1 &&
          std::string(rows.rows()[0].name) == "resize");
    auto kept = grid->map().compact_active_blocks();
    CHECK(kept.ok() && kept->size() == 23);
  }

  // Growth off; the ceiling clamps a doubling and then stops it; a caller's
  // larger size wins over a doubling; a negative ceiling is refused.
  {
    auto grid = filled(dev, alloc, 23);
    CHECK(grid.ok());
    vol::GrowthPolicy off;
    off.enabled = false;
    auto event = vol::GridGrowth(off).grow_ahead(*grid);
    CHECK(event.ok() && event->outcome == GrowthOutcome::Disabled);
    CHECK(grid->grid().num_buckets == 4);

    vol::GrowthPolicy capped;
    capped.max_buckets = 6;
    vol::GridGrowth growth(capped);
    event = growth.grow_ahead(*grid);
    CHECK(event.ok() && event->outcome == GrowthOutcome::Grew);
    CHECK(grid->grid().num_buckets == 6);
    event = growth.grow(*grid);
    CHECK(event.ok() && event->outcome == GrowthOutcome::AtCeiling);
    CHECK(event->to_buckets == 0 && grid->grid().num_buckets == 6);

    event = vol::GridGrowth().grow(*grid, 20);
    CHECK(event.ok() && event->grew() && event->to_buckets == 20);
    CHECK(grid->grid().num_buckets == 20);

    vol::GrowthPolicy negative;
    negative.max_buckets = -1;
    CHECK(vol::GridGrowth(negative).grow(*grid).status().domain() ==
          vkc::Status::Code::InvalidArgument);
  }

  // The headroom: a known reading short of the grown grid declines it -- 0,
  // at the limit, included -- and one that covers it, or an unknown one, does
  // not. A decline holds at its size, for either entry point and without
  // asking again, for retry_after ticks; then the grow is asked again.
  {
    auto grid = filled(dev, alloc, 23);
    CHECK(grid.ok());
    const std::uint64_t needed = grid->bytes_at(8);
    std::optional<std::uint64_t> reading = 0;
    int asked = 0;
    vol::GrowthPolicy policy;
    policy.retry_after = 3;
    policy.headroom = [&] {
      ++asked;
      return reading;
    };
    vol::GridGrowth growth(policy);
    auto event = growth.grow_ahead(*grid);
    CHECK(event.ok() && event->outcome == GrowthOutcome::DeclinedForMemory);
    CHECK(event->headroom_bytes == 0 && event->needed_bytes == needed);
    CHECK(asked == 1 && grid->grid().num_buckets == 4);
    reading = needed - 1;
    for (int tick = 0; tick < 3; ++tick) {
      event = growth.grow_ahead(*grid);
      CHECK(event.ok() && event->outcome == GrowthOutcome::Waiting);
      event = growth.grow(*grid);
      CHECK(event.ok() && event->outcome == GrowthOutcome::Waiting);
      CHECK(asked == 1);
      grid->map().advance_tick();
    }
    event = growth.grow_ahead(*grid);
    CHECK(event.ok() && event->outcome == GrowthOutcome::DeclinedForMemory);
    CHECK(event->headroom_bytes == needed - 1 && asked == 2);

    // At another size the refusal no longer holds: the grid grew by itself.
    CHECK(grid->resize(5).ok());
    reading = std::nullopt;  // unknown: never declines
    event = growth.grow(*grid);
    CHECK(event.ok() && event->grew() && asked == 3);
    CHECK(grid->grid().num_buckets == 10);
    reading = grid->bytes_at(20);  // exactly covers the next doubling
    event = growth.grow(*grid);
    CHECK(event.ok() && event->grew() && asked == 4);
    CHECK(grid->grid().num_buckets == 20);
  }

  // A resize that runs out of memory is a refusal that expires, not an
  // error, and not a cap for the rest of the scan. Five blocks in five
  // buckets of two all land in one bucket of nine, where two slots and a
  // chain of two hold four, so growing 8 -> 9 cannot rehash them.
  {
    vol::VoxelGridParams tight = params(8);
    tight.bucket_size = 2;
    tight.num_blocks = 16;
    tight.max_chain = 2;
    auto grid = make_grid(dev, alloc, tight);
    CHECK(grid.ok());
    std::vector<vol::BlockIndex> clash;
    std::set<std::uint32_t> old_buckets;
    for (int i = 0; i < 4096 && clash.size() < 5; ++i) {
      const vr::Vec3i c(i % 64, i / 64, 0);
      if (vol::hash_bucket(c, 9) == 0 &&
          old_buckets.insert(vol::hash_bucket(c, 8)).second) {
        vol::BlockIndex block{};
        block.coord = c;
        clash.push_back(block);
      }
    }
    CHECK(clash.size() == 5);
    auto placed = grid->map().allocate(clash.data(), 5);
    CHECK(placed.ok() && placed.value() == 0);
    vol::GrowthPolicy policy;
    policy.max_buckets = 9;
    policy.retry_after = 2;
    vol::GridGrowth growth(policy);
    auto event = growth.grow(*grid);
    CHECK(event.ok() && event->outcome == GrowthOutcome::ResizeFailed);
    CHECK(event->to_buckets == 9);
    CHECK(event->error.domain() == vkc::Status::Code::OutOfMemory);
    CHECK(grid->grid().num_buckets == 8);
    for (int tick = 0; tick < 2; ++tick) {
      event = growth.grow(*grid);
      CHECK(event.ok() && event->outcome == GrowthOutcome::Waiting);
      grid->map().advance_tick();
    }
    event = growth.grow(*grid);
    CHECK(event.ok() && event->outcome == GrowthOutcome::ResizeFailed);
    auto kept = grid->map().compact_active_blocks();
    CHECK(kept.ok() && kept->size() == 5);
  }

  std::puts("volume_grid_growth: OK");
  return 0;
}
