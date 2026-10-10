// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The stateless growth helper preserves blocks, honours a requested minimum,
// and refuses invalid or unreachable sizes before changing the grid.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
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
    CHECK(vol::grow_grid(*grid).domain() == vkc::Status::Code::InvalidArgument);
    CHECK(grid->max_num_buckets() == 0 && grid->bytes_at(4) == 0);
    CHECK(moved.max_num_buckets() == std::int32_t(blocks / 8));
  }

  {
    auto grid = filled(dev, alloc, 23);
    CHECK(grid.ok());
    vkc::StageMetrics rows;
    CHECK(vol::grow_grid(*grid, 0, &rows).ok());
    CHECK(grid->grid().num_buckets == 8);
    CHECK(grid->map().load_factor().value() == 23.0f / 64.0f);
    CHECK(rows.rows().size() == 1 &&
          std::string(rows.rows()[0].name) == "resize");
    auto kept = grid->map().compact_active_blocks();
    CHECK(kept.ok() && kept->size() == 23);

    // Codec callers can jump directly to a known larger minimum.
    CHECK(vol::grow_grid(*grid, 20).ok());
    CHECK(grid->grid().num_buckets == 20);
    kept = grid->map().compact_active_blocks();
    CHECK(kept.ok() && kept->size() == 23);

    CHECK(vol::grow_grid(*grid, -1).domain() ==
          vkc::Status::Code::InvalidArgument);
    CHECK(vol::grow_grid(*grid, grid->max_num_buckets() + 1).domain() ==
          vkc::Status::Code::OutOfMemory);
    CHECK(grid->grid().num_buckets == 20);
    kept = grid->map().compact_active_blocks();
    CHECK(kept.ok() && kept->size() == 23);
  }

  std::puts("volume_grid_growth: OK");
  return 0;
}
