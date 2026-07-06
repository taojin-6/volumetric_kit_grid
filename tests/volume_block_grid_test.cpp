// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for VoxelBlockGrid: declare independent per-voxel attributes (SoA),
// verify they are distinct, correctly-sized device buffers keyed by the block
// pool, that the composed VoxelHashMap still allocates, and the error / move
// paths. Runs on the real driver (MoltenVK / NVIDIA); exits 0 (skip) where no
// device is present.

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

  // A block's ptr keys into the attribute arrays (ptr + local voxel index).
  const std::int32_t ptr = active.value().front().ptr;
  CHECK(ptr >= 0 &&
        static_cast<std::uint64_t>(ptr) + grid.voxels_per_block <= voxels);
  tsdf_data[ptr] = -0.02f;
  CHECK(tsdf_data[ptr] == -0.02f && weight_data[ptr] == 0.0f);

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
  moved = std::move(bare).value();  // move-assign over a live grid
  CHECK(moved.valid() && !moved.has_attribute("tsdf"));
  vol::VoxelBlockGrid* alias = &moved;
  moved = std::move(*alias);  // self-move must leave it usable
  CHECK(moved.valid());

  std::printf(
      "recon volume block grid test passed: 2 SoA attributes (%llu voxels "
      "each), independent storage, composed map allocates\n",
      static_cast<unsigned long long>(voxels));
  return 0;
}
