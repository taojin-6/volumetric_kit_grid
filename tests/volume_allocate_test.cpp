// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for depth / point-cloud block allocation: unproject a synthetic
// one-pixel depth frame (identity pose, principal point on the valid pixel) and
// a small world-space point set, then verify that exactly the expected
// (2*tb+1)^3 truncation-band block cube around each surface hit is allocated --
// the camera unprojection, the band dilation, and the DepthCameraParams
// scalar-layout upload, all on the real driver (MoltenVK on Apple, the NVIDIA
// ICD on the Linux CI box). The expected cubes are derived with the same host
// coordinate math the shader mirrors (world_to_block + truncation_blocks), so
// the check stays correct whatever truncation_blocks rounds to. Exits 0 (skip)
// where no device is present.

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
#include "volumetric_kit/recon/volume/voxel_coords.hpp"
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

// Insert the solid (2*tb+1)^3 block cube centred on `center` into `want`, tb =
// truncation_blocks(grid) -- the band the depth/point kernels dilate a surface
// block into. Uses the host coordinate math the shader mirrors.
void insert_cube(const vol::VoxelGridParams& grid, vr::Vec3i center,
                 std::set<Coord>& want) {
  const int tb = vol::truncation_blocks(grid);
  for (int dz = -tb; dz <= tb; ++dz) {
    for (int dy = -tb; dy <= tb; ++dy) {
      for (int dx = -tb; dx <= tb; ++dx) {
        want.insert({center.x + dx, center.y + dy, center.z + dz});
      }
    }
  }
}

// Compact the map's active set into a coord set + assert each block drew a
// distinct, valid heap slot. Returns 1 (fail) via CHECK on any mismatch.
int collect_active(vol::VoxelHashMap& map, std::set<Coord>& got) {
  vr::Result<std::vector<vol::BlockIndex>> active = map.compact_active_blocks();
  CHECK(active.ok());
  std::set<int> ptrs;
  for (const vol::BlockIndex& block : active.value()) {
    got.insert({block.coord.x, block.coord.y, block.coord.z});
    CHECK(block.ptr >= 0);
    ptrs.insert(block.ptr);
  }
  // Distinct slots: a double-pop would hand two coords the same block.
  CHECK(ptrs.size() == active.value().size());
  return 0;
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

  // A small grid: 1024 buckets x 8, 8192-block heap -- ample for a couple of
  // 27-block bands, and cheap to init.
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

  // --- allocate-from-depth --------------------------------------------------
  // A 3x3 depth image with one valid sample at the principal point (u,v)=(1,1);
  // with an identity pose it unprojects to world (0,0,depth), independent of
  // the focal length (the (u-cx) term is zero). Every other pixel is 0 -- below
  // min_depth, so dropped.
  vol::DepthCameraParams cam{};
  cam.fx = 100.0f;
  cam.fy = 100.0f;
  cam.cx = 1.0f;
  cam.cy = 1.0f;
  cam.min_depth = 0.1f;
  cam.max_depth = 10.0f;
  cam.width = 3;
  cam.height = 3;
  cam.cam_to_world = vr::Mat4f(1.0f);  // identity (glm default-init is garbage)

  const float kDepth = 1.0f;
  std::vector<float> depth(9, 0.0f);
  depth[1 * 3 + 1] = kDepth;  // pixel (u=1, v=1)

  // Null input is rejected without touching the device.
  CHECK(!map.allocate_from_depth(nullptr, cam).ok());

  vr::Result<std::uint32_t> depth_fail =
      map.allocate_from_depth(depth.data(), cam);
  CHECK(depth_fail.ok());
  CHECK(depth_fail.value() == 0);

  // The one valid pixel -> world (0,0,kDepth) -> its block -> the band cube.
  std::set<Coord> depth_want;
  insert_cube(grid, vol::world_to_block(vr::Vec3f(0.0f, 0.0f, kDepth), grid),
              depth_want);

  std::set<Coord> depth_got;
  if (collect_active(map, depth_got) != 0) return 1;
  CHECK(depth_got == depth_want);

  // Idempotent: re-running the same frame allocates nothing new.
  vr::Result<std::uint32_t> depth_fail2 =
      map.allocate_from_depth(depth.data(), cam);
  CHECK(depth_fail2.ok() && depth_fail2.value() == 0);
  std::set<Coord> depth_got2;
  if (collect_active(map, depth_got2) != 0) return 1;
  CHECK(depth_got2 == depth_want);

  // --- allocate-from-points -------------------------------------------------
  // Fresh table; two world points far enough apart that their bands are
  // disjoint, so the active set is the union of two full cubes.
  CHECK(map.clear().ok());

  std::vector<vr::Vec3f> points = {vr::Vec3f(0.0f, 0.0f, 1.0f),
                                   vr::Vec3f(0.0f, 1.0f, 0.0f)};

  // Zero count is a no-op success.
  vr::Result<std::uint32_t> points_zero =
      map.allocate_from_points(points.data(), 0);
  CHECK(points_zero.ok() && points_zero.value() == 0);

  vr::Result<std::uint32_t> points_fail = map.allocate_from_points(
      points.data(), static_cast<std::uint32_t>(points.size()));
  CHECK(points_fail.ok());
  CHECK(points_fail.value() == 0);

  std::set<Coord> points_want;
  for (const vr::Vec3f& p : points) {
    insert_cube(grid, vol::world_to_block(p, grid), points_want);
  }

  std::set<Coord> points_got;
  if (collect_active(map, points_got) != 0) return 1;
  CHECK(points_got == points_want);

  std::printf(
      "recon volume allocate test passed: depth band (%zu blocks) + "
      "point-cloud "
      "bands (%zu blocks) allocated + compacted on-device\n",
      depth_want.size(), points_want.size());
  return 0;
}
