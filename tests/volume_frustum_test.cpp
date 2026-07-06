// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for frustum-culled compaction: allocate blocks at known world
// positions -- one dead-centre in front of the camera, the others behind it,
// beyond the far plane, and off to the side -- then verify
// compact_active_blocks_in_frustum returns only the in-view block while plain
// compact_active_blocks returns all four. Exercises make_frustum_planes + the
// on-device p-vertex AABB cull on the real driver (MoltenVK on Apple, the
// NVIDIA ICD on the Linux CI box). Exits 0 (skip) where no device is present.

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
#include "volumetric_kit/recon/volume/frustum.hpp"
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

std::set<Coord> to_set(const std::vector<vol::BlockIndex>& blocks) {
  std::set<Coord> s;
  for (const vol::BlockIndex& b : blocks) {
    s.insert({b.coord.x, b.coord.y, b.coord.z});
  }
  return s;
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

  // Four blocks. A block's world min-corner is coord * block_size * voxel_size
  // = coord * 0.04 m. Camera sits at the origin looking down +Z (fx=fy=100,
  // cx=cy=50, 100x100, near 0.1, far 5), so only the first block is in view.
  const vr::Vec3i kInside(0, 0, 25);   // world (0, 0, 1.0): dead centre
  const vr::Vec3i kBehind(0, 0, -25);  // world (0, 0, -1.0): behind the camera
  const vr::Vec3i kFar(0, 0, 200);    // world (0, 0, 8.0): beyond the far plane
  const vr::Vec3i kSide(100, 0, 25);  // world (4.0, 0, 1.0): outside the fov
  std::vector<vol::BlockIndex> coords;
  for (const vr::Vec3i& c : {kInside, kBehind, kFar, kSide}) {
    vol::BlockIndex b{};
    b.coord = c;
    coords.push_back(b);
  }
  vr::Result<std::uint32_t> fail =
      map.allocate(coords.data(), static_cast<std::uint32_t>(coords.size()));
  CHECK(fail.ok());
  CHECK(fail.value() == 0);

  // Plain compaction: all four blocks are active.
  vr::Result<std::vector<vol::BlockIndex>> all = map.compact_active_blocks();
  CHECK(all.ok());
  CHECK(all.value().size() == 4);

  const std::set<Coord> want = {{0, 0, 25}};

  // Frustum compaction with explicit planes: only the in-view block survives.
  const vol::FrustumPlanes planes = vol::make_frustum_planes(
      100.0f, 100.0f, 50.0f, 50.0f, 100, 100, 0.1f, 5.0f, vr::Mat4f(1.0f));
  vr::Result<std::vector<vol::BlockIndex>> visible =
      map.compact_active_blocks_in_frustum(planes);
  CHECK(visible.ok());
  CHECK(to_set(visible.value()) == want);

  // The DepthCameraParams convenience derives the same frustum (min/max_depth
  // as near/far) and culls identically.
  vol::DepthCameraParams cam{};
  cam.fx = 100.0f;
  cam.fy = 100.0f;
  cam.cx = 50.0f;
  cam.cy = 50.0f;
  cam.min_depth = 0.1f;
  cam.max_depth = 5.0f;
  cam.width = 100;
  cam.height = 100;
  cam.cam_to_world = vr::Mat4f(1.0f);
  vr::Result<std::vector<vol::BlockIndex>> visible_cam =
      map.compact_active_blocks_in_frustum(cam);
  CHECK(visible_cam.ok());
  CHECK(to_set(visible_cam.value()) == want);

  std::printf(
      "recon volume frustum test passed: %zu of 4 blocks kept in-frustum "
      "(behind / far / lateral culled), camera convenience matched\n",
      visible.value().size());
  return 0;
}
