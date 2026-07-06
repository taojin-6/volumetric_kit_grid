// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for depth / point-cloud block allocation: unproject a posed depth
// frame (off-centre principal point, a non-identity pose, every pixel valid)
// and a small world-space point set, then verify that exactly the expected
// (2*tb+1)^3 truncation-band block cubes are allocated on the real driver
// (MoltenVK on Apple, the NVIDIA ICD on the Linux CI box). The expected blocks
// are derived by unprojecting each pixel on the HOST (unproject_to_block, the
// glm/C++ mirror of the shader) and dilating with the host coord math, so the
// shader's unprojection + band dilation are genuinely under test -- a wrong
// intrinsic or a transposed pose would diverge. Also covers many overlapping
// pixels (lock contention + retry), negative-bias blocks, idempotent re-run,
// clear(), and null/zero-count guards. Exits 0 (skip) where no device is
// present.

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

// Host mirror of the depth kernel's unprojection (pinhole intrinsics + pose),
// so the expected block for a pixel is derived INDEPENDENTLY of the shader --
// a wrong fx/fy/cx/cy or a transposed cam_to_world changes this result and the
// on-device set no longer matches. (The device path is GLSL; this is glm/C++.)
vr::Vec3i unproject_to_block(const vol::DepthCameraParams& cam,
                             const vol::VoxelGridParams& grid, std::uint32_t u,
                             std::uint32_t v, float d) {
  const float x = (static_cast<float>(u) - cam.cx) * d / cam.fx;
  const float y = (static_cast<float>(v) - cam.cy) * d / cam.fy;
  const vr::Vec3f world =
      vr::Vec3f(cam.cam_to_world * vr::Vec4f(x, y, d, 1.0f));
  return vol::world_to_block(world, grid);
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
  // A 4x4 frame, every pixel valid at depth 1 m, viewed through a NON-identity
  // pose and an off-centre principal point -- so the unprojection is
  // non-degenerate (a wrong fx/fy/cx/cy or a transposed cam_to_world changes
  // the world point, unlike a principal-point pixel under identity where it
  // cancels to (0,0,depth)). The 16 adjacent pixels also land in overlapping
  // bands that hammer the same buckets -- the multi-thread lock contention the
  // depth kernel
  // + retry exist for -- and the pose lands the blocks at a negative Y,
  // exercising the negative-bias voxel->block floor.
  vol::DepthCameraParams cam{};
  cam.fx = 100.0f;
  cam.fy = 120.0f;  // fx != fy, so a swapped focal length is caught
  cam.cx = 1.5f;
  cam.cy = 1.5f;
  cam.min_depth = 0.1f;
  cam.max_depth = 10.0f;
  cam.width = 4;
  cam.height = 4;
  // 90 deg about Z (exact 0/+-1 entries -> no rounding to race the shader) + a
  // translation. Column-major (glm): col0=(0,1,0,0), col1=(-1,0,0,0),
  // col2=(0,0,1,0), col3=translate. (glm default-init is garbage.)
  cam.cam_to_world =
      vr::Mat4f(0.0f, 1.0f, 0.0f, 0.0f,    // column 0
                -1.0f, 0.0f, 0.0f, 0.0f,   // column 1
                0.0f, 0.0f, 1.0f, 0.0f,    // column 2
                0.2f, -0.1f, 0.3f, 1.0f);  // column 3 (translate)

  const float kDepth = 1.0f;
  std::vector<float> depth(static_cast<std::size_t>(cam.width) * cam.height,
                           kDepth);

  // Null input is rejected without touching the device.
  CHECK(!map.allocate_from_depth(nullptr, cam).ok());

  vr::Result<std::uint32_t> depth_fail =
      map.allocate_from_depth(depth.data(), cam);
  CHECK(depth_fail.ok());
  CHECK(depth_fail.value() == 0);

  // want = the union of every valid pixel's band, unprojected on the HOST (so
  // the shader's unprojection is what's under test, not just the band
  // dilation).
  std::set<Coord> depth_want;
  for (std::uint32_t v = 0; v < cam.height; ++v) {
    for (std::uint32_t u = 0; u < cam.width; ++u) {
      insert_cube(grid, unproject_to_block(cam, grid, u, v, kDepth),
                  depth_want);
    }
  }

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
  // Fresh table. clear() must actually empty it -- assert that directly, since
  // reusing the map below could otherwise let a no-op clear() slip through.
  CHECK(map.clear().ok());
  vr::Result<std::vector<vol::BlockIndex>> after_clear =
      map.compact_active_blocks();
  CHECK(after_clear.ok() && after_clear.value().empty());

  // World points, including a negative one so the points path also exercises
  // the negative-bias voxel->block floor. Bands may overlap; the set union
  // folds it.
  std::vector<vr::Vec3f> points = {vr::Vec3f(0.0f, 0.0f, 1.0f),
                                   vr::Vec3f(0.0f, 1.0f, 0.0f),
                                   vr::Vec3f(-0.03f, -0.02f, 0.5f)};

  // Null input rejected, zero count a no-op success -- both without touching
  // the set (the depth path checks null too; cover points symmetrically).
  CHECK(!map.allocate_from_points(nullptr, 1).ok());
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
