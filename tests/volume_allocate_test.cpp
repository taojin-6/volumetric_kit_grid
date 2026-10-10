// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for depth block allocation: unproject a posed depth frame
// (off-centre principal point, a non-identity pose, every pixel valid), then
// verify that exactly the expected
// (2*tb+1)^3 truncation-band block cubes are allocated on the real driver
// (MoltenVK on Apple, the NVIDIA ICD on the Linux CI box). The expected blocks
// are derived by unprojecting each pixel on the HOST (unproject_to_block, the
// glm/C++ mirror of the shader) and dilating with the host coord math, so the
// shader's unprojection + band dilation are genuinely under test -- a wrong
// intrinsic or a transposed pose would diverge. Also covers a frame of several
// ragged tiles at two band widths, lock contention + retry on a one-bucket
// table, negative-bias blocks, idempotent re-run, clear(), and null/zero-count
// guards. Skips where no device is present.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/voxel_coords.hpp"
#include "volumetric_kit/recon/volume/voxel_grid.hpp"
#include "volumetric_kit/recon/volume/voxel_hash_map.hpp"

#include "gpu_test.hpp"
#include "grid_truncation_comp.spv.hpp"

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
  vkc::Result<std::vector<vol::BlockIndex>> active =
      map.compact_active_blocks();
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
vr::Vec3i unproject_to_block(const vr::DepthCameraParams& cam,
                             const vol::VoxelGridParams& grid, std::uint32_t u,
                             std::uint32_t v, float d) {
  const float x = (static_cast<float>(u) - cam.cx) * d / cam.fx;
  const float y = (static_cast<float>(v) - cam.cy) * d / cam.fy;
  const vr::Vec3f world =
      vr::Vec3f(cam.cam_to_world * vr::Vec4f(x, y, d, 1.0f));
  return vol::world_to_block(world, grid);
}

// Exercise the real shared GLSL helper at the largest representable depth
// candidate count. A tiny compute pass checks host/device parity without
// enumerating the band, which would require billions of candidate visits.
int truncation_bounds_case(const vr_test::Gpu& gpu) {
  const vol::VoxelGridParams base{0.125f, 8, 512, 1.0f, 8, 8, 64, 128};
  std::vector<vol::VoxelGridParams> grids(6, base);
  grids[0].trunc_dist = 0.25f;
  grids[1].trunc_dist = 2.5f;
  grids[2].trunc_dist = std::nextafter(127.0f, 0.0f);
  grids[3].trunc_dist = 127.0f;
  grids[4].voxel_size = std::numeric_limits<float>::min();
  grids[4].trunc_dist = 8.0f * grids[4].voxel_size;
  grids[5].voxel_size = std::numeric_limits<float>::max() / 16.0f;
  grids[5].trunc_dist = 8.0f * grids[5].voxel_size;
  for (const auto& grid : grids) CHECK(grid.validate().ok());
  const auto count = static_cast<std::uint32_t>(grids.size());

  vkc::ComputeKernel kernel;
  vkc::KernelSetBuilder builder(gpu.device);
  CHECK(builder
            .add(kernel, "grid_truncation", vr_grid_truncation_comp_spv,
                 vr_grid_truncation_comp_spv_size, 2)
            .ok());
  auto pool = builder.build();
  auto input = vr_test::upload_device_buffer(
      gpu.device, gpu.allocator, grids.data(), count * sizeof(base));
  auto output = vkc::device_storage_buffer(gpu.allocator,
                                           count * 2 * sizeof(std::uint32_t));
  CHECK(pool.ok() && input.ok() && output.ok());
  kernel.set.write_storage_buffer(0, input->handle(), 0, VK_WHOLE_SIZE);
  kernel.set.write_storage_buffer(1, output->handle(), 0, VK_WHOLE_SIZE);
  CHECK(vkc::dispatch(gpu.device, kernel, nullptr, 0, count,
                      gpu.device.caps().limits().maxComputeWorkGroupCount[0])
            .ok());
  auto got = vr_test::read_back<std::uint32_t>(gpu.device, gpu.allocator,
                                               *output, count * 2);
  CHECK(got.ok());
  for (std::size_t i = 0; i < grids.size(); ++i) {
    const auto tb =
        static_cast<std::uint32_t>(vol::truncation_blocks(grids[i]));
    const std::uint64_t side = 2 * tb + 1;
    const std::uint64_t items = 256 * side * side * side;
    CHECK(items + 255 <= std::numeric_limits<std::uint32_t>::max());
    CHECK(got.value()[2 * i] == tb);
    CHECK(got.value()[2 * i + 1] == items);
  }
  return 0;
}

int gpu_main(vr_test::GpuContext& gpu) {
  CHECK(truncation_bounds_case(gpu) == 0);
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

  vkc::Result<vol::VoxelHashMap> map_result =
      vol::VoxelHashMap::create(gpu.device, gpu.allocator, grid);
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
  // cancels to (0,0,depth)). The pose lands the blocks at a negative Y,
  // exercising the negative-bias voxel->block floor.
  vr::DepthCameraParams cam{};
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

  vkc::Result<std::uint32_t> depth_fail =
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
  vkc::Result<std::uint32_t> depth_fail2 =
      map.allocate_from_depth(depth.data(), cam);
  CHECK(depth_fail2.ok() && depth_fail2.value() == 0);
  std::set<Coord> depth_got2;
  if (collect_active(map, depth_got2) != 0) return 1;
  CHECK(depth_got2 == depth_want);

  // --- lock contention + retry on one bucket --------------------------------
  // Every block lands in the one bucket, so every lane making one queues on
  // one spin lock and a round can lose races that the next round, or another
  // call, settles. Whether any are lost is the driver's (lavapipe serves the
  // lock fast enough to lose none), so the checks are the ones that hold
  // either way: every failure is a lock failure, and re-driving the frame ends
  // at exactly its band.
  {
    vol::VoxelGridParams one = grid;
    one.num_buckets = 1;
    one.bucket_size = 64;
    one.num_blocks = 64;
    vkc::Result<vol::VoxelHashMap> made =
        vol::VoxelHashMap::create(gpu.device, gpu.allocator, one);
    CHECK(made.ok());
    std::uint32_t left = 1;
    int passes = 0;
    for (; passes < 8 && left != 0; ++passes) {
      vol::AllocFailures failures{};
      vkc::Result<std::uint32_t> r =
          made.value().allocate_from_depth(depth.data(), cam, &failures);
      CHECK(r.ok());
      CHECK(failures.lock == r.value() && failures.terminal == 0);
      left = r.value();
    }
    CHECK(left == 0);
    std::printf("  one bucket: settled in %d call(s)\n", passes);
    std::set<Coord> one_got;
    if (collect_active(made.value(), one_got) != 0) return 1;
    CHECK(one_got == depth_want);
  }

  // --- allocate-from-depth over several tiles -------------------------------
  // The kernel works a 16 x 16 tile a workgroup and dilates each distinct
  // block once, so a frame several tiles wide and ragged at both edges, with
  // holes, a smooth half (many pixels a block) and a jagged half (many blocks
  // a tile), must still allocate exactly the union of every pixel's band --
  // at tb = 1, and at tb = 2, whose 125-block band outnumbers the lanes.
  // Every value is dyadic -- focal length 128, voxel 1/128 m, depths in 32nds,
  // a power-of-two pose -- so the host's unprojection is the shader's exactly
  // and no sample can round across a block boundary differently.
  for (const float trunc : {0.04f, 0.1f}) {
    vol::VoxelGridParams tiled = grid;
    tiled.voxel_size = 1.0f / 128.0f;  // a block is 1/16 m
    tiled.trunc_dist = trunc;
    vkc::Result<vol::VoxelHashMap> tiled_map =
        vol::VoxelHashMap::create(gpu.device, gpu.allocator, tiled);
    CHECK(tiled_map.ok());
    vr::DepthCameraParams wide = cam;
    wide.fx = 128.0f;
    wide.fy = 128.0f;
    wide.cx = 0.5f;
    wide.cy = 0.5f;
    wide.min_depth = 0.2f;
    wide.max_depth = 2.0f;
    wide.width = 37;   // 3 tiles, the last 5 pixels wide
    wide.height = 23;  // 2 tiles, the last 7 pixels tall
    wide.cam_to_world[3] = vr::Vec4f(0.25f, -0.125f, 0.375f, 1.0f);
    std::vector<float> wide_depth(std::size_t{wide.width} * wide.height);
    std::set<Coord> tiled_want;
    for (std::uint32_t v = 0; v < wide.height; ++v) {
      for (std::uint32_t u = 0; u < wide.width; ++u) {
        float d =
            u < 18 ? 0.5f + static_cast<float>(v) / 64.0f
                   : 0.25f + static_cast<float>((u * 5 + v * 3) % 24) / 32.0f;
        if ((u + 2 * v) % 7 == 0) d = 0.0f;   // no return
        if ((u * 3 + v) % 11 == 0) d = 4.0f;  // past max_depth
        wide_depth[std::size_t{v} * wide.width + u] = d;
        if (d >= wide.min_depth && d <= wide.max_depth) {
          insert_cube(tiled, unproject_to_block(wide, tiled, u, v, d),
                      tiled_want);
        }
      }
    }
    vkc::Result<std::uint32_t> tiled_fail =
        tiled_map.value().allocate_from_depth(wide_depth.data(), wide);
    CHECK(tiled_fail.ok() && tiled_fail.value() == 0);
    std::set<Coord> tiled_got;
    if (collect_active(tiled_map.value(), tiled_got) != 0) return 1;
    std::printf("  tiled depth, tb %d: %zu blocks\n",
                vol::truncation_blocks(tiled), tiled_got.size());
    CHECK(tiled_got == tiled_want);
  }

  // --- clear -----------------------------------------------------------------
  // clear() must actually empty the table.
  CHECK(map.clear().ok());
  vkc::Result<std::vector<vol::BlockIndex>> after_clear =
      map.compact_active_blocks();
  CHECK(after_clear.ok() && after_clear.value().empty());

  std::printf(
      "recon volume allocate test passed: depth band (%zu blocks) allocated + "
      "compacted on-device\n",
      depth_want.size());
  return 0;
}

}  // namespace

int main() { return vr_test::run_on_gpu(gpu_main); }
