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
// guards. Pins the blocks, heap count and stamps three posed frames allocate,
// one frame alone and two as a set, blocking and recorded, as recorded on main
// at d8b8bda1. Skips where no device is present.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "fnv1a.hpp"
#include "test_check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
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

// --- allocate_from_depth's result, pinned -----------------------------------
// The grid the pinned frames allocate into: 1 cm voxels, 8 cm blocks, a band
// of one block, and room for every frame's.
vol::VoxelGridParams golden_grid() {
  vol::VoxelGridParams g{};
  g.voxel_size = 0.01f;
  g.block_size = 8;
  g.voxels_per_block = 512;
  g.trunc_dist = 0.04f;
  g.bucket_size = 8;
  g.num_buckets = 4096;
  g.num_blocks = 8 * 4096;
  g.max_chain = 128;
  return g;
}

// The rotation of the integer quaternion (a, b, c, d), each entry one rounded
// division, and translation t: the same bits from every compiler.
vr::Mat4f golden_pose(const int (&q)[4], vr::Vec3f t) {
  const int a = q[0], b = q[1], c = q[2], d = q[3];
  const auto e = [n = a * a + b * b + c * c + d * d](int num) {
    return static_cast<float>(num) / static_cast<float>(n);
  };
  vr::Mat4f m(1.0f);  // column-major: m[column][row]
  m[0] = vr::Vec4f(e(a * a + b * b - c * c - d * d), e(2 * (b * c + a * d)),
                   e(2 * (b * d - a * c)), 0.0f);
  m[1] = vr::Vec4f(e(2 * (b * c - a * d)), e(a * a - b * b + c * c - d * d),
                   e(2 * (c * d + a * b)), 0.0f);
  m[2] = vr::Vec4f(e(2 * (b * d + a * c)), e(2 * (c * d - a * b)),
                   e(a * a - b * b - c * c + d * d), 0.0f);
  m[3] = vr::Vec4f(t, 1.0f);
  return m;
}

// Whether the world point at pixel (u, v) and depth d lies more than a
// hundredth of a voxel from every boundary between blocks, in double. They sit
// half a voxel short of each block_size-th voxel, as worldToVoxel rounds.
// Drivers round the kernel's float arithmetic a few ulps apart, a millionth of
// a metre here, so a point this clear lands in one block on every device.
bool clear_of_boundaries(const vr::DepthCameraParams& c,
                         const vol::VoxelGridParams& g, std::uint32_t u,
                         std::uint32_t v, float d) {
  const double z = d;
  const double x = (static_cast<double>(u) - c.cx) * z / c.fx;
  const double y = (static_cast<double>(v) - c.cy) * z / c.fy;
  for (int r = 0; r < 3; ++r) {
    const double w = c.cam_to_world[0][r] * x + c.cam_to_world[1][r] * y +
                     c.cam_to_world[2][r] * z + c.cam_to_world[3][r];
    const double blocks = (w / g.voxel_size + 0.5) / g.block_size;
    if (std::abs(blocks - std::round(blocks)) * g.block_size < 0.01) {
      return false;
    }
  }
  return true;
}

// Frame `index` of three: a pinhole of its own, depth from integer arithmetic
// (exact in float), a smooth plane on the left half and a rough surface on the
// right, with holes, negative, NaN, infinite and out-of-range samples, and
// both range bounds. A sample near a boundary between blocks is a hole
// (clear_of_boundaries).
struct GoldenFrame {
  vr::DepthCameraParams cam{};
  std::vector<float> depth;
};

GoldenFrame golden_frame(int index, const vol::VoxelGridParams& g) {
  // (width, height), (fx, fy, cx, cy), the pose's quaternion and translation.
  // 100 x 75 and 161 x 83 are no multiples of the kernel's 16 x 16 groups.
  const std::uint32_t sizes[3][2] = {{128, 96}, {100, 75}, {161, 83}};
  const vr::Vec4f intrinsics[3] = {vr::Vec4f(211.37f, 209.81f, 63.29f, 47.61f),
                                   vr::Vec4f(150.5f, 151.25f, 49.75f, 37.125f),
                                   vr::Vec4f(300.3f, 299.7f, 80.9f, 41.2f)};
  const int quaternions[3][4] = {{5, 1, 2, 1}, {3, -1, 1, 2}, {4, 2, -1, 3}};
  const vr::Vec3f translations[3] = {vr::Vec3f(0.13f, -0.27f, 0.41f),
                                     vr::Vec3f(-0.62f, 0.08f, 0.35f),
                                     vr::Vec3f(0.21f, 0.33f, -0.57f)};
  GoldenFrame f;
  f.cam.fx = intrinsics[index].x;
  f.cam.fy = intrinsics[index].y;
  f.cam.cx = intrinsics[index].z;
  f.cam.cy = intrinsics[index].w;
  f.cam.min_depth = 0.25f;
  f.cam.max_depth = 3.0f;
  f.cam.width = sizes[index][0];
  f.cam.height = sizes[index][1];
  f.cam.cam_to_world = golden_pose(quaternions[index], translations[index]);
  f.depth.resize(std::size_t{f.cam.width} * f.cam.height);
  const auto i = static_cast<std::uint32_t>(index);
  for (std::uint32_t v = 0; v < f.cam.height; ++v) {
    for (std::uint32_t u = 0; u < f.cam.width; ++u) {
      const std::uint32_t k = u < f.cam.width / 2
                                  ? 1200 + 3 * u + 2 * v
                                  : 1024 + (37 * u + 23 * v + 101 * i) % 997;
      float d = static_cast<float>(k) / 1024.0f;
      const auto at = [u, v](std::uint32_t a, std::uint32_t b,
                             std::uint32_t m) {
        return (a * u + b * v) % m == 0;
      };
      if (at(7, 3, 41)) {
        d = 0.0f;  // no return
      } else if (at(1, 5, 53)) {
        d = -d;
      } else if (at(3, 1, 59)) {
        d = std::numeric_limits<float>::quiet_NaN();
      } else if (at(1, 1, 61)) {
        d = std::numeric_limits<float>::infinity();
      } else if (at(5, 2, 67)) {
        d = 0.125f;  // nearer than min_depth
      } else if (at(2, 7, 71)) {
        d = 3.5f;  // farther than max_depth
      } else if (at(4, 1, 73)) {
        d = f.cam.min_depth;  // the bounds are kept
      } else if (at(1, 4, 79)) {
        d = f.cam.max_depth;
      }
      const bool kept =
          d > 0.0f && d >= f.cam.min_depth && d <= f.cam.max_depth;
      if (kept && !clear_of_boundaries(f.cam, g, u, v, d)) d = 0.0f;
      f.depth[std::size_t{v} * f.cam.width + u] = d;
    }
  }
  return f;
}

// What an allocation left: the blocks, the heap's count, and FNV-1a over the
// sorted coordinates and over each sorted block's coordinate and stamp.
struct Pinned {
  std::size_t blocks = 0;
  std::uint32_t allocated = 0;
  std::uint32_t coords = 0;
  std::uint32_t stamps = 0;

  bool operator==(const Pinned& o) const {
    return blocks == o.blocks && allocated == o.allocated &&
           coords == o.coords && stamps == o.stamps;
  }
};

vkc::Result<Pinned> pinned(const char* what, vol::VoxelHashMap& map) {
  VKC_ASSIGN(std::vector<vol::BlockIndex> active, map.compact_active_blocks());
  VKC_ASSIGN(const std::vector<vol::BlockStamp> stamps,
             map.read_block_stamps());
  std::sort(active.begin(), active.end(),
            [](const vol::BlockIndex& a, const vol::BlockIndex& b) {
              return vol::coord_less(a.coord, b.coord);
            });
  std::vector<std::int32_t> coords;
  std::vector<std::uint32_t> stamped;
  for (const vol::BlockIndex& b : active) {
    coords.insert(coords.end(), {b.coord.x, b.coord.y, b.coord.z});
    const vol::BlockStamp& s =
        stamps[std::size_t(b.ptr) / std::size_t(map.grid().voxels_per_block)];
    stamped.insert(
        stamped.end(),
        {std::uint32_t(b.coord.x), std::uint32_t(b.coord.y),
         std::uint32_t(b.coord.z), s.requested, s.weighted, s.changed});
  }
  const Pinned p{active.size(), map.allocated_blocks(), vr_test::fnv1a(coords),
                 vr_test::fnv1a(stamped)};
  std::printf("  pinned %s: %zu blocks, heap %u, coords %08x, stamps %08x\n",
              what, p.blocks, p.allocated, p.coords, p.stamps);
  return p;
}

// Recorded on main at d8b8bda1: the same on an Apple M5 Max (MoltenVK), an
// RTX 5090 and lavapipe.
constexpr Pinned kOneFrame{3376, 3376, 0x606d08c9, 0xf44db715};
constexpr Pinned kSet{8740, 8740, 0x0b3645e7, 0xedec4629};

// The first frame alone at tick 3, from a host array and from a device buffer;
// then the other two as a set at tick 5 over its blocks, blocking from host
// arrays and recorded from device buffers. Every path gives the pinned values.
int golden_case(vr_test::Gpu& gpu) {
  const vol::VoxelGridParams grid = golden_grid();
  std::vector<GoldenFrame> frames;
  std::vector<vkc::Buffer> device;
  for (int i = 0; i < 3; ++i) {
    frames.push_back(golden_frame(i, grid));
    const std::vector<float>& depth = frames.back().depth;
    vkc::Result<vkc::Buffer> up = vr_test::upload_device_buffer(
        gpu.device, gpu.allocator, depth.data(), depth.size() * sizeof(float));
    CHECK(up.ok());
    device.push_back(std::move(up).value());
  }
  const auto input = [&](int i, bool on_device) {
    return vol::DepthInput{on_device
                               ? vkc::StorageInput(device[i])
                               : vkc::StorageInput(frames[i].depth.data()),
                           frames[i].cam};
  };
  const auto at_tick_3 = [&]() {
    vkc::Result<vol::VoxelHashMap> map =
        vol::VoxelHashMap::create(gpu.device, gpu.allocator, grid);
    if (map.ok()) {
      map->advance_tick();
      map->advance_tick();
    }
    return map;
  };

  for (const bool on_device : {false, true}) {
    vkc::Result<vol::VoxelHashMap> map = at_tick_3();
    CHECK(map.ok());
    vol::AllocFailures f;
    vkc::Result<std::uint32_t> left =
        on_device ? map->allocate_from_depth(device[0], frames[0].cam, &f)
                  : map->allocate_from_depth(frames[0].depth.data(),
                                             frames[0].cam, &f);
    CHECK(left.ok() && left.value() == 0);
    vkc::Result<Pinned> got = pinned("one frame", *map);
    CHECK(got.ok() && got.value() == kOneFrame);
  }

  for (const bool recorded : {false, true}) {
    vkc::Result<vol::VoxelHashMap> map = at_tick_3();
    CHECK(map.ok());
    CHECK(map->allocate_from_depth({input(0, recorded)}).value() == 0);
    map->advance_tick();
    map->advance_tick();
    const std::vector<vol::DepthInput> set{input(1, recorded),
                                           input(2, recorded)};
    if (!recorded) {
      CHECK(map->allocate_from_depth(set).value() == 0);
    } else {
      vol::AllocFailures f;
      for (int pass = 0; pass < 8; ++pass) {
        vkc::CommandBatch batch(gpu.device, gpu.allocator);
        vkc::Result<vol::RecordedRounds> r =
            map->record_allocate_from_depth(batch, set);
        CHECK(r.ok() && batch.submit().ok());
        f = map->settle(*r).value();
        if (f.lock == 0) break;
      }
      CHECK(f.total == 0 && f.terminal == 0);
    }
    vkc::Result<Pinned> got = pinned(recorded ? "set, recorded" : "set", *map);
    CHECK(got.ok() && got.value() == kSet);
  }
  return 0;
}

int gpu_main(vr_test::GpuContext& gpu) {
  CHECK(truncation_bounds_case(gpu) == 0);
  CHECK(golden_case(gpu) == 0);
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
  // exercising the negative-bias voxel->block floor. fx and fy, and cx and cy,
  // differ by enough to move the band: the kernel packs them as
  // vec4(fx, fy, cx, cy), and swapping either pair changes the blocks, which
  // the host checks below.
  vr::DepthCameraParams cam{};
  cam.fx = 40.0f;
  cam.fy = 120.0f;
  cam.cx = 2.5f;
  cam.cy = 0.5f;
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
  const auto band_of = [&](const vr::DepthCameraParams& c) {
    std::set<Coord> want;
    for (std::uint32_t v = 0; v < c.height; ++v) {
      for (std::uint32_t u = 0; u < c.width; ++u) {
        insert_cube(grid, unproject_to_block(c, grid, u, v, kDepth), want);
      }
    }
    return want;
  };
  const std::set<Coord> depth_want = band_of(cam);
  // The frame tells the packing apart: swapped focal lengths, or a swapped
  // principal point, would allocate other blocks.
  vr::DepthCameraParams swapped_f = cam;
  std::swap(swapped_f.fx, swapped_f.fy);
  vr::DepthCameraParams swapped_c = cam;
  std::swap(swapped_c.cx, swapped_c.cy);
  CHECK(band_of(swapped_f) != depth_want && band_of(swapped_c) != depth_want);

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

  // --- a frame over capacity ends its rounds at the limit --------------------
  // The band outnumbers a 16-block table. The first call fills it, in as many
  // rounds as its lock races take to reach the limit; the second meets the
  // limit in its first round and stops there. More rounds would only repeat
  // the failed inserts, whose queued threads' lock races kept a full table
  // re-dispatching until progress stalled.
  {
    vol::VoxelGridParams small = grid;
    small.num_buckets = 2;
    small.num_blocks = 16;
    CHECK(depth_want.size() > 2 * std::size_t(small.num_blocks));
    vkc::Result<vol::VoxelHashMap> made =
        vol::VoxelHashMap::create(gpu.device, gpu.allocator, small);
    CHECK(made.ok());
    for (int call = 0; call < 2; ++call) {
      vol::AllocFailures failures{};
      vkc::Result<std::uint32_t> r =
          made.value().allocate_from_depth(depth.data(), cam, &failures);
      CHECK(r.ok() && r.value() > 0 && failures.capacity_limited());
      std::printf("  over capacity, call %d: %u round(s), %u lock, %u heap\n",
                  call, failures.rounds, failures.lock, failures.heap);
      if (call == 1) CHECK(failures.rounds == 1);
    }
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
