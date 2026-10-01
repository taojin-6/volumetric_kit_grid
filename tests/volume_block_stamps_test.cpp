// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The block stamps and the GC over them. Every allocation path stamps the
// blocks it asks for -- inserted or already there -- with the map's tick
// (`requested`); the grid's block pass stamps the blocks holding weight
// (`weighted`); and free_stale_blocks frees exactly the blocks that have been
// neither for max_age ticks. A freed slot's record is zeroed, resize keeps
// every record in its slot (the rehash does not restamp), and clear zeroes
// them all. Exits 0 (skip) where no device is present.

#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <tuple>
#include <vector>

#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/camera_params.hpp"
#include "volumetric_kit/recon/core/device.hpp"
#include "volumetric_kit/recon/core/instance.hpp"
#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

#include "grid_readback.hpp"

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

vol::VoxelGridParams params() {
  vol::VoxelGridParams g{};
  g.voxel_size = 0.01f;
  g.block_size = 8;
  g.voxels_per_block = 512;
  g.trunc_dist = 0.04f;
  g.bucket_size = 8;
  g.num_buckets = 256;
  g.num_blocks = 256 * 8;
  g.max_chain = 32;
  return g;
}

// Each active block's slot, by coordinate.
vr::Result<std::map<Coord, std::uint32_t>> slots(vol::VoxelBlockGrid& g) {
  VR_ASSIGN(const std::vector<vol::BlockIndex> active,
            g.map().compact_active_blocks());
  std::map<Coord, std::uint32_t> out;
  for (const vol::BlockIndex& b : active) {
    out[Coord{b.coord.x, b.coord.y, b.coord.z}] =
        static_cast<std::uint32_t>(b.ptr) / 512u;
  }
  return out;
}

vol::BlockIndex at(int x) {
  vol::BlockIndex b{};
  b.coord = vr::Vec3i(x, 0, 0);
  return b;
}

// Whether every block a call added since `before` is stamped requested at the
// map's tick, and there is at least one: what each allocation path must do.
vr::Result<bool> new_blocks_requested_now(
    vol::VoxelBlockGrid& g, const std::map<Coord, std::uint32_t>& before) {
  VR_ASSIGN(const auto after, slots(g));
  VR_ASSIGN(const std::vector<vol::BlockStamp> st, g.map().read_block_stamps());
  std::size_t added = 0;
  for (const auto& [coord, slot] : after) {
    if (before.count(coord) != 0) continue;
    ++added;
    if (st[slot].requested != g.map().tick() || st[slot].weighted != 0) {
      return false;
    }
  }
  return added > 0;
}

}  // namespace

int main() {
  vr::Result<vr::Instance> instance = vr::Instance::create({});
  if (!instance) {
    std::fprintf(stderr, "no Vulkan instance; skipping\n");
    return 0;
  }
  vr::Result<VkPhysicalDevice> gpu = instance.value().select_physical_device();
  if (!gpu) {
    std::fprintf(stderr, "no compute-capable device; skipping\n");
    return 0;
  }
  vr::Result<vr::Device> device =
      vr::Device::create(instance.value(), gpu.value(), {});
  CHECK(device.ok());
  vr::Result<vr::Allocator> allocator =
      vr::Allocator::create(instance.value().handle(), device.value());
  CHECK(allocator.ok());
  vr::Device& dev = device.value();
  vr::Allocator& alloc = allocator.value();

  const vol::AttributeSpec attrs[] = {{"tsdf", sizeof(float)},
                                      {"weight", sizeof(float)}};
  auto made = vol::VoxelBlockGrid::create(dev, alloc, params(), attrs, 2);
  CHECK(made.ok());
  vol::VoxelBlockGrid& grid = made.value();
  vol::VoxelHashMap& map = grid.map();
  CHECK(map.tick() == 1);

  // Tick 1: three blocks, requested now and never weighted.
  const vol::BlockIndex three[] = {at(0), at(1), at(2)};
  CHECK(map.allocate(three, 3).ok());
  auto slot = slots(grid);
  CHECK(slot.ok() && slot.value().size() == 3);
  const std::uint32_t a = slot.value()[Coord{0, 0, 0}];
  const std::uint32_t b = slot.value()[Coord{1, 0, 0}];
  const std::uint32_t c = slot.value()[Coord{2, 0, 0}];
  auto st = map.read_block_stamps();
  CHECK(st.ok());
  for (const std::uint32_t s : {a, b, c}) {
    CHECK(st.value()[s].requested == 1 && st.value()[s].weighted == 0);
  }

  // The first block holds weight; at tick 3 the pass stamps it, and asking
  // for the third again restamps that one.
  {
    auto w = vr_test::read_attribute<float>(dev, alloc, grid, "weight");
    CHECK(w.ok());
    w.value()[std::size_t{a} * 512 + 7] = 1.0f;
    CHECK(vr_test::write_attribute(dev, alloc, grid, "weight", w.value()).ok());
  }
  map.advance_tick();
  map.advance_tick();
  CHECK(grid.stamp_blocks().ok());
  CHECK(map.allocate(&three[2], 1).ok());
  st = map.read_block_stamps();
  CHECK(st.ok());
  CHECK(st.value()[a].requested == 1 && st.value()[a].weighted == 3);
  CHECK(st.value()[b].requested == 1 && st.value()[b].weighted == 0);
  CHECK(st.value()[c].requested == 3 && st.value()[c].weighted == 0);

  // Tick 3, max_age 3: the second block is two ticks old, so nothing goes.
  auto freed = grid.free_stale_blocks(3);
  CHECK(freed.ok() && freed.value() == 0);

  // Tick 4: the second block is three ticks old and goes, its record zeroed;
  // the weighted one and the one asked for at tick 3 stay.
  map.advance_tick();
  freed = grid.free_stale_blocks(3);
  CHECK(freed.ok() && freed.value() == 1);
  slot = slots(grid);
  CHECK(slot.ok() && slot.value().size() == 2 &&
        slot.value().count(Coord{1, 0, 0}) == 0);
  st = map.read_block_stamps();
  CHECK(st.ok());
  CHECK(st.value()[b].requested == 0 && st.value()[b].weighted == 0);
  CHECK(st.value()[a].weighted == 4);

  // Every allocation path stamps what it asks for, through a binding of its
  // own: coords (above), points, triangles and depth.
  map.advance_tick();  // 5
  auto before = slots(grid);
  CHECK(before.ok());
  const vr::Vec3f point(1.0f, 1.0f, 1.0f);
  CHECK(map.allocate_from_points(&point, 1).ok());
  auto stamped = new_blocks_requested_now(grid, before.value());
  CHECK(stamped.ok() && stamped.value());

  map.advance_tick();  // 6
  before = slots(grid);
  CHECK(before.ok());
  const vr::Vec3f tri[] = {
      {-1.0f, 1.0f, 0.0f}, {-0.9f, 1.0f, 0.0f}, {-1.0f, 1.1f, 0.0f}};
  const std::uint32_t idx[] = {0, 1, 2};
  CHECK(map.allocate_from_triangles(tri, 3, idx, 1).ok());
  stamped = new_blocks_requested_now(grid, before.value());
  CHECK(stamped.ok() && stamped.value());

  map.advance_tick();  // 7
  before = slots(grid);
  CHECK(before.ok());
  vr::DepthCameraParams cam{};
  cam.fx = cam.fy = 16.0f;
  cam.cx = cam.cy = 8.0f;
  cam.min_depth = 0.1f;
  cam.max_depth = 5.0f;
  cam.width = cam.height = 16;
  cam.cam_to_world = vr::Mat4f(1.0f);
  cam.cam_to_world[3] = vr::Vec4f(0.0f, -1.0f, 0.0f, 1.0f);
  const std::vector<float> depth(16 * 16, 0.5f);
  CHECK(map.allocate_from_depth(depth.data(), cam).ok());
  stamped = new_blocks_requested_now(grid, before.value());
  CHECK(stamped.ok() && stamped.value());

  // resize keeps each record in its slot -- the rehash does not restamp at
  // tick 8 -- and the new slots are zeroed.
  map.advance_tick();  // 8
  st = map.read_block_stamps();
  CHECK(st.ok());
  CHECK(grid.resize(params().num_buckets * 2).ok());
  auto grown = map.read_block_stamps();
  CHECK(grown.ok() && grown.value().size() == 2 * st.value().size());
  for (std::size_t i = 0; i < st.value().size(); ++i) {
    CHECK(grown.value()[i].requested == st.value()[i].requested &&
          grown.value()[i].weighted == st.value()[i].weighted);
  }
  CHECK(grown.value().back().requested == 0);

  // clear zeroes every record.
  CHECK(grid.clear().ok());
  auto cleared = map.read_block_stamps();
  CHECK(cleared.ok());
  for (const vol::BlockStamp& s : cleared.value()) {
    CHECK(s.requested == 0 && s.weighted == 0);
  }

  // Refusals: a max_age of 0, and a grid with no weight.
  CHECK(grid.free_stale_blocks(0).status().domain() ==
        vr::Status::Code::InvalidArgument);
  const vol::AttributeSpec no_weight[] = {{"tsdf", sizeof(float)}};
  auto bare = vol::VoxelBlockGrid::create(dev, alloc, params(), no_weight, 1);
  CHECK(bare.ok());
  CHECK(bare->map().allocate(three, 1).ok());
  CHECK(bare->stamp_blocks().domain() == vr::Status::Code::InvalidArgument);

  std::puts("volume_block_stamps: OK");
  return 0;
}
