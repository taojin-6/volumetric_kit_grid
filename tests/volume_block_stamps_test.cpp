// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The block stamps and the GC over them. Every allocation path stamps the
// blocks it asks for -- inserted or already there -- with the map's tick
// (`requested`); the grid's block pass stamps the blocks holding an observed
// voxel (`weighted`); and free_stale_blocks frees exactly the blocks that have
// been neither for more than max_age ticks, zeroing their voxels. A freed
// slot's record is zeroed, resize keeps every record in its slot (the rehash
// does not restamp), and clear zeroes them all. Exits 0 (skip) where no device
// is present.

#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <tuple>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/recon/core/camera_params.hpp"
#include "volumetric_kit/recon/core/device_requirements.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

#include "grid_readback.hpp"
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
vkc::Result<std::map<Coord, std::uint32_t>> slots(vol::VoxelBlockGrid& g) {
  VKC_ASSIGN(const std::vector<vol::BlockIndex> active,
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
vkc::Result<bool> new_blocks_requested_now(
    vol::VoxelBlockGrid& g, const std::map<Coord, std::uint32_t>& before) {
  VKC_ASSIGN(const auto after, slots(g));
  VKC_ASSIGN(const std::vector<vol::BlockStamp> st,
             g.map().read_block_stamps());
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
  vkc::Result<vkc::Instance> instance = vkc::Instance::create({});
  if (!instance) {
    return vr_test::no_device("no Vulkan instance",
                              instance.status().message());
  }
  vkc::Result<vkc::PhysicalDeviceInfo> gpu =
      instance.value().select_physical_device(vr::device_requirements());
  if (!gpu) {
    return vr_test::no_device("no compute-capable device",
                              gpu.status().message());
  }
  vkc::Result<vkc::Device> device = vkc::Device::create(
      instance.value(), gpu.value(), vr::device_requirements());
  CHECK(device.ok());
  vkc::Result<vkc::Allocator> allocator =
      vkc::Allocator::create(instance.value().handle(), device.value());
  CHECK(allocator.ok());
  vkc::Device& dev = device.value();
  vkc::Allocator& alloc = allocator.value();

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

  // The first block holds weight. The second holds a tsdf and a weight under
  // kObservedWeight, which the mesher and the codec read as none, and so does
  // the pass. At tick 3 the pass stamps the first, and asking for the third
  // again restamps that one.
  {
    auto w = vr_test::read_attribute<float>(dev, alloc, grid, "weight");
    auto t = vr_test::read_attribute<float>(dev, alloc, grid, "tsdf");
    CHECK(w.ok() && t.ok());
    w.value()[std::size_t{a} * 512 + 7] = 1.0f;
    w.value()[std::size_t{b} * 512 + 7] = 0.5f * vol::kObservedWeight;
    t.value()[std::size_t{a} * 512 + 7] = 0.25f;
    for (std::size_t k = 0; k < 512; ++k) {
      t.value()[std::size_t{b} * 512 + k] = 0.5f;
    }
    CHECK(vr_test::write_attribute(dev, alloc, grid, "weight", w.value()).ok());
    CHECK(vr_test::write_attribute(dev, alloc, grid, "tsdf", t.value()).ok());
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

  // Tick 3, max_age 2: the second block is two ticks old, no more, so nothing
  // goes.
  auto freed = grid.free_stale_blocks(2);
  CHECK(freed.ok() && freed.value() == 0);

  // Tick 4: the second block is three ticks old and goes, its record zeroed --
  // `changed` too, set here as a writer would so the zero has something to
  // clear; the weighted one and the one asked for at tick 3 stay.
  {
    auto all = map.read_block_stamps();
    CHECK(all.ok());
    all.value()[b].changed = 3;
    vkc::CommandBatch batch(dev, alloc);
    CHECK(batch
              .upload(map.stamps_buffer(), 0, all.value().data(),
                      all.value().size() * sizeof(vol::BlockStamp))
              .ok());
    CHECK(batch.submit().ok());
  }
  map.advance_tick();
  freed = grid.free_stale_blocks(2);
  CHECK(freed.ok() && freed.value() == 1);
  slot = slots(grid);
  CHECK(slot.ok() && slot.value().size() == 2 &&
        slot.value().count(Coord{1, 0, 0}) == 0);
  st = map.read_block_stamps();
  CHECK(st.ok());
  CHECK(st.value()[b].requested == 0 && st.value()[b].weighted == 0 &&
        st.value()[b].changed == 0);
  CHECK(st.value()[a].weighted == 4);
  // Its voxels are zeroed, every attribute, and the kept block's stay.
  {
    auto w = vr_test::read_attribute<float>(dev, alloc, grid, "weight");
    auto t = vr_test::read_attribute<float>(dev, alloc, grid, "tsdf");
    CHECK(w.ok() && t.ok());
    for (std::size_t k = 0; k < 512; ++k) {
      CHECK(w.value()[std::size_t{b} * 512 + k] == 0.0f &&
            t.value()[std::size_t{b} * 512 + k] == 0.0f);
    }
    CHECK(w.value()[std::size_t{a} * 512 + 7] == 1.0f &&
          t.value()[std::size_t{a} * 512 + 7] == 0.25f);
  }

  // Move construction carries the block pass along and empties the source.
  {
    auto built = vol::VoxelBlockGrid::create(dev, alloc, params(), attrs, 2);
    CHECK(built.ok() && built->stamp_blocks().ok());
    vol::VoxelBlockGrid moved = std::move(built).value();
    CHECK(moved.stamp_blocks().ok());
    CHECK(moved.free_stale_blocks(1).ok());
    // NOLINTNEXTLINE(bugprone-use-after-move) -- asserting it is empty
    CHECK(built->stamp_blocks().domain() == vkc::Status::Code::InvalidArgument);
  }

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
          grown.value()[i].weighted == st.value()[i].weighted &&
          grown.value()[i].changed == st.value()[i].changed);
  }
  CHECK(grown.value().back().requested == 0);

  // clear zeroes every record.
  CHECK(grid.clear().ok());
  auto cleared = map.read_block_stamps();
  CHECK(cleared.ok());
  for (const vol::BlockStamp& s : cleared.value()) {
    CHECK(s.requested == 0 && s.weighted == 0 && s.changed == 0);
  }

  // Repeated GC grows then reuses a device stale-list buffer. A later short
  // list must delete only its live prefix, without stale entries from the
  // longer previous pass or a host coordinate copy.
  for (const int count : {7, 120, 3}) {
    std::vector<vol::BlockIndex> coords;
    for (int i = 0; i < count; ++i) coords.push_back(at(1000 + i));
    auto placed =
        map.allocate(coords.data(), static_cast<std::uint32_t>(count));
    CHECK(placed.ok() && placed.value() == 0);
    map.advance_tick();
    map.advance_tick();
    auto gone = grid.free_stale_blocks(1);
    CHECK(gone.ok() && gone.value() == static_cast<std::uint32_t>(count));
    auto remaining = slots(grid);
    CHECK(remaining.ok() && remaining.value().empty());
    auto again = grid.free_stale_blocks(1);
    CHECK(again.ok() && again.value() == 0);
  }

  // Refusals: a max_age of 0, and a grid with no weight.
  CHECK(grid.free_stale_blocks(0).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  const vol::AttributeSpec no_weight[] = {{"tsdf", sizeof(float)}};
  auto bare = vol::VoxelBlockGrid::create(dev, alloc, params(), no_weight, 1);
  CHECK(bare.ok());
  CHECK(bare->map().allocate(three, 1).ok());
  CHECK(bare->stamp_blocks().domain() == vkc::Status::Code::InvalidArgument);

  std::puts("volume_block_stamps: OK");
  return 0;
}
