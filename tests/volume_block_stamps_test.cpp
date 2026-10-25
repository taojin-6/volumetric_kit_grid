// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The block stamps and the GC over them. Every allocation path stamps the
// blocks it asks for -- inserted or already there -- with the map's tick
// (`requested`); the grid's block pass stamps the blocks holding an observed
// voxel (`weighted`); and free_stale_blocks frees exactly the blocks that have
// been neither for more than max_age ticks, zeroing their voxels. A freed
// slot's record is zeroed, resize keeps every record in its slot (the rehash
// does not restamp), and clear zeroes them all. The first timed pass makes
// the grid's timer and times its row. record_free_stale, in a caller's batch,
// frees what the blocking pass frees, zeroing first, in one warm submit, and
// leaves a contended block for the next pass. Skips where no device is
// present.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <tuple>
#include <vector>

#include "device_blocks.hpp"
#include "test_check.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/gpu_timer.hpp"
#include "volumetric_kit/recon/core/camera_params.hpp"
#include "volumetric_kit/recon/core/dispatch.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"

#include "gpu_test.hpp"
#include "grid_readback.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace vol = volumetric_kit::recon::volume;

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

int gpu_main(vr_test::GpuContext& gpu) {
  vkc::Device& dev = gpu.device;
  vkc::Allocator& alloc = gpu.allocator;

  const vol::AttributeSpec attrs[] = {{"tsdf", sizeof(float)},
                                      {"weight", sizeof(float)}};
  auto made = vol::VoxelBlockGrid::create(dev, alloc, params(), attrs, 2);
  CHECK(made.ok());
  vol::VoxelBlockGrid& grid = made.value();
  vol::VoxelHashMap& map = grid.map();
  CHECK(map.tick() == 1);
  // The clock skips 0 both ways.
  map.rewind_tick();
  CHECK(map.tick() == ~std::uint32_t{0});
  map.advance_tick();
  CHECK(map.tick() == 1);

  // Tick 1: three blocks, requested now and never weighted.
  const vol::BlockIndex three[] = {at(0), at(1), at(2)};
  CHECK(vr_test::allocate_blocks(gpu, map, three, 3).ok());
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
  // The block pass, freeing nothing at this age.
  auto kept = grid.free_stale_blocks(1000);
  CHECK(kept.ok() && kept.value() == 0);
  CHECK(vr_test::allocate_blocks(gpu, map, &three[2], 1).ok());
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
  {
    // A compaction of the map's own list, so the pass below compacts afresh
    // rather than taking back the list the pass before left.
    vkc::CommandBatch batch(dev, alloc);
    CHECK(map.record_compaction(batch, nullptr).ok());
    CHECK(batch.submit().ok());
  }
  // Timed, the first timed pass, which makes the grid's timer: the stamps and
  // the frees in one row.
  vkc::StageMetrics gc;
  freed = grid.free_stale_blocks(2, &gc);
  CHECK(freed.ok() && freed.value() == 1);
  {
    CHECK(!gc.empty());
    const vkc::StageRow& pass = gc.rows()[0];
    CHECK(std::strcmp(pass.name, "block stamps") == 0);
    CHECK(pass.wait_ms <= pass.cpu_ms);
    // Its compaction, named apart from a fuse's "  ..active set" and an
    // encode's "  ..encode set", which a sender times into the same rows.
    CHECK(gc.rows().size() >= 2 &&
          std::strcmp(gc.rows()[1].name, "  ..stale set") == 0);
    vkc::Result<vkc::GpuTimer> probe = vkc::GpuTimer::create(dev);
    CHECK(probe.ok());
    if (probe.value().available()) {
      CHECK(pass.device == vkc::DeviceTime::Measured && pass.gpu_ms > 0.0 &&
            pass.gpu_ms < pass.cpu_ms);
    }
  }
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
    CHECK(built.ok() && built->free_stale_blocks(1).ok());
    vol::VoxelBlockGrid moved = std::move(built).value();
    CHECK(moved.free_stale_blocks(1).ok());
    // NOLINTNEXTLINE(bugprone-use-after-move) -- asserting it is empty
    CHECK(built->free_stale_blocks(1).status().domain() ==
          vkc::Status::Code::InvalidArgument);
  }

  // Every allocation path stamps what it asks for, through a binding of its
  // own: coords (above), triangles and depth.
  map.advance_tick();  // 5
  auto before = slots(grid);
  CHECK(before.ok());
  const vr::Vec3f tri[] = {
      {-1.0f, 1.0f, 0.0f}, {-0.9f, 1.0f, 0.0f}, {-1.0f, 1.1f, 0.0f}};
  const std::uint32_t idx[] = {0, 1, 2};
  CHECK(map.allocate_from_triangles(tri, 3, idx, 1).ok());
  auto stamped = new_blocks_requested_now(grid, before.value());
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
  // longer previous pass or a host coordinate copy. At 16 blocks a dispatch,
  // the 120-block passes split as a list past the device's limit does.
  vr::detail::set_test_groups_per_dispatch(16);
  for (const int count : {7, 120, 3}) {
    std::vector<vol::BlockIndex> coords;
    for (int i = 0; i < count; ++i) coords.push_back(at(1000 + i));
    auto placed = vr_test::allocate_blocks(gpu, map, coords.data(),
                                           static_cast<std::uint32_t>(count));
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
  vr::detail::set_test_groups_per_dispatch(0);

  // A completed recording may still be unsettled when the blocking free
  // pass starts. Count only this pass's removals: an earlier removal must not
  // be counted twice, and an earlier allocation must not hide a removal.
  for (const bool removes : {false, true}) {
    auto g = vol::VoxelBlockGrid::create(dev, alloc, params(), attrs, 2);
    CHECK(g.ok());
    const vol::BlockIndex block = at(1900);
    auto list = vr_test::detail::device_coords(gpu, &block, 1);
    const std::uint32_t one = 1;
    auto count = vr_test::upload_device_buffer(dev, alloc, &one, sizeof(one));
    CHECK(list.ok() && count.ok());
    if (removes) {
      CHECK(vr_test::allocate_blocks(gpu, g->map(), &block, 1).value() == 0);
    }
    vkc::CommandBatch batch(dev, alloc);
    const vol::DeviceCount n{&*count, 0, one};
    auto earlier = removes ? g->record_remove(batch, *list, n)
                           : g->map().record_allocate(batch, *list, n);
    CHECK(earlier.ok() && batch.submit().ok());
    CHECK(g->map().allocated_blocks() == (removes ? 1u : 0u));
    auto before = slots(*g);
    CHECK(before.ok() && before->size() == (removes ? 0u : 1u));
    g->map().advance_tick();
    g->map().advance_tick();
    auto freed = g->free_stale_blocks(1);
    CHECK(freed.ok() && freed.value() == (removes ? 0u : 1u));
    auto after = slots(*g);
    CHECK(after.ok() && after->empty() && g->map().allocated_blocks() == 0);
    CHECK(g->map().settle(*earlier).value().total == 0);
    CHECK(g->map().allocated_blocks() == 0);
  }

  // The record form is the blocking pass: of two grids holding 40 blocks, 10
  // weighted, one freed by free_stale_blocks and one by record_free_stale in
  // a caller's batch keep the same 10. A block the batch allocates after the
  // pass reads zeros, every voxel having held 1: each stale block is zeroed
  // before its index is freed. A warm pass is one submit, no staging buffer.
  {
    auto a = vol::VoxelBlockGrid::create(dev, alloc, params(), attrs, 2);
    auto b = vol::VoxelBlockGrid::create(dev, alloc, params(), attrs, 2);
    CHECK(a.ok() && b.ok());
    std::vector<vol::BlockIndex> coords;
    for (int i = 0; i < 40; ++i) coords.push_back(at(2000 + i));
    for (vol::VoxelBlockGrid* g : {&a.value(), &b.value()}) {
      CHECK(
          vr_test::allocate_blocks(gpu, g->map(), coords.data(), 40).value() ==
          0);
      auto t = vr_test::read_attribute<float>(dev, alloc, *g, "tsdf");
      auto w = vr_test::read_attribute<float>(dev, alloc, *g, "weight");
      auto at_slot = slots(*g);
      CHECK(t.ok() && w.ok() && at_slot.ok());
      for (float& v : t.value()) v = 1.0f;
      for (const auto& [coord, s] : at_slot.value()) {
        if (std::get<0>(coord) < 2010) w.value()[std::size_t{s} * 512] = 1.0f;
      }
      CHECK(vr_test::write_attribute(dev, alloc, *g, "tsdf", t.value()).ok());
      CHECK(vr_test::write_attribute(dev, alloc, *g, "weight", w.value()).ok());
      g->map().advance_tick();
      g->map().advance_tick();
    }
    auto freed_a = a->free_stale_blocks(1);
    CHECK(freed_a.ok() && freed_a.value() == 30);

    std::vector<vol::BlockIndex> fresh;
    for (int i = 0; i < 30; ++i) fresh.push_back(at(3000 + i));
    auto fresh_list = vr_test::detail::device_coords(gpu, fresh.data(), 30);
    const std::uint32_t thirty = 30;
    auto count =
        vr_test::upload_device_buffer(dev, alloc, &thirty, sizeof(thirty));
    CHECK(fresh_list.ok() && count.ok());
    vkc::CommandBatch batch(dev, alloc);
    auto removed = b->record_free_stale(batch, 1);
    auto placed = b->map().record_allocate(batch, *fresh_list,
                                           vol::DeviceCount{&*count, 0, 30});
    CHECK(removed.ok() && placed.ok() && batch.submit().ok());
    auto r = b->map().settle(*removed);
    CHECK(r.ok() && r->total == 0 && r->terminal == 0);
    CHECK(b->map().allocated_blocks() == 10);
    auto p = b->map().settle(*placed);
    CHECK(p.ok() && p->total == 0 && b->map().allocated_blocks() == 40);

    auto kept_a = slots(*a);
    auto kept_b = slots(*b);
    CHECK(kept_a.ok() && kept_b.ok() && kept_a->size() == 10);
    auto t = vr_test::read_attribute<float>(dev, alloc, *b, "tsdf");
    CHECK(t.ok());
    for (const auto& [coord, s] : kept_a.value()) {
      CHECK(kept_b->count(coord) == 1);
    }
    for (const auto& [coord, s] : kept_b.value()) {
      const float want = std::get<0>(coord) >= 3000 ? 0.0f : 1.0f;
      for (std::size_t k = 0; k < 512; ++k) {
        CHECK(t.value()[std::size_t{s} * 512 + k] == want);
      }
    }

    for (int pass = 0; pass < 2; ++pass) {
      const vkc::DeviceCounters was = dev.counters();
      vkc::CommandBatch warm(dev, alloc);
      auto again = a->record_free_stale(warm, 1);
      CHECK(again.ok() && warm.submit().ok());
      const vkc::DeviceCounters now = dev.counters();
      CHECK(a->map().settle(*again).ok() && a->map().allocated_blocks() == 10);
      CHECK(now.submits - was.submits == 1);
      CHECK(now.staging_buffers == was.staging_buffers);
    }
  }

  // Contention past the removal's rounds: eight stale blocks in one 8-slot
  // bucket, every free queued on its lock. A block a lost race keeps stays
  // for the next pass, which frees it; lavapipe frees all eight at once.
  {
    vol::VoxelGridParams one = params();
    one.num_buckets = 1;
    one.num_blocks = 8;
    auto g = vol::VoxelBlockGrid::create(dev, alloc, one, attrs, 2);
    CHECK(g.ok());
    std::vector<vol::BlockIndex> eight;
    for (int i = 0; i < 8; ++i) eight.push_back(at(i));
    CHECK(vr_test::allocate_blocks(gpu, g->map(), eight.data(), 8).value() ==
          0);
    g->map().advance_tick();
    g->map().advance_tick();
    vkc::CommandBatch batch(dev, alloc);
    auto removed = g->record_free_stale(batch, 1);
    CHECK(removed.ok() && batch.submit().ok());
    auto f = g->map().settle(*removed);
    CHECK(f.ok() && f->total == f->lock && f->terminal == 0);
    CHECK(g->map().allocated_blocks() == f->lock);
    std::printf("  a contended pass left %u of 8 blocks\n", f->lock);
    for (int pass = 0; pass < 8 && g->map().allocated_blocks() != 0; ++pass) {
      CHECK(g->free_stale_blocks(1).ok());
    }
    CHECK(g->map().allocated_blocks() == 0);
  }

  // Refusals: a max_age of 0, and a grid with no weight.
  CHECK(grid.free_stale_blocks(0).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  const vol::AttributeSpec no_weight[] = {{"tsdf", sizeof(float)}};
  auto bare = vol::VoxelBlockGrid::create(dev, alloc, params(), no_weight, 1);
  CHECK(bare.ok());
  CHECK(vr_test::allocate_blocks(gpu, bare->map(), three, 1).ok());
  CHECK(bare->free_stale_blocks(1).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  {
    vkc::CommandBatch refused(dev, alloc);
    CHECK(bare->record_free_stale(refused, 1).status().domain() ==
          vkc::Status::Code::InvalidArgument);
    CHECK(grid.record_free_stale(refused, 0).status().domain() ==
          vkc::Status::Code::InvalidArgument);
    CHECK(refused.submit().ok());
  }

  std::puts("volume_block_stamps: OK");
  return 0;
}

}  // namespace

int main() { return vr_test::run_on_gpu(gpu_main); }
