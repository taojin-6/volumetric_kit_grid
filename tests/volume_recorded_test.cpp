// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for the map's recorded rounds (VoxelHashMap::record_allocate,
// record_remove and settle) and VoxelBlockGrid::record_remove: what a caller's
// batch leaves is what the blocking calls leave, on the cube, delete and sheet
// fixtures, stamps and failures included; the count comes from the device,
// clamped to its bound; a closed gate changes nothing; a capacity limit, or
// contention past the rounds, closes it; a removal and an allocation share
// one batch, each settling its own outcome; the rounds split past a
// dispatch's groups; a recording makes cached lists stale; a steady
// recording costs one submit and no staging buffer; the result slots run
// out at the fifth recording held; and a grid zeroes a block before its index
// is freed. Skips with no device.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "buffer_readback.hpp"
#include "device_blocks.hpp"
#include "test_check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/recon/core/dispatch.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"
#include "volumetric_kit/recon/volume/voxel_grid.hpp"
#include "volumetric_kit/recon/volume/voxel_hash_map.hpp"

#include "gpu_test.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace vol = volumetric_kit::recon::volume;
using C = vkc::Status::Code;

namespace {

using Blocks = std::vector<vol::BlockIndex>;
using Coord = std::tuple<int, int, int>;

constexpr std::uint32_t kRounds = vol::VoxelHashMap::kRecordedRounds;

vol::VoxelGridParams params(std::int32_t bucket_size,
                            std::int32_t num_buckets) {
  vol::VoxelGridParams g{};
  g.voxel_size = 0.005f;
  g.block_size = 8;
  g.voxels_per_block = 512;
  g.trunc_dist = 0.04f;
  g.bucket_size = bucket_size;
  g.num_buckets = num_buckets;
  g.num_blocks = bucket_size * num_buckets;
  g.max_chain = 128;
  return g;
}

vol::BlockIndex at(int x, int y, int z) {
  vol::BlockIndex b{};
  b.coord = vr::Vec3i(x, y, z);
  return b;
}

// The 3x3x3 cube about the origin, and its 8 corners.
Blocks cube() {
  Blocks out;
  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      for (int z = -1; z <= 1; ++z) out.push_back(at(x, y, z));
    }
  }
  return out;
}

Blocks corners() {
  Blocks out;
  for (const vol::BlockIndex& b : cube()) {
    if (b.coord.x != 0 && b.coord.y != 0 && b.coord.z != 0) out.push_back(b);
  }
  return out;
}

// A w x h sheet at z.
Blocks sheet(int w, int h, int z = 0) {
  Blocks out;
  for (int i = 0; i < w * h; ++i) out.push_back(at(i % w, i / w, z));
  return out;
}

// The map's active set as coordinates, sorted.
vkc::Result<std::vector<Coord>> coords_of(vol::VoxelHashMap& map) {
  VKC_ASSIGN(const Blocks active, map.compact_active_blocks());
  std::vector<Coord> out;
  for (const vol::BlockIndex& b : active) {
    out.emplace_back(b.coord.x, b.coord.y, b.coord.z);
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<Coord> sorted(const Blocks& blocks) {
  std::vector<Coord> out;
  for (const vol::BlockIndex& b : blocks) {
    out.emplace_back(b.coord.x, b.coord.y, b.coord.z);
  }
  std::sort(out.begin(), out.end());
  return out;
}

// Everything a gated-off recording must leave as it was, bit for bit: the
// active blocks with their pointers, in the compaction's order, the heap's
// host copy, and every slot's stamp.
struct State {
  std::vector<std::tuple<int, int, int, std::int32_t>> blocks;
  std::uint32_t allocated = 0;
  std::vector<std::uint32_t> stamps;

  bool operator==(const State& o) const {
    return blocks == o.blocks && allocated == o.allocated && stamps == o.stamps;
  }
};

vkc::Result<State> state_of(vol::VoxelHashMap& map) {
  State s;
  VKC_ASSIGN(const Blocks active, map.compact_active_blocks());
  for (const vol::BlockIndex& b : active) {
    s.blocks.emplace_back(b.coord.x, b.coord.y, b.coord.z, b.ptr);
  }
  std::sort(s.blocks.begin(), s.blocks.end());
  s.allocated = map.allocated_blocks();
  VKC_ASSIGN(const std::vector<vol::BlockStamp> stamps,
             map.read_block_stamps());
  for (const vol::BlockStamp& st : stamps) {
    s.stamps.push_back(st.requested);
    s.stamps.push_back(st.weighted);
    s.stamps.push_back(st.changed);
  }
  return s;
}

// Words on the device, each a count or a gate a kernel could have written.
vkc::Result<vkc::Buffer> words(vr_test::Gpu& gpu,
                               const std::vector<std::uint32_t>& values) {
  return vr_test::upload_device_buffer(gpu.device, gpu.allocator, values.data(),
                                       values.size() * sizeof(std::uint32_t));
}

vkc::Result<std::uint32_t> word_at(vr_test::Gpu& gpu, const vkc::Buffer& b,
                                   std::size_t index) {
  VKC_ASSIGN(const std::vector<std::uint32_t> w,
             vr_test::read_back<std::uint32_t>(gpu.device, gpu.allocator, b, 1,
                                               index));
  return w[0];
}

// One recording of @p blocks in a batch of its own, the count a device word,
// settled: what a caller that records a frame's removal or allocation does.
vkc::Result<vol::AllocFailures> recorded(
    vr_test::Gpu& gpu, vol::VoxelHashMap& map, const Blocks& blocks,
    bool removes, const vol::DeviceCount* gate = nullptr) {
  VKC_ASSIGN(const vkc::Buffer list,
             vr_test::detail::device_coords(gpu, blocks.data(),
                                            std::uint32_t(blocks.size())));
  VKC_ASSIGN(const vkc::Buffer count,
             words(gpu, {std::uint32_t(blocks.size())}));
  const vol::DeviceCount n{&count, 0, std::uint32_t(blocks.size())};
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  VKC_ASSIGN(const vol::RecordedRounds rounds,
             removes ? map.record_remove(batch, list, n, gate)
                     : map.record_allocate(batch, list, n, gate));
  VKC_TRY(batch.submit());
  return map.settle(rounds);
}

// Recorded until no lost race is left, as the blocking call repeats.
vkc::Result<vol::AllocFailures> recorded_all(vr_test::Gpu& gpu,
                                             vol::VoxelHashMap& map,
                                             const Blocks& blocks,
                                             bool removes) {
  vol::AllocFailures f;
  for (int pass = 0; pass < 8; ++pass) {
    VKC_ASSIGN(f, recorded(gpu, map, blocks, removes));
    if (f.lock == 0) break;
  }
  return f;
}

// The blocking call, every lost race re-driven as a caller does; the last
// call's failures land in @p failures.
vkc::Result<std::uint32_t> blocking_all(
    vr_test::Gpu& gpu, vol::VoxelHashMap& map, const Blocks& blocks,
    bool removes, vol::AllocFailures* failures = nullptr) {
  std::uint32_t left = 0;
  for (int pass = 0; pass < 8; ++pass) {
    VKC_ASSIGN(left,
               removes ? vr_test::remove_blocks(gpu, map, blocks, failures)
                       : vr_test::allocate_blocks(gpu, map, blocks, failures));
    if (left == 0) break;
  }
  return left;
}

// Every block slot's stamp, sorted: what two maps' slots hold, wherever
// their blocks landed.
vkc::Result<
    std::vector<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>>>
stamps_of(const vol::VoxelHashMap& map) {
  VKC_ASSIGN(const std::vector<vol::BlockStamp> stamps,
             map.read_block_stamps());
  std::vector<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>> out;
  for (const vol::BlockStamp& s : stamps) {
    out.emplace_back(s.requested, s.weighted, s.changed);
  }
  std::sort(out.begin(), out.end());
  return out;
}

// The failures two maps' calls report alike: all but the lost races and the
// rounds, which contention decides.
bool same_failures(const vol::AllocFailures& a, const vol::AllocFailures& b) {
  return a.total == b.total && a.chain == b.chain && a.heap == b.heap &&
         a.table == b.table && a.terminal == b.terminal;
}

vkc::Result<vol::VoxelHashMap> make_map(vr_test::Gpu& gpu,
                                        const vol::VoxelGridParams& g) {
  return vol::VoxelHashMap::create(gpu.device, gpu.allocator, g);
}

// --- Recorded and blocking agree. --------------------------------------------

// Two maps, one driven by the blocking calls and one by recordings, through
// the cube's allocation, the corners' delete (chains of 2-slot buckets) and
// a 1 024-block sheet in a wide table: the same blocks, heap count, slot
// stamps and failures.
int agrees_case(vr_test::Gpu& gpu) {
  for (const auto& [g, all, gone] :
       {std::make_tuple(params(2, 1024), cube(), corners()),
        std::make_tuple(params(8, 4096), sheet(32, 32), sheet(16, 32))}) {
    vkc::Result<vol::VoxelHashMap> a = make_map(gpu, g);
    vkc::Result<vol::VoxelHashMap> b = make_map(gpu, g);
    CHECK(a.ok() && b.ok());
    vol::AllocFailures fa;
    CHECK(blocking_all(gpu, *a, all, false, &fa).value() == 0);
    vkc::Result<vol::AllocFailures> placed = recorded_all(gpu, *b, all, false);
    CHECK(placed.ok() && placed->total == 0 && placed->terminal == 0);
    CHECK(placed->rounds >= 1 && placed->rounds <= kRounds);
    CHECK(same_failures(fa, *placed));
    CHECK(coords_of(*a).value() == sorted(all));
    CHECK(coords_of(*b).value() == sorted(all));
    CHECK(a->allocated_blocks() == b->allocated_blocks());
    CHECK(b->allocated_blocks() == all.size());
    CHECK(stamps_of(*a).value() == stamps_of(*b).value());

    CHECK(blocking_all(gpu, *a, gone, true, &fa).value() == 0);
    vkc::Result<vol::AllocFailures> removed = recorded_all(gpu, *b, gone, true);
    CHECK(removed.ok() && removed->total == 0 && removed->terminal == 0);
    CHECK(same_failures(fa, *removed));
    CHECK(coords_of(*a).value() == coords_of(*b).value());
    CHECK(a->allocated_blocks() == b->allocated_blocks());
    CHECK(b->allocated_blocks() == all.size() - gone.size());
    CHECK(stamps_of(*a).value() == stamps_of(*b).value());

    // The freed blocks go back to the heap: allocating them again restores
    // the set, recorded as blocking.
    CHECK(blocking_all(gpu, *a, gone, false).value() == 0);
    CHECK(recorded_all(gpu, *b, gone, false).value().total == 0);
    CHECK(coords_of(*b).value() == sorted(all));
    CHECK(a->allocated_blocks() == b->allocated_blocks());
  }
  return 0;
}

// --- The count, the gate and what closes it. ---------------------------------

// The count is the device's word, at its offset, at most the bound; a bound
// of 0 records nothing and settles to zeros.
int count_case(vr_test::Gpu& gpu) {
  const Blocks all = cube();
  vkc::Result<vkc::Buffer> list = vr_test::detail::device_coords(
      gpu, all.data(), std::uint32_t(all.size()));
  CHECK(list.ok());
  vkc::Result<vkc::Buffer> counts = words(gpu, {7, 0, 20, 1000});
  CHECK(counts.ok());
  for (const auto& [offset, want] :
       {std::make_pair(VkDeviceSize(8), std::size_t(20)),
        std::make_pair(VkDeviceSize(12), all.size()),
        std::make_pair(VkDeviceSize(4), std::size_t(0))}) {
    vkc::Result<vol::VoxelHashMap> map = make_map(gpu, params(2, 1024));
    CHECK(map.ok());
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    const vol::DeviceCount n{&*counts, offset, std::uint32_t(all.size())};
    vkc::Result<vol::RecordedRounds> r = map->record_allocate(batch, *list, n);
    CHECK(r.ok() && batch.submit().ok());
    vkc::Result<vol::AllocFailures> f = map->settle(*r);
    CHECK(f.ok() && f->total == 0);
    CHECK(want != 0 ? f->rounds >= 1 && f->rounds <= kRounds : f->rounds == 0);
    CHECK(coords_of(*map).value() ==
          sorted(Blocks(all.begin(), all.begin() + std::ptrdiff_t(want))));
    CHECK(map->allocated_blocks() == want);
  }

  vkc::Result<vol::VoxelHashMap> map = make_map(gpu, params(2, 1024));
  CHECK(map.ok());
  vkc::CommandBatch none(gpu.device, gpu.allocator);
  vkc::Result<vol::RecordedRounds> nothing =
      map->record_allocate(none, *list, vol::DeviceCount{&*counts, 0, 0});
  CHECK(nothing.ok() && !nothing->result.valid());
  vkc::Result<vol::AllocFailures> zeros = map->settle(*nothing);
  CHECK(zeros.ok() && zeros->total == 0 && zeros->rounds == 0);
  CHECK(none.submit().ok());

  // Refused with nothing recorded: no buffer, a misaligned or past-the-end
  // word, a list shorter than the bound.
  const vkc::Buffer empty;
  vkc::CommandBatch refused(gpu.device, gpu.allocator);
  const auto domain = [&](const vol::DeviceCount& n,
                          const vol::DeviceCount* gate = nullptr) {
    return map->record_allocate(refused, *list, n, gate).status().domain();
  };
  CHECK(domain({nullptr, 0, 1}) == C::InvalidArgument);
  CHECK(domain({&empty, 0, 1}) == C::InvalidArgument);
  CHECK(domain({&*counts, 2, 1}) == C::InvalidArgument);
  CHECK(domain({&*counts, 16, 1}) == C::InvalidArgument);
  CHECK(domain({&*counts, 0, std::uint32_t(all.size()) + 1}) ==
        C::InvalidArgument);
  const vol::DeviceCount bad_gate{&*counts, 3, 1};
  CHECK(domain({&*counts, 0, 1}, &bad_gate) == C::InvalidArgument);
  CHECK(refused.submit().ok());
  CHECK(map->allocated_blocks() == 0);
  return 0;
}

// A closed gate runs no round of either kind: the map, its heap and its
// stamps are bit-identical, the settle is zeros and the gate stays 0. An
// open gate that meets no failure stays open.
int gate_case(vr_test::Gpu& gpu) {
  vkc::Result<vol::VoxelHashMap> map = make_map(gpu, params(2, 1024));
  CHECK(map.ok());
  const Blocks all = cube();
  CHECK(blocking_all(gpu, *map, corners(), false).value() == 0);
  vkc::Result<State> before = state_of(*map);
  CHECK(before.ok());
  vkc::Result<vkc::Buffer> gates = words(gpu, {1, 0});
  CHECK(gates.ok());
  const vol::DeviceCount closed{&*gates, 4, 1};
  for (const bool removes : {false, true}) {
    vkc::Result<vol::AllocFailures> f =
        recorded(gpu, *map, removes ? corners() : all, removes, &closed);
    CHECK(f.ok() && f->total == 0 && f->terminal == 0 && f->rounds == 0);
    CHECK(state_of(*map).value() == *before);
    CHECK(word_at(gpu, *gates, 1).value() == 0);
  }
  const vol::DeviceCount open{&*gates, 0, 1};
  vkc::Result<vol::AllocFailures> f = recorded(gpu, *map, all, false, &open);
  CHECK(f.ok() && f->total == 0 && f->rounds >= 1);
  CHECK(word_at(gpu, *gates, 0).value() == 1);
  CHECK(coords_of(*map).value() == sorted(all));
  return 0;
}

// A capacity limit ends the rounds after the one that met it and closes the
// gate: 16 coords for an 8-slot table.
int capacity_case(vr_test::Gpu& gpu) {
  vkc::Result<vol::VoxelHashMap> map = make_map(gpu, params(8, 1));
  CHECK(map.ok());
  vkc::Result<vkc::Buffer> gate = words(gpu, {1});
  CHECK(gate.ok());
  const vol::DeviceCount g{&*gate, 0, 1};
  const Blocks many = sheet(16, 1);
  vkc::Result<vol::AllocFailures> f = recorded(gpu, *map, many, false, &g);
  CHECK(f.ok());
  CHECK(f->capacity_limited() && f->rounds <= kRounds);
  CHECK(word_at(gpu, *gate, 0).value() == 0);
  CHECK(map->allocated_blocks() <= 8);
  return 0;
}

// Contention past the rounds: one 8-slot bucket, 8 blocks in it, and a
// removal of 1 024 coords, every thread queued on one lock (the absent coords
// still take it to learn they are absent). A recording runs at most K rounds
// and closes the gate on the races it leaves; the blocking call runs at most
// 10 rounds, one submit for every K, and loses nothing. lavapipe settles
// them all and the checks hold trivially there.
int contention_case(vr_test::Gpu& gpu) {
  Blocks coords;
  for (int i = 0; i < 1024; ++i) coords.push_back(at(i, 0, 0));
  const Blocks placed(coords.begin(), coords.begin() + 8);
  vkc::Result<vkc::Buffer> gate = words(gpu, {1});
  CHECK(gate.ok());
  {
    vkc::Result<vol::VoxelHashMap> map = make_map(gpu, params(8, 1));
    CHECK(map.ok() && blocking_all(gpu, *map, placed, false).value() == 0);
    const vol::DeviceCount g{&*gate, 0, 1};
    vkc::Result<vol::AllocFailures> f = recorded(gpu, *map, coords, true, &g);
    CHECK(f.ok() && f->terminal == 0 && f->total == f->lock);
    CHECK(f->rounds >= 1 && f->rounds <= kRounds);
    CHECK(word_at(gpu, *gate, 0).value() == (f->lock != 0 ? 0u : 1u));
    CHECK(blocking_all(gpu, *map, placed, true).value() == 0);
    CHECK(map->allocated_blocks() == 0);
  }
  vkc::Result<vol::VoxelHashMap> map = make_map(gpu, params(8, 1));
  CHECK(map.ok() && blocking_all(gpu, *map, placed, false).value() == 0);
  vkc::Result<vkc::Buffer> list = vr_test::detail::device_coords(
      gpu, coords.data(), std::uint32_t(coords.size()));
  CHECK(list.ok());
  const vkc::DeviceCounters before = gpu.device.counters();
  vol::AllocFailures f;
  vkc::Result<std::uint32_t> left =
      map->remove(*list, std::uint32_t(coords.size()), &f);
  const vkc::DeviceCounters after = gpu.device.counters();
  CHECK(left.ok() && left.value() == f.lock && f.terminal == 0);
  CHECK(f.rounds >= 1 && f.rounds <= 10);
  CHECK(after.submits - before.submits == (f.rounds + kRounds - 1) / kRounds);
  CHECK(blocking_all(gpu, *map, placed, true).value() == 0);
  CHECK(map->allocated_blocks() == 0);

  // A few coords on the one lock settle within the blocking call's rounds.
  vkc::Result<vol::VoxelHashMap> few = make_map(gpu, params(8, 1));
  CHECK(few.ok());
  vol::AllocFailures g;
  vkc::Result<std::uint32_t> missed =
      vr_test::allocate_blocks(gpu, *few, placed, &g);
  CHECK(missed.ok() && missed.value() == 0 && g.rounds <= 10);
  CHECK(few->allocated_blocks() == placed.size());
  return 0;
}

// --- Batches.
// -----------------------------------------------------------------

// A removal and an allocation in one batch, under sync validation where the
// run asks for it: the corners go, eight new blocks come, and the grid zeroes
// every removed block before its index is freed, so a new block that draws a
// freed pointer reads zeros although the batch allocates after the removal.
int one_batch_case(vr_test::Gpu& gpu) {
  const vol::AttributeSpec attrs[] = {{"tsdf", 4}, {"weight", 4}};
  vkc::Result<vol::VoxelBlockGrid> grid = vol::VoxelBlockGrid::create(
      gpu.device, gpu.allocator, params(2, 1024), attrs, 2);
  CHECK(grid.ok());
  const Blocks all = cube();
  CHECK(blocking_all(gpu, grid->map(), all, false).value() == 0);
  vkc::Result<vol::AttributeView> tsdf = grid->attribute("tsdf");
  CHECK(tsdf.ok());
  {
    vkc::CommandBatch fill(gpu.device, gpu.allocator);
    CHECK(fill.fill(*tsdf->buffer, 0, tsdf->buffer->size(), 0x3F800000u).ok());
    CHECK(fill.submit().ok());
  }
  const Blocks gone = corners();
  Blocks fresh;
  for (int i = 0; i < 8; ++i) fresh.push_back(at(10 + i, 0, 0));
  vkc::Result<vkc::Buffer> gone_list = vr_test::detail::device_coords(
      gpu, gone.data(), std::uint32_t(gone.size()));
  vkc::Result<vkc::Buffer> fresh_list = vr_test::detail::device_coords(
      gpu, fresh.data(), std::uint32_t(fresh.size()));
  vkc::Result<vkc::Buffer> counts = words(gpu, {8, 8, 1});
  CHECK(gone_list.ok() && fresh_list.ok() && counts.ok());
  const vol::DeviceCount gate{&*counts, 8, 1};
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  vkc::Result<vol::RecordedRounds> removed = grid->record_remove(
      batch, *gone_list, vol::DeviceCount{&*counts, 0, 8}, &gate);
  vkc::Result<vol::RecordedRounds> placed = grid->map().record_allocate(
      batch, *fresh_list, vol::DeviceCount{&*counts, 4, 8}, &gate);
  CHECK(removed.ok() && placed.ok() && batch.submit().ok());
  // The removal's outcome is its own, gathered before the allocation's
  // rounds: the heap as the corners left it, before the new blocks drew.
  vkc::Result<vol::AllocFailures> r = grid->map().settle(*removed);
  CHECK(grid->map().allocated_blocks() == all.size() - gone.size());
  vkc::Result<vol::AllocFailures> p = grid->map().settle(*placed);
  CHECK(r.ok() && r->total == 0 && r->terminal == 0);
  CHECK(p.ok() && p->total == 0);
  CHECK(word_at(gpu, *counts, 2).value() == 1);
  Blocks want;
  for (const vol::BlockIndex& b : all) {
    if (!(b.coord.x != 0 && b.coord.y != 0 && b.coord.z != 0)) {
      want.push_back(b);
    }
  }
  want.insert(want.end(), fresh.begin(), fresh.end());
  CHECK(coords_of(grid->map()).value() == sorted(want));
  CHECK(grid->map().allocated_blocks() == want.size());
  vkc::Result<Blocks> active = grid->map().compact_active_blocks();
  CHECK(active.ok());
  for (const vol::BlockIndex& b : *active) {  // ptr: the first voxel
    vkc::Result<std::vector<float>> voxels = vr_test::read_back<float>(
        gpu.device, gpu.allocator, *tsdf->buffer, 512, std::size_t(b.ptr));
    CHECK(voxels.ok());
    const float want_value = b.coord.x >= 10 ? 0.0f : 1.0f;
    for (const float v : *voxels) CHECK(v == want_value);
  }
  return 0;
}

// Past a dispatch's groups: a round of 1 024 coords split into four slices
// of one group each places what one dispatch places.
int slices_case(vr_test::Gpu& gpu) {
  vr::detail::set_test_groups_per_dispatch(1);
  const Blocks all = sheet(32, 32);
  vkc::Result<vol::VoxelHashMap> map = make_map(gpu, params(8, 4096));
  CHECK(map.ok());
  vkc::Result<vol::AllocFailures> f = recorded_all(gpu, *map, all, false);
  CHECK(f.ok() && f->total == 0);
  CHECK(coords_of(*map).value() == sorted(all));
  vkc::Result<vol::AllocFailures> g = recorded_all(gpu, *map, all, true);
  CHECK(g.ok() && g->total == 0 && map->allocated_blocks() == 0);
  vr::detail::set_test_groups_per_dispatch(0);
  return 0;
}

// A recording of either kind makes every list compacted before it stale,
// even one that changes nothing.
int stale_case(vr_test::Gpu& gpu) {
  vkc::Result<vol::VoxelHashMap> map = make_map(gpu, params(2, 1024));
  CHECK(map.ok());
  const Blocks all = cube();
  CHECK(blocking_all(gpu, *map, all, false).value() == 0);
  for (const bool removes : {false, true}) {
    vkc::Result<vol::DeviceBlockList> list =
        map->compact_active_blocks_on_device();
    CHECK(list.ok() && map->check_device_block_list(*list, "test").ok());
    CHECK(map->held_device_list().has_value() &&
          map->held_device_list()->serial == list->serial);
    const std::uint64_t epoch = map->topology_epoch();
    const Blocks absent = {at(50, 50, 50)};
    CHECK(recorded(gpu, *map, absent, removes).ok());
    CHECK(!map->check_device_block_list(*list, "test").ok());
    CHECK(!map->held_device_list().has_value());
    CHECK((map->topology_epoch() != epoch) == removes);
  }
  return 0;
}

// --- Steady state and the result slots. --------------------------------------

// Once warm, a batch holding a removal and an allocation costs one submit,
// no staging buffer, and the rounds' dispatches: K rounds and K + 1 one-thread
// kernels a kind.
int counters_case(vr_test::Gpu& gpu) {
  vkc::Result<vol::VoxelHashMap> map = make_map(gpu, params(2, 1024));
  CHECK(map.ok());
  const Blocks all = cube();
  const Blocks gone = corners();
  vkc::Result<vkc::Buffer> all_list = vr_test::detail::device_coords(
      gpu, all.data(), std::uint32_t(all.size()));
  vkc::Result<vkc::Buffer> gone_list = vr_test::detail::device_coords(
      gpu, gone.data(), std::uint32_t(gone.size()));
  vkc::Result<vkc::Buffer> counts = words(gpu, {27, 8});
  CHECK(all_list.ok() && gone_list.ok() && counts.ok());
  const vol::DeviceCount n_all{&*counts, 0, 27};
  const vol::DeviceCount n_gone{&*counts, 4, 8};
  for (int frame = 0; frame < 3; ++frame) {
    const vkc::DeviceCounters before = gpu.device.counters();
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    vkc::Result<vol::RecordedRounds> r =
        map->record_remove(batch, *gone_list, n_gone);
    vkc::Result<vol::RecordedRounds> a =
        map->record_allocate(batch, *all_list, n_all);
    CHECK(r.ok() && a.ok() && batch.submit().ok());
    const vkc::DeviceCounters after = gpu.device.counters();
    CHECK(map->settle(*r).ok() && map->settle(*a).value().total == 0);
    CHECK(coords_of(*map).value() == sorted(all));
    if (frame == 0) continue;  // the first makes the result slots
    CHECK(after.submits - before.submits == 1);
    CHECK(after.staging_buffers - before.staging_buffers == 0);
    CHECK(after.dispatches - before.dispatches == 2 * (2 * kRounds + 1));
  }
  return 0;
}

// kResultSlots recordings may hold their outcome at once; the next is
// refused, with nothing recorded, until one is dropped.
int slots_case(vr_test::Gpu& gpu) {
  vkc::Result<vol::VoxelHashMap> map = make_map(gpu, params(2, 1024));
  CHECK(map.ok());
  const Blocks all = cube();
  vkc::Result<vkc::Buffer> list = vr_test::detail::device_coords(
      gpu, all.data(), std::uint32_t(all.size()));
  vkc::Result<vkc::Buffer> counts = words(gpu, {27});
  CHECK(list.ok() && counts.ok());
  const vol::DeviceCount n{&*counts, 0, 27};
  std::vector<vol::RecordedRounds> held;
  for (std::uint32_t i = 0; i < vol::VoxelHashMap::kResultSlots; ++i) {
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    vkc::Result<vol::RecordedRounds> r = map->record_allocate(batch, *list, n);
    CHECK(r.ok() && batch.submit().ok());
    held.push_back(std::move(r).value());
  }
  {
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    CHECK(map->record_allocate(batch, *list, n).status().domain() ==
          C::InvalidArgument);
    CHECK(batch.submit().ok());
  }
  for (const vol::RecordedRounds& r : held) CHECK(map->settle(r).ok());
  held.pop_back();
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  vkc::Result<vol::RecordedRounds> again =
      map->record_allocate(batch, *list, n);
  CHECK(again.ok() && batch.submit().ok() && map->settle(*again).ok());
  CHECK(coords_of(*map).value() == sorted(all));
  return 0;
}

int gpu_main(vr_test::GpuContext& gpu) {
  if (agrees_case(gpu) != 0) return 1;
  if (count_case(gpu) != 0) return 1;
  if (gate_case(gpu) != 0) return 1;
  if (capacity_case(gpu) != 0) return 1;
  if (contention_case(gpu) != 0) return 1;
  if (one_batch_case(gpu) != 0) return 1;
  if (slices_case(gpu) != 0) return 1;
  if (stale_case(gpu) != 0) return 1;
  if (counters_case(gpu) != 0) return 1;
  if (slots_case(gpu) != 0) return 1;
  std::printf("recon volume recorded test passed\n");
  return 0;
}

}  // namespace

int main() { return vr_test::run_on_gpu(gpu_main); }
