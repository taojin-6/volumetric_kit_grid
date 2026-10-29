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
// is freed. A set's depth round (record_allocate_from_depth) places what the
// blocking call places, closes its gate on a capacity limit, and is that
// call's one submit a round; recordings settle in any order, a blocking call
// overtaking them. A compaction records through a const map (record_compaction
// is const): the whole set and a frustum's subset, each the blocking
// compaction's list, in the caller's one submit with no staging buffer; a held
// list of its kind goes stale and the other kind's holds, and no block,
// pointer, stamp, heap count, epoch or tick moves. A grid grown while a
// reader's batch is pending refuses that batch at its wait, its count
// unwritten, with nothing in use freed. Skips with no device.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "volumetric_kit/core/testing/buffer_readback.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/grid/base/dispatch.hpp"
#include "volumetric_kit/grid/base/math/vector_types.hpp"
#include "volumetric_kit/grid/testing/device_blocks.hpp"
#include "volumetric_kit/grid/volume/frustum.hpp"
#include "volumetric_kit/grid/volume/hash_types.hpp"
#include "volumetric_kit/grid/volume/voxel_block_grid.hpp"
#include "volumetric_kit/grid/volume/voxel_grid.hpp"
#include "volumetric_kit/grid/volume/voxel_hash_map.hpp"

#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/grid/base/device_requirements.hpp"
#include "volumetric_kit/grid/testing/sphere_scene.hpp"

namespace vgrid = volumetric_kit::grid;
namespace vkc = volumetric_kit::core;
namespace vol = volumetric_kit::grid::volume;
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
  b.coord = vgrid::Vec3i(x, y, z);
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

// Blocks with their pointers, sorted: what two lists of one map's blocks
// share, whatever order each was compacted in.
using Placed = std::vector<std::tuple<int, int, int, std::int32_t>>;

Placed with_ptrs(const Blocks& blocks) {
  Placed out;
  for (const vol::BlockIndex& b : blocks) {
    out.emplace_back(b.coord.x, b.coord.y, b.coord.z, b.ptr);
  }
  std::sort(out.begin(), out.end());
  return out;
}

// Everything a gated-off recording must leave as it was, bit for bit: the
// active blocks with their pointers, in the compaction's order, the heap's
// host copy, and every slot's stamp.
struct State {
  Placed blocks;
  std::uint32_t allocated = 0;
  std::vector<std::uint32_t> stamps;

  bool operator==(const State& o) const {
    return blocks == o.blocks && allocated == o.allocated && stamps == o.stamps;
  }
};

vkc::Result<State> state_of(vol::VoxelHashMap& map) {
  State s;
  VKC_ASSIGN(const Blocks active, map.compact_active_blocks());
  s.blocks = with_ptrs(active);
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
vkc::Result<vkc::Buffer> words(vkc::test::Gpu& gpu,
                               const std::vector<std::uint32_t>& values) {
  return vkc::test::upload_device_buffer(gpu.device, gpu.allocator,
                                         values.data(),
                                         values.size() * sizeof(std::uint32_t));
}

vkc::Result<std::uint32_t> word_at(vkc::test::Gpu& gpu, const vkc::Buffer& b,
                                   std::size_t index) {
  VKC_ASSIGN(const std::vector<std::uint32_t> w,
             vkc::test::read_back<std::uint32_t>(gpu.device, gpu.allocator, b,
                                                 1, index));
  return w[0];
}

// One recording of @p blocks in a batch of its own, the count a device word,
// settled: what a caller that records a frame's removal or allocation does.
vkc::Result<vol::AllocFailures> recorded(
    vkc::test::Gpu& gpu, vol::VoxelHashMap& map, const Blocks& blocks,
    bool removes, const vol::DeviceCount* gate = nullptr) {
  VKC_ASSIGN(const vkc::Buffer list,
             vgrid_test::detail::device_coords(gpu, blocks.data(),
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
vkc::Result<vol::AllocFailures> recorded_all(vkc::test::Gpu& gpu,
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
    vkc::test::Gpu& gpu, vol::VoxelHashMap& map, const Blocks& blocks,
    bool removes, vol::AllocFailures* failures = nullptr) {
  std::uint32_t left = 0;
  for (int pass = 0; pass < 8; ++pass) {
    VKC_ASSIGN(left,
               removes
                   ? vgrid_test::remove_blocks(gpu, map, blocks, failures)
                   : vgrid_test::allocate_blocks(gpu, map, blocks, failures));
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

vkc::Result<vol::VoxelHashMap> make_map(vkc::test::Gpu& gpu,
                                        const vol::VoxelGridParams& g) {
  return vol::VoxelHashMap::create(gpu.device, gpu.allocator, g);
}

// --- Recorded and blocking agree. --------------------------------------------

// Two maps, one driven by the blocking calls and one by recordings, through
// the cube's allocation, the corners' delete (chains of 2-slot buckets) and
// a 1 024-block sheet in a wide table: the same blocks, heap count, slot
// stamps and failures.
int agrees_case(vkc::test::Gpu& gpu) {
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
int count_case(vkc::test::Gpu& gpu) {
  const Blocks all = cube();
  vkc::Result<vkc::Buffer> list = vgrid_test::detail::device_coords(
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

// Count and gate words must be complete storage words with known local
// memory. Refusal preserves the map and its lists, runs no first-round
// callback, and leaves the same batch available for a valid retry.
int word_metadata_case(vkc::test::Gpu& gpu) {
  const Blocks one = {at(0, 0, 0)};
  vkc::Result<vkc::Buffer> list =
      vgrid_test::detail::device_coords(gpu, one.data(), 1);
  vkc::Result<vkc::Buffer> counts = words(gpu, {0, 1});
  vkc::Result<vkc::Buffer> short_word =
      vkc::device_storage_buffer(gpu.allocator, 3);
  vkc::Result<vkc::Buffer> short_tail =
      vkc::device_storage_buffer(gpu.allocator, 7);
  vkc::BufferDesc desc;
  desc.size = sizeof(std::uint32_t);
  desc.usage =
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  vkc::Result<vkc::Buffer> transfer = gpu.allocator.create_buffer(desc);
  CHECK(list.ok() && counts.ok() && short_word.ok() && short_tail.ok() &&
        transfer.ok());
  const vkc::Buffer unknown(counts->handle(), counts->size(), counts->usage(),
                            counts->sharing_mode(), nullptr, {}, std::nullopt);
  CHECK(counts->memory_info().has_value());
  vkc::MemoryInfo nonlocal_memory = *counts->memory_info();
  nonlocal_memory.properties &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  const vkc::Buffer nonlocal(counts->handle(), counts->size(), counts->usage(),
                             counts->sharing_mode(), nullptr, {},
                             nonlocal_memory);
  const vkc::Buffer oversized(
      counts->handle(), vkc::max_storage_buffer_range(gpu.device) + 4,
      counts->usage(), counts->sharing_mode(), nullptr, [] {},
      counts->memory_info());
  const vol::DeviceCount good{&*counts, 4, 1};
  const vol::DeviceCount bad_words[] = {
      {&*transfer, 0, 1},
      {&*short_word, 0, 1},
      {&*short_tail, 4, 1},
      {&unknown, 0, 1},
      {&nonlocal, 0, 1},
      {&oversized, 0, 1},
      {&*counts, std::numeric_limits<VkDeviceSize>::max() - 3, 1}};
  for (const bool removes : {false, true}) {
    vkc::Result<vol::VoxelHashMap> map = make_map(gpu, params(2, 1024));
    CHECK(map.ok());
    if (removes) CHECK(blocking_all(gpu, *map, one, false).value() == 0);
    vkc::Result<State> before = state_of(*map);
    vkc::Result<vol::DeviceBlockList> held =
        map->compact_active_blocks_on_device();
    CHECK(before.ok() && held.ok());
    const std::uint64_t epoch = map->topology_epoch();
    bool prepared = false;
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    const auto record = [&](const vol::DeviceCount& count,
                            const vol::DeviceCount* gate) {
      return removes ? map->record_remove(batch, *list, count, gate, nullptr,
                                          [&](vkc::CommandBatch&) {
                                            prepared = true;
                                            return vkc::Status{};
                                          })
                     : map->record_allocate(batch, *list, count, gate);
    };
    for (const vol::DeviceCount& bad : bad_words) {
      CHECK(record(bad, &good).status().domain() == C::InvalidArgument);
      CHECK(record(good, &bad).status().domain() == C::InvalidArgument);
      CHECK(!prepared && map->topology_epoch() == epoch);
      CHECK(map->check_device_block_list(*held, "test").ok());
    }
    CHECK(state_of(*map).value() == *before);

    // A rejected call has not consumed a result slot or poisoned the batch.
    vkc::Result<vol::RecordedRounds> retry = record(good, &good);
    CHECK(retry.ok() && batch.submit().ok());
    vkc::Result<vol::AllocFailures> result = map->settle(*retry);
    CHECK(result.ok() && result->total == 0 && result->terminal == 0);
    CHECK(prepared == removes);
    CHECK(coords_of(*map).value() ==
          (removes ? std::vector<Coord>{} : sorted(one)));
  }
  return 0;
}

// A closed gate runs no round of either kind: the map, its heap and its
// stamps are bit-identical, the settle is zeros and the gate stays 0. An
// open gate that meets no failure stays open.
int gate_case(vkc::test::Gpu& gpu) {
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
int capacity_case(vkc::test::Gpu& gpu) {
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
int contention_case(vkc::test::Gpu& gpu) {
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
  vkc::Result<vkc::Buffer> list = vgrid_test::detail::device_coords(
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
      vgrid_test::allocate_blocks(gpu, *few, placed, &g);
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
int one_batch_case(vkc::test::Gpu& gpu) {
  const vol::AttributeSpec attrs[] = {{vol::kTsdfAttribute, 4},
                                      {vol::kWeightAttribute, 4}};
  vkc::Result<vol::VoxelBlockGrid> grid = vol::VoxelBlockGrid::create(
      gpu.device, gpu.allocator, params(2, 1024), attrs, 2);
  CHECK(grid.ok());
  const Blocks all = cube();
  CHECK(blocking_all(gpu, grid->map(), all, false).value() == 0);
  vkc::Result<vol::AttributeView> tsdf = grid->attribute(vol::kTsdfAttribute);
  CHECK(tsdf.ok());
  {
    vkc::CommandBatch fill(gpu.device, gpu.allocator);
    CHECK(fill.fill(*tsdf->buffer, 0, tsdf->buffer->size(), 0x3F800000u).ok());
    CHECK(fill.submit().ok());
  }
  const Blocks gone = corners();
  Blocks fresh;
  for (int i = 0; i < 8; ++i) fresh.push_back(at(10 + i, 0, 0));
  vkc::Result<vkc::Buffer> gone_list = vgrid_test::detail::device_coords(
      gpu, gone.data(), std::uint32_t(gone.size()));
  vkc::Result<vkc::Buffer> fresh_list = vgrid_test::detail::device_coords(
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
    vkc::Result<std::vector<float>> voxels = vkc::test::read_back<float>(
        gpu.device, gpu.allocator, *tsdf->buffer, 512, std::size_t(b.ptr));
    CHECK(voxels.ok());
    const float want_value = b.coord.x >= 10 ? 0.0f : 1.0f;
    for (const float v : *voxels) CHECK(v == want_value);
  }
  return 0;
}

// Past a dispatch's groups: a round of 1 024 coords split into four slices
// of one group each places what one dispatch places.
int slices_case(vkc::test::Gpu& gpu) {
  vgrid::detail::set_test_groups_per_dispatch(1);
  const Blocks all = sheet(32, 32);
  vkc::Result<vol::VoxelHashMap> map = make_map(gpu, params(8, 4096));
  CHECK(map.ok());
  vkc::Result<vol::AllocFailures> f = recorded_all(gpu, *map, all, false);
  CHECK(f.ok() && f->total == 0);
  CHECK(coords_of(*map).value() == sorted(all));
  vkc::Result<vol::AllocFailures> g = recorded_all(gpu, *map, all, true);
  CHECK(g.ok() && g->total == 0 && map->allocated_blocks() == 0);
  vgrid::detail::set_test_groups_per_dispatch(0);
  return 0;
}

// A recording of either kind makes every list compacted before it stale,
// even one that changes nothing.
int stale_case(vkc::test::Gpu& gpu) {
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

// --- A reader holding the map read-only. -------------------------------------

static_assert(
    std::is_same_v<
        decltype(std::declval<const vol::VoxelHashMap&>().record_compaction(
            std::declval<vkc::CommandBatch&>(), nullptr)),
        vkc::Result<vol::RecordedBlockList>>,
    "a const map records a compaction");

// A camera 1 m behind the cube, looking down +Z, its far plane 0.99 m ahead:
// it keeps the cube's z = -1 slab, which ends 1 m ahead, alone.
vol::FrustumPlanes front_frustum() {
  vgrid::Mat4f pose(1.0f);
  pose[3] = vgrid::Vec4f(0.0f, 0.0f, -1.0f, 1.0f);
  return vol::make_frustum_planes(100.0f, 100.0f, 50.0f, 50.0f, 100, 100, 0.1f,
                                  0.99f, pose);
}

// Through a const map, as a reader handed the map or its grid read-only
// holds it: the whole active set, then the blocks one frustum keeps, each
// recorded into the reader's own batch -- one submit, one dispatch, no
// staging buffer -- and each the blocking compaction's list, pointers
// included, in that list's buffer. A held list of the kind recorded goes
// stale and the other kind's still holds; the blocks, pointers, stamps, heap
// count, epoch and tick are as they were.
int const_case(vkc::test::Gpu& gpu) {
  vkc::Result<vol::VoxelHashMap> map = make_map(gpu, params(2, 1024));
  CHECK(map.ok());
  const Blocks all = cube();
  CHECK(blocking_all(gpu, *map, all, false).value() == 0);
  vkc::Result<State> before = state_of(*map);
  CHECK(before.ok() && before->blocks.size() == all.size());
  const std::uint64_t epoch = map->topology_epoch();
  const std::uint32_t tick = map->tick();
  const std::vector<vol::FrustumPlanes> frustum = {front_frustum()};
  Blocks front;
  for (const vol::BlockIndex& b : all) {
    if (b.coord.z == -1) front.push_back(b);
  }
  const vol::VoxelHashMap& reader = *map;
  for (const bool culled : {false, true}) {
    // A list of each kind from the blocking compactions, the frustum's read
    // back: what the reader's must list.
    vkc::Result<vol::DeviceBlockList> whole =
        map->compact_active_blocks_on_device();
    vkc::Result<vol::DeviceBlockList> kept =
        map->compact_active_blocks_in_frusta_on_device(frustum);
    CHECK(whole.ok() && kept.ok() && whole->count == all.size());
    vkc::Result<Blocks> kept_blocks = vkc::test::read_back<vol::BlockIndex>(
        gpu.device, gpu.allocator, *kept->buffer, kept->count);
    CHECK(kept_blocks.ok() && sorted(*kept_blocks) == sorted(front));

    const vkc::DeviceCounters was = gpu.device.counters();
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    vkc::Result<vol::RecordedBlockList> list =
        reader.record_compaction(batch, culled ? &frustum : nullptr);
    CHECK(list.ok() && batch.submit().ok());
    const vkc::DeviceCounters now = gpu.device.counters();
    CHECK(now.submits - was.submits == 1);
    CHECK(now.dispatches - was.dispatches == 1);
    CHECK(now.staging_buffers == was.staging_buffers);

    const vol::DeviceBlockList& mine = culled ? *kept : *whole;
    const vol::DeviceBlockList& other = culled ? *whole : *kept;
    CHECK(list->buffer == mine.buffer && list->count != nullptr);
    vkc::Result<std::uint32_t> n = word_at(gpu, *list->count, 0);
    CHECK(n.ok() && n.value() == mine.count);
    CHECK(culled || n.value() == reader.allocated_blocks());
    vkc::Result<Blocks> got = vkc::test::read_back<vol::BlockIndex>(
        gpu.device, gpu.allocator, *list->buffer, n.value());
    CHECK(got.ok() && with_ptrs(*got) ==
                          (culled ? with_ptrs(*kept_blocks) : before->blocks));

    CHECK(!reader.check_device_block_subset(mine, "test").ok());
    CHECK(reader.check_device_block_subset(other, "test").ok());
    CHECK(reader.held_device_list().has_value() == culled);
    CHECK(reader.topology_epoch() == epoch && reader.tick() == tick);
  }
  CHECK(state_of(*map).value() == *before);
  CHECK(map->topology_epoch() == epoch && map->tick() == tick);
  return 0;
}

// record_compaction's precondition broken by the grid's owner: a reader's
// compaction of each kind, recorded through a const grid and submitted, not
// yet waited on, while the owner grows the grid. The resize waits for the
// map's queue before it replaces a buffer, so the validation layer sees no
// buffer or set in use freed or rewritten; the core refuses the reader's
// batch at its wait, its count unwritten; and the reader's next batch,
// recorded after the resize, counts the blocks again.
int resize_pending_case(vkc::test::Gpu& gpu) {
  const vol::AttributeSpec attrs[] = {{vol::kTsdfAttribute, 4},
                                      {vol::kWeightAttribute, 4}};
  vkc::Result<vol::VoxelBlockGrid> grid = vol::VoxelBlockGrid::create(
      gpu.device, gpu.allocator, params(2, 1024), attrs, 2);
  CHECK(grid.ok());
  CHECK(blocking_all(gpu, grid->map(), cube(), false).value() == 0);
  const std::vector<vol::FrustumPlanes> frustum = {front_frustum()};
  const vol::VoxelBlockGrid& view = *grid;
  constexpr std::uint32_t kUnwritten = 0xffffffffu;
  for (const bool culled : {false, true}) {
    const std::vector<vol::FrustumPlanes>* frusta = culled ? &frustum : nullptr;
    std::uint32_t count = kUnwritten;
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    vkc::Result<vol::RecordedBlockList> list =
        view.map().record_compaction(batch, frusta);
    CHECK(list.ok());
    CHECK(batch.readback(*list->count, 0, sizeof(count), &count).ok());
    vkc::Result<vkc::PendingBatch> pending = batch.submit_async({}, {});
    CHECK(pending.ok());
    CHECK(grid->resize(2 * view.map().grid().num_buckets).ok());
    const vkc::Status refused = pending->wait();
    CHECK(refused.domain() == C::InvalidArgument && count == kUnwritten);

    vkc::CommandBatch again(gpu.device, gpu.allocator);
    vkc::Result<vol::RecordedBlockList> next =
        view.map().record_compaction(again, frusta);
    CHECK(next.ok() && again.submit().ok());
    vkc::Result<std::uint32_t> n = word_at(gpu, *next->count, 0);
    CHECK(n.ok() && n.value() == (culled ? 9u : 27u));
  }
  return 0;
}

// --- Steady state and the result slots. --------------------------------------

// Once warm, a batch holding a removal and an allocation costs one submit,
// no staging buffer, and the rounds' dispatches: K rounds and K + 1 one-thread
// kernels a kind.
int counters_case(vkc::test::Gpu& gpu) {
  vkc::Result<vol::VoxelHashMap> map = make_map(gpu, params(2, 1024));
  CHECK(map.ok());
  const Blocks all = cube();
  const Blocks gone = corners();
  vkc::Result<vkc::Buffer> all_list = vgrid_test::detail::device_coords(
      gpu, all.data(), std::uint32_t(all.size()));
  vkc::Result<vkc::Buffer> gone_list = vgrid_test::detail::device_coords(
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
int slots_case(vkc::test::Gpu& gpu) {
  vkc::Result<vol::VoxelHashMap> map = make_map(gpu, params(2, 1024));
  CHECK(map.ok());
  const Blocks all = cube();
  vkc::Result<vkc::Buffer> list = vgrid_test::detail::device_coords(
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

// --- A set's depth frames. ---------------------------------------------------

// Three cameras round a sphere, their depth on the host and on the device.
struct Sphere {
  std::vector<vgrid_test::SphereView<vol::DepthPinhole>> views;
  std::vector<vkc::Buffer> device;

  std::vector<vol::DepthInput> frames(std::size_t cameras,
                                      bool on_device) const {
    std::vector<vol::DepthInput> out;
    for (std::size_t c = 0; c < cameras; ++c) {
      out.push_back({on_device ? vkc::StorageInput(device[c])
                               : vkc::StorageInput(views[c].depth.data()),
                     views[c].cam});
    }
    return out;
  }
};

vkc::Result<Sphere> sphere(vkc::test::Gpu& gpu) {
  Sphere s;
  for (int c = 0; c < 3; ++c) {
    s.views.push_back(vgrid_test::sphere_view<vol::DepthPinhole>(c));
    const std::vector<float>& depth = s.views.back().depth;
    VKC_ASSIGN(vkc::Buffer buffer, vkc::test::upload_device_buffer(
                                       gpu.device, gpu.allocator, depth.data(),
                                       depth.size() * sizeof(float)));
    s.device.push_back(std::move(buffer));
  }
  return s;
}

// One round of a set's bands recorded, as the blocking call rounds them: the
// same blocks, heap count and stamps, on one camera and on three, from host
// arrays and from device buffers; each recording is one round.
int depth_case(vkc::test::Gpu& gpu) {
  vkc::Result<Sphere> s = sphere(gpu);
  CHECK(s.ok());
  for (const bool on_device : {false, true}) {
    for (const std::size_t cameras : {std::size_t{1}, std::size_t{3}}) {
      const std::vector<vol::DepthInput> frames = s->frames(cameras, on_device);
      vkc::Result<vol::VoxelHashMap> a = make_map(gpu, params(8, 4096));
      vkc::Result<vol::VoxelHashMap> b = make_map(gpu, params(8, 4096));
      CHECK(a.ok() && b.ok());
      vol::AllocFailures fa;
      CHECK(a->allocate_from_depth(frames, &fa).value() == 0);
      vol::AllocFailures fb;
      for (int pass = 0; pass < 8; ++pass) {
        vkc::CommandBatch batch(gpu.device, gpu.allocator);
        vkc::Result<vol::RecordedRounds> r =
            b->record_allocate_from_depth(batch, frames);
        CHECK(r.ok() && batch.submit().ok());
        fb = b->settle(*r).value();
        CHECK(fb.rounds == 1);
        if (fb.lock == 0) break;
      }
      CHECK(fb.total == 0 && fb.terminal == 0 && same_failures(fa, fb));
      CHECK(coords_of(*a).value() == coords_of(*b).value());
      CHECK(a->allocated_blocks() == b->allocated_blocks());
      CHECK(a->allocated_blocks() > 100);
      CHECK(stamps_of(*a).value() == stamps_of(*b).value());
    }
  }

  // A frame with no pixels records nothing, and neither does a set of them.
  vkc::Result<vol::VoxelHashMap> map = make_map(gpu, params(8, 4096));
  CHECK(map.ok());
  std::vector<vol::DepthInput> none = s->frames(1, true);
  none[0].camera.width = 0;
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  vkc::Result<vol::RecordedRounds> nothing =
      map->record_allocate_from_depth(batch, none);
  CHECK(nothing.ok() && !nothing->result.valid());
  CHECK(batch.submit().ok() && map->settle(*nothing).value().rounds == 0);
  CHECK(map->allocated_blocks() == 0);
  return 0;
}

// A gate the round meets a capacity limit on is closed for what is gated on
// it, the rounds' gate words as record_allocate's; one it completes on stays
// open. The gate is the round's output alone: a closed one still has the
// round run and judged. A gate word past its buffer is refused with nothing
// recorded.
int depth_gate_case(vkc::test::Gpu& gpu) {
  vkc::Result<Sphere> s = sphere(gpu);
  vkc::Result<vkc::Buffer> gates = words(gpu, {1, 1, 0});
  CHECK(s.ok() && gates.ok());
  const std::vector<vol::DepthInput> frames = s->frames(3, true);
  for (const bool room : {false, true}) {
    vkc::Result<vol::VoxelHashMap> map =
        make_map(gpu, room ? params(8, 4096) : params(8, 1));
    CHECK(map.ok());
    const vol::DeviceCount gate{&*gates, room ? 4u : 0u, 1};
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    vkc::Result<vol::RecordedRounds> r =
        map->record_allocate_from_depth(batch, frames, &gate);
    CHECK(r.ok() && batch.submit().ok());
    vkc::Result<vol::AllocFailures> f = map->settle(*r);
    CHECK(f.ok() && f->rounds == 1);
    if (room) {
      CHECK(word_at(gpu, *gates, 1).value() == (f->total == 0 ? 1u : 0u));
    } else {
      CHECK(f->capacity_limited() && map->allocated_blocks() <= 8);
      CHECK(word_at(gpu, *gates, 0).value() == 0);
    }
  }
  vkc::Result<vol::VoxelHashMap> map = make_map(gpu, params(8, 4096));
  CHECK(map.ok());
  {
    const vol::DeviceCount closed{&*gates, 8, 1};
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    vkc::Result<vol::RecordedRounds> r =
        map->record_allocate_from_depth(batch, frames, &closed);
    CHECK(r.ok() && batch.submit().ok());
    vkc::Result<vol::AllocFailures> f = map->settle(*r);
    CHECK(f.ok() && f->rounds == 1 && f->terminal == 0);
    CHECK(map->allocated_blocks() > 100 &&
          word_at(gpu, *gates, 2).value() == 0);
  }
  const vol::DeviceCount past{&*gates, 12, 1};
  vkc::CommandBatch refused(gpu.device, gpu.allocator);
  CHECK(map->record_allocate_from_depth(refused, frames, &past)
            .status()
            .domain() == C::InvalidArgument);
  CHECK(refused.submit().ok());
  return 0;
}

// Recordings settle in any order, and a blocking call since overtakes one:
// the heap's host copy is the latest's. A set fused again, blocking, before
// its round settles, or a free pass, cannot be undone by that settle.
int settle_order_case(vkc::test::Gpu& gpu) {
  vkc::Result<vol::VoxelHashMap> map = make_map(gpu, params(2, 1024));
  CHECK(map.ok());
  const Blocks first = cube();
  const Blocks second = sheet(4, 4, 5);
  const Blocks third = sheet(4, 4, 9);
  vkc::Result<vkc::Buffer> first_list = vgrid_test::detail::device_coords(
      gpu, first.data(), std::uint32_t(first.size()));
  vkc::Result<vkc::Buffer> second_list = vgrid_test::detail::device_coords(
      gpu, second.data(), std::uint32_t(second.size()));
  vkc::Result<vkc::Buffer> third_list = vgrid_test::detail::device_coords(
      gpu, third.data(), std::uint32_t(third.size()));
  vkc::Result<vkc::Buffer> counts = words(gpu, {27, 16});
  CHECK(first_list.ok() && second_list.ok() && third_list.ok() && counts.ok());
  const auto record = [&](const vkc::Buffer& list, VkDeviceSize offset,
                          std::uint32_t n) -> vkc::Result<vol::RecordedRounds> {
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    VKC_ASSIGN(vol::RecordedRounds r,
               map->record_allocate(batch, list,
                                    vol::DeviceCount{&*counts, offset, n}));
    VKC_TRY(batch.submit());
    return r;
  };
  vkc::Result<vol::RecordedRounds> a = record(*first_list, 0, 27);
  vkc::Result<vol::RecordedRounds> b = record(*second_list, 4, 16);
  CHECK(a.ok() && b.ok());
  CHECK(map->settle(*b).value().total == 0);
  CHECK(map->allocated_blocks() == 27 + 16);
  CHECK(map->settle(*a).value().total == 0);
  CHECK(map->allocated_blocks() == 27 + 16);

  vkc::Result<vol::RecordedRounds> c = record(*third_list, 4, 16);
  CHECK(c.ok());
  CHECK(blocking_all(gpu, *map, corners(), true).value() == 0);
  CHECK(map->allocated_blocks() == 27 + 16 + 16 - 8);
  CHECK(map->settle(*c).value().total == 0);
  CHECK(map->allocated_blocks() == 27 + 16 + 16 - 8);
  CHECK(coords_of(*map).value().size() == map->allocated_blocks());
  return 0;
}

// The blocking call over a set is one submit a round and, warm, no staging
// buffer: each round its recording, the count word staying on the device.
int depth_counters_case(vkc::test::Gpu& gpu) {
  vkc::Result<Sphere> s = sphere(gpu);
  vkc::Result<vol::VoxelHashMap> map = make_map(gpu, params(8, 4096));
  CHECK(s.ok() && map.ok());
  const std::vector<vol::DepthInput> frames = s->frames(3, true);
  for (int pass = 0; pass < 3; ++pass) {
    const vkc::DeviceCounters was = gpu.device.counters();
    vol::AllocFailures f;
    vkc::Result<std::uint32_t> left = map->allocate_from_depth(frames, &f);
    const vkc::DeviceCounters now = gpu.device.counters();
    CHECK(left.ok() && left.value() == 0 && f.rounds >= 1);
    if (pass == 0) continue;  // the first makes the result slots
    CHECK(now.submits - was.submits == f.rounds);
    // A wait that finds its round done blocks nothing and counts none.
    CHECK(now.host_waits - was.host_waits <= f.rounds);
    CHECK(now.staging_buffers == was.staging_buffers);
  }
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  if (agrees_case(gpu) != 0) return 1;
  if (count_case(gpu) != 0) return 1;
  if (word_metadata_case(gpu) != 0) return 1;
  if (gate_case(gpu) != 0) return 1;
  if (capacity_case(gpu) != 0) return 1;
  if (contention_case(gpu) != 0) return 1;
  if (one_batch_case(gpu) != 0) return 1;
  if (slices_case(gpu) != 0) return 1;
  if (stale_case(gpu) != 0) return 1;
  if (const_case(gpu) != 0) return 1;
  if (resize_pending_case(gpu) != 0) return 1;
  if (counters_case(gpu) != 0) return 1;
  if (slots_case(gpu) != 0) return 1;
  if (depth_case(gpu) != 0) return 1;
  if (depth_gate_case(gpu) != 0) return 1;
  if (settle_order_case(gpu) != 0) return 1;
  if (depth_counters_case(gpu) != 0) return 1;
  std::printf("grid volume recorded test passed\n");
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(vgrid::device_requirements(), gpu_main);
}
