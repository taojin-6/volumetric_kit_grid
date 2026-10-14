// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for the codec's block lists (block_lists.hpp): the sort against
// std::stable_sort by coord_less, over entries whose ptrs are all distinct,
// which pins stability -- at counts on and off the tile and level boundaries,
// over all of int32, clustered, all equal, a few keys duplicated across
// tiles, sorted and reversed -- with its input untouched, and its refusals;
// the merge against the oracle's map (codec_frames::merge_oracle), every
// mode mix and empty side, int32's extremes and the worked example, and row
// 13's first collision against a walk in frame order; the grid diff against
// std::set_difference both ways, in overflow chains and across a resize; and
// garbage -- unsorted lists, a block count that disagrees -- writing nothing
// past an output's end. Internal headers through the oracle library. Skips
// without a device.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "bitstream.hpp"
#include "block_lists.hpp"
#include "buffer_readback.hpp"
#include "codec_fixture.hpp"
#include "codec_frames.hpp"
#include "test_check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"

#include "gpu_test.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace vol = volumetric_kit::recon::volume;
namespace d = volumetric_kit::recon::codec::detail;
using codec_frames::kMax32;
using codec_frames::kMin32;
using codec_frames::Lcg;
using codec_frames::ZyxLess;
using C = vkc::Status::Code;
using Mode = d::BlockMode;

namespace {

using Blocks = std::vector<vol::BlockIndex>;

bool same(const Blocks& a, const Blocks& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!(a[i].coord == b[i].coord) || a[i].ptr != b[i].ptr) return false;
  }
  return true;
}

// @p coords as entries, each ptr its index: distinct, so a reordered tie
// shows.
Blocks entries(const std::vector<vr::Vec3i>& coords) {
  Blocks out(coords.size());
  for (std::size_t i = 0; i < coords.size(); ++i) {
    out[i] = vol::BlockIndex{coords[i], std::int32_t(i)};
  }
  return out;
}

// --- The sort. --------------------------------------------------------------

// The device's sort of @p in, read back, with @p in read back too.
vkc::Result<Blocks> device_sort(vr_test::Gpu& gpu, d::BlockLists& lists,
                                const Blocks& in, Blocks* input_after) {
  const auto n = static_cast<std::uint32_t>(in.size());
  vkc::Buffer list;
  if (n != 0) {
    VKC_ASSIGN(list, vr_test::upload_device_buffer(
                         gpu.device, gpu.allocator, in.data(),
                         in.size() * sizeof(vol::BlockIndex)));
  }
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  VKC_ASSIGN(const vkc::Buffer* sorted, lists.record_sort(batch, list, n));
  VKC_TRY(batch.submit());
  if (n == 0) return Blocks{};
  if (input_after != nullptr) {
    VKC_ASSIGN(*input_after, vr_test::read_back<vol::BlockIndex>(
                                 gpu.device, gpu.allocator, list, n));
  }
  return vr_test::read_back<vol::BlockIndex>(gpu.device, gpu.allocator, *sorted,
                                             n);
}

// @p coords sorted on the device as std::stable_sort sorts them, the input
// left as it was.
int sorts_as_std(vr_test::Gpu& gpu, d::BlockLists& lists,
                 const std::vector<vr::Vec3i>& coords, const char* what) {
  const Blocks in = entries(coords);
  Blocks expected = in;
  std::stable_sort(expected.begin(), expected.end(),
                   [](const vol::BlockIndex& a, const vol::BlockIndex& b) {
                     return d::coord_less(a.coord, b.coord);
                   });
  Blocks after;
  const vkc::Result<Blocks> got = device_sort(gpu, lists, in, &after);
  if (!got.ok() || !same(got.value(), expected) ||
      (!in.empty() && !same(after, in))) {
    std::fprintf(stderr, "sort, %s, n %zu: %s\n", what, coords.size(),
                 got.ok() ? "differs" : got.status().message().c_str());
    return 1;
  }
  return 0;
}

int sort_case(vr_test::Gpu& gpu, d::BlockLists& lists) {
  Lcg rng{11};
  const std::int32_t picks[] = {kMin32, kMin32 + 1, -1,    0,
                                1,      kMax32 - 1, kMax32};
  const std::vector<vr::Vec3i> extreme = codec_frames::extreme_frame().coords;
  const std::size_t counts[] = {0,   1,   2,    255,   256,
                                257, 511, 4097, 65537, 200000};
  for (const std::size_t n : counts) {
    // All of int32, each axis's extremes and extreme_frame's coordinates
    // among them.
    std::vector<vr::Vec3i> wide(n);
    for (std::size_t i = 0; i < n; ++i) {
      const auto any = [&]() {
        return rng.below(4) == 0 ? picks[rng.below(7)]
                                 : static_cast<std::int32_t>(rng.next());
      };
      wide[i] =
          i < extreme.size() ? extreme[i] : vr::Vec3i(any(), any(), any());
    }
    CHECK(sorts_as_std(gpu, lists, wide, "int32") == 0);
    // A room's worth of space, with many repeats at the larger counts.
    std::vector<vr::Vec3i> room(n);
    for (vr::Vec3i& c : room) {
      c = vr::Vec3i(int(rng.below(120)) - 60, int(rng.below(40)) - 10,
                    int(rng.below(120)) - 60);
    }
    CHECK(sorts_as_std(gpu, lists, room, "room") == 0);
    // All one key, and seven keys, each repeated across tiles.
    CHECK(sorts_as_std(gpu, lists, std::vector<vr::Vec3i>(n, {3, -2, 7}),
                       "equal") == 0);
    std::vector<vr::Vec3i> seven(n);
    for (vr::Vec3i& c : seven) {
      const auto k = static_cast<std::int32_t>(rng.below(7));
      c = vr::Vec3i(k % 2 == 0 ? kMax32 : -k, k, k == 3 ? kMin32 : 0);
    }
    CHECK(sorts_as_std(gpu, lists, seven, "seven keys") == 0);
    // Already sorted, and reversed.
    std::vector<vr::Vec3i> sorted = room;
    std::sort(sorted.begin(), sorted.end(), d::coord_less);
    CHECK(sorts_as_std(gpu, lists, sorted, "sorted") == 0);
    std::reverse(sorted.begin(), sorted.end());
    CHECK(sorts_as_std(gpu, lists, sorted, "reversed") == 0);
  }
  return 0;
}

// A list it cannot read is refused before anything is recorded; none to
// sort records nothing.
int sort_refusals_case(vr_test::Gpu& gpu, d::BlockLists& lists) {
  const Blocks two = entries({{1, 0, 0}, {0, 0, 0}});
  vkc::Result<vkc::Buffer> list = vr_test::upload_device_buffer(
      gpu.device, gpu.allocator, two.data(), 2 * sizeof(vol::BlockIndex));
  CHECK(list.ok());
  vkc::BufferDesc desc;
  desc.size = 2 * sizeof(vol::BlockIndex);
  desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  desc.memory = vkc::MemoryUsage::DeviceOnly;
  vkc::Result<vkc::Buffer> transfer_only = gpu.allocator.create_buffer(desc);
  CHECK(transfer_only.ok());
  const vkc::Buffer empty;
  struct Refused {
    const vkc::Buffer* list;
    std::uint32_t n;
  };
  for (const Refused& r :
       {Refused{&list.value(), 3}, Refused{&transfer_only.value(), 2},
        Refused{&empty, 1}}) {
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    const vkc::Status s = lists.record_sort(batch, *r.list, r.n).status();
    CHECK(s.domain() == C::InvalidArgument &&
          s.message().rfind("BlockLists::record_sort: the list", 0) == 0);
    CHECK(batch.submit().ok());
  }
  vkc::CommandBatch none(gpu.device, gpu.allocator);
  CHECK(lists.record_sort(none, empty, 0).ok());
  CHECK(none.submit().ok());
  // And it sorts after them.
  CHECK(sorts_as_std(gpu, lists, {{1, 0, 0}, {0, 0, 0}}, "after refusals") ==
        0);
  return 0;
}

// --- The merge. -------------------------------------------------------------

// @p v uploaded, one default element when it is empty, so a side a case
// lacks still binds a buffer.
template <typename T>
vkc::Result<vkc::Buffer> upload(vr_test::Gpu& gpu, std::vector<T> v) {
  if (v.empty()) v.resize(1);
  return vr_test::upload_device_buffer(gpu.device, gpu.allocator, v.data(),
                                       v.size() * sizeof(T));
}

Blocks listed(const std::vector<vr::Vec3i>& coords) {
  Blocks out;
  for (const vr::Vec3i& c : coords) out.push_back(vol::BlockIndex{c, 0});
  return out;
}

// What the merge read and wrote: the frame's N blocks and sources, the
// collision, and kGuard entries of each past N, which must stay as filled.
constexpr std::size_t kGuard = 8;
constexpr std::int32_t kGuardPtr = 0x5EED;

struct Merged {
  std::vector<vr::Vec3i> coords;
  std::vector<d::ApplySource> sources;
  d::BlockLists::Verdict verdict;
  bool guards_kept = true;
};

// The survivors before each reference entry, and after the last.
std::vector<std::uint32_t> ranks_of(const std::vector<Mode>& modes) {
  std::vector<std::uint32_t> survives(modes.size() + 1, 0);
  for (std::size_t e = 0; e < modes.size(); ++e) {
    survives[e] = modes[e] == Mode::kRemove ? 0 : 1;
  }
  std::vector<std::uint32_t> ranks(survives.size());
  std::exclusive_scan(survives.begin(), survives.end(), ranks.begin(), 0u);
  return ranks;
}

vkc::Result<Merged> device_merge(vr_test::Gpu& gpu, d::BlockLists& lists,
                                 const std::vector<vr::Vec3i>& reference,
                                 const std::vector<Mode>& modes,
                                 const std::vector<vr::Vec3i>& added,
                                 std::uint32_t n,
                                 const std::vector<std::uint32_t>& ranks) {
  std::vector<std::uint32_t> mode_words;
  for (const Mode m : modes) mode_words.push_back(std::uint32_t(m));
  VKC_ASSIGN(const vkc::Buffer ref_buf, upload(gpu, listed(reference)));
  VKC_ASSIGN(const vkc::Buffer added_buf, upload(gpu, listed(added)));
  VKC_ASSIGN(const vkc::Buffer mode_buf, upload(gpu, mode_words));
  VKC_ASSIGN(const vkc::Buffer rank_buf, upload(gpu, ranks));
  const Blocks frame_fill(n + kGuard, vol::BlockIndex{{7, 7, 7}, kGuardPtr});
  const std::vector<d::ApplySource> source_fill(n + kGuard, {0x5EED, 0xFEED});
  VKC_ASSIGN(const vkc::Buffer frame_buf, upload(gpu, frame_fill));
  VKC_ASSIGN(const vkc::Buffer source_buf, upload(gpu, source_fill));
  d::ResidentBlocks decoded;
  decoded.list = &added_buf;
  decoded.modes = modes.empty() ? nullptr : &mode_buf;
  decoded.ranks = &rank_buf;
  decoded.count = static_cast<std::uint32_t>(reference.size() + added.size());
  decoded.reference_count = static_cast<std::uint32_t>(reference.size());
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  VKC_TRY(lists.record_merge(batch, decoded, ref_buf,
                             d::FrameBuffers{&frame_buf, &source_buf}, n));
  VKC_TRY(batch.submit());
  VKC_ASSIGN(const Blocks frame,
             vr_test::read_back<vol::BlockIndex>(gpu.device, gpu.allocator,
                                                 frame_buf, n + kGuard));
  VKC_ASSIGN(const std::vector<d::ApplySource> sources,
             vr_test::read_back<d::ApplySource>(gpu.device, gpu.allocator,
                                                source_buf, n + kGuard));
  Merged out;
  out.verdict = lists.verdict();
  for (std::size_t i = 0; i < n + kGuard; ++i) {
    if (i < n) {
      out.coords.push_back(frame[i].coord);
      out.sources.push_back(sources[i]);
      continue;
    }
    out.guards_kept = out.guards_kept && frame[i].ptr == kGuardPtr &&
                      frame[i].coord == vr::Vec3i(7, 7, 7) &&
                      sources[i].reference == 0x5EED &&
                      sources[i].payload == 0xFEED;
  }
  return out;
}

bool same_sources(const std::vector<d::ApplySource>& a,
                  const std::vector<d::ApplySource>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].reference != b[i].reference || a[i].payload != b[i].payload) {
      return false;
    }
  }
  return true;
}

// The device's merge of a sound frame is the oracle's, with no collision and
// nothing written past N.
int merges_as_oracle(vr_test::Gpu& gpu, d::BlockLists& lists,
                     const std::vector<vr::Vec3i>& reference,
                     const std::vector<Mode>& modes,
                     const std::vector<vr::Vec3i>& added, const char* what) {
  const codec_frames::Merged want =
      codec_frames::merge_oracle(reference, modes, added);
  const vkc::Result<Merged> got = device_merge(
      gpu, lists, reference, modes, added,
      static_cast<std::uint32_t>(want.coords.size()), ranks_of(modes));
  if (!got.ok() || got.value().coords != want.coords ||
      !same_sources(got.value().sources, want.sources) ||
      got.value().verdict.collision_q != d::kNoSource ||
      !got.value().guards_kept) {
    std::fprintf(stderr, "merge, %s (M %zu, A %zu): %s\n", what,
                 reference.size(), added.size(),
                 got.ok() ? "differs" : got.status().message().c_str());
    return 1;
  }
  return 0;
}

// @p total distinct coordinates, sorted, from a box with int32's extremes
// mixed in, split at random into a reference and new blocks.
void split_coords(Lcg& rng, std::size_t total, std::uint32_t added_per_mille,
                  std::vector<vr::Vec3i>& reference,
                  std::vector<vr::Vec3i>& added) {
  const std::int32_t picks[] = {kMin32, -3, 0, 5, kMax32};
  std::set<vr::Vec3i, ZyxLess> all;
  while (all.size() < total) {
    const auto axis = [&](std::uint32_t span) {
      return rng.below(16) == 0 ? picks[rng.below(5)]
                                : std::int32_t(rng.below(span)) - 40;
    };
    all.insert(vr::Vec3i(axis(80), axis(30), axis(80)));
  }
  reference.clear();
  added.clear();
  for (const vr::Vec3i& c : all) {
    (rng.below(1000) < added_per_mille ? added : reference).push_back(c);
  }
}

std::vector<Mode> modes_of(Lcg& rng, std::size_t m, const Mode* pool,
                           std::size_t pool_size) {
  std::vector<Mode> modes(m);
  for (Mode& mode : modes) mode = pool[rng.below(std::uint32_t(pool_size))];
  return modes;
}

int merge_case(vr_test::Gpu& gpu, d::BlockLists& lists) {
  // The worked example: a <- (0, 0), c <- (none, 2), b <- (1, none).
  {
    const std::vector<vr::Vec3i> reference = {{0, 0, 0}, {4, 0, 0}};
    const std::vector<Mode> modes = {Mode::kKeep, Mode::kSkip};
    const std::vector<vr::Vec3i> added = {{2, 0, 0}};
    vkc::Result<Merged> got =
        device_merge(gpu, lists, reference, modes, added, 3, ranks_of(modes));
    CHECK(got.ok());
    CHECK(got.value().coords ==
          std::vector<vr::Vec3i>({{0, 0, 0}, {2, 0, 0}, {4, 0, 0}}));
    const std::uint32_t no = d::kNoSource;
    CHECK(same_sources(got.value().sources, {{0, 0}, {no, 2}, {1, no}}));
  }
  Lcg rng{41};
  const Mode every[] = {Mode::kKeep, Mode::kSkip, Mode::kZero, Mode::kRemove};
  const Mode survive[] = {Mode::kKeep, Mode::kSkip, Mode::kZero};
  const Mode keep[] = {Mode::kKeep};
  const Mode skip[] = {Mode::kSkip};
  const Mode zero[] = {Mode::kZero};
  const Mode remove[] = {Mode::kRemove};
  struct Mix {
    const char* what;
    const Mode* pool;
    std::size_t size;
  };
  const Mix mixes[] = {{"every mode", every, 4}, {"survivors", survive, 3},
                       {"all keep", keep, 1},    {"all skip", skip, 1},
                       {"all zero", zero, 1},    {"all remove", remove, 1}};
  std::vector<vr::Vec3i> reference;
  std::vector<vr::Vec3i> added;
  for (const Mix& mix : mixes) {
    // Both sides, no new blocks (A = 0), no reference (M = 0), and nothing.
    for (const std::uint32_t added_per_mille : {300u, 0u, 1000u}) {
      for (const std::size_t total : {std::size_t(0), std::size_t(1),
                                      std::size_t(300), std::size_t(2000)}) {
        split_coords(rng, total, added_per_mille, reference, added);
        CHECK(merges_as_oracle(
                  gpu, lists, reference,
                  modes_of(rng, reference.size(), mix.pool, mix.size), added,
                  mix.what) == 0);
      }
    }
  }
  // int32's extremes alone.
  {
    const std::vector<vr::Vec3i> extreme = codec_frames::extreme_frame().coords;
    reference.clear();
    added.clear();
    for (std::size_t i = 0; i < extreme.size(); ++i) {
      (i % 3 == 1 ? added : reference).push_back(extreme[i]);
    }
    CHECK(merges_as_oracle(gpu, lists, reference,
                           modes_of(rng, reference.size(), every, 4), added,
                           "int32 extremes") == 0);
  }
  // A thousand random frames, and one past a dispatch's 65,535 groups' worth
  // of nothing but a large reference.
  for (int trial = 0; trial < 1000; ++trial) {
    split_coords(rng, rng.below(200), rng.below(1001), reference, added);
    CHECK(merges_as_oracle(gpu, lists, reference,
                           modes_of(rng, reference.size(), every, 4), added,
                           "random") == 0);
  }
  split_coords(rng, 70000, 50, reference, added);
  CHECK(merges_as_oracle(gpu, lists, reference,
                         modes_of(rng, reference.size(), every, 4), added,
                         "large") == 0);
  return 0;
}

// The first new block on a reference block in frame order, by a walk: the
// pair row 13 names.
std::pair<std::uint32_t, std::uint32_t> first_collision(
    const std::vector<vr::Vec3i>& reference,
    const std::vector<vr::Vec3i>& added) {
  std::size_t e = 0;
  std::size_t q = 0;
  while (e < reference.size() && q < added.size()) {
    if (d::coord_less(reference[e], added[q])) {
      ++e;
    } else if (d::coord_less(added[q], reference[e])) {
      ++q;
    } else {
      return {std::uint32_t(q), std::uint32_t(e)};
    }
  }
  return {d::kNoSource, d::kNoSource};
}

// Row 13: new blocks on reference blocks -- one removed, one skipped, and
// several at once -- name the first in frame order, whatever their modes.
int collision_case(vr_test::Gpu& gpu, d::BlockLists& lists) {
  Lcg rng{43};
  const Mode every[] = {Mode::kKeep, Mode::kSkip, Mode::kZero, Mode::kRemove};
  for (int trial = 0; trial < 60; ++trial) {
    std::vector<vr::Vec3i> reference;
    std::vector<vr::Vec3i> added;
    split_coords(rng, 50 + rng.below(400), 300, reference, added);
    CHECK(reference.size() > 8);
    std::vector<Mode> modes = modes_of(rng, reference.size(), every, 4);
    // On a removed entry, on a skipped one, then on several.
    const std::size_t on = trial % 3 == 2 ? 1 + rng.below(4) : 1;
    for (std::size_t i = 0; i < on; ++i) {
      const std::size_t e = rng.below(std::uint32_t(reference.size()));
      if (trial % 3 == 0) modes[e] = Mode::kRemove;
      if (trial % 3 == 1) modes[e] = Mode::kSkip;
      const auto at = std::lower_bound(added.begin(), added.end(), reference[e],
                                       d::coord_less);
      if (at == added.end() || !(*at == reference[e])) {
        added.insert(at, reference[e]);
      }
    }
    const auto [q, e] = first_collision(reference, added);
    CHECK(q != d::kNoSource);
    const std::vector<std::uint32_t> ranks = ranks_of(modes);
    const auto n = static_cast<std::uint32_t>(ranks.back() + added.size());
    vkc::Result<Merged> got =
        device_merge(gpu, lists, reference, modes, added, n, ranks);
    CHECK(got.ok() && got.value().guards_kept);
    if (got.value().verdict.collision_q != q ||
        got.value().verdict.collision_e != e) {
      std::fprintf(stderr, "collision, trial %d: (%u, %u), not (%u, %u)\n",
                   trial, got.value().verdict.collision_q,
                   got.value().verdict.collision_e, q, e);
      return 1;
    }
  }
  return 0;
}

// --- The grid diff. ---------------------------------------------------------

// @p coords into @p grid, in the order given, a few at a time and again
// while lock races leave any out.
vkc::Status place(vol::VoxelBlockGrid& grid,
                  const std::vector<vr::Vec3i>& coords) {
  const Blocks blocks = listed(coords);
  for (std::size_t first = 0; first < blocks.size(); first += 16) {
    const auto count = static_cast<std::uint32_t>(
        std::min<std::size_t>(16, blocks.size() - first));
    std::uint32_t failed = 1;
    for (int call = 0; call < 16 && failed != 0; ++call) {
      VKC_ASSIGN(failed, grid.map().allocate(blocks.data() + first, count));
    }
    if (failed != 0) return vkc::Status::io_error("place: lock races");
  }
  return {};
}

std::vector<vr::Vec3i> sorted_coords(const Blocks& blocks) {
  std::vector<vr::Vec3i> out;
  for (const vol::BlockIndex& b : blocks) out.push_back(b.coord);
  std::sort(out.begin(), out.end(), d::coord_less);
  return out;
}

// The device's diff of @p grid against @p frame (sorted): the gone and
// missing lists, read back.
struct Diffed {
  Blocks gone;
  Blocks missing;
};

vkc::Result<Diffed> device_diff(vr_test::Gpu& gpu, d::BlockLists& lists,
                                const vol::VoxelBlockGrid& grid,
                                const std::vector<vr::Vec3i>& frame) {
  const auto n = static_cast<std::uint32_t>(frame.size());
  VKC_ASSIGN(const vkc::Buffer frame_buf, upload(gpu, listed(frame)));
  const vol::VoxelGridParams& gp = grid.grid();
  VKC_TRY(lists.reserve(
      n, static_cast<std::uint32_t>(gp.num_buckets * gp.bucket_size)));
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  VKC_TRY(lists.record_diff(batch, grid, frame_buf, n));
  VKC_TRY(batch.submit());
  const d::BlockLists::Verdict found = lists.verdict();
  Diffed out;
  if (found.gone != 0) {
    VKC_ASSIGN(out.gone,
               vr_test::read_back<vol::BlockIndex>(gpu.device, gpu.allocator,
                                                   lists.gone(), found.gone));
  }
  if (found.missing != 0) {
    VKC_ASSIGN(out.missing,
               vr_test::read_back<vol::BlockIndex>(
                   gpu.device, gpu.allocator, lists.missing(), found.missing));
  }
  return out;
}

// The diff is std::set_difference both ways, in any order, each gone block
// with the grid's ptr and each missing one with 0.
int diffs_as_std(vr_test::Gpu& gpu, d::BlockLists& lists,
                 vol::VoxelBlockGrid& grid, const std::vector<vr::Vec3i>& frame,
                 const char* what) {
  vkc::Result<Blocks> active = grid.map().compact_active_blocks();
  CHECK(active.ok());
  std::map<vr::Vec3i, std::int32_t, ZyxLess> ptr_of;
  for (const vol::BlockIndex& b : active.value()) ptr_of[b.coord] = b.ptr;
  const std::vector<vr::Vec3i> held = sorted_coords(active.value());
  std::vector<vr::Vec3i> gone;
  std::vector<vr::Vec3i> missing;
  std::set_difference(held.begin(), held.end(), frame.begin(), frame.end(),
                      std::back_inserter(gone), ZyxLess{});
  std::set_difference(frame.begin(), frame.end(), held.begin(), held.end(),
                      std::back_inserter(missing), ZyxLess{});
  const vkc::Result<Diffed> got = device_diff(gpu, lists, grid, frame);
  bool ptrs = got.ok();
  if (got.ok()) {
    for (const vol::BlockIndex& b : got.value().gone) {
      ptrs = ptrs && ptr_of.count(b.coord) == 1 && ptr_of[b.coord] == b.ptr;
    }
    for (const vol::BlockIndex& b : got.value().missing) {
      ptrs = ptrs && b.ptr == 0;
    }
  }
  if (!got.ok() || sorted_coords(got.value().gone) != gone ||
      sorted_coords(got.value().missing) != missing || !ptrs) {
    std::fprintf(stderr, "diff, %s (G %zu, N %zu): %s\n", what, held.size(),
                 frame.size(),
                 got.ok() ? "differs" : got.status().message().c_str());
    return 1;
  }
  return 0;
}

int diff_case(vr_test::Gpu& gpu, d::BlockLists& lists) {
  codec_fixture::GridShape tiny;  // overflow chains
  tiny.bucket_size = 2;
  tiny.num_buckets = 64;
  tiny.max_chain = 64;
  Lcg rng{47};
  for (const codec_fixture::GridShape& shape :
       {codec_fixture::GridShape{}, tiny}) {
    const auto room = static_cast<std::size_t>(shape.bucket_size) *
                      static_cast<std::size_t>(shape.num_buckets) / 2;
    std::vector<vr::Vec3i> grid_coords;
    std::vector<vr::Vec3i> frame;
    split_coords(rng, room, 500, grid_coords, frame);
    // Some of the frame's blocks in the grid too: an overlap.
    std::vector<vr::Vec3i> shared;
    for (std::size_t i = 0; i < grid_coords.size(); i += 3) {
      shared.push_back(grid_coords[i]);
    }
    std::vector<vr::Vec3i> overlap = frame;
    overlap.insert(overlap.end(), shared.begin(), shared.end());
    std::sort(overlap.begin(), overlap.end(), d::coord_less);
    std::vector<vr::Vec3i> shuffled = grid_coords;
    for (std::size_t i = shuffled.size(); i > 1; --i) {
      std::swap(shuffled[i - 1], shuffled[rng.below(std::uint32_t(i))]);
    }
    vkc::Result<vol::VoxelBlockGrid> g = codec_fixture::make_grid(gpu, shape);
    CHECK(g.ok());
    vol::VoxelBlockGrid& grid = g.value();
    // An empty grid, against nothing and against a frame.
    CHECK(diffs_as_std(gpu, lists, grid, {}, "nothing") == 0);
    CHECK(diffs_as_std(gpu, lists, grid, frame, "empty grid") == 0);
    CHECK(place(grid, shuffled).ok());
    CHECK(diffs_as_std(gpu, lists, grid, {}, "empty frame") == 0);
    CHECK(diffs_as_std(gpu, lists, grid, grid_coords, "equal") == 0);
    CHECK(diffs_as_std(gpu, lists, grid, frame, "disjoint") == 0);
    CHECK(diffs_as_std(gpu, lists, grid, overlap, "overlap") == 0);
    // A resize replaces the table the diff binds.
    CHECK(grid.resize(shape.num_buckets * 2).ok());
    CHECK(diffs_as_std(gpu, lists, grid, overlap, "resized") == 0);
  }
  return 0;
}

// --- Garbage. ---------------------------------------------------------------

// A frame rows 11 and 12 refuse -- new blocks out of order, a block count
// that disagrees with the survivors, ranks of anything -- still merges
// within its outputs, and its searches end; an unsorted frame list diffs
// within the lists' bounds and leaves the list as it was.
int garbage_case(vr_test::Gpu& gpu, d::BlockLists& lists) {
  Lcg rng{53};
  const Mode every[] = {Mode::kKeep, Mode::kSkip, Mode::kZero, Mode::kRemove};
  for (int trial = 0; trial < 40; ++trial) {
    std::vector<vr::Vec3i> reference;
    std::vector<vr::Vec3i> added;
    split_coords(rng, 20 + rng.below(300), 400, reference, added);
    for (std::size_t i = added.size(); i > 1; --i) {
      std::swap(added[i - 1], added[rng.below(std::uint32_t(i))]);
    }
    std::vector<Mode> modes = modes_of(rng, reference.size(), every, 4);
    if (!modes.empty()) modes[0] = static_cast<Mode>(7);  // past the alphabet
    std::vector<std::uint32_t> ranks = ranks_of(modes);
    if (trial % 2 == 1) {
      for (std::uint32_t& r : ranks) r = rng.next();
    }
    const auto n = static_cast<std::uint32_t>(
        rng.below(std::uint32_t(reference.size() + added.size() + 1)));
    vkc::Result<Merged> got =
        device_merge(gpu, lists, reference, modes, added, n, ranks);
    CHECK(got.ok() && got.value().guards_kept);
  }
  vkc::Result<vol::VoxelBlockGrid> g = codec_fixture::make_grid(gpu);
  CHECK(g.ok());
  std::vector<vr::Vec3i> held;
  std::vector<vr::Vec3i> frame;
  split_coords(rng, 600, 500, held, frame);
  CHECK(place(g.value(), held).ok());
  frame.insert(frame.end(), held.begin(), held.begin() + 100);
  for (std::size_t i = frame.size(); i > 1; --i) {
    std::swap(frame[i - 1], frame[rng.below(std::uint32_t(i))]);
  }
  const auto n = static_cast<std::uint32_t>(frame.size());
  vkc::Result<vkc::Buffer> frame_buf = upload(gpu, listed(frame));
  CHECK(frame_buf.ok());
  const vol::VoxelGridParams& gp = g.value().grid();
  const auto slots =
      static_cast<std::uint32_t>(gp.num_buckets * gp.bucket_size);
  CHECK(lists.reserve(n, slots).ok());
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  CHECK(lists.record_diff(batch, g.value(), frame_buf.value(), n).ok());
  CHECK(batch.submit().ok());
  CHECK(lists.verdict().gone <= held.size() && lists.verdict().missing <= n);
  vkc::Result<Blocks> after = vr_test::read_back<vol::BlockIndex>(
      gpu.device, gpu.allocator, frame_buf.value(), n);
  CHECK(after.ok() &&
        sorted_coords(after.value()) == sorted_coords(listed(frame)));
  return 0;
}

// The merge's and the diff's refusals, before anything is recorded.
int lists_refusals_case(vr_test::Gpu& gpu, d::BlockLists& lists) {
  vkc::Result<vkc::Buffer> one = upload(gpu, Blocks(1));
  CHECK(one.ok());
  d::ResidentBlocks decoded;
  decoded.list = &one.value();
  decoded.ranks = &one.value();
  decoded.count = 4;  // four new blocks in a buffer of one
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  const vkc::Status merge =
      lists.record_merge(batch, decoded, one.value(),
                         d::FrameBuffers{&one.value(), &one.value()}, 4);
  CHECK(merge.domain() == C::InvalidArgument &&
        merge.message().rfind("BlockLists::record_merge: ", 0) == 0);
  vkc::Result<vol::VoxelBlockGrid> g = codec_fixture::make_grid(gpu);
  CHECK(g.ok());
  CHECK(lists.reserve(1, 1).ok());
  const vkc::Status diff = lists.record_diff(batch, g.value(), one.value(), 2);
  CHECK(diff.domain() == C::InvalidArgument &&
        diff.message().find("more than were reserved") != std::string::npos);
  vol::VoxelBlockGrid moved = std::move(g).value();
  vol::VoxelBlockGrid taken = std::move(moved);
  CHECK(lists.record_diff(batch, moved, one.value(), 0).domain() ==  // NOLINT
        C::InvalidArgument);
  CHECK(lists.reserve(1u << 30, 1).domain() == C::InvalidArgument);
  CHECK(batch.submit().ok());
  return 0;
}

int gpu_main(vr_test::GpuContext& gpu) {
  vkc::Result<std::unique_ptr<d::BlockLists>> made =
      d::BlockLists::create(gpu.device, gpu.allocator);
  CHECK(made.ok());
  d::BlockLists& lists = *made.value();
  if (sort_case(gpu, lists) != 0) return 1;
  if (sort_refusals_case(gpu, lists) != 0) return 1;
  if (merge_case(gpu, lists) != 0) return 1;
  if (collision_case(gpu, lists) != 0) return 1;
  if (diff_case(gpu, lists) != 0) return 1;
  if (garbage_case(gpu, lists) != 0) return 1;
  if (lists_refusals_case(gpu, lists) != 0) return 1;
  std::printf("codec block lists: OK\n");
  return 0;
}

}  // namespace

int main() { return vr_test::run_on_gpu(gpu_main); }
