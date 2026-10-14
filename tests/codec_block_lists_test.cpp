// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for the codec's block lists (block_lists.hpp): the sort against
// std::stable_sort by coord_less, over entries whose ptrs are all distinct,
// which pins stability -- at counts on and off the tile and level boundaries,
// over all of int32, clustered, all equal, a few keys duplicated across
// tiles, sorted and reversed -- with its input untouched, and its refusals.
// Internal headers through the oracle library. Skips without a device.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "bitstream.hpp"
#include "block_lists.hpp"
#include "buffer_readback.hpp"
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
using C = vkc::Status::Code;

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

int gpu_main(vr_test::GpuContext& gpu) {
  vkc::Result<std::unique_ptr<d::BlockLists>> made =
      d::BlockLists::create(gpu.device, gpu.allocator);
  CHECK(made.ok());
  d::BlockLists& lists = *made.value();
  if (sort_case(gpu, lists) != 0) return 1;
  if (sort_refusals_case(gpu, lists) != 0) return 1;
  std::printf("codec block lists: OK\n");
  return 0;
}

}  // namespace

int main() { return vr_test::run_on_gpu(gpu_main); }
