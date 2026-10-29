// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for the device frame writer's step layout (step_layout.hpp)
// against the CPU oracle's (oracle::step_layout): every entry's start, every
// segment's slot offset and both totals, for the steps the oracle counts in
// real frames -- block counts on and off the scan's 256-entry chunks, K of 1,
// 64 and 512, segments of 1 to 1,024 blocks, int32's coordinate extremes,
// the worked example and C2's P-frame matrix at every update and segment
// size, every mode among them -- and for any steps up to the bound, at every
// mix of update and new-block segments; each also as a bounded P-frame's
// count leaves it, garbage past N that the surplus word takes off, and a
// surplus past the new blocks clamped. The step total stops at the first
// value whose steps reach 4 GiB, also after more steps than a u32 holds, and
// is exact one step below it. Then the refusals, which record nothing.
// Internal headers through the oracle library. Skips without a device.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "bitstream.hpp"
#include "codec_frames.hpp"
#include "codec_oracle.hpp"
#include "step_layout.hpp"
#include "volumetric_kit/core/testing/buffer_readback.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"

#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/recon/core/device_requirements.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace codec = volumetric_kit::recon::codec;
namespace d = volumetric_kit::recon::codec::detail;
namespace oracle = codec_oracle;
using codec_frames::Lcg;
using codec_frames::make_frame;
using C = vkc::Status::Code;
using Steps = std::vector<std::uint32_t>;

namespace {

// The bound step_layout.hpp holds each entry's steps to, and the grammar's
// realistic ceiling: an entry takes at most 1,685.
constexpr std::uint32_t kMaxSteps = 1u << 20;
constexpr std::uint32_t kGrammarSteps = 4096;
// The first step total whose steps, 4 bytes each, reach 4 GiB: where the
// total stops, and the writer refuses.
constexpr std::uint64_t kStepCap =
    (std::uint64_t(std::numeric_limits<std::uint32_t>::max()) + 1) / 4;

// The device's layout of @p steps, read back.
struct Laid {
  std::vector<std::uint32_t> starts;
  std::vector<std::uint32_t> slot_offsets;
  d::StepLayout::Totals totals;
};

// @p steps laid out by the device, followed by @p tail's past them with
// @p surplus as the surplus word, as a bounded P-frame's count leaves them;
// read back as far as @p steps' own layout reaches. Uploads, layout and
// readbacks in one batch.
vkc::Result<Laid> device_layout(vkc::test::Gpu& gpu, d::StepLayout& layout,
                                const Steps& steps, std::uint32_t m,
                                std::uint32_t u, std::uint32_t r,
                                const Steps& tail, std::uint32_t surplus) {
  const auto n = static_cast<std::uint32_t>(steps.size());
  Steps bound = steps;
  bound.insert(bound.end(), tail.begin(), tail.end());
  const VkDeviceSize bytes = bound.size() * sizeof(std::uint32_t);
  vkc::Buffer input;
  if (!bound.empty()) {
    VKC_ASSIGN(input, vkc::device_storage_buffer(gpu.allocator, bytes));
  }
  // The frame words: the surplus, and every block of the bound.
  const std::uint32_t words[2] = {
      surplus, static_cast<std::uint32_t>(bound.size()) -
                   std::min<std::uint32_t>(m, std::uint32_t(bound.size()))};
  VKC_ASSIGN(const vkc::Buffer word,
             vkc::device_storage_buffer(gpu.allocator, sizeof(words)));
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  if (!bound.empty()) {
    VKC_TRY(batch.upload(input, 0, bound.data(), bytes));
  }
  VKC_TRY(batch.upload(word, 0, words, sizeof(words)));
  VKC_TRY(layout.record(batch, input, word,
                        static_cast<std::uint32_t>(bound.size()), m, u, r));
  Laid out;
  if (n != 0) {
    const std::uint64_t segments =
        d::frame_segment_count(m, u) + d::frame_segment_count(n - m, r);
    out.starts.resize(std::size_t(n) + 1);
    out.slot_offsets.resize(segments + 1);
    VKC_TRY(batch.readback(layout.starts(), 0,
                           out.starts.size() * sizeof(std::uint32_t),
                           out.starts.data()));
    VKC_TRY(batch.readback(layout.slot_offsets(), 0,
                           out.slot_offsets.size() * sizeof(std::uint32_t),
                           out.slot_offsets.data()));
  }
  VKC_TRY(batch.readback(layout.totals(), 0, sizeof(out.totals), &out.totals));
  VKC_TRY(batch.submit());
  return out;
}

template <typename A, typename B>
bool same(const std::vector<A>& a, const std::vector<B>& b) {
  return std::equal(a.begin(), a.end(), b.begin(), b.end(),
                    [](A x, B y) { return std::uint64_t(x) == y; });
}

// @p steps laid out on the device as the oracle lays them out, whatever
// @p tail and @p surplus follow them (device_layout): every start and slot
// offset, each offset even, and both totals; or, for a total at the cap or
// past it, the step total at the cap.
int lays_out_as_oracle(vkc::test::Gpu& gpu, d::StepLayout& layout,
                       const Steps& steps, std::uint32_t m, std::uint32_t u,
                       std::uint32_t r, const std::string& what,
                       const Steps& tail = {}, std::uint32_t surplus = 0) {
  const oracle::StepLayout expected = oracle::step_layout(steps, m, u, r);
  const vkc::Result<Laid> got =
      device_layout(gpu, layout, steps, m, u, r, tail, surplus);
  if (!got.ok()) {
    std::fprintf(stderr, "%s: %s\n", what.c_str(),
                 got.status().message().c_str());
    return 1;
  }
  const Laid& g = got.value();
  const std::uint64_t total = expected.starts.back();
  bool ok = true;
  if (total >= kStepCap) {
    ok = g.totals.steps == kStepCap;
  } else {
    ok = g.totals.steps == total &&
         g.totals.slot_words == expected.slot_offsets.back();
    if (!steps.empty()) {
      ok = ok && same(g.starts, expected.starts) &&
           same(g.slot_offsets, expected.slot_offsets) &&
           std::all_of(g.slot_offsets.begin(), g.slot_offsets.end(),
                       [](std::uint32_t w) { return (w & 1u) == 0; });
    }
  }
  if (!ok) {
    std::fprintf(stderr,
                 "%s: N %zu, M %u, U %u, R %u, tail %zu, surplus %u: device "
                 "totals %u / %u, oracle %llu / %llu\n",
                 what.c_str(), steps.size(), m, u, r, tail.size(), surplus,
                 g.totals.steps, g.totals.slot_words,
                 static_cast<unsigned long long>(total),
                 static_cast<unsigned long long>(expected.slot_offsets.back()));
    return 1;
  }
  return 0;
}

// @p steps laid out as the oracle lays them out, alone and as a bounded
// P-frame's count leaves them (ResidentBlocks::bounded): g steps of garbage
// past them, each up to the bound, with surplus g, for g of 1, 7 and M.
int lays_out_bounded(vkc::test::Gpu& gpu, d::StepLayout& layout,
                     const Steps& steps, std::uint32_t m, std::uint32_t u,
                     std::uint32_t r, const std::string& what) {
  Lcg rng{steps.size() * 7 + m};
  for (const std::uint32_t g : std::set<std::uint32_t>{0, 1, 7, m}) {
    Steps tail(g);
    for (std::uint32_t& s : tail) s = rng.below(kMaxSteps + 1);
    if (lays_out_as_oracle(gpu, layout, steps, m, u, r,
                           what + ", " + std::to_string(g) + " past it", tail,
                           g) != 0) {
      return 1;
    }
  }
  return 0;
}

// A frame's own steps, as the writer counts them, laid out at its sizes.
int frame_lays_out(vkc::test::Gpu& gpu, d::StepLayout& layout,
                   const oracle::Frame& frame, std::uint32_t u, std::uint32_t r,
                   const std::string& what) {
  const auto m = static_cast<std::uint32_t>(frame.modes.size());
  return lays_out_bounded(gpu, layout, oracle::entry_steps(frame, r), m,
                          m != 0 ? u : 0, r, what);
}

// Intra frames on and off the chunks, every K, segments of 1 to 1,024, and
// int32's extremes; the worked example; C2's matrix at every size.
int frames_case(vkc::test::Gpu& gpu, d::StepLayout& layout) {
  for (std::size_t n :
       {std::size_t(0), std::size_t(1), std::size_t(63), std::size_t(64),
        std::size_t(65), std::size_t(255), std::size_t(256), std::size_t(257),
        std::size_t(700), std::size_t(codec::kMaxDeviceDecodeSegmentSize)}) {
    for (std::uint32_t k : {1u, 64u, codec::kVoxelsPerBlock}) {
      const oracle::Frame f = make_frame(n, k, 17 * n + k);
      for (std::uint32_t r :
           {1u, 16u, 64u, codec::kMaxDeviceDecodeSegmentSize}) {
        CHECK(frame_lays_out(gpu, layout, f, 0, r, "intra") == 0);
      }
    }
  }
  CHECK(frame_lays_out(gpu, layout, make_frame(65537, 8, 5), 0, 64,
                       "65,537 blocks") == 0);
  const oracle::Frame extreme = codec_frames::extreme_frame();
  for (std::uint32_t r : {1u, 2u, 64u}) {
    CHECK(frame_lays_out(gpu, layout, extreme, 0, r, "int32 extremes") == 0);
  }

  const d::FrameReference example = codec_frames::reference_of(
      oracle::write_frame(codec_frames::example_reference()).value());
  const std::vector<std::uint8_t> bytes =
      codec_frames::example_bytes(example.hash);
  const oracle::Frame p =
      oracle::read_frame(bytes.data(), bytes.size(), 3, &example).value();
  for (std::uint32_t u : {1u, 64u}) {
    CHECK(frame_lays_out(gpu, layout, p, u, 64, "the worked example") == 0);
  }
  int laid = 0;
  for (const codec_frames::InterCode& c : codec_frames::inter_matrix()) {
    for (std::uint32_t u : codec_frames::kInterSizes) {
      for (std::uint32_t r : codec_frames::kInterSizes) {
        CHECK(frame_lays_out(gpu, layout, c.p, u, r, c.what) == 0);
        ++laid;
      }
    }
  }
  std::printf("frames: the worked example and %d of the P-frame matrix\n",
              laid);
  return 0;
}

// Any steps: random up to the grammar's ceiling at N on and off the chunks,
// M of none, some and all, and every update and segment size; every entry
// at the bound (the largest sums a chunk scans), none at all (a slot of the
// state alone), and odd and even alternating (every rounding).
int steps_case(vkc::test::Gpu& gpu, d::StepLayout& layout) {
  Lcg rng{43};
  const std::uint32_t sizes[] = {1, 7, 64, codec::kMaxDeviceDecodeSegmentSize};
  for (std::uint32_t n : {1u, 255u, 256u, 257u, 511u, 513u, 4097u, 65537u}) {
    Steps steps(n);
    for (std::uint32_t& s : steps) {
      s = rng.below(4) == 0 ? kGrammarSteps : rng.below(kGrammarSteps + 1);
    }
    for (std::uint32_t m : {0u, n / 3, n}) {
      for (std::uint32_t u : sizes) {
        for (std::uint32_t r : sizes) {
          CHECK(lays_out_bounded(gpu, layout, steps, m, m != 0 ? u : 0, r,
                                 "random") == 0);
        }
      }
    }
  }
  const Steps full(1023, kMaxSteps);
  for (std::uint32_t r : {1u, codec::kMaxDeviceDecodeSegmentSize}) {
    CHECK(lays_out_as_oracle(gpu, layout, full, 0, 0, r, "at the bound") == 0);
  }
  CHECK(lays_out_as_oracle(gpu, layout, full, 400, 7, 64,
                           "at the bound, with updates") == 0);
  const Steps none(70000, 0);
  for (std::uint32_t r : {1u, 64u}) {
    CHECK(lays_out_as_oracle(gpu, layout, none, 35000, 7, r, "none") == 0);
  }
  Steps alternating(70000);
  for (std::size_t i = 0; i < alternating.size(); ++i) {
    alternating[i] = static_cast<std::uint32_t>(i % 2 == 0 ? 3 : 4);
  }
  for (std::uint32_t r : {1u, 2u, 3u}) {
    CHECK(lays_out_as_oracle(gpu, layout, alternating, 0, 0, r,
                             "odd and even") == 0);
  }
  return 0;
}

// The step total at the cap: exact one step below it, stopped at it, and
// stopped after more steps than a u32 holds -- entries at the bound, and a
// million at the grammar's ceiling -- which would otherwise wrap to a total
// that fits.
int cap_case(vkc::test::Gpu& gpu, d::StepLayout& layout) {
  for (const std::uint32_t each : {kMaxSteps, kGrammarSteps}) {
    Steps steps(static_cast<std::size_t>(kStepCap / each), each);
    steps.back() -= 1;
    for (std::uint32_t r : {1u, codec::kMaxDeviceDecodeSegmentSize}) {
      CHECK(lays_out_as_oracle(gpu, layout, steps, 0, 0, r, "below the cap") ==
            0);
    }
    steps.back() += 1;
    CHECK(lays_out_as_oracle(gpu, layout, steps, 0, 0, 64, "at the cap") == 0);
    steps.push_back(1);
    CHECK(lays_out_as_oracle(gpu, layout, steps, 0, 0, 64, "past the cap") ==
          0);
    const Steps wraps(static_cast<std::size_t>(4 * kStepCap / each + 1000),
                      each);
    CHECK(std::uint64_t(wraps.size()) * each >
          std::numeric_limits<std::uint32_t>::max());
    CHECK(lays_out_as_oracle(gpu, layout, wraps,
                             static_cast<std::uint32_t>(wraps.size() / 4), 64,
                             64, "past a u32") == 0);
  }
  return 0;
}

// A surplus past the new blocks, which no frame has: the device clamps it,
// laying out the M reference entries alone, and with none of them nothing,
// so every read stays inside the bound.
int clamp_case(vkc::test::Gpu& gpu, d::StepLayout& layout) {
  Lcg rng{53};
  Steps reference(100);
  Steps added(200);
  for (std::uint32_t& s : reference) s = rng.below(kGrammarSteps + 1);
  for (std::uint32_t& s : added) s = rng.below(kGrammarSteps + 1);
  for (const std::uint32_t surplus :
       {201u, 300u, std::numeric_limits<std::uint32_t>::max()}) {
    CHECK(lays_out_as_oracle(gpu, layout, reference, 100, 7, 64,
                             "the reference entries", added, surplus) == 0);
    CHECK(lays_out_as_oracle(gpu, layout, Steps{}, 0, 0, 64, "nothing", added,
                             surplus) == 0);
  }
  return 0;
}

// What record refuses, recording nothing: a segment or update size past the
// rule, more reference entries than entries, and steps or frame words (two)
// the input does not hold. No entries zeroes the totals alone.
int refusals_case(vkc::test::Gpu& gpu, d::StepLayout& layout) {
  const Steps steps(100, 9);
  vkc::Result<vkc::Buffer> input =
      vkc::test::upload_device_buffer(gpu.device, gpu.allocator, steps.data(),
                                      steps.size() * sizeof(std::uint32_t));
  const std::uint32_t zero = 0;
  vkc::Result<vkc::Buffer> word = vkc::test::upload_device_buffer(
      gpu.device, gpu.allocator, &zero, sizeof(zero));
  vkc::Result<vkc::Buffer> half_word =
      vkc::device_storage_buffer(gpu.allocator, 2);
  CHECK(input.ok() && word.ok() && half_word.ok());
  auto refused = [&](std::uint32_t n, std::uint32_t m, std::uint32_t u,
                     std::uint32_t r, const vkc::Buffer& surplus,
                     const char* says) {
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    const vkc::Status s =
        layout.record(batch, input.value(), surplus, n, m, u, r);
    if (s.domain() != C::InvalidArgument || s.message().rfind(says, 0) != 0) {
      std::fprintf(stderr, "%s: %s\n", says,
                   s.ok() ? "not refused" : s.message().c_str());
      return 1;
    }
    CHECK(batch.submit().ok());
    return 0;
  };
  const std::uint32_t over = codec::kMaxDeviceDecodeSegmentSize + 1;
  struct Refusal {
    std::uint32_t n, m, u, r;
    const char* says;
  };
  const Refusal refusals[] = {
      {100, 0, 0, 0, "StepLayout::record: segment_size 0 is not in"},
      {100, 0, 0, over, "StepLayout::record: segment_size 1025 is not in"},
      {100, 10, 0, 64, "StepLayout::record: update_size 0 is not in"},
      {100, 10, over, 64, "StepLayout::record: update_size 1025 is not in"},
      {100, 101, 64, 64,
       "StepLayout::record: 101 reference entries, more than the 100"},
      {101, 0, 0, 64, "StepLayout::record: the steps"},
  };
  for (const Refusal& f : refusals) {
    CHECK(refused(f.n, f.m, f.u, f.r, word.value(), f.says) == 0);
  }
  CHECK(refused(100, 0, 0, 64, half_word.value(),
                "StepLayout::record: the frame words") == 0);
  CHECK(refused(100, 0, 0, 64, word.value(),
                "StepLayout::record: the frame words") == 0);
  const auto totals = [&]() {
    return vkc::test::read_back<std::uint32_t>(gpu.device, gpu.allocator,
                                               layout.totals(), 2);
  };
  CHECK(lays_out_as_oracle(gpu, layout, steps, 0, 0, 64, "after refusals") ==
        0);
  CHECK(totals().value()[0] == 900);
  CHECK(lays_out_as_oracle(gpu, layout, Steps{}, 0, 0, 64, "no entries") == 0);
  CHECK(totals().value() == std::vector<std::uint32_t>({0, 0}));
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  vkc::Result<std::unique_ptr<d::StepLayout>> created =
      d::StepLayout::create(gpu.device, gpu.allocator);
  CHECK(created.ok());
  d::StepLayout& layout = *created.value();
  if (frames_case(gpu, layout) != 0) return 1;
  if (steps_case(gpu, layout) != 0) return 1;
  if (cap_case(gpu, layout) != 0) return 1;
  if (clamp_case(gpu, layout) != 0) return 1;
  if (refusals_case(gpu, layout) != 0) return 1;
  std::puts("codec step layout: OK");
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(vr::device_requirements(), gpu_main);
}
