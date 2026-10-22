// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for the decoder-side deblock. detail::Deblock: its chains, each
// stage alone and every rule its filters have -- the 3-sample candidacy, the
// saturated samples, the clips, the [-1, 1] clamp, unobserved taps, the
// Gaussian's 26-block halo -- against the CPU oracle
// (codec_deblock_oracle.hpp); workgroups run one at a time in list order and
// in reverse, and in chunks, each the same bits, so a write another
// workgroup reads is a deterministic failure; the stamps; recording without
// submitting; the skip word; the refusals; and the step-keyed rule on the
// host. The fixtures' blocks sit at slots past twice their count, and their
// lists carry another block's ptr, so a kernel that trusts a ptr or indexes
// by slot fails. And the Decoder's switch: off is the exact decode; on, each
// mask-free frame is the oracle's filter of it, under the filter its header
// picks, for I- and P-frames, toggled between frames, across filters in one
// decoder and rebuilt while paused; a frame with an unobserved voxel, a
// single one included, is the exact decode, reported "off", decoded or
// rebuilt, interleaved with mask-free frames, and a true-SDF frame is
// deblocked unless it holds a cut; the stamps; and the rows. Skips with no
// device.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "codec_deblock_oracle.hpp"
#include "codec_fixture.hpp"
#include "deblock.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/recon/codec/decode_grid.hpp"
#include "volumetric_kit/recon/codec/decoder.hpp"
#include "volumetric_kit/recon/codec/encoder.hpp"
#include "volumetric_kit/recon/core/dispatch.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"

#include "gpu_test.hpp"

namespace vkc = volumetric_kit::core;
namespace codec = volumetric_kit::recon::codec;
namespace d = volumetric_kit::recon::codec::detail;
namespace oracle = codec_deblock_oracle;
using namespace codec_fixture;

namespace {

// A power of two, so v * trunc and tsdf / trunc are exact: a fixture's field
// is what the kernels read, to the bit.
constexpr float kBand = 0.0625f;
// Device against oracle, in units of the band: float against double over a
// few passes is a few ulps of |v| < 1; a filtering error is far larger.
constexpr double kTolerance = 2e-6;

const d::DeblockRow& row(const char* filter) {
  for (const d::DeblockRow& r : d::kDeblockRule) {
    if (std::strcmp(r.filter, filter) == 0) return r;
  }
  return d::kDeblockRule[0];
}

d::DeblockPasses linfit_only(float t) {
  d::DeblockPasses p = d::deblock_passes(row("LF"), t);
  return p;
}

d::DeblockPasses gauss_only(float t, const char* filter = "R") {
  d::DeblockPasses p = d::deblock_passes(row(filter), t);
  p.linfit = false;
  p.line_clip = 0.0f;
  return p;
}

// A voxel of a fixture's field: observed with value v (normalized), or not.
struct Voxel {
  bool observed = true;
  float v = 0.0f;
};

// Local voxel (x, y, z) of block @p b; global voxel g = 8 b + local.
struct Where {
  vr::Vec3i block;
  int x, y, z;
  vr::Vec3i global() const {
    return vr::Vec3i(block.x * 8 + x, block.y * 8 + y, block.z * 8 + z);
  }
};

using Field = std::function<Voxel(const Where&)>;

// A deterministic hash in [0, 1).
float hash01(std::uint32_t a, std::uint32_t b = 0, std::uint32_t c = 0) {
  std::uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u ^
                    (c + 0x165667B1u) * 0xC2B2AE3Du;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  return float(h & 0xFFFFFFu) / float(0x1000000u);
}

// The decoded field as a quantizer leaves it: a ramp, plus each block's own
// offset and slope along x (the DC and first-AC errors, a few T at
// T = 0.01), plus noise; 10% of voxels and the plane x = 13 unobserved.
Voxel ramp_field(const Where& at) {
  const vr::Vec3i g = at.global();
  const vr::Vec3i& b = at.block;
  const std::uint32_t bx = std::uint32_t(b.x + 16),
                      by = std::uint32_t(b.y + 16),
                      bz = std::uint32_t(b.z + 16);
  const std::uint32_t gx = std::uint32_t(g.x + 64),
                      gy = std::uint32_t(g.y + 64),
                      gz = std::uint32_t(g.z + 64);
  if (g.x == 13 || hash01(gx, gy, gz) < 0.1f) return Voxel{false, 0.0f};
  const float offset = 0.04f * (hash01(bx, by, bz + 7) - 0.5f);
  const float slope = 0.004f * (hash01(bx + 3, by, bz) - 0.5f);
  const float noise = 0.002f * (hash01(gx + 11, gy, gz) - 0.5f);
  return Voxel{true, 0.004f * float(g.x) - 0.003f * float(g.y) +
                         0.002f * float(g.z) - 0.02f + offset +
                         slope * (float(at.x) - 3.5f) + noise};
}

// The ramp fixture's blocks: 4 x 3 x 3, less a face neighbour, an edge
// neighbour and a corner neighbour of their blocks.
std::vector<vr::Vec3i> ramp_blocks() {
  std::vector<vr::Vec3i> out;
  for (int z = 0; z < 3; ++z) {
    for (int y = 0; y < 3; ++y) {
      for (int x = 0; x < 4; ++x) {
        const vr::Vec3i c(x, y, z);
        if (c == vr::Vec3i(2, 1, 2) || c == vr::Vec3i(0, 0, 1) ||
            c == vr::Vec3i(0, 2, 2)) {
          continue;
        }
        out.push_back(c);
      }
    }
  }
  return out;
}

// A grid holding @p coords with @p field, allocated after 2n filler blocks
// that are then removed, in a shuffled order, so slots and list positions
// differ and lie past 2n; and the frame's list -- sorted (z, y, x), reversed
// with @p reverse, each entry carrying another block's ptr -- and its masks,
// by list position.
struct Fixture {
  vol::VoxelBlockGrid grid;
  std::vector<vr::Vec3i> coords;  // the list's order
  vkc::Buffer list;
  vkc::Buffer masks;
  d::ResidentBlocks frame() const {
    d::ResidentBlocks r;
    r.list = &list;
    r.masks = &masks;
    r.count = std::uint32_t(coords.size());
    r.coefficient_count = 1;
    return r;
  }
};

vkc::Result<Fixture> make_fixture(Gpu& gpu,
                                  const std::vector<vr::Vec3i>& coords,
                                  const Field& field, bool reverse = false) {
  const auto fail = [](const char* why) {
    return vkc::Status::invalid_argument(std::string("fixture: ") + why);
  };
  GridShape shape;
  shape.trunc_dist = kBand;
  VKC_ASSIGN(vol::VoxelBlockGrid grid, make_grid(gpu, shape));
  const std::size_t n = coords.size();
  std::vector<vr::Vec3i> fillers;
  for (std::size_t i = 0; i < 2 * n; ++i) {
    fillers.push_back(vr::Vec3i(100 + int(i), 100, 100));
  }
  VKC_TRY(allocate(gpu, grid, fillers));
  std::vector<vr::Vec3i> shuffled = coords;
  for (std::size_t i = shuffled.size(); i > 1; --i) {
    std::swap(shuffled[i - 1],
              shuffled[std::size_t(hash01(std::uint32_t(i)) * float(i))]);
  }
  VKC_TRY(allocate(gpu, grid, shuffled));
  std::vector<vol::BlockIndex> gone(fillers.size());
  for (std::size_t i = 0; i < fillers.size(); ++i) gone[i].coord = fillers[i];
  VKC_ASSIGN(const std::uint32_t failed,
             vr_test::remove_blocks(gpu, grid, gone));
  if (failed != 0) return fail("fillers stayed");

  VKC_ASSIGN(const std::vector<vol::BlockIndex> active, active_sorted(grid));
  if (active.size() != n) return fail("block count");
  VKC_ASSIGN(
      std::vector<float> tsdf,
      vr_test::read_attribute<float>(gpu.device, gpu.allocator, grid, "tsdf"));
  VKC_ASSIGN(std::vector<float> weight,
             vr_test::read_attribute<float>(gpu.device, gpu.allocator, grid,
                                            "weight"));
  std::vector<vol::BlockIndex> list = active;
  if (reverse) std::reverse(list.begin(), list.end());
  std::vector<std::uint32_t> masks(n * codec::kMaskWordsPerBlock, 0u);
  std::vector<vr::Vec3i> order;
  for (std::size_t e = 0; e < n; ++e) {
    const vol::BlockIndex& b = list[e];
    if (std::uint32_t(b.ptr) / kVpb < 2 * n) return fail("a slot below 2n");
    for (std::uint32_t l = 0; l < kVpb; ++l) {
      const Voxel v =
          field(Where{b.coord, int(l % 8), int(l / 8 % 8), int(l / 64)});
      tsdf[std::uint32_t(b.ptr) + l] = v.observed ? v.v * kBand : 0.0f;
      weight[std::uint32_t(b.ptr) + l] = v.observed ? 1.0f : 0.0f;
      if (v.observed) {
        masks[e * codec::kMaskWordsPerBlock + l / 32] |= 1u << (l % 32);
      }
    }
    order.push_back(b.coord);
  }
  VKC_TRY(
      vr_test::write_attribute(gpu.device, gpu.allocator, grid, "tsdf", tsdf));
  VKC_TRY(vr_test::write_attribute(gpu.device, gpu.allocator, grid, "weight",
                                   weight));
  // Each entry names its block by coordinate alone; the ptr it carries is
  // its mirror's, which a kernel must never read.
  std::vector<vol::BlockIndex> poisoned = list;
  for (std::size_t e = 0; e < n; ++e) poisoned[e].ptr = list[n - 1 - e].ptr;
  VKC_ASSIGN(
      vkc::Buffer list_buffer,
      vr_test::upload_device_buffer(gpu.device, gpu.allocator, poisoned.data(),
                                    n * sizeof(vol::BlockIndex)));
  VKC_ASSIGN(
      vkc::Buffer mask_buffer,
      vr_test::upload_device_buffer(gpu.device, gpu.allocator, masks.data(),
                                    masks.size() * sizeof(std::uint32_t)));
  return Fixture{std::move(grid), std::move(order), std::move(list_buffer),
                 std::move(mask_buffer)};
}

// One deblock of @p fx under @p passes, at a fresh tick, in a batch of its
// own, gated by @p skip when given.
vkc::Status run(Gpu& gpu, d::Deblock& deblock, Fixture& fx,
                const d::DeblockPasses& passes,
                const vkc::Buffer* skip = nullptr) {
  fx.grid.map().advance_tick();
  VKC_TRY(deblock.reserve(std::uint32_t(fx.coords.size()), passes));
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  VKC_TRY(deblock.record(batch, fx.grid, fx.frame(), passes, skip));
  return batch.submit();
}

// The device's result against the oracle's: the same blocks and weights, an
// unobserved voxel's tsdf to the bit, an observed one's within kTolerance.
// The largest difference, in units of the band, goes to @p max_diff.
int matches(const Snapshot& dev, const Snapshot& want, double* max_diff) {
  CHECK(dev.coords == want.coords);
  CHECK(dev.weight == want.weight);
  double worst = 0.0;
  for (std::size_t i = 0; i < dev.tsdf.size(); ++i) {
    if (dev.weight[i] < vol::kObservedWeight) {
      CHECK(std::memcmp(&dev.tsdf[i], &want.tsdf[i], sizeof(float)) == 0);
    } else {
      worst = std::max(worst,
                       std::fabs(double(dev.tsdf[i]) - double(want.tsdf[i])) /
                           double(kBand));
    }
  }
  if (max_diff != nullptr) *max_diff = worst;
  CHECK(worst <= kTolerance);
  return 0;
}

// The largest change between two snapshots, in units of the band.
double largest_change(const Snapshot& a, const Snapshot& b) {
  double worst = 0.0;
  for (std::size_t i = 0; i < a.tsdf.size(); ++i) {
    worst = std::max(worst, std::fabs(double(a.tsdf[i]) - double(b.tsdf[i])) /
                                double(kBand));
  }
  return worst;
}

// @p passes on a fresh ramp fixture against the oracle, the filter moving
// the field by far more than the tolerance. The result goes to @p out.
int against_oracle(Gpu& gpu, d::Deblock& deblock,
                   const d::DeblockPasses& passes, const char* name,
                   Snapshot* out = nullptr) {
  vkc::Result<Fixture> made = make_fixture(gpu, ramp_blocks(), ramp_field);
  CHECK(made.ok());
  Fixture fx = std::move(made).value();
  vkc::Result<Snapshot> before = snapshot(gpu, fx.grid);
  CHECK(before.ok());
  CHECK(run(gpu, deblock, fx, passes).ok());
  vkc::Result<Snapshot> dev = snapshot(gpu, fx.grid);
  CHECK(dev.ok());
  Snapshot want = before.value();
  oracle::apply(want, passes, kBand);
  double diff = 0.0;
  CHECK(matches(dev.value(), want, &diff) == 0);
  const double moved = largest_change(dev.value(), before.value());
  std::printf("%s: moves the field by up to %.3g, the oracle within %.3g\n",
              name, moved, diff);
  CHECK(moved >= 100 * kTolerance);
  if (out != nullptr) *out = std::move(dev).value();
  return 0;
}

// 1, 2. LF, LG and R, and each stage alone, against the oracle on the ramp:
// partial masks, a missing face, edge and corner neighbour, an unobserved
// plane. LF writes no voxel a line cannot reach: local 2..5 on every axis.
int chains_case(Gpu& gpu, d::Deblock& deblock) {
  const float t = 0.01f;
  for (const char* filter : {"LF", "LG", "R"}) {
    Snapshot dev;
    CHECK(against_oracle(gpu, deblock, d::deblock_passes(row(filter), t),
                         filter, &dev) == 0);
    if (std::strcmp(filter, "LF") == 0) {
      vkc::Result<Fixture> made = make_fixture(gpu, ramp_blocks(), ramp_field);
      CHECK(made.ok());
      vkc::Result<Snapshot> before = snapshot(gpu, made.value().grid);
      CHECK(before.ok());
      for (std::size_t i = 0; i < dev.tsdf.size(); ++i) {
        const std::size_t l = i % kVpb;
        const bool inner = l % 8 >= 2 && l % 8 <= 5 && l / 8 % 8 >= 2 &&
                           l / 8 % 8 <= 5 && l / 64 >= 2 && l / 64 <= 5;
        if (inner) {
          CHECK(std::memcmp(&dev.tsdf[i], &before.value().tsdf[i],
                            sizeof(float)) == 0);
        }
      }
    }
  }
  CHECK(against_oracle(gpu, deblock, linfit_only(t), "linfit alone") == 0);
  CHECK(against_oracle(gpu, deblock, gauss_only(t), "Gaussian alone") == 0);
  return 0;
}

// Two blocks along x, a zero field but for lines set by hand, which only the
// x pass reaches.
struct Lines {
  float p[3];  // block (0,0,0)'s voxels at x = 7, 6, 5
  float q[3];  // block (1,0,0)'s at x = 0, 1, 2
  bool p_observed[3] = {true, true, true};
  bool q_observed[3] = {true, true, true};
};
// Lines (u, w) = (2 + i, 3) of the two blocks: y = 2 + i, z = 3.
Field lines_field(const std::vector<Lines>& lines) {
  return [lines](const Where& at) {
    const int i = at.y - 2;
    if (at.z != 3 || i < 0 || i >= int(lines.size())) return Voxel{true, 0.0f};
    const Lines& line = lines[std::size_t(i)];
    const bool left = at.block.x == 0;
    const int s = left ? 7 - at.x : at.x;
    if (s > 2) return Voxel{true, 0.0f};
    return left ? Voxel{line.p_observed[s], line.p[s]}
                : Voxel{line.q_observed[s], line.q[s]};
  };
}

// The lines after the x pass: block (0,0,0)'s x = 7, 6, 5 then block
// (1,0,0)'s x = 0, 1, 2, normalized.
std::vector<float> read_line(const Snapshot& s, int i) {
  std::vector<float> out;
  const std::uint32_t row_base = 8u * std::uint32_t(2 + i) + 64u * 3u;
  for (int k = 0; k < 3; ++k) {
    out.push_back(s.tsdf[row_base + std::uint32_t(7 - k)] / kBand);
  }
  for (int k = 0; k < 3; ++k) {
    out.push_back(s.tsdf[kVpb + row_base + std::uint32_t(k)] / kBand);
  }
  return out;
}

// 3, 4. Candidacy over all three fitted samples a side; a fitted sample at
// exactly 1 - 1e-5 skips its line and one an ulp below does not; the clamp
// to [-1, 1]; and the line clip, which moves p0 by exactly 0.375 of it.
int line_rules_case(Gpu& gpu, d::Deblock& deblock) {
  const float sat = 1.0f - 1e-5f;
  for (const bool at_kink : {true, false}) {
    std::vector<Lines> lines;
    // 0: the clamp. q's line extrapolates past 1, so p0 + 0.375 delta > 1.
    lines.push_back(Lines{{0.995f, 0.99f, 0.985f}, {0.99998f, 0.95f, 0.9f}});
    // 1: q2 at the kink, or an ulp below it.
    lines.push_back(
        Lines{{0.2f, 0.2f, 0.2f},
              {0.3f, 0.3f, at_kink ? sat : std::nextafter(sat, 0.0f)}});
    // 2, 3: p2, then q2, unobserved: not a candidate, though p0..1 and
    // q0..1 are.
    Lines no_p2{{0.1f, 0.1f, 0.0f}, {0.3f, 0.3f, 0.3f}};
    no_p2.p_observed[2] = false;
    lines.push_back(no_p2);
    Lines no_q2{{0.1f, 0.1f, 0.1f}, {0.3f, 0.3f, 0.0f}};
    no_q2.q_observed[2] = false;
    lines.push_back(no_q2);
    // 4: a jump of 0.5, clipped.
    lines.push_back(Lines{{0.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f}});

    vkc::Result<Fixture> made = make_fixture(
        gpu, {vr::Vec3i(0, 0, 0), vr::Vec3i(1, 0, 0)}, lines_field(lines));
    CHECK(made.ok());
    Fixture fx = std::move(made).value();
    vkc::Result<Snapshot> before = snapshot(gpu, fx.grid);
    CHECK(before.ok());
    const float t = 0.05f;  // line clip 0.2: lines 0..3 unclipped
    const d::DeblockPasses passes = linfit_only(t);
    CHECK(run(gpu, deblock, fx, passes).ok());
    vkc::Result<Snapshot> after = snapshot(gpu, fx.grid);
    CHECK(after.ok());
    Snapshot want = before.value();
    oracle::apply(want, passes, kBand);
    CHECK(matches(after.value(), want, nullptr) == 0);

    // Every voxel off the lines' x = 5..7 / 0..2 is as it was, to the bit.
    std::size_t changed = 0;
    for (std::size_t i = 0; i < after.value().tsdf.size(); ++i) {
      if (std::memcmp(&after.value().tsdf[i], &before.value().tsdf[i],
                      sizeof(float)) != 0) {
        ++changed;
      }
    }
    const std::vector<float> clamp_line = read_line(after.value(), 0);
    CHECK(clamp_line[0] == 1.0f);  // 0.995 + 0.375 * 0.0275, clamped
    const std::vector<float> kink = read_line(after.value(), 1);
    const std::vector<float> kink_before = read_line(before.value(), 1);
    CHECK((kink == kink_before) == at_kink);
    CHECK(read_line(after.value(), 2) == read_line(before.value(), 2));
    CHECK(read_line(after.value(), 3) == read_line(before.value(), 3));
    const std::vector<float> clipped = read_line(after.value(), 4);
    CHECK(std::fabs(double(clipped[0]) - 0.375 * double(passes.line_clip)) <=
          kTolerance);
    CHECK(std::fabs(double(clipped[1]) - 0.125 * double(passes.line_clip)) <=
          kTolerance);
    CHECK(std::fabs(double(clipped[3]) -
                    (0.5 - 0.375 * double(passes.line_clip))) <= kTolerance);
    CHECK(clipped[2] == 0.0f && clipped[5] == 0.5f);
    // Lines 0 and 4 write four voxels each, line 1 four unless at the kink.
    CHECK(changed == (at_kink ? 8u : 12u));
  }
  return 0;
}

// x alone varies: a ramp along x, and with @p plane the plane x = 4 of block
// (0,0,0) unobserved; or a spike.
Voxel x_ramp_field(const Where& at) {
  const vr::Vec3i g = at.global();
  if (g.x == 4) return Voxel{false, 0.0f};
  return Voxel{true, 0.02f * float(g.x)};
}
Voxel spike_field(const Where& at) {
  if (at.block == vr::Vec3i(0, 0, 0) && at.x == 3 && at.y == 3 && at.z == 3) {
    return Voxel{true, 0.5f};
  }
  return Voxel{true, 0.0f};
}
Voxel corner_field(const Where& at) {
  if (at.block == vr::Vec3i(0, 0, 0) && at.x == 7 && at.y == 7 && at.z == 7) {
    return Voxel{true, 0.3f};
  }
  return Voxel{true, 0.0f};
}

// 3, 5, 6. The Gaussian: an unobserved tap drops out of both sums (a hand
// value); each pass's change is clipped, so a spike loses three clips; and
// a block's corner reaches its diagonal neighbour's through the 26-block
// halo.
int gauss_rules_case(Gpu& gpu, d::Deblock& deblock) {
  {
    vkc::Result<Fixture> made = make_fixture(
        gpu, {vr::Vec3i(0, 0, 0), vr::Vec3i(1, 0, 0)}, x_ramp_field);
    CHECK(made.ok());
    Fixture fx = std::move(made).value();
    const d::DeblockPasses passes = gauss_only(0.1f);
    CHECK(run(gpu, deblock, fx, passes).ok());
    vkc::Result<Snapshot> after = snapshot(gpu, fx.grid);
    CHECK(after.ok());
    // x = 5 of block (0,0,0): its x - 1 tap is the unobserved plane.
    const double g1 = passes.gauss_g1;
    const double want = (0.1 + g1 * 0.12) / (1.0 + g1);
    const double got = after.value().tsdf[5 + 8 * 4 + 64 * 4] / kBand;
    std::printf("unobserved tap: %.9f, by hand %.9f\n", got, want);
    CHECK(std::fabs(got - want) <= kTolerance);
    // Counted as the 0 stored there, it would be far off.
    CHECK(std::fabs((g1 * 0.0 + 0.1 + g1 * 0.12) / (1.0 + 2 * g1) - want) >=
          100 * kTolerance);
  }
  {
    vkc::Result<Fixture> made =
        make_fixture(gpu, {vr::Vec3i(0, 0, 0)}, spike_field);
    CHECK(made.ok());
    Fixture fx = std::move(made).value();
    const d::DeblockPasses passes = gauss_only(0.0133f);
    vkc::Result<Snapshot> before = snapshot(gpu, fx.grid);
    CHECK(before.ok());
    CHECK(run(gpu, deblock, fx, passes).ok());
    vkc::Result<Snapshot> after = snapshot(gpu, fx.grid);
    CHECK(after.ok());
    Snapshot want = before.value();
    oracle::apply(want, passes, kBand);
    CHECK(matches(after.value(), want, nullptr) == 0);
    const double spike = after.value().tsdf[3 + 8 * 3 + 64 * 3] / kBand;
    CHECK(std::fabs(spike - (0.5 - 3.0 * double(passes.gauss_clip))) <=
          kTolerance);
  }
  {
    std::vector<vr::Vec3i> cube;
    for (int i = 0; i < 8; ++i)
      cube.push_back(vr::Vec3i(i & 1, i >> 1 & 1, i >> 2));
    vkc::Result<Fixture> made = make_fixture(gpu, cube, corner_field);
    CHECK(made.ok());
    Fixture fx = std::move(made).value();
    const d::DeblockPasses passes = gauss_only(0.2f);
    vkc::Result<Snapshot> before = snapshot(gpu, fx.grid);
    CHECK(before.ok());
    CHECK(run(gpu, deblock, fx, passes).ok());
    vkc::Result<Snapshot> after = snapshot(gpu, fx.grid);
    CHECK(after.ok());
    Snapshot want = before.value();
    oracle::apply(want, passes, kBand);
    CHECK(matches(after.value(), want, nullptr) == 0);
    // Block (1,1,1)'s corner (0,0,0), diagonal to the spike.
    const double corner = after.value().tsdf[7 * kVpb] / kBand;
    std::printf("diagonal corner: %.6g\n", corner);
    CHECK(corner >= 1000 * kTolerance);
  }
  return 0;
}

// 7, 8. Every workgroup alone, after the last one's writes, in list order
// and in reverse, and in chunks of 2 and 3 and one dispatch: each run of a
// chain the same bits, and the oracle's within the tolerance. A write
// another workgroup of its pass reads -- a shell voxel written in place, a
// line reaching past 4 -- reads differently in one order or the other.
int order_case(Gpu& gpu) {
  const float t = 0.01f;
  struct Run {
    std::uint32_t per_dispatch;
    bool reverse;
  };
  const Run runs[] = {{1, false}, {1, true}, {2, false}, {3, true}, {0, false}};
  for (const char* filter : {"LF", "LG", "R"}) {
    const d::DeblockPasses passes = d::deblock_passes(row(filter), t);
    Snapshot first;
    for (const Run& r : runs) {
      vkc::Result<std::unique_ptr<d::Deblock>> made_deblock =
          d::Deblock::create(gpu.device, gpu.allocator);
      CHECK(made_deblock.ok());
      vkc::Result<Fixture> made =
          make_fixture(gpu, ramp_blocks(), ramp_field, r.reverse);
      CHECK(made.ok());
      Fixture fx = std::move(made).value();
      vkc::Result<Snapshot> before = snapshot(gpu, fx.grid);
      CHECK(before.ok());
      // 0: the device's limit.
      vr::detail::set_test_groups_per_dispatch(r.per_dispatch);
      const vkc::Status ran = run(gpu, *made_deblock.value(), fx, passes);
      vr::detail::set_test_groups_per_dispatch(0);
      CHECK(ran.ok());
      vkc::Result<Snapshot> after = snapshot(gpu, fx.grid);
      CHECK(after.ok());
      Snapshot want = before.value();
      oracle::apply(want, passes, kBand);
      CHECK(matches(after.value(), want, nullptr) == 0);
      if (first.coords.empty()) {
        first = after.value();
      } else {
        CHECK(after.value() == first);
      }
    }
  }
  return 0;
}

Voxel stamps_field(const Where& at) {
  return Voxel{true, at.block == vr::Vec3i(0, 0, 0) ? 0.1f : 0.0f};
}

// 9. A block the deblock writes carries the pass's tick, whoever wrote it:
// a line's +axis block, written by its neighbour's workgroup alone, and a
// block whose shell alone the Gaussian changes, which the store writes. A
// block it leaves as it was keeps its stamp.
int stamps_case(Gpu& gpu, d::Deblock& deblock) {
  for (const bool gaussian : {false, true}) {
    vkc::Result<Fixture> made = make_fixture(
        gpu, {vr::Vec3i(0, 0, 0), vr::Vec3i(1, 0, 0), vr::Vec3i(5, 5, 5)},
        stamps_field);
    CHECK(made.ok());
    Fixture fx = std::move(made).value();
    vkc::Result<Snapshot> before = snapshot(gpu, fx.grid);
    CHECK(before.ok());
    const d::DeblockPasses passes =
        gaussian ? gauss_only(0.01f) : linfit_only(0.01f);
    CHECK(run(gpu, deblock, fx, passes).ok());
    const std::uint32_t tick = fx.grid.map().tick();
    vkc::Result<Snapshot> after = snapshot(gpu, fx.grid);
    vkc::Result<std::vector<vol::BlockStamp>> stamps =
        fx.grid.map().read_block_stamps();
    vkc::Result<std::vector<vol::BlockIndex>> active = active_sorted(fx.grid);
    CHECK(after.ok() && stamps.ok() && active.ok());
    for (std::size_t b = 0; b < 3; ++b) {
      const bool changed = std::memcmp(&after.value().tsdf[b * kVpb],
                                       &before.value().tsdf[b * kVpb],
                                       kVpb * sizeof(float)) != 0;
      const std::uint32_t slot = std::uint32_t(active.value()[b].ptr) / kVpb;
      CHECK(changed == (b < 2));
      CHECK((stamps.value()[slot].changed == tick) == changed);
    }
    if (gaussian) {
      // Block (1,0,0)'s interior is untouched: its stamp is the store's.
      for (std::uint32_t l = 0; l < kVpb; ++l) {
        const std::uint32_t x = l % 8, y = l / 8 % 8, z = l / 64;
        if (x > 0 && x < 7 && y > 0 && y < 7 && z > 0 && z < 7) {
          CHECK(after.value().tsdf[kVpb + l] == 0.0f);
        }
      }
    }
  }
  return 0;
}

// 10. Recorded, not run: until the batch is submitted the grid is as it was;
// a filter that runs nothing, or no blocks, records nothing; and the
// refusals, each before anything is recorded.
int record_case(Gpu& gpu, d::Deblock& deblock) {
  vkc::Result<Fixture> made = make_fixture(gpu, ramp_blocks(), ramp_field);
  CHECK(made.ok());
  Fixture fx = std::move(made).value();
  vkc::Result<Snapshot> before = snapshot(gpu, fx.grid);
  CHECK(before.ok());
  const d::DeblockPasses passes = d::deblock_passes(row("LG"), 0.01f);
  const std::uint32_t n = std::uint32_t(fx.coords.size());
  CHECK(deblock.reserve(n, passes).ok());
  {
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    CHECK(deblock.record(batch, fx.grid, fx.frame(), passes).ok());
    vkc::Result<Snapshot> pending = snapshot(gpu, fx.grid);
    CHECK(pending.ok() && pending.value() == before.value());
    CHECK(batch.submit().ok());
    vkc::Result<Snapshot> after = snapshot(gpu, fx.grid);
    CHECK(after.ok() && !(after.value() == before.value()));
  }
  vkc::Result<Snapshot> filtered = snapshot(gpu, fx.grid);
  CHECK(filtered.ok());
  {
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    CHECK(deblock.record(batch, fx.grid, fx.frame(), d::DeblockPasses{}).ok());
    d::ResidentBlocks none = fx.frame();
    none.count = 0;
    CHECK(deblock.record(batch, fx.grid, none, passes).ok());
    CHECK(batch.submit().ok());
    vkc::Result<Snapshot> after = snapshot(gpu, fx.grid);
    CHECK(after.ok() && after.value() == filtered.value());
  }
  const auto refused = [&](const d::ResidentBlocks& frame,
                           const d::DeblockPasses& p,
                           vol::VoxelBlockGrid& grid) {
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    const vkc::Status s = deblock.record(batch, grid, frame, p);
    return s.domain() == vkc::Status::Code::InvalidArgument;
  };
  d::DeblockPasses unclipped = passes;
  unclipped.line_clip = 0.0f;
  CHECK(refused(fx.frame(), unclipped, fx.grid));
  unclipped = passes;
  unclipped.gauss_clip = 0.0f;
  CHECK(refused(fx.frame(), unclipped, fx.grid));
  d::ResidentBlocks inter = fx.frame();
  inter.reference_count = 1;
  CHECK(refused(inter, passes, fx.grid));
  d::ResidentBlocks short_list = fx.frame();
  short_list.count = n + 1;
  CHECK(refused(short_list, linfit_only(0.01f), fx.grid));
  d::ResidentBlocks no_masks = fx.frame();
  no_masks.masks = nullptr;
  CHECK(refused(no_masks, passes, fx.grid));
  // More blocks than reserve made room for.
  vkc::Result<std::unique_ptr<d::Deblock>> fresh =
      d::Deblock::create(gpu.device, gpu.allocator);
  CHECK(fresh.ok());
  {
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    const vkc::Status s =
        fresh.value()->record(batch, fx.grid, fx.frame(), passes);
    CHECK(s.domain() == vkc::Status::Code::InvalidArgument &&
          s.message().find("reserve") != std::string::npos);
  }
  // A grid of another block size.
  GridShape four;
  four.block_size = 4;
  vkc::Result<vol::VoxelBlockGrid> g4 = make_grid(gpu, four);
  CHECK(g4.ok());
  CHECK(refused(fx.frame(), passes, g4.value()));
  return 0;
}

// A device word holding @p value, as a skip word.
vkc::Result<vkc::Buffer> word(Gpu& gpu, std::uint32_t value) {
  return vr_test::upload_device_buffer(gpu.device, gpu.allocator, &value,
                                       sizeof(value));
}

// 10b. The skip word: set, every filter leaves the grid and its stamps as
// they were -- the store too, though the snapshot holds the last Gaussian's
// shell -- and zero runs the filter as no word does; a buffer without a word
// is refused.
int skip_case(Gpu& gpu, d::Deblock& deblock) {
  vkc::Result<vkc::Buffer> one = word(gpu, 1u);
  vkc::Result<vkc::Buffer> zero = word(gpu, 0u);
  CHECK(one.ok() && zero.ok());
  const float t = 0.01f;
  for (const char* filter : {"LF", "LG", "R"}) {
    const d::DeblockPasses passes = d::deblock_passes(row(filter), t);
    // The snapshot primed with a filtered shell, entry for entry.
    CHECK(against_oracle(gpu, deblock, passes, filter) == 0);
    vkc::Result<Fixture> made = make_fixture(gpu, ramp_blocks(), ramp_field);
    CHECK(made.ok());
    Fixture fx = std::move(made).value();
    vkc::Result<Snapshot> before = snapshot(gpu, fx.grid);
    vkc::Result<std::vector<vol::BlockStamp>> stamps_before =
        fx.grid.map().read_block_stamps();
    CHECK(before.ok() && stamps_before.ok());
    CHECK(run(gpu, deblock, fx, passes, &one.value()).ok());
    vkc::Result<Snapshot> after = snapshot(gpu, fx.grid);
    vkc::Result<std::vector<vol::BlockStamp>> stamps_after =
        fx.grid.map().read_block_stamps();
    CHECK(after.ok() && stamps_after.ok());
    CHECK(after.value() == before.value());
    CHECK(stamps_after.value().size() == stamps_before.value().size());
    for (std::size_t i = 0; i < stamps_after.value().size(); ++i) {
      CHECK(stamps_after.value()[i].changed ==
            stamps_before.value()[i].changed);
    }
    // Zero runs it: the oracle's filter.
    CHECK(run(gpu, deblock, fx, passes, &zero.value()).ok());
    vkc::Result<Snapshot> ran = snapshot(gpu, fx.grid);
    CHECK(ran.ok());
    Snapshot want = before.value();
    oracle::apply(want, passes, kBand);
    CHECK(matches(ran.value(), want, nullptr) == 0);
    CHECK(largest_change(ran.value(), before.value()) >= 100 * kTolerance);
  }
  vkc::Result<Fixture> made = make_fixture(gpu, ramp_blocks(), ramp_field);
  const std::uint16_t half = 1;
  vkc::Result<vkc::Buffer> short_word = vr_test::upload_device_buffer(
      gpu.device, gpu.allocator, &half, sizeof(half));
  CHECK(made.ok() && short_word.ok());
  const d::DeblockPasses passes = d::deblock_passes(row("LG"), t);
  CHECK(
      deblock.reserve(std::uint32_t(made.value().coords.size()), passes).ok());
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  const vkc::Status s =
      deblock.record(batch, made.value().grid, made.value().frame(), passes,
                     &short_word.value());
  CHECK(s.domain() == vkc::Status::Code::InvalidArgument &&
        s.message().find("skip") != std::string::npos);
  return 0;
}

// 11. The rule, on the host: T in the study's arithmetic, to the bit; the
// rows' bounds, each inclusive below; and the filters it resolves.
int rule_case() {
  codec::CodecParams p;
  CHECK(d::step_scale(p) == oracle::study_step_scale(p));
  // Uniform 0.2: a first AC's peak, 0.2 x 0.0612991.
  CHECK(std::fabs(double(d::step_scale(p)) - 0.2 * 0.0612991) <= 1e-7);
  for (const std::uint32_t k : {1u, 2u, 3u, 4u, 56u, 512u}) {
    codec::CodecParams q;
    q.coefficient_count = k;
    q.quantization_scale = 0.040877f;
    q.quantization_weights[0] = 1.5f;
    q.quantization_weights[1] = 0.7f;
    q.quantization_weights[8] = 2.3f;
    q.quantization_weights[64] = 1.9f;
    CHECK(d::step_scale(q) == oracle::study_step_scale(q));
  }
  codec::CodecParams dc;
  dc.coefficient_count = 1;
  // DC alone: (1/sqrt 8)^3 of the step.
  CHECK(std::fabs(double(d::step_scale(dc)) - 0.2 * 0.0441942) <= 1e-7);

  const auto filter = [](std::uint32_t k, double t_mm) -> std::string {
    const d::DeblockRow* r = d::deblock_rule(k, t_mm);
    return r == nullptr ? "none" : r->filter;
  };
  CHECK(filter(1, 0.0) == "R");
  CHECK(filter(35, 0.0) == "R");
  CHECK(filter(36, 0.074) == "none");
  CHECK(filter(36, 0.075) == "LF");
  CHECK(filter(55, 0.3) == "LG");
  CHECK(filter(56, 0.6) == "R");
  CHECK(filter(56, std::nextafter(0.6, 0.0)) == "LG");
  CHECK(filter(56, std::nextafter(0.3, 0.0)) == "LF");
  CHECK(filter(64, std::nextafter(0.075, 0.0)) == "none");
  CHECK(filter(512, 10.0) == "R");

  // Room0's default at a 4 cm band: T 0.49 mm, LG.
  const double t_mm = double(d::step_scale(p)) * 0.04 * 1e3;
  CHECK(filter(64, t_mm) == "LG");
  const float t = d::step_scale(p);
  const d::DeblockPasses lg = d::deblock_passes(row("LG"), t);
  CHECK(lg.linfit && lg.gauss);
  CHECK(lg.line_clip == float(4.0 * double(t)));
  CHECK(lg.gauss_g1 == float(std::exp(-2.0)));
  CHECK(lg.gauss_clip == float(0.375 * double(t)));
  const d::DeblockPasses r = d::deblock_passes(row("R"), t);
  CHECK(r.gauss_g1 == float(std::exp(-1.0 / 1.125)));
  CHECK(r.gauss_clip == float(0.75 * double(t)));
  const d::DeblockPasses lf = d::deblock_passes(row("LF"), t);
  CHECK(lf.linfit && !lf.gauss);
  return 0;
}

// --- The Decoder's switch. --------------------------------------------------

const vkc::StageRow* find_row(const vkc::StageMetrics& m, const char* name) {
  for (const vkc::StageRow& row : m.rows()) {
    if (std::strcmp(row.name, name) == 0) return &row;
  }
  return nullptr;
}

using Bytes = std::vector<std::uint8_t>;

// A frame of @p grid at K @p k and uniform step @p scale, intra.
vkc::Result<Bytes> encode_at(Gpu& gpu, vol::VoxelBlockGrid& grid,
                             std::uint32_t k, float scale) {
  codec::EncoderConfig config;
  config.params.coefficient_count = k;
  config.params.quantization_scale = scale;
  VKC_ASSIGN(codec::Encoder enc,
             codec::Encoder::create(gpu.device, gpu.allocator, config));
  return enc.encode(grid);
}

vkc::Result<codec::Decoder> decoder(Gpu& gpu, bool deblock) {
  VKC_ASSIGN(codec::Decoder dec,
             codec::Decoder::create(gpu.device, gpu.allocator));
  dec.set_deblock(deblock);
  return dec;
}

// A grid of the codec fixture's geometry, its table @p num_buckets wide.
vkc::Result<vol::VoxelBlockGrid> player_grid(Gpu& gpu,
                                             std::int32_t num_buckets = 512) {
  GridShape shape;
  shape.num_buckets = num_buckets;
  return make_grid(gpu, shape);
}

// A mask-free fixture: the band blocks of @p spheres, every voxel observed,
// the nearer surface's distance clamped to the band, so every mask is all
// ones -- a frame the deblock runs on. The fixture's sphere_grid and
// two_sphere_grid leave the voxels past the band unobserved, as fused depth
// does: a masked frame, which it skips.
vkc::Result<vol::VoxelBlockGrid> full_grid(Gpu& gpu,
                                           const std::vector<Sphere>& spheres) {
  VKC_ASSIGN(vol::VoxelBlockGrid grid, make_grid(gpu));
  VKC_TRY(allocate(gpu, grid, union_blocks(spheres)));
  VKC_TRY(write_field(
      gpu, grid,
      [&spheres](const vr::Vec3f& p) {
        float d = spheres[0].sdf(p);
        for (const Sphere& s : spheres) d = std::min(d, s.sdf(p));
        return d;
      },
      true));
  return grid;
}

// Whether @p s holds an unobserved voxel: a masked frame's decode.
bool has_unobserved(const Snapshot& s) {
  return std::find(s.weight.begin(), s.weight.end(), 0.0f) != s.weight.end();
}

// The filter the rule picks for a frame of @p info, resolved: what the
// decoder should have run.
const d::DeblockRow* picked(const codec::FrameInfo& info, float* t) {
  *t = d::step_scale(info.params);
  return d::deblock_rule(info.params.coefficient_count,
                         double(*t) * double(info.trunc_dist) * 1e3);
}

// @p on, a decode with the deblock on, against @p exact, the same frame
// decoded with it off: the oracle's filter of it under @p row at T @p t, the
// unobserved voxels, weights and blocks exactly the exact decode's, and the
// field moved.
int deblocked(const Snapshot& on, const Snapshot& exact,
              const d::DeblockRow& row, float t, float trunc) {
  Snapshot want = exact;
  oracle::apply(want, d::deblock_passes(row, t), trunc);
  CHECK(on.coords == want.coords && on.weight == want.weight);
  double worst = 0.0, moved = 0.0;
  for (std::size_t i = 0; i < on.tsdf.size(); ++i) {
    if (on.weight[i] < vol::kObservedWeight) {
      CHECK(std::memcmp(&on.tsdf[i], &exact.tsdf[i], sizeof(float)) == 0);
      continue;
    }
    worst = std::max(
        worst, std::fabs(double(on.tsdf[i] - want.tsdf[i])) / double(trunc));
    moved = std::max(
        moved, std::fabs(double(on.tsdf[i] - exact.tsdf[i])) / double(trunc));
  }
  CHECK(worst <= kTolerance);
  CHECK(moved >= 100 * kTolerance);
  return 0;
}

// The largest difference of two decodes' fields, in units of the band.
double field_distance(const Snapshot& a, const Snapshot& b, float trunc) {
  double worst = 0.0;
  for (std::size_t i = 0; i < a.tsdf.size(); ++i) {
    worst = std::max(worst,
                     std::fabs(double(a.tsdf[i] - b.tsdf[i])) / double(trunc));
  }
  return worst;
}

// Off by default, and off is the exact decode; on, each frame takes the
// filter its header picks -- none at a fine step, LF, LG, R by T, and R at
// a small K -- the oracle's, and not the filter of the next row; the switch
// read at each decode.
int switch_case(Gpu& gpu) {
  vkc::Result<codec::Decoder> made_off = decoder(gpu, false);
  vkc::Result<codec::Decoder> made_on = decoder(gpu, true);
  CHECK(made_off.ok() && made_on.ok());
  codec::Decoder off = std::move(made_off).value();
  codec::Decoder on = std::move(made_on).value();
  vkc::Result<codec::Decoder> fresh =
      codec::Decoder::create(gpu.device, gpu.allocator);
  CHECK(fresh.ok() && !fresh.value().deblock());
  CHECK(std::strcmp(fresh.value().last_deblock().filter, "off") == 0 &&
        fresh.value().last_deblock().step == 0.0f);
  CHECK(on.deblock());

  const Sphere s{vr::Vec3f(0.006f, -0.004f, 0.002f), 0.09f};
  vkc::Result<vol::VoxelBlockGrid> source = full_grid(gpu, {s});
  CHECK(source.ok());
  struct Setting {
    std::uint32_t k;
    float scale;
    const char* filter;
    const char* next;  // the row a wrong rule would pick
  };
  // At the fixture's 4 cm band, T is 2.45 mm a unit of scale.
  const Setting settings[] = {{64, 0.02f, "off", "LF"},
                              {64, 0.1f, "LF", "LG"},
                              {64, 0.2f, "LG", "LF"},
                              {64, 0.3f, "R", "LG"},
                              {20, 0.2f, "R", "LG"}};
  for (const Setting& setting : settings) {
    vkc::Result<Bytes> frame =
        encode_at(gpu, source.value(), setting.k, setting.scale);
    CHECK(frame.ok());
    vkc::Result<codec::FrameInfo> info =
        codec::read_frame_info(frame.value().data(), frame.value().size());
    CHECK(info.ok());
    vkc::Result<vol::VoxelBlockGrid> a = player_grid(gpu);
    vkc::Result<vol::VoxelBlockGrid> b = player_grid(gpu, 1024);
    CHECK(a.ok() && b.ok());
    CHECK(
        off.decode(frame.value().data(), frame.value().size(), a.value()).ok());
    CHECK(
        on.decode(frame.value().data(), frame.value().size(), b.value()).ok());
    vkc::Result<Snapshot> exact = snapshot(gpu, a.value());
    vkc::Result<Snapshot> smoothed = snapshot(gpu, b.value());
    CHECK(exact.ok() && smoothed.ok());
    CHECK(!has_unobserved(exact.value()));
    float t = 0.0f;
    const d::DeblockRow* chosen = picked(info.value(), &t);
    const codec::DeblockInfo& got = on.last_deblock();
    std::printf("K %u, scale %g: T %.3f mm, %s\n", setting.k,
                double(setting.scale), double(got.step) * 1e3, got.filter);
    CHECK(std::strcmp(got.filter, setting.filter) == 0);
    CHECK(got.step == t * kTrunc);
    CHECK(std::strcmp(off.last_deblock().filter, "off") == 0 &&
          off.last_deblock().step == got.step);
    if (chosen == nullptr) {
      CHECK(std::strcmp(setting.filter, "off") == 0);
      CHECK(smoothed.value() == exact.value());
      continue;
    }
    CHECK(std::strcmp(chosen->filter, setting.filter) == 0);
    CHECK(deblocked(smoothed.value(), exact.value(), *chosen, t, kTrunc) == 0);
    Snapshot other = exact.value();
    oracle::apply(other, d::deblock_passes(row(setting.next), t), kTrunc);
    CHECK(field_distance(smoothed.value(), other, kTrunc) >= 100 * kTolerance);
    // The switch is read at each decode: off again, the exact decode.
    on.set_deblock(false);
    CHECK(
        on.decode(frame.value().data(), frame.value().size(), b.value()).ok());
    vkc::Result<Snapshot> again = snapshot(gpu, b.value());
    CHECK(again.ok() && again.value() == exact.value());
    CHECK(std::strcmp(on.last_deblock().filter, "off") == 0);
    on.set_deblock(true);
  }
  // An empty frame, on.
  vkc::Result<vol::VoxelBlockGrid> empty_source = player_grid(gpu);
  CHECK(empty_source.ok());
  vkc::Result<Bytes> empty = encode_at(gpu, empty_source.value(), 64, 0.2f);
  vkc::Result<vol::VoxelBlockGrid> g = player_grid(gpu);
  CHECK(empty.ok() && g.ok());
  CHECK(on.decode(empty.value().data(), empty.value().size(), g.value()).ok());
  vkc::Result<Snapshot> none = snapshot(gpu, g.value());
  CHECK(none.ok() && none.value().coords.empty());
  return 0;
}

// A sphere that moves and grows beside one that stays, at step @p t:
// mask-free, or with @p full false masked.
vkc::Result<vol::VoxelBlockGrid> moving_grid(Gpu& gpu, int t,
                                             bool full = true) {
  const Sphere moving{
      vr::Vec3f(0.004f * float(t), -0.002f * float(t), 0.003f * float(t)),
      0.07f + 0.002f * float(t)};
  const Sphere still{vr::Vec3f(-0.5f, 0.1f, 0.0f), 0.05f};
  if (full) return full_grid(gpu, {moving, still});
  return two_sphere_grid(gpu, moving, still);
}

// Mask-free I- and P-frames, frame after frame into one grid, with the
// switch toggled between frames: each frame on is the oracle's filter of
// the exact decode
// -- a P-frame decodes from its reference's coefficients, so the filter
// never compounds -- each frame off is the exact decode, and a second
// decoder on, into a grid of another table, gets the same bits.
int chain_case(Gpu& gpu) {
  codec::EncoderConfig config;
  config.intra_period = 4;
  vkc::Result<codec::Encoder> enc =
      codec::Encoder::create(gpu.device, gpu.allocator, config);
  vkc::Result<codec::Decoder> exact_dec = decoder(gpu, false);
  vkc::Result<codec::Decoder> toggled_dec = decoder(gpu, true);
  vkc::Result<codec::Decoder> on_dec = decoder(gpu, true);
  vkc::Result<vol::VoxelBlockGrid> exact_grid = player_grid(gpu);
  vkc::Result<vol::VoxelBlockGrid> toggled_grid = player_grid(gpu);
  vkc::Result<vol::VoxelBlockGrid> on_grid = player_grid(gpu, 2048);
  CHECK(enc.ok() && exact_dec.ok() && toggled_dec.ok() && on_dec.ok());
  CHECK(exact_grid.ok() && toggled_grid.ok() && on_grid.ok());
  std::size_t p_frames = 0;
  for (int t = 0; t < 7; ++t) {
    vkc::Result<vol::VoxelBlockGrid> g = moving_grid(gpu, t);
    CHECK(g.ok());
    vkc::Result<Bytes> frame = enc.value().encode(g.value());
    CHECK(frame.ok());
    vkc::Result<codec::FrameInfo> info =
        codec::read_frame_info(frame.value().data(), frame.value().size());
    CHECK(info.ok());
    p_frames += info.value().type == codec::FrameType::kInter ? 1 : 0;
    const bool toggled_on = t % 3 != 1;
    toggled_dec.value().set_deblock(toggled_on);
    for (codec::Decoder* dec :
         {&exact_dec.value(), &toggled_dec.value(), &on_dec.value()}) {
      vol::VoxelBlockGrid& grid =
          dec == &exact_dec.value()
              ? exact_grid.value()
              : (dec == &on_dec.value() ? on_grid.value()
                                        : toggled_grid.value());
      CHECK(dec->decode(frame.value().data(), frame.value().size(), grid).ok());
    }
    vkc::Result<Snapshot> exact = snapshot(gpu, exact_grid.value());
    vkc::Result<Snapshot> toggled = snapshot(gpu, toggled_grid.value());
    vkc::Result<Snapshot> on = snapshot(gpu, on_grid.value());
    CHECK(exact.ok() && toggled.ok() && on.ok());
    float step = 0.0f;
    const d::DeblockRow* row = picked(info.value(), &step);
    CHECK(row != nullptr && std::strcmp(row->filter, "LG") == 0);
    CHECK(deblocked(on.value(), exact.value(), *row, step, kTrunc) == 0);
    if (toggled_on) {
      CHECK(toggled.value() == on.value());
    } else {
      CHECK(toggled.value() == exact.value());
    }
  }
  CHECK(p_frames == 5);
  return 0;
}

// One decoder, one grid, each frame's filter another and its block count
// larger: none, LG, LF, R. Each the oracle's: the snapshot grows, and no
// state of a frame's filter is left for the next.
int mixed_filters_case(Gpu& gpu) {
  struct Frame {
    float radius;
    std::uint32_t k;
    float scale;
    const char* filter;
  };
  const Frame frames[] = {{0.05f, 64, 0.02f, "off"},
                          {0.07f, 64, 0.2f, "LG"},
                          {0.09f, 64, 0.1f, "LF"},
                          {0.11f, 20, 0.2f, "R"}};
  vkc::Result<codec::Decoder> on_dec = decoder(gpu, true);
  vkc::Result<codec::Decoder> exact_dec = decoder(gpu, false);
  vkc::Result<vol::VoxelBlockGrid> grid = player_grid(gpu, 2048);
  CHECK(on_dec.ok() && exact_dec.ok() && grid.ok());
  std::size_t last_blocks = 0;
  for (const Frame& f : frames) {
    vkc::Result<vol::VoxelBlockGrid> source =
        full_grid(gpu, {Sphere{vr::Vec3f(0.0f), f.radius}});
    CHECK(source.ok());
    vkc::Result<Bytes> frame = encode_at(gpu, source.value(), f.k, f.scale);
    vkc::Result<vol::VoxelBlockGrid> exact_grid = player_grid(gpu, 2048);
    CHECK(frame.ok() && exact_grid.ok());
    CHECK(on_dec.value()
              .decode(frame.value().data(), frame.value().size(), grid.value())
              .ok());
    CHECK(exact_dec.value()
              .decode(frame.value().data(), frame.value().size(),
                      exact_grid.value())
              .ok());
    vkc::Result<Snapshot> on = snapshot(gpu, grid.value());
    vkc::Result<Snapshot> exact = snapshot(gpu, exact_grid.value());
    vkc::Result<codec::FrameInfo> info =
        codec::read_frame_info(frame.value().data(), frame.value().size());
    CHECK(on.ok() && exact.ok() && info.ok());
    CHECK(on.value().coords.size() > last_blocks);
    last_blocks = on.value().coords.size();
    CHECK(std::strcmp(on_dec.value().last_deblock().filter, f.filter) == 0);
    float t = 0.0f;
    const d::DeblockRow* row = picked(info.value(), &t);
    if (row == nullptr) {
      CHECK(on.value() == exact.value());
    } else {
      CHECK(deblocked(on.value(), exact.value(), *row, t, kTrunc) == 0);
    }
  }
  return 0;
}

// Paused: the frame decoded last, rebuilt from the decoder's reference with
// the switch as it is now, is the decode with the switch so; the reference
// is untouched, so the next P-frame decodes as it would have. NotFound
// before a decode; a grid without exactly the frame's blocks, or of other
// geometry, refused untouched; the row.
int rebuild_case(Gpu& gpu) {
  codec::EncoderConfig config;
  config.intra_period = 30;
  vkc::Result<codec::Encoder> enc =
      codec::Encoder::create(gpu.device, gpu.allocator, config);
  CHECK(enc.ok());
  std::vector<Bytes> frames;
  for (int t = 0; t < 3; ++t) {
    vkc::Result<vol::VoxelBlockGrid> g = moving_grid(gpu, t);
    CHECK(g.ok());
    vkc::Result<Bytes> frame = enc.value().encode(g.value());
    CHECK(frame.ok());
    frames.push_back(std::move(frame).value());
  }
  // What a decode of frames 0..1 leaves with the switch off, and on.
  vkc::Result<codec::Decoder> exact_dec = decoder(gpu, false);
  vkc::Result<codec::Decoder> on_dec = decoder(gpu, true);
  vkc::Result<vol::VoxelBlockGrid> exact_grid = player_grid(gpu);
  vkc::Result<vol::VoxelBlockGrid> on_grid = player_grid(gpu);
  CHECK(exact_dec.ok() && on_dec.ok() && exact_grid.ok() && on_grid.ok());
  for (int t = 0; t < 2; ++t) {
    const Bytes& f = frames[std::size_t(t)];
    CHECK(
        exact_dec.value().decode(f.data(), f.size(), exact_grid.value()).ok());
    CHECK(on_dec.value().decode(f.data(), f.size(), on_grid.value()).ok());
  }
  vkc::Result<Snapshot> exact = snapshot(gpu, exact_grid.value());
  vkc::Result<Snapshot> on = snapshot(gpu, on_grid.value());
  CHECK(exact.ok() && on.ok());

  vkc::Result<codec::Decoder> made = decoder(gpu, false);
  vkc::Result<vol::VoxelBlockGrid> made_grid = player_grid(gpu, 1024);
  CHECK(made.ok() && made_grid.ok());
  codec::Decoder& dec = made.value();
  vol::VoxelBlockGrid& grid = made_grid.value();
  CHECK(dec.rebuild(grid).domain() == vkc::Status::Code::NotFound);
  for (int t = 0; t < 2; ++t) {
    const Bytes& f = frames[std::size_t(t)];
    CHECK(dec.decode(f.data(), f.size(), grid).ok());
  }
  dec.set_deblock(true);
  vkc::StageMetrics m;
  CHECK(dec.rebuild(grid, &m).ok());
  vkc::Result<Snapshot> rebuilt_on = snapshot(gpu, grid);
  CHECK(rebuilt_on.ok() && rebuilt_on.value() == on.value());
  CHECK(std::strcmp(dec.last_deblock().filter, "LG") == 0);
  CHECK(find_row(m, "codec decode") != nullptr);
  dec.set_deblock(false);
  CHECK(dec.rebuild(grid).ok());
  vkc::Result<Snapshot> rebuilt_off = snapshot(gpu, grid);
  CHECK(rebuilt_off.ok() && rebuilt_off.value() == exact.value());
  CHECK(std::strcmp(dec.last_deblock().filter, "off") == 0);

  // Refused, untouched: a block too many, one too few, other geometry.
  vol::BlockIndex extra{};
  extra.coord = vr::Vec3i(40, 40, 40);
  vkc::Result<std::uint32_t> added =
      vr_test::allocate_blocks(gpu, grid.map(), {extra});
  CHECK(added.ok() && added.value() == 0);
  vkc::Result<Snapshot> with_extra = snapshot(gpu, grid);
  CHECK(with_extra.ok());
  dec.set_deblock(true);
  CHECK(dec.rebuild(grid).domain() == vkc::Status::Code::InvalidArgument);
  vkc::Result<Snapshot> still = snapshot(gpu, grid);
  CHECK(still.ok() && still.value() == with_extra.value());
  vkc::Result<std::uint32_t> dropped =
      vr_test::remove_blocks(gpu, grid, {extra});
  CHECK(dropped.ok() && dropped.value() == 0);
  vol::BlockIndex missing{};
  missing.coord = exact.value().coords[3];
  dropped = vr_test::remove_blocks(gpu, grid, {missing});
  CHECK(dropped.ok() && dropped.value() == 0);
  vkc::Result<Snapshot> short_one = snapshot(gpu, grid);
  CHECK(short_one.ok());
  CHECK(dec.rebuild(grid).domain() == vkc::Status::Code::InvalidArgument);
  still = snapshot(gpu, grid);
  CHECK(still.ok() && still.value() == short_one.value());
  GridShape wide;
  wide.trunc_dist = 2 * kTrunc;
  vkc::Result<vol::VoxelBlockGrid> other = make_grid(gpu, wide);
  CHECK(other.ok());
  CHECK(dec.rebuild(other.value()).domain() ==
        vkc::Status::Code::InvalidArgument);

  // The reference is the decode's: frame 2, a P-frame, decodes on it into
  // the grid, short a block, as the decoder that never paused decodes it.
  const Bytes& f2 = frames[2];
  CHECK(dec.decode(f2.data(), f2.size(), grid).ok());
  CHECK(on_dec.value().decode(f2.data(), f2.size(), on_grid.value()).ok());
  vkc::Result<Snapshot> next = snapshot(gpu, grid);
  vkc::Result<Snapshot> next_on = snapshot(gpu, on_grid.value());
  CHECK(next.ok() && next_on.ok() && next.value() == next_on.value());
  return 0;
}

// Stamps, on: block B alone, then B unchanged beside a new block B - x. The
// inverse leaves B as it was, so only the deblock -- B - x's line pass,
// writing B as its +x neighbour -- can stamp it. Off, B keeps frame 1's
// stamp.
int decoder_stamps_case(Gpu& gpu) {
  const vr::Vec3i b(1, 0, 0);
  const vr::Vec3i left(0, 0, 0);
  // A plane, crossing both blocks, with B - x's half lifted 4 mm.
  const auto plane = [](float lift) {
    return [lift](const vr::Vec3f& p) {
      return 0.6f * p.x + 0.8f * p.y - 0.02f + (p.x < 0.04f ? lift : 0.0f);
    };
  };
  vkc::Result<vol::VoxelBlockGrid> one = make_grid(gpu);
  vkc::Result<vol::VoxelBlockGrid> two = make_grid(gpu);
  CHECK(one.ok() && two.ok());
  CHECK(allocate(gpu, one.value(), {b}).ok());
  CHECK(allocate(gpu, two.value(), {left, b}).ok());
  CHECK(write_field(gpu, one.value(), plane(0.0f), true).ok());
  CHECK(write_field(gpu, two.value(), plane(0.004f), true).ok());
  vkc::Result<Bytes> f1 = encode_at(gpu, one.value(), 64, 0.2f);
  vkc::Result<Bytes> f2 = encode_at(gpu, two.value(), 64, 0.2f);
  CHECK(f1.ok() && f2.ok());
  for (const bool deblock : {false, true}) {
    vkc::Result<codec::Decoder> dec = decoder(gpu, deblock);
    vkc::Result<vol::VoxelBlockGrid> grid = player_grid(gpu);
    CHECK(dec.ok() && grid.ok());
    CHECK(dec.value()
              .decode(f1.value().data(), f1.value().size(), grid.value())
              .ok());
    const std::uint32_t tick1 = grid.value().map().tick();
    vkc::Result<Snapshot> first = snapshot(gpu, grid.value());
    CHECK(first.ok());
    CHECK(dec.value()
              .decode(f2.value().data(), f2.value().size(), grid.value())
              .ok());
    const std::uint32_t tick2 = grid.value().map().tick();
    vkc::Result<Snapshot> second = snapshot(gpu, grid.value());
    vkc::Result<std::vector<vol::BlockStamp>> stamps =
        grid.value().map().read_block_stamps();
    vkc::Result<std::vector<vol::BlockIndex>> active =
        active_sorted(grid.value());
    CHECK(second.ok() && stamps.ok() && active.ok());
    CHECK(active.value().size() == 2 && active.value()[1].coord == b);
    const bool changed =
        std::memcmp(&second.value().tsdf[kVpb], first.value().tsdf.data(),
                    kVpb * sizeof(float)) != 0;
    CHECK(changed == deblock);
    const std::uint32_t stamp =
        stamps.value()[std::uint32_t(active.value()[1].ptr) / kVpb].changed;
    CHECK(stamp == (deblock ? tick2 : tick1));
  }
  return 0;
}

// A grid's stamps, block by block in (z, y, x) order.
vkc::Result<std::vector<std::uint32_t>> stamps_by_coord(
    vol::VoxelBlockGrid& grid) {
  VKC_ASSIGN(const std::vector<vol::BlockIndex> active, active_sorted(grid));
  VKC_ASSIGN(const std::vector<vol::BlockStamp> stamps,
             grid.map().read_block_stamps());
  std::vector<std::uint32_t> out;
  for (const vol::BlockIndex& b : active) {
    out.push_back(stamps[std::uint32_t(b.ptr) / kVpb].changed);
  }
  return out;
}

// Masked frames -- the moving spheres with the voxels past the band
// unobserved, as fused depth leaves them -- with the switch on: I- and
// P-frames each the exact decode to the bit, their stamps too, reported
// "off" with the frame's T, though the rule picks a filter; and rebuilt with
// the switch on, the exact decode again.
int masked_case(Gpu& gpu) {
  codec::EncoderConfig config;
  config.intra_period = 4;
  vkc::Result<codec::Encoder> enc =
      codec::Encoder::create(gpu.device, gpu.allocator, config);
  vkc::Result<codec::Decoder> on_dec = decoder(gpu, true);
  vkc::Result<codec::Decoder> off_dec = decoder(gpu, false);
  vkc::Result<vol::VoxelBlockGrid> on_grid = player_grid(gpu);
  vkc::Result<vol::VoxelBlockGrid> off_grid = player_grid(gpu);
  CHECK(enc.ok() && on_dec.ok() && off_dec.ok() && on_grid.ok() &&
        off_grid.ok());
  std::size_t p_frames = 0;
  for (int t = 0; t < 6; ++t) {
    vkc::Result<vol::VoxelBlockGrid> g = moving_grid(gpu, t, false);
    CHECK(g.ok());
    vkc::Result<Bytes> frame = enc.value().encode(g.value());
    CHECK(frame.ok());
    vkc::Result<codec::FrameInfo> info =
        codec::read_frame_info(frame.value().data(), frame.value().size());
    CHECK(info.ok());
    p_frames += info.value().type == codec::FrameType::kInter ? 1 : 0;
    CHECK(
        on_dec.value()
            .decode(frame.value().data(), frame.value().size(), on_grid.value())
            .ok());
    CHECK(off_dec.value()
              .decode(frame.value().data(), frame.value().size(),
                      off_grid.value())
              .ok());
    vkc::Result<Snapshot> on = snapshot(gpu, on_grid.value());
    vkc::Result<Snapshot> exact = snapshot(gpu, off_grid.value());
    vkc::Result<std::vector<std::uint32_t>> on_stamps =
        stamps_by_coord(on_grid.value());
    vkc::Result<std::vector<std::uint32_t>> off_stamps =
        stamps_by_coord(off_grid.value());
    CHECK(on.ok() && exact.ok() && on_stamps.ok() && off_stamps.ok());
    CHECK(has_unobserved(exact.value()));
    CHECK(on.value() == exact.value());
    CHECK(on_stamps.value() == off_stamps.value());
    float step = 0.0f;
    const d::DeblockRow* row = picked(info.value(), &step);
    CHECK(row != nullptr);
    const codec::DeblockInfo& got = on_dec.value().last_deblock();
    CHECK(std::strcmp(got.filter, "off") == 0 && got.step == step * kTrunc);
    CHECK(off_dec.value().last_deblock().step == got.step);
  }
  CHECK(p_frames == 4);
  CHECK(on_dec.value().rebuild(on_grid.value()).ok());
  vkc::Result<Snapshot> rebuilt = snapshot(gpu, on_grid.value());
  vkc::Result<Snapshot> exact = snapshot(gpu, off_grid.value());
  CHECK(rebuilt.ok() && exact.ok() && rebuilt.value() == exact.value());
  CHECK(std::strcmp(on_dec.value().last_deblock().filter, "off") == 0);
  return 0;
}

// A single unobserved voxel skips the whole frame: a mask-free frame is the
// oracle's filter of its exact decode, and the same frame with one voxel of
// one block unobserved is its exact decode in every block, reported "off".
int one_voxel_case(Gpu& gpu) {
  const Sphere s{vr::Vec3f(0.006f, -0.004f, 0.002f), 0.09f};
  vkc::Result<vol::VoxelBlockGrid> full = full_grid(gpu, {s});
  vkc::Result<vol::VoxelBlockGrid> one = full_grid(gpu, {s});
  CHECK(full.ok() && one.ok());
  {
    vol::VoxelBlockGrid& grid = one.value();
    vkc::Result<std::vector<vol::BlockIndex>> active = active_sorted(grid);
    vkc::Result<std::vector<float>> tsdf =
        vr_test::read_attribute<float>(gpu.device, gpu.allocator, grid, "tsdf");
    vkc::Result<std::vector<float>> weight = vr_test::read_attribute<float>(
        gpu.device, gpu.allocator, grid, "weight");
    CHECK(active.ok() && tsdf.ok() && weight.ok());
    const vol::BlockIndex& b = active.value()[active.value().size() / 2];
    const std::uint32_t v = std::uint32_t(b.ptr) + 3 + 8 * 4 + 64 * 5;
    tsdf.value()[v] = 0.0f;
    weight.value()[v] = 0.0f;
    CHECK(vr_test::write_attribute(gpu.device, gpu.allocator, grid, "tsdf",
                                   tsdf.value())
              .ok());
    CHECK(vr_test::write_attribute(gpu.device, gpu.allocator, grid, "weight",
                                   weight.value())
              .ok());
  }
  for (const bool masked : {false, true}) {
    vkc::Result<Bytes> frame =
        encode_at(gpu, masked ? one.value() : full.value(), 64, 0.2f);
    CHECK(frame.ok());
    vkc::Result<codec::FrameInfo> info =
        codec::read_frame_info(frame.value().data(), frame.value().size());
    vkc::Result<codec::Decoder> on_dec = decoder(gpu, true);
    vkc::Result<codec::Decoder> off_dec = decoder(gpu, false);
    vkc::Result<vol::VoxelBlockGrid> on_grid = player_grid(gpu);
    vkc::Result<vol::VoxelBlockGrid> off_grid = player_grid(gpu);
    CHECK(info.ok() && on_dec.ok() && off_dec.ok() && on_grid.ok() &&
          off_grid.ok());
    CHECK(
        on_dec.value()
            .decode(frame.value().data(), frame.value().size(), on_grid.value())
            .ok());
    CHECK(off_dec.value()
              .decode(frame.value().data(), frame.value().size(),
                      off_grid.value())
              .ok());
    vkc::Result<Snapshot> on = snapshot(gpu, on_grid.value());
    vkc::Result<Snapshot> exact = snapshot(gpu, off_grid.value());
    CHECK(on.ok() && exact.ok());
    CHECK(std::size_t(std::count(exact.value().weight.begin(),
                                 exact.value().weight.end(), 0.0f)) ==
          (masked ? 1u : 0u));
    float step = 0.0f;
    const d::DeblockRow* row = picked(info.value(), &step);
    CHECK(row != nullptr && std::strcmp(row->filter, "LG") == 0);
    const char* reported = on_dec.value().last_deblock().filter;
    if (masked) {
      CHECK(on.value() == exact.value());
      CHECK(std::strcmp(reported, "off") == 0);
    } else {
      CHECK(deblocked(on.value(), exact.value(), *row, step, kTrunc) == 0);
      CHECK(std::strcmp(reported, "LG") == 0);
    }
  }
  return 0;
}

// One decoder, the switch on, over a chain whose frames alternate between
// mask-free and masked, I- and P-frames of each: every frame the oracle's
// filter of its exact decode or the exact decode, as its own masks say,
// with its report, so nothing of one frame's gate is left for the next.
int interleaved_case(Gpu& gpu) {
  codec::EncoderConfig config;
  config.intra_period = 3;
  vkc::Result<codec::Encoder> enc =
      codec::Encoder::create(gpu.device, gpu.allocator, config);
  vkc::Result<codec::Decoder> on_dec = decoder(gpu, true);
  vkc::Result<codec::Decoder> off_dec = decoder(gpu, false);
  vkc::Result<vol::VoxelBlockGrid> on_grid = player_grid(gpu);
  vkc::Result<vol::VoxelBlockGrid> off_grid = player_grid(gpu);
  CHECK(enc.ok() && on_dec.ok() && off_dec.ok() && on_grid.ok() &&
        off_grid.ok());
  // Frames by (masked, P).
  std::size_t kinds[2][2] = {{0, 0}, {0, 0}};
  for (int t = 0; t < 7; ++t) {
    const bool masked = t % 2 == 1;
    vkc::Result<vol::VoxelBlockGrid> g = moving_grid(gpu, t, !masked);
    CHECK(g.ok());
    vkc::Result<Bytes> frame = enc.value().encode(g.value());
    CHECK(frame.ok());
    vkc::Result<codec::FrameInfo> info =
        codec::read_frame_info(frame.value().data(), frame.value().size());
    CHECK(info.ok());
    ++kinds[masked][info.value().type == codec::FrameType::kInter];
    CHECK(
        on_dec.value()
            .decode(frame.value().data(), frame.value().size(), on_grid.value())
            .ok());
    CHECK(off_dec.value()
              .decode(frame.value().data(), frame.value().size(),
                      off_grid.value())
              .ok());
    vkc::Result<Snapshot> on = snapshot(gpu, on_grid.value());
    vkc::Result<Snapshot> exact = snapshot(gpu, off_grid.value());
    CHECK(on.ok() && exact.ok());
    CHECK(has_unobserved(exact.value()) == masked);
    float step = 0.0f;
    const d::DeblockRow* row = picked(info.value(), &step);
    CHECK(row != nullptr);
    const char* reported = on_dec.value().last_deblock().filter;
    if (masked) {
      CHECK(on.value() == exact.value());
      CHECK(std::strcmp(reported, "off") == 0);
    } else {
      CHECK(deblocked(on.value(), exact.value(), *row, step, kTrunc) == 0);
      CHECK(std::strcmp(reported, row->filter) == 0);
    }
  }
  for (const auto& by_type : kinds) {
    CHECK(by_type[0] >= 1 && by_type[1] >= 1);
  }
  return 0;
}

// The block ceiling the player's helpers take: far past any fixture's.
constexpr std::uint32_t kMaxBlocks = 1u << 20;

// True-SDF frames (EncoderConfig::true_sdf) at the mesh templates' 16-voxel
// band: a sphere's, every mask all ones, is the oracle's filter of its exact
// decode; a curtain's, whose sign errors are cut and coded unobserved, is
// its exact decode, reported "off".
int true_sdf_case(Gpu& gpu) {
  codec::EncoderConfig config;
  config.true_sdf = true;
  vkc::Result<codec::Encoder> enc =
      codec::Encoder::create(gpu.device, gpu.allocator, config);
  vkc::Result<vol::VoxelBlockGrid> sphere = sphere_grid(
      gpu, Sphere{vr::Vec3f(0.006f, -0.004f, 0.002f), 0.07f}, wide_band());
  vkc::Result<vol::VoxelBlockGrid> curtain = curtain_grid(
      gpu, Curtain{Sphere{vr::Vec3f(0.013f, -0.007f, 0.004f), 0.09f}, false});
  CHECK(enc.ok() && sphere.ok() && curtain.ok());
  for (const bool cut : {false, true}) {
    vkc::Result<Bytes> frame =
        enc.value().encode(cut ? curtain.value() : sphere.value());
    CHECK(frame.ok());
    const Bytes& f = frame.value();
    vkc::Result<codec::FrameInfo> info =
        codec::read_frame_info(f.data(), f.size());
    vkc::Result<codec::Decoder> on_dec = decoder(gpu, true);
    vkc::Result<codec::Decoder> off_dec = decoder(gpu, false);
    vkc::Result<vol::VoxelBlockGrid> on_grid = codec::make_decode_grid(
        gpu.device, gpu.allocator, f.data(), f.size(), kMaxBlocks);
    vkc::Result<vol::VoxelBlockGrid> off_grid = codec::make_decode_grid(
        gpu.device, gpu.allocator, f.data(), f.size(), kMaxBlocks);
    CHECK(info.ok() && on_dec.ok() && off_dec.ok() && on_grid.ok() &&
          off_grid.ok());
    CHECK(codec::decode_growing(on_dec.value(), f.data(), f.size(),
                                on_grid.value(), kMaxBlocks)
              .ok());
    CHECK(codec::decode_growing(off_dec.value(), f.data(), f.size(),
                                off_grid.value(), kMaxBlocks)
              .ok());
    vkc::Result<Snapshot> on = snapshot(gpu, on_grid.value());
    vkc::Result<Snapshot> exact = snapshot(gpu, off_grid.value());
    CHECK(on.ok() && exact.ok());
    CHECK(has_unobserved(exact.value()) == cut);
    float step = 0.0f;
    const d::DeblockRow* row = picked(info.value(), &step);
    CHECK(row != nullptr);
    const char* reported = on_dec.value().last_deblock().filter;
    if (cut) {
      CHECK(on.value() == exact.value());
      CHECK(std::strcmp(reported, "off") == 0);
    } else {
      CHECK(deblocked(on.value(), exact.value(), *row, step,
                      info.value().trunc_dist) == 0);
      CHECK(std::strcmp(reported, row->filter) == 0);
    }
  }
  return 0;
}

// On, the rows are the decode's: "codec decode", its device half the
// deblock's dispatches too, over the same three breakdowns, no other.
int rows_case(Gpu& gpu) {
  vkc::Result<codec::Decoder> dec = decoder(gpu, true);
  vkc::Result<vol::VoxelBlockGrid> source =
      full_grid(gpu, {Sphere{vr::Vec3f(0.0f), 0.08f}});
  CHECK(dec.ok() && source.ok());
  vkc::Result<Bytes> frame = encode_at(gpu, source.value(), 64, 0.2f);
  vkc::Result<vol::VoxelBlockGrid> grid = player_grid(gpu);
  CHECK(frame.ok() && grid.ok());
  vkc::StageMetrics m;
  CHECK(
      dec.value()
          .decode(frame.value().data(), frame.value().size(), grid.value(), &m)
          .ok());
  CHECK(std::strcmp(dec.value().last_deblock().filter, "LG") == 0);
  std::vector<std::string> names;
  for (const vkc::StageRow& row : m.rows()) names.push_back(row.name);
  const std::vector<std::string> want = {"codec decode", "  ..rans decode",
                                         "  ..apply", "  ..inverse"};
  CHECK(names == want);
  vkc::Result<vkc::GpuTimer> probe = vkc::GpuTimer::create(gpu.device);
  CHECK(probe.ok());
  if (probe.value().available()) {
    CHECK(find_row(m, "codec decode")->has_gpu);
  }
  return 0;
}

int gpu_main(vr_test::GpuContext& gpu) {
  if (rule_case() != 0) return 1;
  vkc::Result<std::unique_ptr<d::Deblock>> made =
      d::Deblock::create(gpu.device, gpu.allocator);
  CHECK(made.ok());
  d::Deblock& deblock = *made.value();
  if (chains_case(gpu, deblock) != 0) return 1;
  if (line_rules_case(gpu, deblock) != 0) return 1;
  if (gauss_rules_case(gpu, deblock) != 0) return 1;
  if (order_case(gpu) != 0) return 1;
  if (stamps_case(gpu, deblock) != 0) return 1;
  if (record_case(gpu, deblock) != 0) return 1;
  if (skip_case(gpu, deblock) != 0) return 1;
  if (switch_case(gpu) != 0) return 1;
  if (chain_case(gpu) != 0) return 1;
  if (mixed_filters_case(gpu) != 0) return 1;
  if (rebuild_case(gpu) != 0) return 1;
  if (decoder_stamps_case(gpu) != 0) return 1;
  if (masked_case(gpu) != 0) return 1;
  if (one_voxel_case(gpu) != 0) return 1;
  if (interleaved_case(gpu) != 0) return 1;
  if (true_sdf_case(gpu) != 0) return 1;
  if (rows_case(gpu) != 0) return 1;
  std::printf("codec deblock: OK\n");
  return 0;
}

}  // namespace

int main() { return vr_test::run_on_gpu(gpu_main); }
