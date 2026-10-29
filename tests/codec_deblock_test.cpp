// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for the decoder-side deblock. detail::Deblock: its chains, each
// stage alone and every rule its filters have -- the saturated samples, the
// clips, the [-1, 1] clamp, a masked neighbour's taps, the Gaussian's
// 26-block halo -- against the CPU oracle (codec_deblock_oracle.hpp);
// workgroups run one at a time in list order and in reverse, and in chunks,
// each the same bits, so a write another workgroup reads is a deterministic
// failure; the stamps; recording without submitting; the refusals; and the
// step-keyed rule on the host. The per-block fallback: a block with one
// unobserved voxel, anywhere, keeps its exact decode and leaves both sides of
// its faces unfitted; one masked block among many is kept to the bit, faces
// and all, the rest the oracle's filter, and differs from the filter without
// the fallback only within the line fit's reach of it; a fully masked frame
// is left as it was, stamps too; and the count of the blocks kept. The
// fixtures' blocks sit at slots past twice their count, and their lists
// carry another block's ptr, so a kernel that trusts a ptr or indexes by
// slot fails. And the Decoder's switch: off is the exact decode; on, each
// frame is the oracle's filter of it, under the filter its header picks, its
// masked blocks kept to the bit: mask-free frames, for I- and P-frames,
// toggled between frames, across filters in one decoder and rebuilt while
// paused; masked chains, interleaved with mask-free frames too; a frame of
// 2n blocks with one masked block, a single unobserved voxel, with n, and
// with every block masked, which is the exact decode; and a true-SDF frame,
// its cut blocks kept; each with its report, the blocks kept among them; the
// stamps; and the rows. Skips with no device.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "codec_deblock_oracle.hpp"
#include "codec_fixture.hpp"
#include "deblock.hpp"
#include "host_slices.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/recon/codec/decode_grid.hpp"
#include "volumetric_kit/recon/codec/decoder.hpp"
#include "volumetric_kit/recon/codec/encoder.hpp"
#include "volumetric_kit/recon/core/dispatch.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"

#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/recon/core/device_requirements.hpp"

namespace vr = volumetric_kit::recon;
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
// T = 0.01), plus noise, every voxel observed.
Voxel quantized_ramp(const Where& at) {
  const vr::Vec3i g = at.global();
  const vr::Vec3i& b = at.block;
  const std::uint32_t bx = std::uint32_t(b.x + 16),
                      by = std::uint32_t(b.y + 16),
                      bz = std::uint32_t(b.z + 16);
  const std::uint32_t gx = std::uint32_t(g.x + 64),
                      gy = std::uint32_t(g.y + 64),
                      gz = std::uint32_t(g.z + 64);
  const float offset = 0.04f * (hash01(bx, by, bz + 7) - 0.5f);
  const float slope = 0.004f * (hash01(bx + 3, by, bz) - 0.5f);
  const float noise = 0.002f * (hash01(gx + 11, gy, gz) - 0.5f);
  return Voxel{true, 0.004f * float(g.x) - 0.003f * float(g.y) +
                         0.002f * float(g.z) - 0.02f + offset +
                         slope * (float(at.x) - 3.5f) + noise};
}

// The ramp fixture's masked blocks (kRampMasked): 10% of block (1,1,1)'s
// voxels, the plane x = 13 of the blocks (1, y, 0), and block (3,0,0)'s voxel
// (0,4,4), on its face to (2,0,0), unobserved; the rest of quantized_ramp.
Voxel ramp_field(const Where& at) {
  const vr::Vec3i g = at.global();
  const vr::Vec3i& b = at.block;
  const bool partial = b == vr::Vec3i(1, 1, 1) &&
                       hash01(std::uint32_t(g.x + 64), std::uint32_t(g.y + 64),
                              std::uint32_t(g.z + 64)) < 0.1f;
  const bool plane = g.x == 13 && b.z == 0;
  const bool single =
      b == vr::Vec3i(3, 0, 0) && at.x == 0 && at.y == 4 && at.z == 4;
  if (partial || plane || single) return Voxel{false, 0.0f};
  return quantized_ramp(at);
}
constexpr std::uint32_t kRampMasked = 5;

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
  VKC_ASSIGN(vkc::Buffer list_buffer,
             vkc::test::upload_device_buffer(gpu.device, gpu.allocator,
                                             poisoned.data(),
                                             n * sizeof(vol::BlockIndex)));
  VKC_ASSIGN(
      vkc::Buffer mask_buffer,
      vkc::test::upload_device_buffer(gpu.device, gpu.allocator, masks.data(),
                                      masks.size() * sizeof(std::uint32_t)));
  return Fixture{std::move(grid), std::move(order), std::move(list_buffer),
                 std::move(mask_buffer)};
}

// @p fx's grid's block slots.
std::uint32_t slots(const Fixture& fx) {
  return std::uint32_t(fx.grid.grid().num_blocks);
}

// The deblock's slices for a frame of @p n blocks, as the decoder's gate
// writes them for a frame it passed.
struct Slices {
  vr_test::HostSlices blocks;
  vr_test::HostSlices flags;
};

vkc::Result<Slices> slices_for(Gpu& gpu, std::uint32_t n) {
  Slices out;
  VKC_ASSIGN(out.blocks, vr_test::host_slices(gpu, n, 1));
  VKC_ASSIGN(out.flags, vr_test::host_slices(gpu, n, d::Deblock::kFlagLanes));
  return out;
}

// @p deblock's record of @p frame under @p passes into @p batch, over
// @p slices.
vkc::Status record(d::Deblock& deblock, vkc::CommandBatch& batch,
                   vol::VoxelBlockGrid& grid, const d::ResidentBlocks& frame,
                   const d::DeblockPasses& passes, const Slices& slices) {
  return deblock.record(batch, grid, frame, passes, slices.blocks.slices(),
                        slices.flags.slices());
}

// The blocks the last record's passes kept, once its batch ran.
vkc::Result<std::uint32_t> exact_of(Gpu& gpu, const d::Deblock& deblock) {
  VKC_ASSIGN(const std::vector<std::uint32_t> words,
             vkc::test::read_back<std::uint32_t>(gpu.device, gpu.allocator,
                                                 deblock.exact(), 1));
  return words[0];
}

// One deblock of @p fx under @p passes, at a fresh tick, in a batch of its
// own; the blocks it kept go to @p exact.
vkc::Status run(Gpu& gpu, d::Deblock& deblock, Fixture& fx,
                const d::DeblockPasses& passes,
                std::uint32_t* exact = nullptr) {
  fx.grid.map().advance_tick();
  const auto n = std::uint32_t(fx.coords.size());
  VKC_TRY(deblock.reserve(n, slots(fx), passes));
  VKC_ASSIGN(const Slices slices, slices_for(gpu, n));
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  VKC_TRY(record(deblock, batch, fx.grid, fx.frame(), passes, slices));
  VKC_TRY(batch.submit());
  if (exact != nullptr) {
    VKC_ASSIGN(*exact, exact_of(gpu, deblock));
  }
  return {};
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

// Whether every masked block of @p before -- one holding an unobserved voxel
// -- holds in @p after exactly the bits it held.
bool masked_kept(const Snapshot& after, const Snapshot& before) {
  for (std::size_t b = 0; b < before.coords.size(); ++b) {
    if (oracle::masked(before, b) &&
        std::memcmp(&after.tsdf[b * kVpb], &before.tsdf[b * kVpb],
                    kVpb * sizeof(float)) != 0) {
      return false;
    }
  }
  return true;
}

// The masked blocks of @p s.
std::uint32_t masked_count(const Snapshot& s) {
  std::uint32_t out = 0;
  for (std::size_t b = 0; b < s.coords.size(); ++b) {
    out += oracle::masked(s, b) ? 1u : 0u;
  }
  return out;
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

// @p passes on a fresh ramp fixture against the oracle, the filter moving
// the field by far more than the tolerance, the masked blocks kept to the
// bit and counted, and the filter without the fallback far off. The result
// goes to @p out.
int against_oracle(Gpu& gpu, d::Deblock& deblock,
                   const d::DeblockPasses& passes, const char* name,
                   Snapshot* out = nullptr) {
  vkc::Result<Fixture> made = make_fixture(gpu, ramp_blocks(), ramp_field);
  CHECK(made.ok());
  Fixture fx = std::move(made).value();
  vkc::Result<Snapshot> before = snapshot(gpu, fx.grid);
  CHECK(before.ok());
  CHECK(masked_count(before.value()) == kRampMasked);
  std::uint32_t exact = 0;
  CHECK(run(gpu, deblock, fx, passes, &exact).ok());
  CHECK(exact == kRampMasked);
  vkc::Result<Snapshot> dev = snapshot(gpu, fx.grid);
  CHECK(dev.ok());
  CHECK(masked_kept(dev.value(), before.value()));
  Snapshot want = before.value();
  oracle::apply(want, passes, kBand);
  double diff = 0.0;
  CHECK(matches(dev.value(), want, &diff) == 0);
  Snapshot unguarded = before.value();
  oracle::apply(unguarded, passes, kBand, false);
  const double moved = largest_change(dev.value(), before.value());
  const double fallback = largest_change(dev.value(), unguarded);
  std::printf(
      "%s: moves the field by up to %.3g, the oracle within %.3g, %.3g from "
      "the filter without the fallback\n",
      name, moved, diff, fallback);
  CHECK(moved >= 100 * kTolerance);
  CHECK(fallback >= 100 * kTolerance);
  if (out != nullptr) *out = std::move(dev).value();
  return 0;
}

// 1, 2. LF, LG and R, and each stage alone, against the oracle on the ramp:
// masked blocks -- 10% unobserved, an unobserved plane, a single voxel --
// beside fully observed ones, a missing face, edge and corner neighbour. LF
// writes no voxel a line cannot reach: local 2..5 on every axis.
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

// Whether @p s, a fixture's grid as it was, is the same grid after a run:
// its blocks and fields, and its stamps against @p stamps.
int unchanged(Gpu& gpu, Fixture& fx, const Snapshot& s,
              const std::vector<vol::BlockStamp>& stamps) {
  vkc::Result<Snapshot> after = snapshot(gpu, fx.grid);
  vkc::Result<std::vector<vol::BlockStamp>> stamps_after =
      fx.grid.map().read_block_stamps();
  CHECK(after.ok() && stamps_after.ok());
  CHECK(after.value() == s);
  CHECK(stamps_after.value().size() == stamps.size());
  for (std::size_t i = 0; i < stamps.size(); ++i) {
    CHECK(stamps_after.value()[i].changed == stamps[i].changed);
  }
  return 0;
}

// Two blocks along x, a zero field but for lines set by hand, which only the
// x pass reaches.
struct Lines {
  float p[3];  // block (0,0,0)'s voxels at x = 7, 6, 5
  float q[3];  // block (1,0,0)'s at x = 0, 1, 2
};
// Lines (u, w) = (2 + i, 3) of the two blocks: y = 2 + i, z = 3; with
// @p hole, that voxel unobserved, which masks its block.
Field lines_field(const std::vector<Lines>& lines,
                  const Where* hole = nullptr) {
  return [lines, hole = hole == nullptr
                            ? std::vector<Where>{}
                            : std::vector<Where>{*hole}](const Where& at) {
    for (const Where& h : hole) {
      if (h.block == at.block && h.x == at.x && h.y == at.y && h.z == at.z) {
        return Voxel{false, 0.0f};
      }
    }
    const int i = at.y - 2;
    if (at.z != 3 || i < 0 || i >= int(lines.size())) return Voxel{true, 0.0f};
    const Lines& line = lines[std::size_t(i)];
    const bool left = at.block.x == 0;
    const int s = left ? 7 - at.x : at.x;
    if (s > 2) return Voxel{true, 0.0f};
    return Voxel{true, left ? line.p[s] : line.q[s]};
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

// 3, 4. A fitted sample at exactly 1 - 1e-5 skips its line and one an ulp
// below does not; the clamp to [-1, 1]; and the line clip, which moves p0 by
// exactly 0.375 of it. Then the fallback on the face: one unobserved voxel
// in either block -- p2 of a line, q2, or the far corner of a block, no
// line's sample -- masks it, and no line of the face is fitted, both sides
// left as they were, to the bit, stamps too, and the block counted; and a
// block the frame does not list is no fully observed neighbour.
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
    // 2: a jump of 0.5, clipped.
    lines.push_back(Lines{{0.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f}});

    vkc::Result<Fixture> made = make_fixture(
        gpu, {vr::Vec3i(0, 0, 0), vr::Vec3i(1, 0, 0)}, lines_field(lines));
    CHECK(made.ok());
    Fixture fx = std::move(made).value();
    vkc::Result<Snapshot> before = snapshot(gpu, fx.grid);
    CHECK(before.ok());
    const float t = 0.05f;  // line clip 0.2: lines 0..1 unclipped
    const d::DeblockPasses passes = linfit_only(t);
    std::uint32_t exact = 1;
    CHECK(run(gpu, deblock, fx, passes, &exact).ok());
    CHECK(exact == 0);
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
    const std::vector<float> clipped = read_line(after.value(), 2);
    CHECK(std::fabs(double(clipped[0]) - 0.375 * double(passes.line_clip)) <=
          kTolerance);
    CHECK(std::fabs(double(clipped[1]) - 0.125 * double(passes.line_clip)) <=
          kTolerance);
    CHECK(std::fabs(double(clipped[3]) -
                    (0.5 - 0.375 * double(passes.line_clip))) <= kTolerance);
    CHECK(clipped[2] == 0.0f && clipped[5] == 0.5f);
    // Lines 0 and 2 write four voxels each, line 1 four unless at the kink.
    CHECK(changed == (at_kink ? 8u : 12u));
  }

  // The fallback: the clipped line, which would move four voxels, beside a
  // hole in either block.
  const std::vector<Lines> jump = {
      Lines{{0.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f}}};
  const Where holes[] = {{vr::Vec3i(0, 0, 0), 5, 2, 3},
                         {vr::Vec3i(1, 0, 0), 2, 2, 3},
                         {vr::Vec3i(0, 0, 0), 0, 0, 0},
                         {vr::Vec3i(1, 0, 0), 7, 7, 7}};
  for (const Where& hole : holes) {
    vkc::Result<Fixture> made =
        make_fixture(gpu, {vr::Vec3i(0, 0, 0), vr::Vec3i(1, 0, 0)},
                     lines_field(jump, &hole));
    CHECK(made.ok());
    Fixture fx = std::move(made).value();
    vkc::Result<Snapshot> before = snapshot(gpu, fx.grid);
    vkc::Result<std::vector<vol::BlockStamp>> stamps =
        fx.grid.map().read_block_stamps();
    CHECK(before.ok() && stamps.ok());
    for (const char* filter : {"LF", "R"}) {
      std::uint32_t exact = 0;
      CHECK(run(gpu, deblock, fx, d::deblock_passes(row(filter), 0.05f), &exact)
                .ok());
      CHECK(exact == 1);
      if (std::strcmp(filter, "LF") == 0) {
        CHECK(unchanged(gpu, fx, before.value(), stamps.value()) == 0);
      }
    }
    // The Gaussian smooths the fully observed block alone.
    vkc::Result<Snapshot> after = snapshot(gpu, fx.grid);
    CHECK(after.ok() && masked_kept(after.value(), before.value()));
    CHECK(!(after.value() == before.value()));
  }

  // A block the frame does not list reads as masked, though an earlier run
  // flagged its slot: after the face is fitted once, a frame of block
  // (0,0,0) alone leaves the grid as it was, where a second fit would move
  // it again.
  vkc::Result<Fixture> made = make_fixture(
      gpu, {vr::Vec3i(0, 0, 0), vr::Vec3i(1, 0, 0)}, lines_field(jump));
  CHECK(made.ok());
  Fixture fx = std::move(made).value();
  const d::DeblockPasses lf = linfit_only(0.05f);
  CHECK(run(gpu, deblock, fx, lf).ok());
  vkc::Result<Snapshot> fitted = snapshot(gpu, fx.grid);
  vkc::Result<std::vector<vol::BlockStamp>> stamps =
      fx.grid.map().read_block_stamps();
  CHECK(fitted.ok() && stamps.ok());
  CHECK(fx.coords.front() == vr::Vec3i(0, 0, 0));
  d::ResidentBlocks first = fx.frame();
  first.count = 1;
  fx.grid.map().advance_tick();
  vkc::Result<Slices> one = slices_for(gpu, 1);
  CHECK(one.ok());
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  CHECK(record(deblock, batch, fx.grid, first, lf, one.value()).ok());
  CHECK(batch.submit().ok());
  vkc::Result<std::uint32_t> exact = exact_of(gpu, deblock);
  CHECK(exact.ok() && exact.value() == 0);
  CHECK(unchanged(gpu, fx, fitted.value(), stamps.value()) == 0);
  CHECK(run(gpu, deblock, fx, lf).ok());
  vkc::Result<Snapshot> twice = snapshot(gpu, fx.grid);
  CHECK(twice.ok() && !(twice.value() == fitted.value()));
  return 0;
}

// x alone varies: a ramp along x, the plane x = 7 of block (0,0,0)
// unobserved for y >= 4, which masks it; or a spike.
Voxel x_ramp_field(const Where& at) {
  const vr::Vec3i g = at.global();
  if (g.x == 7 && g.y >= 4) return Voxel{false, 0.0f};
  return Voxel{true, 0.02f * float(g.x)};
}
// A spike in masked block (0,1,0), at (3,0,3), on its face to (0,0,0).
Voxel masked_spike_field(const Where& at) {
  if (at.block == vr::Vec3i(0, 1, 0)) {
    if (at.x == 7 && at.y == 7 && at.z == 7) return Voxel{false, 0.0f};
    if (at.x == 3 && at.y == 0 && at.z == 3) return Voxel{true, 0.5f};
  }
  return Voxel{true, 0.0f};
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

// 3, 5, 6. The Gaussian: a masked neighbour's unobserved tap drops out of
// both sums, and its observed one counts as decoded (hand values); a masked
// neighbour's voxels keep their values through every pass, so a full
// block's y pass reads a spike there as decoded, not as the x pass would
// have left it; each pass's change is clipped, so a spike loses three
// clips; and a block's corner reaches its diagonal neighbour's through the
// 26-block halo.
int gauss_rules_case(Gpu& gpu, d::Deblock& deblock) {
  {
    vkc::Result<Fixture> made = make_fixture(
        gpu, {vr::Vec3i(0, 0, 0), vr::Vec3i(1, 0, 0)}, x_ramp_field);
    CHECK(made.ok());
    Fixture fx = std::move(made).value();
    vkc::Result<Snapshot> before = snapshot(gpu, fx.grid);
    CHECK(before.ok());
    const d::DeblockPasses passes = gauss_only(0.1f);
    std::uint32_t exact = 0;
    CHECK(run(gpu, deblock, fx, passes, &exact).ok());
    CHECK(exact == 1);
    vkc::Result<Snapshot> after = snapshot(gpu, fx.grid);
    CHECK(after.ok() && masked_kept(after.value(), before.value()));
    // x = 0 of block (1,0,0), at y 6 and 2: its x - 1 tap is the masked
    // block's unobserved voxel, then its observed one, 0.14. y and z run
    // along lines constant around both.
    const double g1 = passes.gauss_g1;
    const double want_hole = (0.16 + g1 * 0.18) / (1.0 + g1);
    const double want_tap = (g1 * 0.14 + 0.16 + g1 * 0.18) / (1.0 + 2 * g1);
    const double hole = after.value().tsdf[kVpb + 8 * 6 + 64 * 4] / kBand;
    const double tap = after.value().tsdf[kVpb + 8 * 2 + 64 * 4] / kBand;
    std::printf("unobserved tap: %.9f, by hand %.9f; observed: %.9f, %.9f\n",
                hole, want_hole, tap, want_tap);
    CHECK(std::fabs(hole - want_hole) <= kTolerance);
    CHECK(std::fabs(tap - want_tap) <= kTolerance);
    // Counted as the 0 stored there, it would be far off.
    CHECK(std::fabs((g1 * 0.0 + 0.16 + g1 * 0.18) / (1.0 + 2 * g1) -
                    want_hole) >= 100 * kTolerance);
  }
  {
    vkc::Result<Fixture> made = make_fixture(
        gpu, {vr::Vec3i(0, 0, 0), vr::Vec3i(0, 1, 0)}, masked_spike_field);
    CHECK(made.ok());
    Fixture fx = std::move(made).value();
    vkc::Result<Snapshot> before = snapshot(gpu, fx.grid);
    CHECK(before.ok());
    const d::DeblockPasses passes = gauss_only(1.0f);  // clips past 0.5
    CHECK(run(gpu, deblock, fx, passes).ok());
    vkc::Result<Snapshot> after = snapshot(gpu, fx.grid);
    CHECK(after.ok() && masked_kept(after.value(), before.value()));
    Snapshot want = before.value();
    oracle::apply(want, passes, kBand);
    CHECK(matches(after.value(), want, nullptr) == 0);
    // Block (0,0,0)'s (3,7,3), below the spike: y reads it as 0.5, then z
    // spreads the result. Read as x would have left it, it is far off.
    const double g1 = passes.gauss_g1;
    const double y_out = g1 * 0.5 / (1.0 + 2 * g1);
    const double z_out = y_out / (1.0 + 2 * g1);
    const double got = after.value().tsdf[3 + 8 * 7 + 64 * 3] / kBand;
    Snapshot unguarded = before.value();
    oracle::apply(unguarded, passes, kBand, false);
    const double filtered_tap = unguarded.tsdf[3 + 8 * 7 + 64 * 3] / kBand;
    std::printf("masked tap: %.9f, by hand %.9f; filtered first: %.9f\n", got,
                z_out, filtered_tap);
    CHECK(std::fabs(got - z_out) <= kTolerance);
    CHECK(std::fabs(filtered_tap - z_out) >= 100 * kTolerance);
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

// Block (0,0,0) at 0.1 among zeros; block (-1,0,0) masked by one voxel.
Voxel stamps_field(const Where& at) {
  if (at.block == vr::Vec3i(-1, 0, 0) && at.x == 3 && at.y == 3 && at.z == 3) {
    return Voxel{false, 0.0f};
  }
  return Voxel{true, at.block == vr::Vec3i(0, 0, 0) ? 0.1f : 0.0f};
}

// 9. A block the deblock writes carries the pass's tick, whoever wrote it:
// a line's +axis block, written by its neighbour's workgroup alone, and a
// block whose shell alone the Gaussian changes, which the store writes. A
// block it leaves as it was keeps its stamp: one far from the step, and a
// masked one beside it.
int stamps_case(Gpu& gpu, d::Deblock& deblock) {
  for (const bool gaussian : {false, true}) {
    vkc::Result<Fixture> made =
        make_fixture(gpu,
                     {vr::Vec3i(-1, 0, 0), vr::Vec3i(0, 0, 0),
                      vr::Vec3i(1, 0, 0), vr::Vec3i(5, 5, 5)},
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
    CHECK(after.value().coords == before.value().coords);
    for (std::size_t b = 0; b < 4; ++b) {
      const vr::Vec3i& c = after.value().coords[b];
      const bool changed = std::memcmp(&after.value().tsdf[b * kVpb],
                                       &before.value().tsdf[b * kVpb],
                                       kVpb * sizeof(float)) != 0;
      const std::uint32_t slot = std::uint32_t(active.value()[b].ptr) / kVpb;
      CHECK(changed == (c == vr::Vec3i(0, 0, 0) || c == vr::Vec3i(1, 0, 0)));
      CHECK((stamps.value()[slot].changed == tick) == changed);
      if (gaussian && c == vr::Vec3i(1, 0, 0)) {
        // Block (1,0,0)'s interior is untouched: its stamp is the store's.
        for (std::uint32_t l = 0; l < kVpb; ++l) {
          const std::uint32_t x = l % 8, y = l / 8 % 8, z = l / 64;
          if (x > 0 && x < 7 && y > 0 && y < 7 && z > 0 && z < 7) {
            CHECK(after.value().tsdf[b * kVpb + l] == 0.0f);
          }
        }
      }
    }
  }
  return 0;
}

// 10. Recorded, not run: until the batch is submitted the grid is as it was,
// and once it is the count of the blocks kept is the masked blocks'; a
// filter that runs nothing, or no blocks, records nothing, not a fill, so
// the decoder counts every block itself (masked_case); and the refusals,
// each before anything is recorded, a reserve that made no room for the
// blocks or for the grid's slots among them.
int record_case(Gpu& gpu, d::Deblock& deblock) {
  vkc::Result<Fixture> made = make_fixture(gpu, ramp_blocks(), ramp_field);
  CHECK(made.ok());
  Fixture fx = std::move(made).value();
  vkc::Result<Snapshot> before = snapshot(gpu, fx.grid);
  CHECK(before.ok());
  const d::DeblockPasses passes = d::deblock_passes(row("LG"), 0.01f);
  const std::uint32_t n = std::uint32_t(fx.coords.size());
  CHECK(deblock.reserve(n, slots(fx), passes).ok());
  vkc::Result<Slices> all = slices_for(gpu, n);
  CHECK(all.ok());
  {
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    CHECK(
        record(deblock, batch, fx.grid, fx.frame(), passes, all.value()).ok());
    vkc::Result<Snapshot> pending = snapshot(gpu, fx.grid);
    CHECK(pending.ok() && pending.value() == before.value());
    CHECK(batch.submit().ok());
    vkc::Result<std::uint32_t> exact = exact_of(gpu, deblock);
    CHECK(exact.ok() && exact.value() == kRampMasked);
    vkc::Result<Snapshot> after = snapshot(gpu, fx.grid);
    CHECK(after.ok() && !(after.value() == before.value()));
  }
  vkc::Result<Snapshot> filtered = snapshot(gpu, fx.grid);
  CHECK(filtered.ok());
  {
    // No pass, and no block: nothing recorded.
    const vkc::DeviceCounters was = gpu.device.counters();
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    CHECK(record(deblock, batch, fx.grid, fx.frame(), d::DeblockPasses{},
                 all.value())
              .ok());
    d::ResidentBlocks none = fx.frame();
    none.count = 0;
    CHECK(record(deblock, batch, fx.grid, none, passes, all.value()).ok());
    CHECK(batch.submit().ok());
    const vkc::DeviceCounters now = gpu.device.counters();
    CHECK(now.dispatches == was.dispatches && now.barriers == was.barriers);
    vkc::Result<std::uint32_t> exact = exact_of(gpu, deblock);
    CHECK(exact.ok() && exact.value() == kRampMasked);  // the last filter's
    vkc::Result<Snapshot> after = snapshot(gpu, fx.grid);
    CHECK(after.ok() && after.value() == filtered.value());
  }
  const auto refused = [&](d::Deblock& db, const d::ResidentBlocks& frame,
                           const d::DeblockPasses& p, vol::VoxelBlockGrid& grid,
                           const char* says = nullptr) {
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    const vkc::Status s = record(db, batch, grid, frame, p, all.value());
    return s.domain() == vkc::Status::Code::InvalidArgument &&
           (says == nullptr || s.message().find(says) != std::string::npos);
  };
  d::DeblockPasses unclipped = passes;
  unclipped.line_clip = 0.0f;
  CHECK(refused(deblock, fx.frame(), unclipped, fx.grid));
  unclipped = passes;
  unclipped.gauss_clip = 0.0f;
  CHECK(refused(deblock, fx.frame(), unclipped, fx.grid));
  d::ResidentBlocks inter = fx.frame();
  inter.reference_count = 1;
  CHECK(refused(deblock, inter, passes, fx.grid));
  d::ResidentBlocks short_list = fx.frame();
  short_list.count = n + 1;
  CHECK(refused(deblock, short_list, linfit_only(0.01f), fx.grid));
  d::ResidentBlocks no_masks = fx.frame();
  no_masks.masks = nullptr;
  CHECK(refused(deblock, no_masks, passes, fx.grid));
  // More blocks, or slots, than reserve made room for: the snapshot under a
  // Gaussian, the flags under any filter.
  vkc::Result<std::unique_ptr<d::Deblock>> fresh =
      d::Deblock::create(gpu.device, gpu.allocator);
  CHECK(fresh.ok());
  d::Deblock& small = *fresh.value();
  CHECK(refused(small, fx.frame(), passes, fx.grid, "reserve"));
  CHECK(refused(small, fx.frame(), linfit_only(0.01f), fx.grid, "slots"));
  CHECK(small.reserve(n, slots(fx), linfit_only(0.01f)).ok());
  CHECK(!refused(small, fx.frame(), linfit_only(0.01f), fx.grid));
  CHECK(refused(small, fx.frame(), passes, fx.grid, "blocks, more than"));
  // A grid of another block size.
  GridShape four;
  four.block_size = 4;
  vkc::Result<vol::VoxelBlockGrid> g4 = make_grid(gpu, four);
  CHECK(g4.ok());
  CHECK(refused(deblock, fx.frame(), passes, g4.value()));
  return 0;
}

// The fallback's cube: 5 x 5 x 5 blocks of quantized_ramp, block kHole's
// voxel (4,3,5) unobserved -- one masked block among 125, list entry 87, in
// the flags kernel's second workgroup, or 37 reversed -- or, with @p all,
// every block's, or none.
const vr::Vec3i kHole(2, 2, 3);

std::vector<vr::Vec3i> cube_blocks() {
  std::vector<vr::Vec3i> out;
  for (int z = 0; z < 5; ++z) {
    for (int y = 0; y < 5; ++y) {
      for (int x = 0; x < 5; ++x) out.push_back(vr::Vec3i(x, y, z));
    }
  }
  return out;
}

enum class Holes { kNone, kOne, kAll };

Field cube_field(Holes holes) {
  return [holes](const Where& at) {
    const bool holed =
        holes == Holes::kAll || (holes == Holes::kOne && at.block == kHole);
    if (holed && at.x == 4 && at.y == 3 && at.z == 5) {
      return Voxel{false, 0.0f};
    }
    return quantized_ramp(at);
  };
}

// Voxel i of a snapshot of @p coords, in global voxels.
vr::Vec3i global_voxel(const std::vector<vr::Vec3i>& coords, std::size_t i) {
  const Where w{coords[i / kVpb], int(i % 8), int(i / 8 % 8),
                int(i % kVpb / 64)};
  return w.global();
}

// The Chebyshev distance, in voxels, from global voxel @p g to block @p b.
int distance_to_block(const vr::Vec3i& g, const vr::Vec3i& b) {
  const auto axis = [](int v, int lo) {
    return v < lo ? lo - v : (v > lo + 7 ? v - lo - 7 : 0);
  };
  return std::max({axis(g.x, 8 * b.x), axis(g.y, 8 * b.y), axis(g.z, 8 * b.z)});
}

// 10b. The per-block fallback on the cube. One masked block, a single
// unobserved voxel: under LF, LG and R, in list order and reversed, in one
// dispatch and a workgroup a dispatch, it is kept to the bit and its stamp
// as it was, the rest is the oracle's filter, the same bits every run, and
// one block is counted kept. Against the filter without the fallback it
// differs, and only within the line fit's reach of the masked block -- 2
// voxels, 5 with a Gaussian -- and under LF the voxels of each face beside
// it that only a line across that face can reach, 1 and 2 voxels from it,
// keep their bits, which the filter without the fallback moves. Every block
// masked, the grid and its stamps are as they were and all 125 counted; none
// masked, the oracle's filter with none counted, as without the fallback.
int fallback_case(Gpu& gpu, d::Deblock& deblock) {
  const float t = 0.01f;
  const std::vector<vr::Vec3i> coords = cube_blocks();
  for (const char* filter : {"LF", "LG", "R"}) {
    const d::DeblockPasses passes = d::deblock_passes(row(filter), t);
    const bool lf = std::strcmp(filter, "LF") == 0;
    Snapshot first;
    for (const bool reverse : {false, true}) {
      for (const std::uint32_t per_dispatch : {0u, 1u}) {
        vkc::Result<Fixture> made =
            make_fixture(gpu, coords, cube_field(Holes::kOne), reverse);
        CHECK(made.ok());
        Fixture fx = std::move(made).value();
        vkc::Result<Snapshot> before = snapshot(gpu, fx.grid);
        vkc::Result<std::vector<std::uint32_t>> stamps_before =
            stamps_by_coord(fx.grid);
        CHECK(before.ok() && stamps_before.ok());
        CHECK(masked_count(before.value()) == 1);
        vr::detail::set_test_groups_per_dispatch(per_dispatch);
        std::uint32_t exact = 0;
        const vkc::Status ran = run(gpu, deblock, fx, passes, &exact);
        vr::detail::set_test_groups_per_dispatch(0);
        CHECK(ran.ok() && exact == 1);
        vkc::Result<Snapshot> after = snapshot(gpu, fx.grid);
        vkc::Result<std::vector<std::uint32_t>> stamps_after =
            stamps_by_coord(fx.grid);
        CHECK(after.ok() && stamps_after.ok());
        const Snapshot& dev = after.value();
        CHECK(masked_kept(dev, before.value()));
        const std::size_t hole = std::size_t(
            std::find(coords.begin(), coords.end(), kHole) - coords.begin());
        CHECK(dev.coords == coords);
        CHECK(stamps_after.value()[hole] == stamps_before.value()[hole]);
        Snapshot want = before.value();
        oracle::apply(want, passes, kBand);
        CHECK(matches(dev, want, nullptr) == 0);
        if (first.coords.empty()) {
          first = dev;
        } else {
          CHECK(dev == first);
        }
        Snapshot unguarded = before.value();
        oracle::apply(unguarded, passes, kBand, false);
        const int reach = lf ? 2 : 5;
        double inside = 0.0, slab_moved = 0.0;
        for (std::size_t i = 0; i < dev.tsdf.size(); ++i) {
          const double diff =
              std::fabs(double(dev.tsdf[i]) - double(unguarded.tsdf[i])) /
              double(kBand);
          if (diff <= kTolerance) continue;
          CHECK(distance_to_block(global_voxel(coords, i), kHole) <= reach);
          inside = std::max(inside, diff);
        }
        CHECK(inside >= 100 * kTolerance);
        if (lf) {
          // Each face neighbour's voxels 1 and 2 from the face, 2..5 along
          // the face: on no other line of any pass.
          for (std::size_t b = 0; b < coords.size(); ++b) {
            const vr::Vec3i o = coords[b] - kHole;
            if (std::abs(o.x) + std::abs(o.y) + std::abs(o.z) != 1) continue;
            for (std::uint32_t l = 0; l < kVpb; ++l) {
              const int p[3] = {int(l % 8), int(l / 8 % 8), int(l / 64)};
              const int n[3] = {o.x, o.y, o.z};
              bool slab = true;
              for (int a = 0; a < 3; ++a) {
                const int from = n[a] > 0 ? p[a] : 7 - p[a];
                slab = slab && (n[a] != 0 ? from <= 1 : p[a] >= 2 && p[a] <= 5);
              }
              if (!slab) continue;
              const std::size_t i = b * kVpb + l;
              CHECK(std::memcmp(&dev.tsdf[i], &before.value().tsdf[i],
                                sizeof(float)) == 0);
              slab_moved = std::max(slab_moved,
                                    std::fabs(double(unguarded.tsdf[i]) -
                                              double(before.value().tsdf[i])) /
                                        double(kBand));
            }
          }
          CHECK(slab_moved >= 100 * kTolerance);
        }
        if (!reverse && per_dispatch == 0) {
          std::printf(
              "%s, one masked block: %.3g from the filter without the "
              "fallback, within %d voxels of it\n",
              filter, inside, reach);
          if (lf) {
            std::printf("  its faces' slabs kept, %.3g without the fallback\n",
                        slab_moved);
          }
        }
      }
    }
  }
  for (const Holes holes : {Holes::kAll, Holes::kNone}) {
    for (const char* filter : {"LF", "LG", "R"}) {
      const d::DeblockPasses passes = d::deblock_passes(row(filter), t);
      vkc::Result<Fixture> made = make_fixture(gpu, coords, cube_field(holes));
      CHECK(made.ok());
      Fixture fx = std::move(made).value();
      vkc::Result<Snapshot> before = snapshot(gpu, fx.grid);
      vkc::Result<std::vector<vol::BlockStamp>> stamps =
          fx.grid.map().read_block_stamps();
      CHECK(before.ok() && stamps.ok());
      vr::detail::set_test_groups_per_dispatch(1);
      std::uint32_t exact = 0;
      const vkc::Status ran = run(gpu, deblock, fx, passes, &exact);
      vr::detail::set_test_groups_per_dispatch(0);
      CHECK(ran.ok());
      if (holes == Holes::kAll) {
        CHECK(exact == coords.size());
        CHECK(unchanged(gpu, fx, before.value(), stamps.value()) == 0);
        continue;
      }
      CHECK(exact == 0);
      vkc::Result<Snapshot> after = snapshot(gpu, fx.grid);
      CHECK(after.ok());
      Snapshot want = before.value();
      oracle::apply(want, passes, kBand);
      Snapshot unguarded = before.value();
      oracle::apply(unguarded, passes, kBand, false);
      CHECK(want == unguarded);
      CHECK(matches(after.value(), want, nullptr) == 0);
      CHECK(largest_change(after.value(), before.value()) >= 100 * kTolerance);
    }
  }
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
// does: a masked frame, whose masked blocks it keeps exact.
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
// masked blocks, weights and blocks exactly the exact decode's, and the
// field moved.
int deblocked(const Snapshot& on, const Snapshot& exact,
              const d::DeblockRow& row, float t, float trunc) {
  Snapshot want = exact;
  oracle::apply(want, d::deblock_passes(row, t), trunc);
  CHECK(on.coords == want.coords && on.weight == want.weight);
  CHECK(masked_kept(on, exact));
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

// @p got, a decoder's report, against @p exact, the frame's exact decode:
// @p filter, the frame's blocks, and of them those kept -- the masked ones,
// or every one when no filter ran.
int reported(const codec::DeblockInfo& got, const Snapshot& exact,
             const char* filter) {
  CHECK(std::strcmp(got.filter, filter) == 0);
  CHECK(got.block_count == exact.coords.size());
  const bool off = std::strcmp(filter, "off") == 0;
  CHECK(got.exact_blocks ==
        (off ? std::uint32_t(exact.coords.size()) : masked_count(exact)));
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

// Off by default, and off is the exact decode; on, each mask-free frame
// takes the filter its header picks -- none at a fine step, LF, LG, R by T,
// and R at a small K -- the oracle's, and not the filter of the next row, no
// block kept; the switch read at each decode; and the reports.
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
        fresh.value().last_deblock().step == 0.0f &&
        fresh.value().last_deblock().block_count == 0 &&
        fresh.value().last_deblock().exact_blocks == 0);
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
    CHECK(reported(got, exact.value(), setting.filter) == 0);
    CHECK(got.step == t * kTrunc);
    CHECK(reported(off.last_deblock(), exact.value(), "off") == 0);
    CHECK(off.last_deblock().step == got.step);
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
    CHECK(reported(on.last_deblock(), exact.value(), "off") == 0);
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
  CHECK(reported(on.last_deblock(), none.value(), "LG") == 0);
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
    CHECK(reported(on_dec.value().last_deblock(), exact.value(), "LG") == 0);
    CHECK(reported(toggled_dec.value().last_deblock(), exact.value(),
                   toggled_on ? "LG" : "off") == 0);
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
    CHECK(reported(on_dec.value().last_deblock(), exact.value(), f.filter) ==
          0);
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
  CHECK(reported(dec.last_deblock(), exact.value(), "LG") == 0);
  CHECK(find_row(m, "codec decode") != nullptr);
  dec.set_deblock(false);
  CHECK(dec.rebuild(grid).ok());
  vkc::Result<Snapshot> rebuilt_off = snapshot(gpu, grid);
  CHECK(rebuilt_off.ok() && rebuilt_off.value() == exact.value());
  CHECK(reported(dec.last_deblock(), exact.value(), "off") == 0);

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

// Masked frames -- the moving spheres with the voxels past the band
// unobserved, as fused depth leaves them, most of their blocks masked --
// with the switch on: I- and P-frames each the oracle's filter of the exact
// decode, the masked blocks kept to the bit and their stamps the exact
// decode's, reported by the rule's filter with the masked blocks kept;
// rebuilt with the switch on, the decode again; and an empty frame after
// them reports no block kept, decoded and rebuilt.
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
  Snapshot last_on;
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
    const std::uint32_t masked = masked_count(exact.value());
    std::printf("masked frame %d: %u of %zu blocks masked\n", t, masked,
                exact.value().coords.size());
    CHECK(2 * masked > exact.value().coords.size());
    float step = 0.0f;
    const d::DeblockRow* row = picked(info.value(), &step);
    CHECK(row != nullptr);
    CHECK(deblocked(on.value(), exact.value(), *row, step, kTrunc) == 0);
    for (std::size_t b = 0; b < exact.value().coords.size(); ++b) {
      if (oracle::masked(exact.value(), b)) {
        CHECK(on_stamps.value()[b] == off_stamps.value()[b]);
      }
    }
    const codec::DeblockInfo& got = on_dec.value().last_deblock();
    CHECK(reported(got, exact.value(), row->filter) == 0);
    CHECK(got.step == step * kTrunc);
    CHECK(off_dec.value().last_deblock().step == got.step);
    last_on = std::move(on).value();
  }
  CHECK(p_frames == 4);
  const codec::DeblockInfo decoded = on_dec.value().last_deblock();
  CHECK(on_dec.value().rebuild(on_grid.value()).ok());
  vkc::Result<Snapshot> rebuilt = snapshot(gpu, on_grid.value());
  CHECK(rebuilt.ok() && rebuilt.value() == last_on);
  const codec::DeblockInfo& again = on_dec.value().last_deblock();
  CHECK(std::strcmp(again.filter, decoded.filter) == 0 &&
        again.exact_blocks == decoded.exact_blocks &&
        again.block_count == decoded.block_count);

  // An empty frame whose header picks a filter: no block, so none kept,
  // whatever the masked frame before left in the deblock's count.
  vkc::Result<vol::VoxelBlockGrid> empty_source = player_grid(gpu);
  CHECK(empty_source.ok());
  vkc::Result<Bytes> empty = encode_at(gpu, empty_source.value(), 64, 0.2f);
  CHECK(empty.ok() && decoded.exact_blocks > 0);
  CHECK(on_dec.value()
            .decode(empty.value().data(), empty.value().size(), on_grid.value())
            .ok());
  const codec::DeblockInfo& none = on_dec.value().last_deblock();
  CHECK(std::strcmp(none.filter, "LG") == 0 && none.block_count == 0 &&
        none.exact_blocks == 0);
  CHECK(on_dec.value().rebuild(on_grid.value()).ok());
  CHECK(on_dec.value().last_deblock().exact_blocks == 0);
  return 0;
}

// The fallback's decoder fixture: a mask-free sphere's band blocks, the last
// dropped when they are odd, so its frames have 2n.
std::vector<vr::Vec3i> share_coords(const Sphere& s) {
  std::vector<vr::Vec3i> coords = union_blocks({s});
  if (coords.size() % 2 != 0) coords.pop_back();
  return coords;
}

// share_coords' grid with @p k of its blocks masked: @p holes voxels of
// each, one or two, in two x-lines, unobserved. The masked blocks are every
// other one in (z, y, x) order, then those between.
vkc::Result<vol::VoxelBlockGrid> share_grid(Gpu& gpu, const Sphere& s,
                                            std::size_t k, std::size_t holes) {
  VKC_ASSIGN(vol::VoxelBlockGrid grid, make_grid(gpu));
  VKC_TRY(allocate(gpu, grid, share_coords(s)));
  VKC_TRY(write_field(
      gpu, grid, [&s](const vr::Vec3f& p) { return s.sdf(p); }, true));
  VKC_ASSIGN(const std::vector<vol::BlockIndex> active, active_sorted(grid));
  VKC_ASSIGN(
      std::vector<float> tsdf,
      vr_test::read_attribute<float>(gpu.device, gpu.allocator, grid, "tsdf"));
  VKC_ASSIGN(std::vector<float> weight,
             vr_test::read_attribute<float>(gpu.device, gpu.allocator, grid,
                                            "weight"));
  const std::size_t n = active.size();
  const std::size_t evens = (n + 1) / 2;
  // Local (3, 4, 5), then (5, 1, 2).
  const std::uint32_t at_local[2] = {3u + 8u * 4u + 64u * 5u, 5u + 8u + 128u};
  for (std::size_t i = 0; i < k && i < n; ++i) {
    const std::size_t at = i < evens ? 2 * i : 2 * (i - evens) + 1;
    const std::uint32_t base = std::uint32_t(active[at].ptr);
    for (std::size_t h = 0; h < holes; ++h) {
      tsdf[base + at_local[h]] = 0.0f;
      weight[base + at_local[h]] = 0.0f;
    }
  }
  VKC_TRY(
      vr_test::write_attribute(gpu.device, gpu.allocator, grid, "tsdf", tsdf));
  VKC_TRY(vr_test::write_attribute(gpu.device, gpu.allocator, grid, "weight",
                                   weight));
  return grid;
}

// The fallback on a decoded frame of 2n blocks: with one block masked by a
// single unobserved voxel, and with n masked, each by two, the frame is the
// oracle's filter of its exact decode, its masked blocks kept to the bit;
// with all 2n masked it is the exact decode. Each reports the rule's filter
// and the masked blocks kept, rebuilt while paused is its decode, and
// rebuilt with the switch off the exact decode, reported "off". Two holes a
// block in two x-lines, so a count by word or line rather than by block
// would report more.
int share_case(Gpu& gpu) {
  const Sphere s{vr::Vec3f(0.006f, -0.004f, 0.002f), 0.09f};
  const std::size_t blocks = share_coords(s).size();
  const std::size_t n = blocks / 2;
  CHECK(n >= 2);
  for (const std::size_t k : {std::size_t(1), n, blocks}) {
    const std::size_t holes = k == 1 ? 1 : 2;
    vkc::Result<vol::VoxelBlockGrid> source = share_grid(gpu, s, k, holes);
    CHECK(source.ok());
    vkc::Result<Bytes> frame = encode_at(gpu, source.value(), 64, 0.2f);
    CHECK(frame.ok());
    vkc::Result<codec::FrameInfo> info =
        codec::read_frame_info(frame.value().data(), frame.value().size());
    vkc::Result<codec::Decoder> on_dec = decoder(gpu, true);
    vkc::Result<codec::Decoder> off_dec = decoder(gpu, false);
    vkc::Result<vol::VoxelBlockGrid> on_grid = player_grid(gpu);
    vkc::Result<vol::VoxelBlockGrid> off_grid = player_grid(gpu);
    CHECK(info.ok() && on_dec.ok() && off_dec.ok() && on_grid.ok() &&
          off_grid.ok());
    CHECK(info.value().block_count == blocks);
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
    CHECK(exact.value().coords.size() == blocks);
    CHECK(masked_count(exact.value()) == k);
    CHECK(std::size_t(std::count(exact.value().weight.begin(),
                                 exact.value().weight.end(), 0.0f)) ==
          holes * k);
    float step = 0.0f;
    const d::DeblockRow* row = picked(info.value(), &step);
    CHECK(row != nullptr && std::strcmp(row->filter, "LG") == 0);
    const codec::DeblockInfo got = on_dec.value().last_deblock();
    std::printf("%zu of %zu blocks masked: %s, %u of %u blocks exact\n", k,
                blocks, got.filter, got.exact_blocks, got.block_count);
    CHECK(reported(got, exact.value(), "LG") == 0);
    if (k == blocks) {
      CHECK(on.value() == exact.value());
      CHECK(got.exact_blocks == got.block_count);
    } else {
      CHECK(deblocked(on.value(), exact.value(), *row, step, kTrunc) == 0);
    }
    CHECK(on_dec.value().rebuild(on_grid.value()).ok());
    vkc::Result<Snapshot> rebuilt = snapshot(gpu, on_grid.value());
    CHECK(rebuilt.ok() && rebuilt.value() == on.value());
    CHECK(reported(on_dec.value().last_deblock(), exact.value(), "LG") == 0);
    on_dec.value().set_deblock(false);
    CHECK(on_dec.value().rebuild(on_grid.value()).ok());
    rebuilt = snapshot(gpu, on_grid.value());
    CHECK(rebuilt.ok() && rebuilt.value() == exact.value());
    CHECK(reported(on_dec.value().last_deblock(), exact.value(), "off") == 0);
  }
  return 0;
}

// One decoder, the switch on, over a chain whose frames alternate between
// mask-free and mostly masked, I- and P-frames of each: every frame the
// oracle's filter of its exact decode, its own masked blocks kept, with its
// report, so nothing of one frame's flags or count is left for the next.
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
    CHECK(deblocked(on.value(), exact.value(), *row, step, kTrunc) == 0);
    CHECK(reported(on_dec.value().last_deblock(), exact.value(), row->filter) ==
          0);
  }
  for (const auto& by_type : kinds) {
    CHECK(by_type[0] >= 1 && by_type[1] >= 1);
  }
  return 0;
}

// The block ceiling the player's helpers take: far past any fixture's.
constexpr std::uint32_t kMaxBlocks = 1u << 20;

// True-SDF frames (EncoderConfig::true_sdf) at the mesh templates' 16-voxel
// band, each the oracle's filter of its exact decode, reported by its
// filter: a sphere's, every mask all ones, and a curtain's, whose sign
// errors are cut and coded unobserved in a minority of its blocks, as the
// mesh templates' frames with a cut have them, those blocks kept.
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
    std::printf("true SDF %s: %u of %zu blocks masked\n",
                cut ? "curtain" : "sphere", masked_count(exact.value()),
                exact.value().coords.size());
    float step = 0.0f;
    const d::DeblockRow* row = picked(info.value(), &step);
    CHECK(row != nullptr);
    CHECK(deblocked(on.value(), exact.value(), *row, step,
                    info.value().trunc_dist) == 0);
    CHECK(reported(on_dec.value().last_deblock(), exact.value(), row->filter) ==
          0);
  }
  return 0;
}

// On, the rows are the decode's: "codec decode", its device half the
// deblock's dispatches too, over the same three breakdowns, no other. Timed,
// the three tile it, so the deblock's dispatches are a breakdown's:
// "  ..inverse"'s, whose submit they share.
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
    const vkc::StageRow* top = find_row(m, "codec decode");
    CHECK(top->has_gpu);
    double sum = 0.0;
    for (std::size_t i = 1; i < want.size(); ++i) {
      const vkc::StageRow* row = find_row(m, want[i].c_str());
      CHECK(row->device == vkc::DeviceTime::Measured);
      sum += row->gpu_ms;
    }
    std::fprintf(stderr, "decode: device %.4f ms, ..inverse %.4f\n",
                 top->gpu_ms, find_row(m, "  ..inverse")->gpu_ms);
    CHECK(std::fabs(sum - top->gpu_ms) < 1e-9);
    CHECK(m.total_gpu_ms() == top->gpu_ms);
  }
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
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
  if (fallback_case(gpu, deblock) != 0) return 1;
  if (switch_case(gpu) != 0) return 1;
  if (chain_case(gpu) != 0) return 1;
  if (mixed_filters_case(gpu) != 0) return 1;
  if (rebuild_case(gpu) != 0) return 1;
  if (decoder_stamps_case(gpu) != 0) return 1;
  if (masked_case(gpu) != 0) return 1;
  if (share_case(gpu) != 0) return 1;
  if (interleaved_case(gpu) != 0) return 1;
  if (true_sdf_case(gpu) != 0) return 1;
  if (rows_case(gpu) != 0) return 1;
  std::printf("codec deblock: OK\n");
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(vr::device_requirements(), gpu_main);
}
