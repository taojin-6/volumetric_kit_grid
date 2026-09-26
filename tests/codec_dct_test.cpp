// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GPU test for the codec's DctTransform: the forward kernel against a
// double-precision host reference of the same transform, the reconstruction
// bounds an orthonormal transform guarantees, the observed mask, batching, and
// every refusal. Runs on the real driver (MoltenVK / NVIDIA); exits 0 (skip)
// where no device is present.
//
// All block content is written in NORMALIZED units s in [-1, 1] (tsdf =
// s * trunc_dist), the units the transform and the steps are defined in.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <utility>
#include <vector>

#include "dct_tables.hpp"
#include "dct_transform.hpp"
#include "volumetric_kit/recon/codec/codec_params.hpp"
#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/device.hpp"
#include "volumetric_kit/recon/core/instance.hpp"
#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/volume/hash_types.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"
#include "volumetric_kit/recon/volume/voxel_grid.hpp"

namespace vr = volumetric_kit::recon;
namespace vol = volumetric_kit::recon::volume;
namespace codec = volumetric_kit::recon::codec;
using codec::detail::DctTransform;
using codec::detail::DctTransformConfig;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

constexpr float kTrunc = 0.04f;
constexpr std::uint32_t kVpb = codec::kVoxelsPerBlock;
using Cube = std::array<double, kVpb>;

vol::VoxelGridParams small_grid(std::int32_t block_size = 8) {
  vol::VoxelGridParams gp{};
  gp.voxel_size = 0.005f;
  gp.block_size = block_size;
  gp.voxels_per_block = block_size * block_size * block_size;
  gp.trunc_dist = kTrunc;
  gp.bucket_size = 8;
  gp.num_buckets = 256;
  gp.num_blocks = 256 * 8;
  gp.max_chain = 128;
  return gp;
}

vr::Result<vol::VoxelBlockGrid> make_grid(vr::Device& device,
                                          vr::Allocator& allocator,
                                          std::int32_t block_size = 8,
                                          bool with_weight = true) {
  const vol::AttributeSpec attrs[] = {{"tsdf", sizeof(float)},
                                      {"weight", sizeof(float)}};
  return vol::VoxelBlockGrid::create(device, allocator, small_grid(block_size),
                                     attrs, with_weight ? 2 : 1);
}

// Allocate blocks (i, 0, 0) for i in [0, count) and return the compacted set
// sorted by x, so entry i is block i whatever order the compaction produced.
vr::Result<std::vector<vol::BlockIndex>> allocate_row(vol::VoxelBlockGrid& grid,
                                                      int count) {
  std::vector<vol::BlockIndex> coords(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    coords[static_cast<std::size_t>(i)].coord = vr::Vec3i(i, 0, 0);
  }
  VR_ASSIGN(std::uint32_t failed,
            grid.map().allocate(coords.data(),
                                static_cast<std::uint32_t>(coords.size())));
  if (failed != 0) {
    return vr::Status::invalid_argument("allocate_row: allocation failed");
  }
  VR_ASSIGN(std::vector<vol::BlockIndex> active,
            grid.map().compact_active_blocks());
  std::sort(active.begin(), active.end(),
            [](const vol::BlockIndex& a, const vol::BlockIndex& b) {
              return a.coord.x < b.coord.x;
            });
  return active;
}

float* attr(vol::VoxelBlockGrid& grid, const char* name) {
  return static_cast<float*>(grid.attribute(name).value().buffer->mapped());
}

// Write a block's content: tsdf = s * trunc, weight = w (per voxel).
void write_block(vol::VoxelBlockGrid& grid, const vol::BlockIndex& b,
                 const Cube& s, const std::function<float(std::uint32_t)>& w) {
  float* tsdf = attr(grid, "tsdf");
  float* weight = attr(grid, "weight");
  for (std::uint32_t v = 0; v < kVpb; ++v) {
    tsdf[std::uint32_t(b.ptr) + v] = static_cast<float>(s[v]) * kTrunc;
    weight[std::uint32_t(b.ptr) + v] = w(v);
  }
}

float observed(std::uint32_t) { return 2.0f; }

// Read a block back in normalized units.
Cube read_block(vol::VoxelBlockGrid& grid, const vol::BlockIndex& b) {
  const float* tsdf = attr(grid, "tsdf");
  Cube s{};
  for (std::uint32_t v = 0; v < kVpb; ++v) {
    s[v] = double(tsdf[std::uint32_t(b.ptr) + v]) / double(kTrunc);
  }
  return s;
}

// --- Block content, in normalized units. -----------------------------------

Cube random_cube(std::uint32_t seed) {
  Cube s{};
  std::uint32_t state = seed * 2654435761u + 1u;
  for (double& x : s) {
    state = state * 1664525u + 1013904223u;
    x = double(state >> 8) / double(1u << 24) * 2.0 - 1.0;  // [-1, 1)
  }
  return s;
}

// A tilted plane: smooth, most energy in the lowest frequencies, and inside
// +-0.6, so no reconstruction below rings past the decoder's +-1 clamp.
Cube plane_cube() {
  Cube s{};
  for (std::uint32_t v = 0; v < kVpb; ++v) {
    const double x = v % 8, z = v / 64;
    s[v] = (z - 3.5) * 0.12 + (x - 3.5) * 0.05;
  }
  return s;
}

Cube sphere_cube() {
  Cube s{};
  for (std::uint32_t v = 0; v < kVpb; ++v) {
    const double x = v % 8 - 3.5, y = (v / 8) % 8 - 3.5, z = v / 64 - 3.5;
    s[v] =
        std::clamp((std::sqrt(x * x + y * y + z * z) - 2.5) * 0.3, -1.0, 1.0);
  }
  return s;
}

Cube constant_cube(double c) {
  Cube s{};
  s.fill(c);
  return s;
}

// --- The double-precision reference: the same separable transform. ---------

Cube reference_dct(const Cube& in, bool inverse) {
  const std::array<double, 64> b = codec::detail::dct_basis<double>();
  Cube a = in;
  Cube out{};
  const std::uint32_t strides[3] = {1, 8, 64};
  for (std::uint32_t stride : strides) {
    for (std::uint32_t v = 0; v < kVpb; ++v) {
      const std::uint32_t pos = (v / stride) % 8;  // this voxel's index on axis
      const std::uint32_t base = v - pos * stride;
      double acc = 0.0;
      for (std::uint32_t i = 0; i < 8; ++i) {
        acc +=
            (inverse ? b[i * 8 + pos] : b[pos * 8 + i]) * a[base + i * stride];
      }
      out[v] = acc;
    }
    a = out;
  }
  return a;
}

double rms_diff(const Cube& a, const Cube& b) {
  double sum = 0.0;
  for (std::uint32_t v = 0; v < kVpb; ++v) {
    sum += (a[v] - b[v]) * (a[v] - b[v]);
  }
  return std::sqrt(sum / kVpb);
}

// Check a forward output against the reference: every quantized value is
// the reference's rounded to nearest-even, except where the reference sits so
// close to a half-way point that float error may round it the other way --
// and even there it is one of the two neighbours. Returns non-zero on failure;
// counts the exact and near-half comparisons.
int check_against_reference(const std::vector<std::int32_t>& coeffs,
                            const Cube* content, std::size_t count,
                            const codec::CodecParams& params, int& exact,
                            int& near_half) {
  const auto zigzag = codec::detail::zigzag_order();
  const std::size_t k = params.coefficient_count;
  CHECK(coeffs.size() == count * k);
  for (std::size_t i = 0; i < count; ++i) {
    const Cube ref = reference_dct(content[i], false);
    for (std::uint32_t j = 0; j < k; ++j) {
      const double step = j == 0 ? params.dc_step : params.ac_step;
      const double r = ref[zigzag[j]] / step;
      const double q = coeffs[i * k + j];
      CHECK(std::fabs(q - r) <= 0.5 + 0.05);
      const double frac = r - std::floor(r);
      if (std::fabs(frac - 0.5) > 0.05) {
        CHECK(q == std::nearbyint(r));  // default rounding: half-to-even
        ++exact;
      } else {
        ++near_half;
      }
    }
  }
  return 0;
}

// --- Cases. ------------------------------------------------------------------

// The kernel computes the reference's coefficients (see
// check_against_reference), over content from white noise to a constant.
int forward_matches_reference_case(vr::Device& device, vr::Allocator& allocator,
                                   DctTransform& t) {
  vr::Result<vol::VoxelBlockGrid> g = make_grid(device, allocator);
  CHECK(g.ok());
  vol::VoxelBlockGrid grid = std::move(g).value();
  vr::Result<std::vector<vol::BlockIndex>> active = allocate_row(grid, 4);
  CHECK(active.ok() && active.value().size() == 4);
  const std::vector<vol::BlockIndex>& blocks = active.value();
  const Cube content[4] = {random_cube(1), plane_cube(), sphere_cube(),
                           constant_cube(0.3)};
  for (int i = 0; i < 4; ++i) {
    write_block(grid, blocks[std::size_t(i)], content[i], observed);
  }

  codec::CodecParams params;
  params.coefficient_count = kVpb;
  params.dc_step = 1e-3f;
  params.ac_step = 2e-3f;
  std::vector<std::int32_t> coeffs;
  std::vector<std::uint32_t> masks;
  CHECK(t.forward(grid, grid.block_list(blocks), params, coeffs, masks).ok());
  CHECK(coeffs.size() == 4u * kVpb);
  CHECK(masks.size() == 4u * codec::kMaskWordsPerBlock);

  int exact = 0;
  int near_half = 0;
  CHECK(check_against_reference(coeffs, content, 4, params, exact, near_half) ==
        0);
  // The exclusion is a sliver, not a loophole.
  CHECK(exact > 20 * near_half);
  // Every voxel was observed.
  for (std::uint32_t m : masks) {
    CHECK(m == 0xFFFFFFFFu);
  }
  return 0;
}

// Orthonormal, so the reconstruction error's norm IS the quantization error's
// norm: with all 512 coefficients kept, each off by at most half its step,
// a block's RMS error is at most sqrt(((dc/2)^2 + 511 (ac/2)^2) / 512).
int round_trip_bound_case(vr::Device& device, vr::Allocator& allocator,
                          DctTransform& t) {
  vr::Result<vol::VoxelBlockGrid> g = make_grid(device, allocator);
  CHECK(g.ok());
  vol::VoxelBlockGrid grid = std::move(g).value();
  vr::Result<std::vector<vol::BlockIndex>> active = allocate_row(grid, 4);
  CHECK(active.ok());
  const std::vector<vol::BlockIndex>& blocks = active.value();
  Cube content[4];
  for (int i = 0; i < 4; ++i) {
    content[i] = random_cube(100u + std::uint32_t(i));
    write_block(grid, blocks[std::size_t(i)], content[i], observed);
  }

  codec::CodecParams params;
  params.coefficient_count = kVpb;
  params.dc_step = 0.02f;
  params.ac_step = 0.01f;
  std::vector<std::int32_t> coeffs;
  std::vector<std::uint32_t> masks;
  const vol::BlockList list = grid.block_list(blocks);
  CHECK(t.forward(grid, list, params, coeffs, masks).ok());
  CHECK(t.inverse(grid, list, params, coeffs, masks).ok());

  const double dc = params.dc_step, ac = params.ac_step;
  const double bound =
      std::sqrt((dc * dc / 4.0 + 511.0 * ac * ac / 4.0) / kVpb) + 1e-5;
  const float* weight = attr(grid, "weight");
  for (int i = 0; i < 4; ++i) {
    const double err =
        rms_diff(read_block(grid, blocks[std::size_t(i)]), content[i]);
    CHECK(err <= bound);
    CHECK(err > 0.0);  // it did quantize
    for (std::uint32_t v = 0; v < kVpb; ++v) {
      CHECK(weight[std::uint32_t(blocks[std::size_t(i)].ptr) + v] ==
            codec::detail::kDecodedWeight);
    }
  }
  return 0;
}

// Keeping only the first K coefficients loses exactly the energy of the rest,
// so the kernel pair's RMS error at each K must match what the reference's
// coefficients predict -- which is what pins both kernels to one zigzag order
// and one prefix. Steps at the floor keep quantization out of the picture.
int truncation_matches_reference_case(vr::Device& device,
                                      vr::Allocator& allocator,
                                      DctTransform& t) {
  vr::Result<vol::VoxelBlockGrid> g = make_grid(device, allocator);
  CHECK(g.ok());
  vol::VoxelBlockGrid grid = std::move(g).value();
  vr::Result<std::vector<vol::BlockIndex>> active = allocate_row(grid, 1);
  CHECK(active.ok());
  const vol::BlockIndex block = active.value()[0];
  const Cube content = plane_cube();
  const Cube ref = reference_dct(content, false);
  const auto zigzag = codec::detail::zigzag_order();

  // The cumulative sizes of the zigzag's total-frequency bands, and all 512.
  const std::uint32_t ks[] = {1, 4, 10, 20, 35, 84, 512};
  double last = 1e9;
  for (std::uint32_t k : ks) {
    write_block(grid, block, content, observed);
    codec::CodecParams params;
    params.coefficient_count = k;
    params.dc_step = codec::kMinStep;
    params.ac_step = codec::kMinStep;
    std::vector<std::int32_t> coeffs;
    std::vector<std::uint32_t> masks;
    const vol::BlockList list = grid.block_list(active.value());
    CHECK(t.forward(grid, list, params, coeffs, masks).ok());
    CHECK(coeffs.size() == k);
    CHECK(t.inverse(grid, list, params, coeffs, masks).ok());

    double dropped = 0.0;
    for (std::uint32_t j = k; j < kVpb; ++j) {
      dropped += ref[zigzag[j]] * ref[zigzag[j]];
    }
    const double predicted = std::sqrt(dropped / kVpb);
    const double err = rms_diff(read_block(grid, block), content);
    CHECK(std::fabs(err - predicted) <= 1e-3);
    CHECK(err <= last + 1e-4);  // more coefficients never hurt
    last = err;
  }
  CHECK(last <= 1e-3);  // K = 512 is near-lossless
  return 0;
}

// The mask marks exactly weight >= kObservedWeight, and the inverse writes an
// observed voxel as decoded and an unobserved one as a fresh block holds it.
int mask_case(vr::Device& device, vr::Allocator& allocator, DctTransform& t) {
  vr::Result<vol::VoxelBlockGrid> g = make_grid(device, allocator);
  CHECK(g.ok());
  vol::VoxelBlockGrid grid = std::move(g).value();
  vr::Result<std::vector<vol::BlockIndex>> active = allocate_row(grid, 2);
  CHECK(active.ok());
  const std::vector<vol::BlockIndex>& blocks = active.value();
  // Block 0 cycles through: none, below the threshold, at it, well above.
  // Block 1 is entirely unobserved -- a mask of all zeros.
  const float cycle[4] = {0.0f, 5e-7f, codec::detail::kObservedWeight, 3.0f};
  write_block(grid, blocks[0], sphere_cube(),
              [&](std::uint32_t v) { return cycle[v % 4]; });
  write_block(grid, blocks[1], random_cube(7),
              [](std::uint32_t) { return 0.0f; });

  codec::CodecParams params;
  std::vector<std::int32_t> coeffs;
  std::vector<std::uint32_t> masks;
  const vol::BlockList list = grid.block_list(blocks);
  CHECK(t.forward(grid, list, params, coeffs, masks).ok());
  for (std::uint32_t v = 0; v < kVpb; ++v) {
    const bool bit = ((masks[v / 32] >> (v % 32)) & 1u) != 0u;
    CHECK(bit == (v % 4 >= 2));
  }
  for (std::uint32_t w = 0; w < codec::kMaskWordsPerBlock; ++w) {
    CHECK(masks[codec::kMaskWordsPerBlock + w] == 0u);
  }

  // Poison both attributes so the inverse must write every voxel.
  float* tsdf = attr(grid, "tsdf");
  float* weight = attr(grid, "weight");
  for (const vol::BlockIndex& b : blocks) {
    for (std::uint32_t v = 0; v < kVpb; ++v) {
      tsdf[std::uint32_t(b.ptr) + v] = 123.0f;
      weight[std::uint32_t(b.ptr) + v] = 123.0f;
    }
  }
  CHECK(t.inverse(grid, list, params, coeffs, masks).ok());
  for (std::size_t i = 0; i < 2; ++i) {
    const std::uint32_t base = std::uint32_t(blocks[i].ptr);
    for (std::uint32_t v = 0; v < kVpb; ++v) {
      const bool obs = i == 0 && v % 4 >= 2;
      if (obs) {
        CHECK(weight[base + v] == codec::detail::kDecodedWeight);
        CHECK(std::fabs(tsdf[base + v]) <= kTrunc);
      } else {
        CHECK(weight[base + v] == 0.0f);
        CHECK(tsdf[base + v] == 0.0f);
      }
    }
  }
  return 0;
}

// A constant block is DC-only, and at the floor step a block saturated at +1
// lands on the clamp's edge without being clamped from further out.
int constant_case(vr::Device& device, vr::Allocator& allocator,
                  DctTransform& t) {
  vr::Result<vol::VoxelBlockGrid> g = make_grid(device, allocator);
  CHECK(g.ok());
  vol::VoxelBlockGrid grid = std::move(g).value();
  vr::Result<std::vector<vol::BlockIndex>> active = allocate_row(grid, 2);
  CHECK(active.ok());
  const std::vector<vol::BlockIndex>& blocks = active.value();
  write_block(grid, blocks[0], constant_cube(0.3), observed);
  write_block(grid, blocks[1], constant_cube(1.0), observed);
  const vol::BlockList list = grid.block_list(blocks);

  codec::CodecParams full;
  full.coefficient_count = kVpb;
  full.dc_step = codec::kMinStep;
  full.ac_step = 1e-3f;
  std::vector<std::int32_t> coeffs;
  std::vector<std::uint32_t> masks;
  CHECK(t.forward(grid, list, full, coeffs, masks).ok());
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::uint32_t j = 1; j < kVpb; ++j) {
      CHECK(coeffs[i * kVpb + j] == 0);
    }
  }
  // sqrt(512) / kMinStep ~= 32767: at the edge, and inside it.
  CHECK(coeffs[kVpb] <= codec::kMaxQuantizedMagnitude);
  CHECK(coeffs[kVpb] >= codec::kMaxQuantizedMagnitude - 1);

  // DC alone reconstructs the constant to within half a DC step, spread over
  // the block: |error| <= dc_step / (2 sqrt(512)).
  codec::CodecParams dc_only;
  dc_only.coefficient_count = 1;
  dc_only.dc_step = 0.01f;
  CHECK(t.forward(grid, list, dc_only, coeffs, masks).ok());
  CHECK(coeffs.size() == 2);
  CHECK(t.inverse(grid, list, dc_only, coeffs, masks).ok());
  const Cube back = read_block(grid, blocks[0]);
  const double tol = 0.01 / (2.0 * std::sqrt(512.0)) + 1e-6;
  for (double s : back) {
    CHECK(std::fabs(s - 0.3) <= tol);
  }
  return 0;
}

// The decoder clamps to +-trunc_dist, the range the integrator keeps: a sharp
// edge reconstructed from few coefficients rings past it (a +-1 step along z
// at K = 4 peaks near 1.26), and a decoded grid must not hand a later fuse or
// a consumer SDF outside the band.
int clamp_case(vr::Device& device, vr::Allocator& allocator, DctTransform& t) {
  vr::Result<vol::VoxelBlockGrid> g = make_grid(device, allocator);
  CHECK(g.ok());
  vol::VoxelBlockGrid grid = std::move(g).value();
  vr::Result<std::vector<vol::BlockIndex>> active = allocate_row(grid, 1);
  CHECK(active.ok());
  const vol::BlockIndex block = active.value()[0];
  Cube edge{};
  for (std::uint32_t v = 0; v < kVpb; ++v) {
    edge[v] = v / 64 < 4 ? 1.0 : -1.0;
  }
  write_block(grid, block, edge, observed);

  codec::CodecParams params;
  params.coefficient_count = 4;
  params.dc_step = codec::kMinStep;
  params.ac_step = codec::kMinStep;
  std::vector<std::int32_t> coeffs;
  std::vector<std::uint32_t> masks;
  const vol::BlockList list = grid.block_list(active.value());
  CHECK(t.forward(grid, list, params, coeffs, masks).ok());
  // The unclamped reconstruction does overshoot -- the premise of the case.
  Cube kept{};
  const Cube ref = reference_dct(edge, false);
  const auto zigzag = codec::detail::zigzag_order();
  for (std::uint32_t j = 0; j < 4; ++j) {
    kept[zigzag[j]] = ref[zigzag[j]];
  }
  const Cube unclamped = reference_dct(kept, true);
  CHECK(*std::max_element(unclamped.begin(), unclamped.end()) > 1.2);

  CHECK(t.inverse(grid, list, params, coeffs, masks).ok());
  const float* tsdf = attr(grid, "tsdf");
  bool saturated = false;
  for (std::uint32_t v = 0; v < kVpb; ++v) {
    const float sdf = tsdf[std::uint32_t(block.ptr) + v];
    CHECK(std::fabs(sdf) <= kTrunc);
    saturated = saturated || std::fabs(sdf) == kTrunc;
  }
  CHECK(saturated);
  return 0;
}

// Output follows the list, not the ptrs: permuting the list permutes the rows.
int order_follows_list_case(vr::Device& device, vr::Allocator& allocator,
                            DctTransform& t) {
  vr::Result<vol::VoxelBlockGrid> g = make_grid(device, allocator);
  CHECK(g.ok());
  vol::VoxelBlockGrid grid = std::move(g).value();
  vr::Result<std::vector<vol::BlockIndex>> active = allocate_row(grid, 3);
  CHECK(active.ok());
  std::vector<vol::BlockIndex> blocks = active.value();
  for (std::size_t i = 0; i < 3; ++i) {
    write_block(grid, blocks[i], random_cube(40u + std::uint32_t(i)), observed);
  }
  codec::CodecParams params;
  std::vector<std::int32_t> a_coeffs, b_coeffs;
  std::vector<std::uint32_t> a_masks, b_masks;
  CHECK(
      t.forward(grid, grid.block_list(blocks), params, a_coeffs, a_masks).ok());
  const std::vector<vol::BlockIndex> rotated = {blocks[2], blocks[0],
                                                blocks[1]};
  CHECK(t.forward(grid, grid.block_list(rotated), params, b_coeffs, b_masks)
            .ok());
  const std::size_t k = params.coefficient_count;
  const std::size_t from[3] = {2, 0, 1};
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < k; ++j) {
      CHECK(b_coeffs[i * k + j] == a_coeffs[from[i] * k + j]);
    }
  }
  return 0;
}

// A list longer than one dispatch's batch runs as several, with the same
// result -- the path a list past maxComputeWorkGroupCount[0] takes.
int batching_case(vr::Device& device, vr::Allocator& allocator,
                  DctTransform& t) {
  DctTransformConfig small;
  small.max_blocks_per_dispatch = 3;
  vr::Result<DctTransform> batched_r =
      DctTransform::create(device, allocator, small);
  CHECK(batched_r.ok());
  DctTransform batched = std::move(batched_r).value();

  vr::Result<vol::VoxelBlockGrid> g = make_grid(device, allocator);
  CHECK(g.ok());
  vol::VoxelBlockGrid grid = std::move(g).value();
  vr::Result<std::vector<vol::BlockIndex>> active = allocate_row(grid, 10);
  CHECK(active.ok() && active.value().size() == 10);
  const std::vector<vol::BlockIndex>& blocks = active.value();
  std::vector<Cube> content;
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    content.push_back(random_cube(200u + std::uint32_t(i)));
    write_block(grid, blocks[i], content.back(), observed);
  }
  const vol::BlockList list = grid.block_list(blocks);
  codec::CodecParams params;
  std::vector<std::int32_t> one_coeffs, many_coeffs;
  std::vector<std::uint32_t> one_masks, many_masks;
  // Batched first, and judged against the reference rather than only against
  // the single run: a freed buffer VMA hands straight back still holds the
  // previous run's output, so equality alone passes a batch that wrote
  // nothing past its first entries.
  CHECK(batched.forward(grid, list, params, many_coeffs, many_masks).ok());
  int exact = 0;
  int near_half = 0;
  CHECK(check_against_reference(many_coeffs, content.data(), content.size(),
                                params, exact, near_half) == 0);
  for (std::uint32_t m : many_masks) {
    CHECK(m == 0xFFFFFFFFu);
  }
  CHECK(t.forward(grid, list, params, one_coeffs, one_masks).ok());
  CHECK(one_coeffs == many_coeffs);
  CHECK(one_masks == many_masks);

  // The inverse too: every block of the last, partial batch is written.
  CHECK(t.inverse(grid, list, params, one_coeffs, one_masks).ok());
  std::vector<Cube> single;
  for (const vol::BlockIndex& b : blocks) {
    single.push_back(read_block(grid, b));
  }
  float* tsdf = attr(grid, "tsdf");
  for (const vol::BlockIndex& b : blocks) {
    for (std::uint32_t v = 0; v < kVpb; ++v) {
      tsdf[std::uint32_t(b.ptr) + v] = 9.0f;
    }
  }
  CHECK(batched.inverse(grid, list, params, many_coeffs, many_masks).ok());
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    CHECK(read_block(grid, blocks[i]) == single[i]);
  }
  return 0;
}

// Every input the kernels would index unchecked is refused on the host.
int refusals_case(vr::Device& device, vr::Allocator& allocator,
                  DctTransform& t) {
  vr::Result<vol::VoxelBlockGrid> g = make_grid(device, allocator);
  CHECK(g.ok());
  vol::VoxelBlockGrid grid = std::move(g).value();
  vr::Result<std::vector<vol::BlockIndex>> active = allocate_row(grid, 3);
  CHECK(active.ok());
  const std::vector<vol::BlockIndex> blocks = active.value();
  const codec::CodecParams params;
  std::vector<std::int32_t> coeffs = {1, 2, 3};
  std::vector<std::uint32_t> masks = {4, 5};

  // An empty list is a no-op success, and it empties the outputs.
  CHECK(t.forward(grid, vol::BlockList{}, params, coeffs, masks).ok());
  CHECK(coeffs.empty() && masks.empty());
  CHECK(t.inverse(grid, vol::BlockList{}, params, coeffs, masks).ok());

  // Invalid params.
  codec::CodecParams bad = params;
  bad.coefficient_count = 0;
  CHECK(!t.forward(grid, grid.block_list(blocks), bad, coeffs, masks).ok());

  // Null with a count.
  vol::BlockList null_list{nullptr, 3, grid.topology_epoch()};
  CHECK(!t.forward(grid, null_list, params, coeffs, masks).ok());

  // A ptr outside the heap, negative, or not on a block boundary. Each would
  // be an out-of-bounds device access, and a write on the inverse.
  const std::int32_t bad_ptrs[] = {small_grid().num_blocks * 512, -512, 1};
  for (std::int32_t p : bad_ptrs) {
    std::vector<vol::BlockIndex> forged = blocks;
    forged[1].ptr = p;
    coeffs = {1};
    CHECK(
        !t.forward(grid, grid.block_list(forged), params, coeffs, masks).ok());
    CHECK(coeffs.empty());  // a refusal leaves the outputs empty
  }

  // Size mismatches on the inverse.
  CHECK(t.forward(grid, grid.block_list(blocks), params, coeffs, masks).ok());
  std::vector<std::int32_t> short_coeffs(coeffs.begin(), coeffs.end() - 1);
  CHECK(!t.inverse(grid, grid.block_list(blocks), params, short_coeffs, masks)
             .ok());
  std::vector<std::uint32_t> short_masks(masks.begin(), masks.end() - 1);
  CHECK(!t.inverse(grid, grid.block_list(blocks), params, coeffs, short_masks)
             .ok());

  // A list compacted before a remove() names a different block through a
  // still in-range ptr, and only the epoch can tell.
  const vol::BlockList stale = grid.block_list(blocks);
  vol::BlockIndex gone{};
  gone.coord = blocks[0].coord;
  vr::Result<std::uint32_t> removed = grid.remove(&gone, 1);
  CHECK(removed.ok());
  CHECK(!t.forward(grid, stale, params, coeffs, masks).ok());
  CHECK(!t.inverse(grid, stale, params, coeffs, masks).ok());

  // A grid with another block size.
  vr::Result<vol::VoxelBlockGrid> g4 = make_grid(device, allocator, 4);
  CHECK(g4.ok());
  vol::VoxelBlockGrid grid4 = std::move(g4).value();
  std::vector<vol::BlockIndex> one(1);
  one[0].coord = vr::Vec3i(0, 0, 0);
  CHECK(grid4.map().allocate(one.data(), 1).ok());
  vr::Result<std::vector<vol::BlockIndex>> active4 =
      grid4.map().compact_active_blocks();
  CHECK(active4.ok());
  CHECK(!t.forward(grid4, grid4.block_list(active4.value()), params, coeffs,
                   masks)
             .ok());

  // A grid without a weight attribute.
  vr::Result<vol::VoxelBlockGrid> gw = make_grid(device, allocator, 8, false);
  CHECK(gw.ok());
  vol::VoxelBlockGrid no_weight = std::move(gw).value();
  CHECK(no_weight.map().allocate(one.data(), 1).ok());
  vr::Result<std::vector<vol::BlockIndex>> active_w =
      no_weight.map().compact_active_blocks();
  CHECK(active_w.ok());
  CHECK(!t.forward(no_weight, no_weight.block_list(active_w.value()), params,
                   coeffs, masks)
             .ok());
  return 0;
}

// The move-only type rules: a moved-from transform is empty, a move-assign
// over a live one works, and a self-move leaves it intact.
int moves_case(vr::Device& device, vr::Allocator& allocator) {
  vr::Result<DctTransform> a_r = DctTransform::create(device, allocator);
  CHECK(a_r.ok());
  DctTransform a = std::move(a_r).value();
  CHECK(a.valid());

  DctTransform b(std::move(a));
  CHECK(b.valid());
  CHECK(!a.valid());  // NOLINT(bugprone-use-after-move): asserting the source

  vr::Result<DctTransform> c_r = DctTransform::create(device, allocator);
  CHECK(c_r.ok());
  DctTransform c = std::move(c_r).value();
  c = std::move(b);  // over a live transform
  CHECK(c.valid());
  CHECK(!b.valid());  // NOLINT(bugprone-use-after-move)

  DctTransform* alias = &c;
  c = std::move(*alias);  // self-move, laundered past -Wself-move
  CHECK(c.valid());

  // The survivor still works, and a moved-from one refuses rather than
  // dispatching through a null pipeline.
  vr::Result<vol::VoxelBlockGrid> g = make_grid(device, allocator);
  CHECK(g.ok());
  vol::VoxelBlockGrid grid = std::move(g).value();
  vr::Result<std::vector<vol::BlockIndex>> active = allocate_row(grid, 1);
  CHECK(active.ok());
  std::vector<std::int32_t> coeffs;
  std::vector<std::uint32_t> masks;
  const vol::BlockList list = grid.block_list(active.value());
  CHECK(c.forward(grid, list, codec::CodecParams{}, coeffs, masks).ok());
  CHECK(!a.forward(grid, list, codec::CodecParams{}, coeffs, masks).ok());
  return 0;
}

}  // namespace

int main() {
  vr::Result<vr::Instance> instance = vr::Instance::create({});
  if (!instance) {
    std::fprintf(stderr, "no Vulkan instance (%s); skipping\n",
                 instance.status().message().c_str());
    return 0;
  }
  vr::Result<VkPhysicalDevice> gpu = instance.value().select_physical_device();
  if (!gpu) {
    std::fprintf(stderr, "no compute-capable device (%s); skipping\n",
                 gpu.status().message().c_str());
    return 0;
  }
  vr::Result<vr::Device> device =
      vr::Device::create(instance.value(), gpu.value(), {});
  if (!device) {
    std::fprintf(stderr, "device create failed: %s\n",
                 device.status().message().c_str());
    return 1;
  }
  vr::Result<vr::Allocator> allocator =
      vr::Allocator::create(instance.value().handle(), device.value());
  if (!allocator) {
    std::fprintf(stderr, "allocator create failed: %s\n",
                 allocator.status().message().c_str());
    return 1;
  }
  vr::Device& dev = device.value();
  vr::Allocator& alloc = allocator.value();

  vr::Result<DctTransform> t_r = DctTransform::create(dev, alloc);
  if (!t_r) {
    std::fprintf(stderr, "DctTransform::create failed: %s\n",
                 t_r.status().message().c_str());
    return 1;
  }
  DctTransform t = std::move(t_r).value();

  if (forward_matches_reference_case(dev, alloc, t) != 0) return 1;
  if (round_trip_bound_case(dev, alloc, t) != 0) return 1;
  if (truncation_matches_reference_case(dev, alloc, t) != 0) return 1;
  if (mask_case(dev, alloc, t) != 0) return 1;
  if (constant_case(dev, alloc, t) != 0) return 1;
  if (clamp_case(dev, alloc, t) != 0) return 1;
  if (order_follows_list_case(dev, alloc, t) != 0) return 1;
  if (batching_case(dev, alloc, t) != 0) return 1;
  if (refusals_case(dev, alloc, t) != 0) return 1;
  if (moves_case(dev, alloc) != 0) return 1;
  std::printf("codec DctTransform: OK\n");
  return 0;
}
