// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Synthetic intra frames for the codec's frame tests: the host writer's
// round trips and the device writer's byte comparison draw the same frames,
// and both writers must write the golden frames' pinned bytes.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <vector>

#include "bitstream.hpp"
#include "fnv1a.hpp"
#include "volumetric_kit/recon/codec/codec_params.hpp"

namespace codec_frames {

namespace vr = volumetric_kit::recon;
namespace codec = volumetric_kit::recon::codec;
namespace d = volumetric_kit::recon::codec::detail;

struct Lcg {
  std::uint64_t state;
  std::uint32_t next() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<std::uint32_t>(state >> 32);
  }
  std::uint32_t below(std::uint32_t n) { return next() % n; }
};

// A partial mask shaped like a fused band's edge: the voxels on one side of a
// plane through the block, so its planes and lines repeat, run full or empty,
// or cut across -- every symbol the mask code has.
inline void slab_mask(Lcg& rng, std::uint32_t* mask) {
  const int a = int(rng.below(7)) - 3;
  const int b = int(rng.below(7)) - 3;
  const int c = int(rng.below(7)) - 3;
  const int d = int(rng.below(40)) - 20;
  for (std::uint32_t w = 0; w < codec::kMaskWordsPerBlock; ++w) {
    mask[w] = 0;
  }
  for (std::uint32_t v = 0; v < codec::kVoxelsPerBlock; ++v) {
    const int x = int(v % 8), y = int(v / 8 % 8), z = int(v / 64);
    if (a * x + b * y + c * z < d) {
      mask[v / 32] |= 1u << (v % 32);
    }
  }
}

// A frame shaped like a real one: blocks in runs along x with jumps between
// them, masks mostly full with some empty and some partial -- noise, or a
// slab's edge -- coefficients small and sparser at higher indices, and the
// odd extreme value.
inline d::IntraFrame make_frame(std::size_t n, std::uint32_t k,
                                std::uint64_t seed) {
  Lcg rng{seed};
  auto less = [](const vr::Vec3i& a, const vr::Vec3i& b) {
    return d::coord_less(a, b);
  };
  std::set<vr::Vec3i, decltype(less)> coords(less);
  while (coords.size() < n) {
    // One draw per statement: the order a function's arguments are evaluated
    // in is unspecified (GCC and Clang disagree), and a fixture must be the
    // same frame on every compiler.
    const int sx = int(rng.below(200)) - 100;
    const int sy = int(rng.below(60)) - 30;
    const int sz = int(rng.below(60)) - 30;
    const vr::Vec3i start(sx, sy, sz);
    const std::uint32_t run = 1 + rng.below(12);
    for (std::uint32_t i = 0; i < run && coords.size() < n; ++i) {
      coords.insert(vr::Vec3i(start.x + int(i), start.y, start.z));
    }
  }
  d::IntraFrame f;
  f.voxel_size = 0.005f;
  f.coords.assign(coords.begin(), coords.end());
  f.blocks.params.coefficient_count = k;
  f.blocks.trunc_dist = 0.04f;
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint32_t cls = rng.below(10);
    std::uint32_t mask[codec::kMaskWordsPerBlock];
    if (cls < 8) {
      for (std::uint32_t& w : mask) {
        w = cls < 6 ? ~0u : cls < 7 ? 0u : rng.next();
      }
    } else {
      slab_mask(rng, mask);
    }
    f.blocks.masks.insert(f.blocks.masks.end(), mask,
                          mask + codec::kMaskWordsPerBlock);
    for (std::uint32_t j = 0; j < k; ++j) {
      std::int32_t v = 0;
      const std::uint32_t spread = 1 + 64 / (1 + j);
      if (rng.below(1 + j / 4) == 0) {
        v = int(rng.below(2 * spread + 1)) - int(spread);
      }
      if (rng.below(500) == 0) {
        v = rng.below(2) == 0 ? codec::kMaxQuantizedMagnitude
                              : -codec::kMaxQuantizedMagnitude;
      }
      f.blocks.coefficients.push_back(static_cast<std::int16_t>(v));
    }
  }
  return f;
}

// Coordinates at the ends of int32, and deltas of 2^32 - 1 on every axis.
inline d::IntraFrame extreme_frame() {
  constexpr std::int32_t kMax = std::numeric_limits<std::int32_t>::max();
  constexpr std::int32_t kMin = std::numeric_limits<std::int32_t>::min();
  d::IntraFrame f = make_frame(0, 4, 1);
  f.coords = {
      {kMin, kMin, kMin}, {kMax, kMin, kMin},  // dx = 2^32 - 1
      {kMin, kMax, kMin},                      // dy, dx back
      {0, 0, kMax},                            // dz
      {1, 0, kMax},       {kMax, kMax, kMax},
  };
  f.blocks.masks.assign(f.coords.size() * codec::kMaskWordsPerBlock, ~0u);
  f.blocks.coefficients.assign(f.coords.size() * 4, 0);
  return f;
}

// What a frame holds, hashed: coordinates, masks and coefficients.
inline std::uint32_t content_hash(const d::IntraFrame& f) {
  std::vector<std::uint32_t> words;
  for (const vr::Vec3i& c : f.coords) {
    words.insert(words.end(), {static_cast<std::uint32_t>(c.x),
                               static_cast<std::uint32_t>(c.y),
                               static_cast<std::uint32_t>(c.z)});
  }
  words.insert(words.end(), f.blocks.masks.begin(), f.blocks.masks.end());
  for (std::int16_t c : f.blocks.coefficients) {
    words.push_back(static_cast<std::uint16_t>(c));
  }
  return vr_test::fnv1a(words);
}

// --- Golden frames. ---------------------------------------------------------
//
// The v3 bytes every writer, host or device, writes for these frames on every
// machine: what pins the format, where a round trip would pass a writer and a
// reader changed together. `content` pins the fixture, so a failure says which
// changed. A new kFrameVersion brings new values; nothing else may.
static_assert(d::kFrameVersion == 3, "the golden frames are v3's");

struct GoldenFrame {
  bool extreme;  // extreme_frame(), else make_frame(blocks, k, seed)
  std::size_t blocks;
  std::uint32_t k;
  std::uint64_t seed;
  std::uint32_t segment_size;
  std::uint32_t content;  // content_hash of the frame
  std::size_t size;       // its bytes
  std::uint32_t hash;     // vr_test::fnv1a of them
};

// The default K and segment size over many segments, every K with segments
// of 16, an odd K with odd segments, and coordinates at the ends of int32,
// as steps of up to 2^32 - 1 and each in full.
inline constexpr GoldenFrame kGoldenFrames[] = {
    {false, 700, 64, 7, 64, 0x3f9773d8u, 17582, 0x411a37d6u},
    {false, 40, codec::kVoxelsPerBlock, 5, 16, 0xebbf784au, 6520, 0x8e023b27u},
    {false, 65, 21, 3, 7, 0x4fb1073du, 1907, 0x29533280u},
    {true, 6, 4, 0, 64, 0x11b607ddu, 198, 0xe081c9b3u},
    {true, 6, 4, 0, 1, 0x11b607ddu, 236, 0x92cc1c51u},
};

inline d::IntraFrame golden_frame(const GoldenFrame& g) {
  return g.extreme ? extreme_frame() : make_frame(g.blocks, g.k, g.seed);
}

}  // namespace codec_frames
